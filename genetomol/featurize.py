"""Compound featurization.

Compounds -> ECFP4 bits + a small block of physicochemical descriptors.

The point of using *structure* rather than a lookup table is that it is the
only thing that lets the model score a compound it never saw in training.
Keep it that way: if you ever find yourself indexing compounds by id inside an
encoder, you have quietly deleted the contribution of the paper.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Descriptors kept deliberately small and cheap. Anything correlated with
# assay artefacts (e.g. molecular weight alone) is a shortcut the contrastive
# loss will happily exploit, so we standardize these on the train split only.
_DESCRIPTOR_NAMES = [
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "BertzCT",
]

N_DESCRIPTORS = len(_DESCRIPTOR_NAMES)


def _require_rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "rdkit is required for molecule featurization: "
            "conda install -c conda-forge rdkit"
        ) from exc
    RDLogger.DisableLog("rdApp.*")
    return Chem, Descriptors, rdFingerprintGenerator, MurckoScaffold


def featurize_molecules(
    smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_descriptors: bool = True,
    include_chirality: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, valid_mask).

    `include_chirality` is off by default because that is what every existing
    artifact was built with, but it is the single cheapest thing that raises
    the achievable ceiling on this bank: without it every stereoisomer folds
    onto one bit vector, and 3,744 of the 3,770 identical-fingerprint groups
    in LINCS 2020 are stereoisomer families. Rebuilding artifacts with it on
    splits those groups and makes them retrievable at all. See
    `duplicate_group_ids`.

    features has shape (n, n_bits + N_DESCRIPTORS) when use_descriptors, else
    (n, n_bits). Rows where RDKit could not parse the SMILES -- or where a
    descriptor came back non-finite -- are left as all zeros and flagged False
    in valid_mask. Drop them upstream rather than training on zero vectors,
    which otherwise collapse into a single spurious cluster.
    """
    Chem, Descriptors, rdFingerprintGenerator, _ = _require_rdkit()
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=include_chirality
    )

    width = n_bits + (N_DESCRIPTORS if use_descriptors else 0)
    out = np.zeros((len(smiles), width), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)

    desc_fns = [getattr(Descriptors, name) for name in _DESCRIPTOR_NAMES]

    for i, smi in enumerate(smiles):
        if not isinstance(smi, str) or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        desc = None
        if use_descriptors:
            # Compute descriptors *before* writing anything, so a failure here
            # leaves the row all-zero and consistent with valid_mask=False
            # rather than half-written with live fingerprint bits.
            try:
                desc = np.asarray([float(fn(mol)) for fn in desc_fns], dtype=np.float32)
            except Exception:  # noqa: BLE001 - a single bad descriptor kills the row
                continue
            if not np.all(np.isfinite(desc)):
                continue

        out[i, :n_bits] = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        if desc is not None:
            out[i, n_bits:] = desc
        valid[i] = True

    n_bad = int((~valid).sum())
    if n_bad:
        logger.warning("featurize_molecules: %d/%d SMILES unusable", n_bad, len(smiles))
    return out, valid


def bemis_murcko_scaffolds(smiles: Sequence[str], include_chirality: bool = False):
    """Generic Bemis-Murcko scaffold per molecule, for scaffold splitting.

    Unparsable molecules get their own singleton scaffold so they can never
    bridge the train/test boundary.
    """
    Chem, _, _, MurckoScaffold = _require_rdkit()
    scaffolds = []
    for i, smi in enumerate(smiles):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smi, includeChirality=include_chirality
            )
            scaffolds.append(scaf if scaf else f"__empty_{i}")
        except Exception:  # noqa: BLE001
            scaffolds.append(f"__invalid_{i}")
    return scaffolds


def tanimoto_matrix(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """Dense Tanimoto between two sets of binary fingerprints.

    Used by the ECFP nearest-neighbor baseline. Pass *only* the bit block,
    not the descriptor block.
    """
    a = (fp_a > 0).astype(np.float32)
    b = (fp_b > 0).astype(np.float32)
    inter = a @ b.T
    na = a.sum(1, keepdims=True)
    nb = b.sum(1, keepdims=True).T
    union = na + nb - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


class DescriptorScaler:
    """Standardize the descriptor block only. Fit on train compounds.

    Fitting on the full bank would leak test-set statistics into training.

    Two guards, both of which fire on held-out data:

    `min_std`. A descriptor can be constant across the *training* compounds and
    not constant on the test ones. Dividing by a near-zero standard deviation
    would turn a held-out value into a feature of ~1e6 that dominates the
    encoder input, so a column whose standard deviation falls below `min_std`
    is mean-centered and left unscaled instead.

    `clip`. Descriptors are unbounded, so one out-of-distribution molecule can
    still dominate through a legitimately extreme value. Clipping to +/- `clip`
    standard deviations bounds that without touching anything in range.
    """

    def __init__(self, n_bits: int = 2048, clip: float | None = 10.0, min_std: float = 1e-3):
        self.n_bits = n_bits
        self.clip = clip
        self.min_std = min_std
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.degenerate_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "DescriptorScaler":
        block = x[:, self.n_bits :]
        if block.shape[1] == 0:
            self.mean_ = np.zeros(0, dtype=np.float32)
            self.std_ = np.ones(0, dtype=np.float32)
            self.degenerate_ = np.zeros(0, dtype=bool)
            return self
        self.mean_ = block.mean(0)
        std = block.std(0)
        self.degenerate_ = std < self.min_std
        # Constant-on-train columns carry no training signal; center them and
        # leave the scale alone rather than exploding held-out deviations.
        self.std_ = np.where(self.degenerate_, 1.0, std).astype(np.float32)
        if self.degenerate_.any():
            idx = np.flatnonzero(self.degenerate_)
            if block.shape[1] == N_DESCRIPTORS:
                names = [_DESCRIPTOR_NAMES[i] for i in idx]
            else:
                names = [f"column {self.n_bits + i}" for i in idx]
            logger.warning(
                "DescriptorScaler: %d descriptor(s) constant across the fit set "
                "(%s) -- centered but not scaled",
                len(idx), ", ".join(names),
            )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("DescriptorScaler.fit must be called first")
        x = x.copy()
        block = x[:, self.n_bits :]
        if block.shape[1]:
            block = (block - self.mean_) / self.std_
            if self.clip is not None:
                block = np.clip(block, -self.clip, self.clip)
            x[:, self.n_bits :] = block
        return x

    def fit_transform(self, x: np.ndarray, fit_rows: np.ndarray | None = None) -> np.ndarray:
        """Fit on `fit_rows` (the train compounds) and transform everything."""
        self.fit(x if fit_rows is None else x[fit_rows])
        return self.transform(x)


def duplicate_group_ids(
    mol_features: np.ndarray,
    n_bits: int | None = None,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Group id per compound the molecule encoder cannot tell apart; -1 if unique.

    `n_bits` restricts grouping to the leading fingerprint block. Pass None
    (the default) to group on the **full feature matrix** -- descriptors and
    any concatenated chemical-LM embedding included -- which is what the
    encoder actually sees, and therefore the only honest definition of
    indistinguishable.

    `tolerance` > 0 groups rows that are merely *near*-identical, by cosine on
    column-standardized features, joined transitively. This path is O(n^2), so
    it is opt-in.

    Two compounds with the same feature vector are the same point as far as the
    molecule encoder is concerned, so they are not negatives for each other in
    any achievable sense -- `train` masks them out of the InfoNCE denominator
    exactly like same-MOA pairs.

    See `recall_ceiling` for the bound these groups place on any structure-only
    encoder.
    """
    block = np.ascontiguousarray(
        mol_features if n_bits is None else mol_features[:, :n_bits]
    )
    if tolerance > 0.0:
        return _near_duplicate_groups(block, tolerance)

    # Binary blocks pack 8x; anything else hashes its raw bytes, which is exact
    # for float32 and still O(n).
    is_binary = np.array_equal(block, block.astype(bool))
    keys = np.packbits(block > 0, axis=1) if is_binary else block
    members: dict[bytes, list[int]] = {}
    for i, row in enumerate(keys):
        members.setdefault(row.tobytes(), []).append(i)
    out = np.full(len(block), -1, dtype=np.int64)
    gid = 0
    for rows in members.values():
        if len(rows) < 2:
            continue
        out[rows] = gid
        gid += 1
    return out


def _near_duplicate_groups(block: np.ndarray, tolerance: float,
                           chunk: int = 512) -> np.ndarray:
    """Transitively group rows with cosine >= 1 - tolerance, chunked."""
    z = block.astype(np.float32)
    z = z - z.mean(0, keepdims=True)
    z = z / np.maximum(z.std(0, keepdims=True), 1e-6)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)

    parent = np.arange(len(z))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    threshold = 1.0 - tolerance
    for start in range(0, len(z), chunk):
        sim = z[start:start + chunk] @ z.T
        for local, row in enumerate(sim):
            i = start + local
            # Upper triangle only: every pair is visited once.
            for j in np.nonzero(row[i + 1:] >= threshold)[0] + i + 1:
                ra, rb = find(i), find(int(j))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(len(z))])
    out = np.full(len(z), -1, dtype=np.int64)
    gid = 0
    for root in np.unique(roots):
        rows = np.nonzero(roots == root)[0]
        if len(rows) < 2:
            continue
        out[rows] = gid
        gid += 1
    return out


def recall_ceiling(duplicate_ids: np.ndarray, positions: np.ndarray | None = None,
                   ks: tuple[int, ...] = (1, 5, 10, 20, 50)) -> dict[str, float]:
    """Best macro recall@k any structure-only encoder can reach on this bank.

    Members of a fingerprint-identical group are interchangeable to the encoder,
    so the true compound is equally likely to be at any position within its
    group's block. That gives `min(1, k / g)` as the per-compound bound, which
    is generous -- it assumes the group is otherwise ranked perfectly.
    """
    g = duplicate_ids if positions is None else duplicate_ids[positions]
    sizes = np.ones(len(g), dtype=np.float64)
    if (g >= 0).any():
        counts = np.bincount(duplicate_ids[duplicate_ids >= 0])
        sizes = np.where(g >= 0, counts[np.clip(g, 0, None)], 1.0)
    return {f"macro_recall@{k}": float(np.minimum(1.0, k / sizes).mean()) for k in ks}


def main():
    """Recompute `mol_features` from the bank's SMILES, without touching signatures.

    A full `prepare` re-run costs 30-60 minutes and re-reads the 33 GB GCTX to
    regenerate data that did not change. Only the molecule block depends on the
    featurization, so this rewrites just that -- ~24k RDKit calls, a couple of
    minutes -- and `train --mol-features` swaps it in.

        python -m genetomol.featurize --data ../lincs/prepared             --out ../lincs/prepared/ecfp_chiral.npz
    """
    import argparse

    from .data import load_artifacts
    from .molembed import save_embeddings

    ap = argparse.ArgumentParser(description="rebuild mol_features from SMILES")
    ap.add_argument("--data", required=True, help="directory holding artifacts.npz")
    ap.add_argument("--out", required=True, help="destination .npz (carries bank ids)")
    ap.add_argument("--n-bits", type=int, default=None, help="defaults to meta.json")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--no-chirality", action="store_true",
                    help="reproduce the original achiral fingerprints")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bank, _, meta = load_artifacts(args.data)
    n_bits = args.n_bits or meta.get("n_bits", 2048)
    chiral = not args.no_chirality

    missing = sum(1 for s in bank.smiles if not s)
    if missing:
        logger.warning("%d/%d compounds have no SMILES", missing, len(bank))

    logger.info("featurizing %d compounds (radius=%d, n_bits=%d, chirality=%s)",
                len(bank), args.radius, n_bits, chiral)
    feats, valid = featurize_molecules(
        bank.smiles, radius=args.radius, n_bits=n_bits,
        use_descriptors=True, include_chirality=chiral,
    )
    if not valid.all():
        # An all-zero row would collapse into one spurious cluster, so refuse
        # rather than write a matrix that trains to a plausible-looking loss.
        raise SystemExit(
            f"{int((~valid).sum())} compounds failed to featurize; the existing "
            "artifacts were built from these same SMILES, so this is a bug worth "
            "understanding rather than papering over"
        )

    before = duplicate_group_ids(bank.mol_features, n_bits)
    after = duplicate_group_ids(feats, n_bits)
    print(f"\nfingerprint-identical compounds: "
          f"{int((before >= 0).sum())} ({(before >= 0).mean():.1%}) -> "
          f"{int((after >= 0).sum())} ({(after >= 0).mean():.1%})")
    print(f"duplicate groups: {int(before.max()) + 1} -> "
          f"{0 if (after < 0).all() else int(after.max()) + 1}")
    for label, ids in (("before", before), ("after", after)):
        c = recall_ceiling(ids)
        print(f"  macro recall ceiling {label:>6}: "
              + "  ".join(f"@{k.split('@')[1]} {v:.4f}" for k, v in c.items()))

    path = save_embeddings(args.out, bank.ids, feats)
    print(f"\nwrote {path}  shape={feats.shape}")
    print(f"use it with:  --mol-features {path}")


if __name__ == "__main__":
    main()

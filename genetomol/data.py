"""L1000 level-5 loading, the compound bank, and split logic. Compounds only.

Two behaviours worth knowing before changing anything here:

1. TAS filtering. Low-activity signatures act as *false positives* for their own
   compound under InfoNCE, so they are dropped rather than kept. Exemplar
   filtering is OFF by default: it removes extra views of the same compound
   across cell lines and doses, which the encoder needs, since it receives
   neither cell line nor dose as input.

2. The batch sampler. If two signatures of the same pert_id land in one batch
   they become in-batch negatives for each other, so `UniquePertBatchSampler`
   emits at most one signature per compound per batch.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

PERT_TYPE_COMPOUND = "trt_cp"


@dataclass
class CompoundBank:
    """The output space: every compound that can be retrieved.

    `mol_features` is aligned 1:1 with `ids` -- row i is the featurization of
    compound `ids[i]`. There is no id-indexed table anywhere in the encoder
    path; that is the whole point. If you ever find yourself keying an
    embedding by compound id outside of `LookupMoleculeEncoder`, you have
    deleted the contribution of the paper.
    """

    ids: list[str]                              # pert_id
    mol_features: np.ndarray                    # (P, F_mol)
    smiles: list[str | None] = field(default_factory=list)
    moa: list[str | None] = field(default_factory=list)
    targets: list[tuple[str, ...]] = field(default_factory=list)
    # ECFP bits for the *baselines*, kept separate from what the model is fed.
    # `ECFPNearestNeighbor` binarizes whatever it is given, so once the model
    # trains on ChemBERTa alone, slicing `mol_features[:, :n_bits]` would
    # threshold embedding floats and compute a meaningless Tanimoto -- a
    # silently wrong bar, which is worse than a broken one. None means "the
    # leading columns of mol_features are the bits", which is the default case.
    fingerprints: np.ndarray | None = None

    def __post_init__(self) -> None:
        p = len(self.ids)
        if self.mol_features.shape[0] != p:
            raise ValueError(
                f"mol_features has {self.mol_features.shape[0]} rows for {p} ids"
            )
        for name in ("smiles", "moa"):
            if not getattr(self, name):
                setattr(self, name, [None] * p)
        if not self.targets:
            self.targets = [() for _ in range(p)]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def n_features(self) -> int:
        return int(self.mol_features.shape[1])

    def index_of(self, pert_id: str) -> int:
        if not hasattr(self, "_lookup"):
            self._lookup = {p: i for i, p in enumerate(self.ids)}
        return self._lookup[pert_id]

    def primary_targets(self) -> list[str | None]:
        return [t[0] if t else None for t in self.targets]

    def moa_ids(self) -> np.ndarray:
        """Integer MOA label per compound, -1 where unannotated.

        Feed this to `train()` so same-MOA compounds are masked out of the
        InfoNCE denominator instead of being treated as negatives.
        """
        table: dict[str, int] = {}
        out = np.full(len(self.ids), -1, dtype=np.int64)
        for i, m in enumerate(self.moa):
            if m:
                out[i] = table.setdefault(m, len(table))
        return out

    def features_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(self.mol_features, device=device)

    def subset(self, positions: Sequence[int]) -> "CompoundBank":
        pos = list(positions)
        return CompoundBank(
            ids=[self.ids[i] for i in pos],
            mol_features=self.mol_features[pos],
            smiles=[self.smiles[i] for i in pos],
            moa=[self.moa[i] for i in pos],
            targets=[self.targets[i] for i in pos],
        )


class SignatureDataset(Dataset):
    """One item = one L1000 level-5 signature and its context."""

    def __init__(
        self,
        signatures: np.ndarray,      # (N, 978) z-scores
        bank_index: np.ndarray,      # (N,) index into CompoundBank
        cell_index: np.ndarray,      # (N,) categorical cell line
        log_dose: np.ndarray,        # (N,) log10(uM + eps)
        log_time: np.ndarray,        # (N,) log10(hours)
        has_dose: np.ndarray | None = None,
    ):
        if not len(signatures) == len(bank_index) == len(cell_index):
            raise ValueError("signatures, bank_index and cell_index must align")
        # A memmap is kept as-is on purpose: `ascontiguousarray` would pull all
        # 1.45 GB into RAM, which is the whole thing `load_artifacts` is
        # avoiding. Subsetting it reads only the rows a split actually wants.
        if isinstance(signatures, np.memmap) and signatures.dtype == np.float32:
            self.signatures = signatures
        else:
            self.signatures = np.ascontiguousarray(signatures, dtype=np.float32)
        self.bank_index = bank_index.astype(np.int64)
        self.cell_index = cell_index.astype(np.int64)
        self.log_dose = log_dose.astype(np.float32)
        self.log_time = log_time.astype(np.float32)
        self.has_dose = (
            np.ones(len(signatures), dtype=np.float32)
            if has_dose is None
            else has_dose.astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.signatures)

    def release(self) -> None:
        """Drop the arrays. Only call this once the subsets have been taken.

        `subset_dataset` copies, so a split holds a second full copy of the
        signature matrix -- 1.4 GB on LINCS, on top of the 1.4 GB parent that
        nothing reads again. This is opt-in because a caller may split the same
        dataset twice; `run_experiment(release_source=True)` is the one that
        knows it is finished with it.
        """
        self.signatures = np.zeros((0, self.signatures.shape[1]), dtype=np.float32)
        self.bank_index = np.zeros(0, dtype=np.int64)
        self.cell_index = np.zeros(0, dtype=np.int64)
        self.log_dose = np.zeros(0, dtype=np.float32)
        self.log_time = np.zeros(0, dtype=np.float32)
        self.has_dose = np.zeros(0, dtype=np.float32)

    def context_matrix(self) -> np.ndarray:
        """(N, 3) context block in the order SignatureEncoder expects."""
        return np.stack([self.log_dose, self.log_time, self.has_dose], axis=1)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        # `.copy()` only when the source is a read-only memmap: sharing memory
        # with one gives torch a tensor it thinks is writable, and an in-place
        # op on it would fault. One row is 3.9 KB and the default collate copies
        # anyway, so this costs nothing measurable.
        row = self.signatures[i]
        return {
            "signature": torch.from_numpy(row if row.flags.writeable else row.copy()),
            "bank_index": torch.tensor(self.bank_index[i]),
            "cell_index": torch.tensor(self.cell_index[i]),
            "context": torch.tensor(
                [self.log_dose[i], self.log_time[i], self.has_dose[i]],
                dtype=torch.float32,
            ),
        }


class UniquePertBatchSampler(Sampler[list[int]]):
    """Batches in which every signature comes from a different compound.

    Without this, dose series and replicate cell lines of the same compound sit
    in the same batch as negatives of one another and InfoNCE spends its
    capacity separating things that are the same molecule.

    `set_epoch` is the only thing that advances the shuffle; iterating does not
    mutate epoch state, so two passes at the same epoch give the same order.
    """

    def __init__(
        self,
        bank_index: np.ndarray,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = True,
        max_per_epoch: int | None = 60,
    ):
        """`max_per_epoch` caps how many signatures each compound contributes
        per epoch, resampling a fresh subset every epoch.

        The cap is about class balance, not memory. On LINCS 2020 the signature
        count per compound runs from 1 to 6,003 with a median of 3, and this
        sampler draws one signature per compound per batch -- so uncapped, a
        single heavily-profiled drug appears in 6,003 batches an epoch while
        half the bank appears in three. The gradient ends up owned by a handful
        of well-studied compounds.

        Capping *per epoch* rather than subsampling once at prep time means no
        signature is ever permanently discarded: each epoch sees a different
        draw, so over training the model still uses all the data while every
        epoch stays balanced. Set to None to disable.
        """
        self.groups: dict[int, list[int]] = defaultdict(list)
        for i, p in enumerate(bank_index):
            self.groups[int(p)].append(i)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.max_per_epoch = max_per_epoch
        self.epoch = 0
        self._len: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        cap = self.max_per_epoch
        pools = {
            p: rng.permutation(v).tolist()[:cap] if cap else rng.permutation(v).tolist()
            for p, v in self.groups.items()
        }
        active = [p for p, v in pools.items() if v]
        while active:
            rng.shuffle(active)
            take = active[: self.batch_size]
            batch = [pools[p].pop() for p in take]
            active = [p for p in active if pools[p]]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """Exact batch count.

        Each round drains one signature from up to `batch_size` non-empty
        groups, so the number of rounds depends only on the multiset of group
        sizes, not on the shuffle. Draining the largest groups keeps the
        simulation deterministic and gives the same count as any other choice.

        Cached: DataLoader asks for this more than once, and the simulation is
        O(rounds * P log P).
        """
        if self._len is not None:
            return self._len
        counts = np.array([len(v) for v in self.groups.values()], dtype=np.int64)
        if self.max_per_epoch:
            counts = np.minimum(counts, self.max_per_epoch)
        n_batches = 0
        while counts.size:
            take = min(self.batch_size, counts.size)
            if take == self.batch_size or not self.drop_last:
                n_batches += 1
            order = np.argsort(-counts)
            counts[order[:take]] -= 1
            counts = counts[counts > 0]
        self._len = n_batches
        return n_batches


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def assign_groups(
    groups: Sequence[Sequence[int]],
    fracs: Sequence[float],
    weights: np.ndarray | None = None,
) -> list[list[int]]:
    """Distribute whole groups across splits, balancing count *and* weight.

    Whole groups always land in one split, so whatever defines a group (a
    scaffold, an MOA) can never straddle the train/test boundary.

    Groups are placed heaviest-first, each going to whichever split is furthest
    from its target on its *worst* axis. Filling the splits in order instead
    would let a group too large for the train budget fall through to valid and
    then to test, so with few large groups the validation set can come out
    empty; balancing on compound count alone would also leave the splits
    unbalanced in signatures.

    Pass `weights` (e.g. the signature count per compound) to balance signatures
    as well as compounds.
    """
    n_items = sum(len(g) for g in groups)
    w = (np.ones(n_items, dtype=np.float64) if weights is None
         else np.asarray(weights, dtype=np.float64))
    total_w = float(w.sum()) or 1.0

    target_n = [f * n_items for f in fracs]
    target_w = [f * total_w for f in fracs]

    # Heaviest first: big groups are the hard ones to place, and placing them
    # while every split still has room is what keeps the result balanced.
    ordered = sorted(groups, key=lambda g: (-float(w[list(g)].sum()), -len(g), g[0]))

    buckets: list[list[int]] = [[] for _ in fracs]
    cur_n = [0.0] * len(fracs)
    cur_w = [0.0] * len(fracs)
    for group in ordered:
        gn, gw = len(group), float(w[list(group)].sum())
        best, best_cost = None, None
        for s in range(len(fracs)):
            if target_n[s] <= 0:
                continue
            fill_n = (cur_n[s] + gn) / max(target_n[s], 1e-9)
            fill_w = (cur_w[s] + gw) / max(target_w[s], 1e-9)
            cost = max(fill_n, fill_w)      # penalize whichever axis overflows first
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        if best is None:
            best = 0
        buckets[best].extend(group)
        cur_n[best] += gn
        cur_w[best] += gw
    return buckets


def scaffold_split(
    scaffolds: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bemis-Murcko scaffold split over compounds (indices are positional).

    No scaffold is ever shared between splits. Pass `weights` -- the number of
    signatures per compound -- to balance signatures across splits as well as
    compound counts; without it, the most heavily profiled compounds pile up in
    one split and recall@k ends up dominated by a handful of them.

    A random split over compounds leaks analogues and will inflate held-out
    numbers substantially -- do not use one for the headline result.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)

    frac_test = max(1.0 - frac_train - frac_valid, 0.0)
    train, valid, test = assign_groups(
        list(groups.values()), [frac_train, frac_valid, frac_test], weights
    )
    return np.array(sorted(train)), np.array(sorted(valid)), np.array(sorted(test))


def random_split(
    n: int, frac_train: float = 0.8, frac_valid: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random compound split. Leaks analogues -- for the leakage ablation only."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    return (
        np.sort(perm[:n_train]),
        np.sort(perm[n_train : n_train + n_valid]),
        np.sort(perm[n_train + n_valid :]),
    )


def moa_disjoint_split(
    moa: Sequence[str | None],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out whole mechanisms of action.

    Harder than a scaffold split: can the model score a molecule whose
    *mechanism* is absent from training? Compounds without an MOA annotation are
    assigned to train, so where annotation coverage is sparse this split holds
    out far fewer compounds than a scaffold split.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    unann: list[int] = []
    for i, m in enumerate(moa):
        if m:
            groups[m].append(i)
        else:
            unann.append(i)

    frac_test = max(1.0 - frac_train - frac_valid, 0.0)
    train, valid, test = assign_groups(
        list(groups.values()), [frac_train, frac_valid, frac_test], weights
    )
    return (
        np.array(sorted(train + unann)),
        np.array(sorted(valid)),
        np.array(sorted(test)),
    )


def signature_mask(bank_index: np.ndarray, compound_positions: Sequence[int]) -> np.ndarray:
    """Boolean mask over signatures whose compound sits in `compound_positions`."""
    wanted = np.zeros(int(bank_index.max()) + 1, dtype=bool) if len(bank_index) else np.zeros(0, dtype=bool)
    for p in np.asarray(compound_positions, dtype=np.int64):
        if 0 <= p < len(wanted):
            wanted[p] = True
    return wanted[bank_index] if len(bank_index) else np.zeros(0, dtype=bool)


def subset_dataset(ds: SignatureDataset, mask: np.ndarray) -> SignatureDataset:
    return SignatureDataset(
        ds.signatures[mask],
        ds.bank_index[mask],
        ds.cell_index[mask],
        ds.log_dose[mask],
        ds.log_time[mask],
        ds.has_dose[mask],
    )


def subset_to_memmap(
    ds: SignatureDataset, mask: np.ndarray, path: str | Path, chunk: int = 16384
) -> SignatureDataset:
    """Write a split to an on-disk .npy and return a dataset mapped onto it.

    `subset_dataset` allocates the split on the heap: 1.17 GB for the LINCS
    train split, which cannot be evicted under pressure no matter how little of
    it is being read at any moment. Writing it to disk and mapping it back
    makes those pages clean and file-backed, so the OS reclaims them instead of
    the process being killed.

    This is only viable because every consumer of `.signatures` already streams
    -- `consensus_signatures` takes 32k-row blocks, `rank_bank` takes 1024-row
    query blocks, and the DataLoader takes one batch at a time. Nothing asks
    for the whole matrix at once, so nothing forces it resident.

    An existing file of the right shape is reused, which makes a repeated run
    on the same split start immediately.
    """
    rows = np.flatnonzero(mask)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_genes = ds.signatures.shape[1]

    reusable = False
    if path.exists():
        try:
            existing = np.load(path, mmap_mode="r")
            reusable = existing.shape == (len(rows), n_genes)
            del existing
        except Exception:  # noqa: BLE001 - a truncated cache is just a rewrite
            reusable = False

    if not reusable:
        out = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(rows), n_genes)
        )
        for start in range(0, len(rows), chunk):
            stop = min(start + chunk, len(rows))
            out[start:stop] = ds.signatures[rows[start:stop]]
        out.flush()
        del out
        logger.info("cached %d x %d split to %s", len(rows), n_genes, path)

    return SignatureDataset(
        np.load(path, mmap_mode="r"),
        ds.bank_index[rows], ds.cell_index[rows],
        ds.log_dose[rows], ds.log_time[rows], ds.has_dose[rows],
    )


def split_cache_dir(ds: SignatureDataset) -> Path | None:
    """Where to cache splits: beside the memmapped source, or nowhere."""
    sig = ds.signatures
    if not isinstance(sig, np.memmap) or not getattr(sig, "filename", None):
        return None
    return Path(sig.filename).parent / "splits"


def consume_subset_dataset(ds: SignatureDataset, mask: np.ndarray) -> SignatureDataset:
    """Take a subset by compacting `ds` in place, destroying it in the process.

    `subset_dataset` copies, so taking the train split holds the 1.45 GB source
    matrix and its 1.17 GB copy at the same time. On an 8 GB box that peak is
    what kills the run -- and it is avoidable, because the source is never read
    again. This moves the wanted rows to the front of the existing buffer and
    shrinks it, so the peak is the source plus one chunk instead of the source
    plus the whole split.

    Take the *other* splits first, with `subset_dataset`; this one leaves `ds`
    empty. The `signatures` array must own its buffer, which is true of
    anything `load_artifacts` produced -- if it does not, this falls back to
    copying rather than corrupting a view.
    """
    rows = np.flatnonzero(mask)
    sig = ds.signatures
    n_genes = sig.shape[1]
    if isinstance(sig, np.memmap):
        # Nothing to compact: the source lives on disk, so the copy IS the
        # only resident allocation.
        out = subset_dataset(ds, mask)
        ds.release()
        return out
    if sig.base is not None or not sig.flags["C_CONTIGUOUS"]:
        logger.warning("signatures is a view; falling back to a copying subset")
        out = subset_dataset(ds, mask)
        ds.release()
        return out

    # Pull the small per-row arrays out before the source is touched.
    columns = (
        ds.bank_index[rows], ds.cell_index[rows],
        ds.log_dose[rows], ds.log_time[rows], ds.has_dose[rows],
    )

    # `rows` is ascending, so rows[d] >= d and a forward pass never overwrites
    # a source row it has not read yet.
    chunk = 8192
    for start in range(0, len(rows), chunk):
        stop = min(start + chunk, len(rows))
        sig[start:stop] = sig[rows[start:stop]]
    sig.resize((len(rows), n_genes), refcheck=False)

    ds.signatures = np.zeros((0, n_genes), dtype=np.float32)
    ds.release()
    return SignatureDataset(sig, *columns)


# --------------------------------------------------------------------------
# L1000 loading
# --------------------------------------------------------------------------

def load_l1000(
    gctx_path: str,
    siginfo_path: str,
    geneinfo_path: str,
    compoundinfo_path: str | None = None,
    tas_min: float = 0.1,
    exemplar_only: bool = False,
    landmark_only: bool = True,
    max_sigs_per_pert: int | None = None,
    seed: int = 0,
):
    """Read LINCS 2020 level-5 compound signatures and the accompanying metadata.

    Returns (signatures, siginfo_df, landmark_gene_ids). Only `trt_cp` rows are
    kept -- the genetic arms belong to a separate model.

    `max_sigs_per_pert` defaults to None: keep every signature on disk and let
    `UniquePertBatchSampler(max_per_epoch=...)` cap exposure per epoch instead.
    A prep-time subsample discards those signatures permanently; a per-epoch cap
    balances the gradient while still showing the model all the data over
    training. Set it only if the stored matrix will not fit.

    Column names follow the LINCS 2020 beta release. If you are on GSE92742 /
    GSE70138 instead, `cell_iname` is `cell_id`, `pert_dose` is `pert_idose`
    and there is no `tas` column -- compute TAS yourself from signature
    strength x replicate correlation, or fall back to `distil_cc_q75` and
    `pct_self_rank_q25`.
    """
    import pandas as pd
    from cmapPy.pandasGEXpress.parse import parse

    gene_info = pd.read_csv(geneinfo_path, sep="\t", dtype=str)
    if landmark_only:
        col = "feature_space" if "feature_space" in gene_info else "pr_is_lm"
        mask = (
            gene_info[col].eq("landmark")
            if col == "feature_space"
            else gene_info[col].eq("1")
        )
        gene_info = gene_info[mask]
    gene_ids = gene_info["gene_id"].astype(str).tolist()

    sig = pd.read_csv(siginfo_path, sep="\t", low_memory=False)
    before = len(sig)
    sig = sig[sig["pert_type"] == PERT_TYPE_COMPOUND]
    logger.info("pert_type == trt_cp keeps %d/%d signatures", len(sig), before)

    if exemplar_only and "is_exemplar_sig" in sig:
        before = len(sig)
        sig = sig[sig["is_exemplar_sig"].astype(str).isin({"1", "1.0", "True", "true"})]
        logger.info("exemplar filter keeps %d/%d signatures", len(sig), before)
    if "tas" in sig:
        before = len(sig)
        sig = sig[pd.to_numeric(sig["tas"], errors="coerce") >= tas_min]
        logger.info("TAS >= %.2f keeps %d/%d signatures", tas_min, len(sig), before)
    else:
        logger.warning("no `tas` column found -- signatures are unfiltered for activity")

    if max_sigs_per_pert:
        rng = np.random.default_rng(seed)
        keep = []
        for _, grp in sig.groupby("pert_id", sort=False):
            idx = grp.index.to_numpy()
            if len(idx) > max_sigs_per_pert:
                idx = rng.choice(idx, size=max_sigs_per_pert, replace=False)
            keep.append(idx)
        sig = sig.loc[np.sort(np.concatenate(keep))].reset_index(drop=True)
        logger.info(
            "capped at %d sigs/compound -> %d signatures over %d compounds",
            max_sigs_per_pert, len(sig), sig["pert_id"].nunique(),
        )

    logger.info(
        "loading %d signatures x %d genes from %s", len(sig), len(gene_ids), gctx_path
    )
    gct = parse(gctx_path, cid=sig["sig_id"].tolist(), rid=gene_ids)
    mat = gct.data_df.T  # signatures x genes

    # cmapPy does not promise to honor the order of `rid` / `cid`, so realign
    # both axes explicitly. A silent gene permutation here would be invisible
    # downstream and would quietly destroy every result.
    present = [g for g in gene_ids if g in mat.columns]
    if len(present) != len(gene_ids):
        logger.warning(
            "%d/%d landmark genes missing from the GCTX",
            len(gene_ids) - len(present), len(gene_ids),
        )
    mat = mat[present]
    sig = sig.set_index("sig_id").loc[mat.index].reset_index()

    if compoundinfo_path:
        cmpd = pd.read_csv(compoundinfo_path, sep="\t", low_memory=False)
        cols = [c for c in ("pert_id", "canonical_smiles", "moa", "target") if c in cmpd]
        sig = sig.merge(cmpd[cols].drop_duplicates("pert_id"), on="pert_id", how="left")

    return mat.to_numpy(dtype=np.float32), sig, present


def robust_zscore_rows(x: np.ndarray) -> np.ndarray:
    """Per-signature robust standardization.

    Apply this to *every* query, including Tahoe pseudobulk and any patient
    differential, so that vectors from different platforms at least share a
    scale before they meet the same encoder.
    """
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    return ((x - med) / (1.4826 * mad + 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------
# Prepared artifacts
# --------------------------------------------------------------------------

_JOIN = "|"


def save_artifacts(
    out_dir: str | Path,
    bank: CompoundBank,
    ds: SignatureDataset,
    cell_lines: Sequence[str] = (),
    gene_ids: Sequence[str] = (),
    n_bits: int = 2048,
) -> Path:
    """Write everything `train.main` needs: one npz plus a metadata json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "artifacts.npz",
        bank_ids=np.array(bank.ids, dtype=np.str_),
        bank_smiles=np.array([s or "" for s in bank.smiles], dtype=np.str_),
        bank_moa=np.array([m or "" for m in bank.moa], dtype=np.str_),
        bank_targets=np.array([_JOIN.join(t) for t in bank.targets], dtype=np.str_),
        mol_features=bank.mol_features,
        signatures=ds.signatures,
        bank_index=ds.bank_index,
        cell_index=ds.cell_index,
        log_dose=ds.log_dose,
        log_time=ds.log_time,
        has_dose=ds.has_dose,
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "n_compounds": len(bank),
                "n_signatures": len(ds),
                "n_genes": int(ds.signatures.shape[1]),
                "n_bits": int(n_bits),
                "n_cell_lines": int(ds.cell_index.max()) + 1 if len(ds) else 0,
                "cell_lines": list(cell_lines),
                "gene_ids": list(gene_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir / "artifacts.npz"


def extract_signatures(data_dir: str | Path) -> Path:
    """Write the signature matrix beside artifacts.npz as a plain .npy.

    Members of a compressed .npz cannot be memory-mapped -- numpy has to
    inflate the whole array into RAM -- so on LINCS every run pays 1.45 GB
    before it does anything. An uncompressed sidecar can be mapped, which
    turns that into disk the OS pages on demand and leaves only the split
    copies resident. Costs 1.45 GB of disk, once.
    """
    data_dir = Path(data_dir)
    out = data_dir / "signatures.npy"
    with np.load(data_dir / "artifacts.npz") as z:
        sig = z["signatures"]
        logger.info("writing %s  shape=%s  %.2f GB", out, sig.shape, sig.nbytes / 1e9)
        np.save(out, np.ascontiguousarray(sig, dtype=np.float32))
    return out


def load_artifacts(
    data_dir: str | Path, signatures: str = "auto"
) -> tuple[CompoundBank, SignatureDataset, dict]:
    """Load the bank and the signature dataset.

    `signatures` picks where the 1.45 GB matrix lives, which is the only
    memory decision in this project that actually matters:

      "ram"   inflate it from artifacts.npz onto the heap. Fastest per epoch
              (~2x), and what every run before the sidecar existed did. Peaks
              at roughly 2.9 GB once the splits are taken, which an 8 GB box
              with a browser open does not have.
      "mmap"  map an uncompressed sidecar, and cache the splits to disk too.
              Batches are read from the page cache, so epochs are slower, but
              the pages are clean and file-backed and the OS can reclaim them
              under pressure instead of the run being killed. Writes the
              sidecar if it is missing.
      "auto"  mmap when the sidecar is already there, ram otherwise. Default.

    Compressed .npz members cannot be mapped -- numpy has to inflate them --
    which is the whole reason the sidecar exists.
    """
    data_dir = Path(data_dir)
    if signatures not in ("auto", "ram", "mmap"):
        raise ValueError(f"signatures must be auto/ram/mmap, got {signatures!r}")
    z = np.load(data_dir / "artifacts.npz")
    sidecar = data_dir / "signatures.npy"

    if signatures == "ram":
        signature_array = z["signatures"]
    else:
        if signatures == "mmap" and not sidecar.exists():
            # One-off, and it does need the array resident for as long as it
            # takes to write it out.
            z.close()
            extract_signatures(data_dir)
            z = np.load(data_dir / "artifacts.npz")
        if sidecar.exists():
            signature_array = np.load(sidecar, mmap_mode="r")
            logger.info("signatures memory-mapped from %s", sidecar)
        else:
            signature_array = z["signatures"]
    bank = CompoundBank(
        ids=[str(x) for x in z["bank_ids"]],
        mol_features=z["mol_features"],
        smiles=[str(x) or None for x in z["bank_smiles"]],
        moa=[str(x) or None for x in z["bank_moa"]],
        targets=[tuple(t for t in str(x).split(_JOIN) if t) for x in z["bank_targets"]],
    )
    ds = SignatureDataset(
        signatures=signature_array,
        bank_index=z["bank_index"],
        cell_index=z["cell_index"],
        log_dose=z["log_dose"],
        log_time=z["log_time"],
        has_dose=z["has_dose"],
    )
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return bank, ds, meta

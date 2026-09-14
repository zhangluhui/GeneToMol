"""Turn the raw LINCS release into the `artifacts.npz` that `train.py` expects.

    python -m genetomol.prepare \\
        --gctx ../lincs/level5_beta_trt_cp_n720216x12328.gctx \\
        --siginfo ../lincs/siginfo_beta.txt \\
        --geneinfo ../lincs/geneinfo_beta.txt \\
        --compoundinfo ../lincs/compoundinfo_beta.txt \\
        --out ../lincs/prepared

Run `census.py` first; this applies exactly the same filter cascade, so the
counts it reports are the counts you get here.

Three things this is careful about, all of which are silent when done wrong:

**Memory.** The output matrix is ~372k x 978 float32 = 1.5 GB, and letting
cmapPy build one DataFrame for all of it adds index and column overhead on top.
The GCTX is read in chunks into a preallocated array instead, so the peak is a
chunk (~80 MB), not the whole thing.

**HDF5 access order.** Fancy-indexing a 33 GB HDF5 dataset with unsorted
positions is pathologically slow. Columns are read in file order and permuted
back afterwards.

**Axis alignment.** cmapPy does not promise to return rows or columns in the
order you asked for. Both axes are realigned explicitly against the ids we
requested. A silent gene permutation here is invisible downstream and destroys
every result that follows.

Features are written **unscaled**. `train.run_experiment` fits the descriptor
scaler on the training compounds only, after the split, so scaling here would
leak test statistics into training.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from .data import (
    PERT_TYPE_COMPOUND,
    CompoundBank,
    SignatureDataset,
    robust_zscore_rows,
    save_artifacts,
)
from .featurize import featurize_molecules

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "1.0", "true", "True", "TRUE", "t", "y", "yes"}
_PLACEHOLDER = {"restricted", "nan", "", "none", "-666"}


def _resolve(df, *names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def landmark_gene_ids(geneinfo_path: str, landmark_only: bool = True) -> list[str]:
    import pandas as pd

    gi = pd.read_csv(geneinfo_path, sep="\t", dtype=str)
    if landmark_only:
        col = "feature_space" if "feature_space" in gi else "pr_is_lm"
        gi = gi[gi[col].eq("landmark") if col == "feature_space" else gi[col].eq("1")]
    ids = gi["gene_id"].astype(str).tolist()
    logger.info("%d landmark genes", len(ids))
    return ids


def select_signatures(
    siginfo_path: str,
    compoundinfo_path: str,
    tas_min: float,
    exemplar_only: bool,
    min_sigs: int,
):
    """Apply the census filter cascade. Returns (siginfo_df, {pert_id: smiles})."""
    import pandas as pd
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    sig = pd.read_csv(siginfo_path, sep="\t", low_memory=False)
    logger.info("siginfo: %d rows", len(sig))

    sig = sig[sig["pert_type"] == PERT_TYPE_COMPOUND]
    logger.info("  trt_cp: %d signatures / %d compounds", len(sig), sig["pert_id"].nunique())

    ex = _resolve(sig, "is_exemplar_sig", "is_exemplar")
    if exemplar_only and ex:
        sig = sig[sig[ex].astype(str).isin(_TRUTHY)]
        logger.info("  exemplar: %d signatures", len(sig))

    if "tas" in sig:
        sig = sig[pd.to_numeric(sig["tas"], errors="coerce") >= tas_min]
        logger.info("  tas>=%.2f: %d signatures / %d compounds",
                    tas_min, len(sig), sig["pert_id"].nunique())
    else:
        logger.warning("  no `tas` column -- signatures unfiltered for activity")

    cmpd = pd.read_csv(compoundinfo_path, sep="\t", low_memory=False)
    cmpd = cmpd.drop_duplicates("pert_id")

    def usable(x) -> bool:
        if not isinstance(x, str) or x.strip().lower() in _PLACEHOLDER:
            return False
        return Chem.MolFromSmiles(x) is not None

    keep = cmpd[cmpd["canonical_smiles"].map(usable)]
    smiles = dict(zip(keep["pert_id"], keep["canonical_smiles"]))
    sig = sig[sig["pert_id"].isin(smiles)]
    logger.info("  parseable SMILES: %d signatures / %d compounds",
                len(sig), sig["pert_id"].nunique())

    if min_sigs > 1:
        counts = sig["pert_id"].value_counts()
        sig = sig[sig["pert_id"].isin(counts[counts >= min_sigs].index)]
        logger.info("  >=%d sigs/compound: %d signatures / %d compounds",
                    min_sigs, len(sig), sig["pert_id"].nunique())

    ann_cols = [c for c in ("moa", "target") if c in cmpd]
    ann = cmpd.set_index("pert_id")[ann_cols] if ann_cols else None
    return sig.reset_index(drop=True), smiles, ann


def read_gctx_chunked(
    gctx_path: str,
    sig_ids: list[str],
    gene_ids: list[str],
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Read (len(sig_ids), len(gene_ids)) from a GCTX without loading it all.

    Returns (matrix, sig_ids_kept, gene_ids_kept) with both axes in exactly the
    order given, restricted to ids that are actually present in the file.
    """
    from cmapPy.pandasGEXpress.parse import parse

    logger.info("reading GCTX metadata from %s", gctx_path)
    col_meta = parse(gctx_path, col_meta_only=True)
    row_meta = parse(gctx_path, row_meta_only=True)
    file_cols = {str(c): i for i, c in enumerate(col_meta.index)}
    file_rows = {str(r): i for i, r in enumerate(row_meta.index)}
    logger.info("  GCTX holds %d signatures x %d genes", len(file_cols), len(file_rows))

    genes_kept = [g for g in gene_ids if g in file_rows]
    if len(genes_kept) != len(gene_ids):
        logger.warning("  %d/%d landmark genes absent from the GCTX",
                       len(gene_ids) - len(genes_kept), len(gene_ids))
    sigs_kept = [s for s in sig_ids if s in file_cols]
    if len(sigs_kept) != len(sig_ids):
        logger.warning("  %d/%d selected signatures absent from the GCTX",
                       len(sig_ids) - len(sigs_kept), len(sig_ids))
    if not sigs_kept or not genes_kept:
        raise ValueError("nothing left to read after intersecting with the GCTX")

    # Ask for rows in file order; the returned labels are mapped back below.
    ridx = sorted(file_rows[g] for g in genes_kept)
    gene_to_out = {g: j for j, g in enumerate(genes_kept)}
    out = np.empty((len(sigs_kept), len(genes_kept)), dtype=np.float32)
    sig_to_out = {s: i for i, s in enumerate(sigs_kept)}

    # Same trick on the column axis: ascending positions, permute afterwards.
    order = sorted(range(len(sigs_kept)), key=lambda i: file_cols[sigs_kept[i]])
    t0 = time.time()
    for start in range(0, len(order), chunk_size):
        block = order[start : start + chunk_size]
        cidx = [file_cols[sigs_kept[i]] for i in block]
        gct = parse(gctx_path, cidx=cidx, ridx=ridx)
        df = gct.data_df                       # genes x signatures

        got_rows = [str(x) for x in df.index]
        got_cols = [str(x) for x in df.columns]
        # cmapPy makes no ordering promise -- map what came back, do not assume.
        r_target = np.array([gene_to_out[g] for g in got_rows], dtype=np.int64)
        c_target = np.array([sig_to_out[s] for s in got_cols], dtype=np.int64)
        block_vals = df.to_numpy(dtype=np.float32).T          # signatures x genes
        out[np.ix_(c_target, r_target)] = block_vals

        done = min(start + chunk_size, len(order))
        rate = done / max(time.time() - t0, 1e-9)
        logger.info("  %d/%d signatures (%.0f/s, eta %s)", done, len(order), rate,
                    time.strftime("%H:%M:%S",
                                  time.gmtime((len(order) - done) / max(rate, 1e-9))))
    return out, sigs_kept, genes_kept


def build(
    gctx: str,
    siginfo: str,
    geneinfo: str,
    compoundinfo: str,
    out_dir: str,
    tas_min: float = 0.1,
    exemplar_only: bool = False,
    min_sigs: int = 1,
    n_bits: int = 2048,
    chunk_size: int = 20_000,
    landmark_only: bool = True,
    apply_robust_zscore: bool = False,
    dry_run: bool = False,
):
    import pandas as pd

    gene_ids = landmark_gene_ids(geneinfo, landmark_only)
    sig, smiles, ann = select_signatures(
        siginfo, compoundinfo, tas_min, exemplar_only, min_sigs
    )

    pert_ids = sorted(sig["pert_id"].unique())
    logger.info("bank: %d compounds, %d signatures", len(pert_ids), len(sig))
    est = len(sig) * len(gene_ids) * 4 / 1e9
    logger.info("signature matrix will be %.2f GB; reading in chunks of %d",
                est, chunk_size)
    if dry_run:
        logger.info("--dry-run: stopping before the GCTX read")
        return None

    # --- signatures ---------------------------------------------------------
    mat, sig_ids_kept, genes_kept = read_gctx_chunked(
        gctx, sig["sig_id"].astype(str).tolist(), gene_ids, chunk_size
    )
    sig = sig.set_index("sig_id").loc[sig_ids_kept].reset_index()
    assert len(sig) == mat.shape[0], "siginfo and matrix rows disagree"

    if apply_robust_zscore:
        logger.info("applying robust_zscore_rows to every signature")
        mat = robust_zscore_rows(mat)

    # Compounds may have vanished if all their signatures were missing from the file.
    pert_ids = sorted(sig["pert_id"].unique())
    pos = {p: i for i, p in enumerate(pert_ids)}
    bank_index = sig["pert_id"].map(pos).to_numpy(dtype=np.int64)

    # --- context (the model ignores it; kept for splitting and diagnostics) --
    cell_col = _resolve(sig, "cell_iname", "cell_id")
    cell_names = sorted(sig[cell_col].astype(str).unique()) if cell_col else ["unknown"]
    cell_lut = {c: i for i, c in enumerate(cell_names)}
    cell_index = (sig[cell_col].astype(str).map(cell_lut).to_numpy(dtype=np.int64)
                  if cell_col else np.zeros(len(sig), dtype=np.int64))

    dose_col = _resolve(sig, "pert_dose", "pert_idose")
    dose = pd.to_numeric(sig[dose_col].astype(str).str.extract(r"([\d.]+)")[0],
                         errors="coerce") if dose_col else pd.Series(np.nan, index=sig.index)
    has_dose = dose.notna().to_numpy().astype(np.float32)
    log_dose = np.log10(dose.fillna(0).to_numpy(dtype=np.float64) + 1e-3).astype(np.float32)

    time_col = _resolve(sig, "pert_time", "pert_itime")
    ptime = pd.to_numeric(sig[time_col].astype(str).str.extract(r"([\d.]+)")[0],
                          errors="coerce") if time_col else pd.Series(np.nan, index=sig.index)
    log_time = np.log10(ptime.fillna(24.0).to_numpy(dtype=np.float64) + 1e-3).astype(np.float32)

    logger.info("%d cell lines; dose present on %.1f%% of signatures",
                len(cell_names), 100 * has_dose.mean())

    # --- compound features (RAW: the scaler is fit after the split) ---------
    logger.info("featurizing %d compounds (ECFP%d + descriptors)", len(pert_ids), n_bits)
    feats, valid = featurize_molecules([smiles[p] for p in pert_ids], n_bits=n_bits)
    if not valid.all():
        bad = np.flatnonzero(~valid)
        logger.warning("dropping %d compounds that failed featurization", len(bad))
        keep_pos = np.flatnonzero(valid)
        remap = {old: new for new, old in enumerate(keep_pos)}
        row_ok = np.array([b in remap for b in bank_index])
        mat, bank_index = mat[row_ok], bank_index[row_ok]
        cell_index, log_dose = cell_index[row_ok], log_dose[row_ok]
        log_time, has_dose = log_time[row_ok], has_dose[row_ok]
        bank_index = np.array([remap[b] for b in bank_index], dtype=np.int64)
        pert_ids = [pert_ids[i] for i in keep_pos]
        feats = feats[keep_pos]

    def _ann(p, col):
        if ann is None or col not in ann.columns or p not in ann.index:
            return None
        v = ann.at[p, col]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        s = str(v).strip()
        return s if s and s.lower() not in _PLACEHOLDER else None

    bank = CompoundBank(
        ids=pert_ids,
        mol_features=feats,
        smiles=[smiles[p] for p in pert_ids],
        moa=[_ann(p, "moa") for p in pert_ids],
        targets=[tuple(t for t in (_ann(p, "target") or "").split("|") if t)
                 for p in pert_ids],
    )
    ds = SignatureDataset(mat, bank_index, cell_index, log_dose, log_time, has_dose)

    path = save_artifacts(out_dir, bank, ds, cell_names, genes_kept, n_bits=n_bits)
    n_moa = sum(1 for m in bank.moa if m)
    logger.info("wrote %s", path)
    logger.info("  %d compounds / %d signatures / %d genes",
                len(bank), len(ds), ds.signatures.shape[1])
    logger.info("  MOA annotated on %d compounds (%.1f%%)",
                n_moa, 100 * n_moa / max(len(bank), 1))
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gctx", required=True)
    ap.add_argument("--siginfo", required=True)
    ap.add_argument("--geneinfo", required=True)
    ap.add_argument("--compoundinfo", required=True)
    ap.add_argument("--out", required=True, help="directory for artifacts.npz")
    ap.add_argument("--tas-min", type=float, default=0.1)
    ap.add_argument("--exemplar-only", action="store_true")
    ap.add_argument("--min-sigs", type=int, default=1)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=20_000)
    ap.add_argument("--all-genes", action="store_true",
                    help="keep all 12,328 genes instead of the 978 landmarks")
    ap.add_argument("--robust-zscore", action="store_true",
                    help="row-standardize every signature. Off by default: level 5 "
                         "is already control-referenced, and this is for putting "
                         "foreign queries on a common scale -- if you use it, use "
                         "it on both training data and queries")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the filters and report sizes without touching the GCTX")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    build(
        gctx=args.gctx, siginfo=args.siginfo, geneinfo=args.geneinfo,
        compoundinfo=args.compoundinfo, out_dir=args.out,
        tas_min=args.tas_min, exemplar_only=args.exemplar_only,
        min_sigs=args.min_sigs, n_bits=args.n_bits, chunk_size=args.chunk_size,
        landmark_only=not args.all_genes,
        apply_robust_zscore=args.robust_zscore, dry_run=args.dry_run,
    )
    print(f"\nelapsed {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}")
    if not args.dry_run:
        print(f"\nNext:\n  python -m genetomol.train --data {args.out} "
              f"--split scaffold --out runs/scaffold")


if __name__ == "__main__":
    main()

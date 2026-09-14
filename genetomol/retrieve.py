"""Rank compounds for a real query signature. No ground truth required.

`evaluate.py` answers "was the right compound in the top k", which needs the
answer. This answers "which compounds should I look at", which does not: the
model is a scoring function, and the ranked list falls out of one matrix
multiply against the encoded bank.

**Input format.** A CSV or TSV with a header. The first column is the gene
identifier; every remaining column is one query.

    gene_symbol,tumour_vs_normal,drug_vs_dmso
    AARS1,1.23,0.44
    ABCB6,-0.41,-1.02

Identifiers may be Entrez ids (what `meta.json` stores), gene symbols, or
Ensembl ids. Symbols and Ensembl ids need `--geneinfo geneinfo_beta.txt` to map
them; Entrez ids need nothing. The column order in your file is irrelevant --
values are reindexed onto the model's 978 landmark genes, which is the step
that silently ruins a query if you do it yourself and get it wrong.

**What the values must be.** The model was trained on LINCS Level 5 z-scores:
treatment against the plate population, positive meaning up-regulated. A
differential on a different scale (raw log2 fold-change, say) is out of
distribution, and nothing in this repo has measured how far that transfers.
`--standardize zscore` rescales a query across genes to zero mean and unit
variance, which makes the magnitude comparable but cannot fix a sign
convention. The report prints the query's own spread so a mismatch is visible.

    python -m genetomol.retrieve --data ../lincs/prepared \\
        --run runs/genetomol \\
        --mol-features ../lincs/prepared/ecfp_chiral.npz \\
        --query my_signature.csv --geneinfo ../lincs/geneinfo_beta.txt --k 25
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Below this, the query is missing so much of the landmark space that the
# zero-fill dominates and the result is not worth printing.
MIN_COVERAGE = 0.5


def gene_aliases(geneinfo: "str | Mapping[str, str] | None") -> dict[str, str]:
    """symbol/ensembl -> Entrez id, from `geneinfo_beta.txt`. Empty without it.

    An already-built mapping is accepted and passed straight through, so a
    serving path that carries its aliases inside the bundle (`serve.Bundle`)
    can reuse `_read` without a LINCS metadata file on disk.
    """
    if not geneinfo:
        return {}
    if isinstance(geneinfo, Mapping):
        return {str(k): str(v) for k, v in geneinfo.items()}
    import pandas as pd

    df = pd.read_csv(geneinfo, sep="\t", dtype=str)
    out: dict[str, str] = {}
    for col in ("gene_symbol", "ensembl_id"):
        if col in df:
            for alias, gid in zip(df[col], df["gene_id"]):
                if isinstance(alias, str) and alias and alias != "-":
                    out.setdefault(alias.strip().upper(), str(gid).strip())
    return out


# A differential is centered near zero and roughly half negative. Raw expression
# is almost all positive, and z-scoring it across genes yields "which genes are
# abundant" rather than "what changed" -- plausible-looking and useless.
MIN_NEGATIVE_FRAC = 0.05


def robust_z(delta: np.ndarray) -> np.ndarray:
    """Scale a contrast to a robust z, with the dispersion pooled ACROSS GENES.

    `delta` is the per-gene change, treated mean minus control mean, shaped
    (n_genes,) or (..., n_genes). Scaling is applied over the last axis.

    Every gene is divided by a single estimate of how much the whole contrast
    varies, so the result expresses how large a change is relative to the
    others. Median and median absolute deviation stand in for mean and standard
    deviation, so a few extreme genes cannot set the scale for the rest.

    This is the default and the scale the model expects. `per_gene_z` is the
    per-gene alternative.
    """
    med = np.median(delta, axis=-1, keepdims=True)
    mad = np.median(np.abs(delta - med), axis=-1, keepdims=True)
    return (delta - med) / np.maximum(1.4826 * mad, 1e-9)


# No automatic switch to per-gene dispersion at any control-group size; pass
# `--dispersion per-gene` to select it explicitly.
PER_GENE_MIN_N = None


def per_gene_z(treated_mean: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Scale a contrast per gene: each divided by its own MAD across controls.

    `treated_mean` is (n_genes,); `control` is (n_controls, n_genes). This is
    the LINCS Level 4 definition.

    It rescales the contrast by the noise structure of your own controls rather
    than preserving LINCS's per-gene scale, and with few controls the
    denominators are themselves noisy. Not the default; `robust_z` is, and the
    caller warns when this is selected.
    """
    med = np.median(control, axis=0)
    mad = np.median(np.abs(control - med), axis=0)
    return (treated_mean - med) / np.maximum(1.4826 * mad, 1e-9)


def rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged. NumPy only, no scipy, no pandas."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    s = a[order]
    starts = np.r_[True, s[1:] != s[:-1]]
    group = starts.cumsum()                       # 1-based tie-group per position
    edge = np.r_[np.flatnonzero(starts), len(a)]  # 0-based group boundaries
    # A group spanning sorted positions [edge[i], edge[i+1]) holds 1-based ranks
    # edge[i]+1 .. edge[i+1]; their mean is (edge[i] + edge[i+1] + 1) / 2.
    avg = 0.5 * (edge[group - 1] + edge[group] + 1)
    out = np.empty(len(a), dtype=np.float64)
    out[order] = avg
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho: Pearson on average-tied ranks.

    Written out rather than taken from scipy (not a dependency) or pandas
    (a dependency only of the file readers -- a core statistic should not fail
    because an optional pandas accelerator is broken).
    """
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def split_half_reliability(T: np.ndarray, C: np.ndarray) -> dict | None:
    """Reliability of ONE contrast, from disjoint treated/control pairings.

    `T` and `C` are (n_replicates, n_genes) in log space.

    Each sample is used at most once per pairing, so the two halves share no
    term. Returns None when there are fewer than two replicates a side.

    `single_contrast_r` is the reliability of one contrast;
    `averaged_r_spearman_brown` projects it to the k replicates actually
    averaged, as k*r / (1 + (k-1)*r), and is the value retrieval depends on.
    """
    from itertools import permutations

    k = min(len(T), len(C))
    if k < 2:
        return None
    rs = []
    for perm in permutations(range(k), 2):
        d = [T[i] - C[perm[i]] for i in range(2)]
        pair = spearman(d[0], d[1])
        if np.isfinite(pair):
            rs.append(float(pair))
    if not rs:
        return None
    r1 = float(np.mean(rs))
    sb = k * r1 / (1 + (k - 1) * r1) if r1 > -1 / (k - 1) else float("nan")
    return {"single_contrast_r": r1, "n_pairings": len(rs),
            "averaged_r_spearman_brown": sb, "k": k}


def describe_reliability(r: float) -> str:
    """Where a query sits against the LINCS ceilings measured in signal.py."""
    if r >= 0.40:
        return ("comparable to a LINCS consensus (0.449 overall); the headline "
                "retrieval numbers are roughly applicable")
    if r >= 0.15:
        return ("thin but usable -- between the 4-5 signature bin (0.168) and "
                "the overall consensus (0.449). Expect worse than 20.1% @10 "
                "and read MOA rather than compound identity")
    return ("below the shallowest LINCS bin (0.168). This contrast is mostly "
            "technical noise; retrieval will return hub compounds, not biology. "
            "More replicates will help more than any modeling")


def _read(path: str, genes: list[str],
          geneinfo: "str | Mapping[str, str] | None" = None):
    """CSV/TSV -> (X, column names, report). X is (n_columns, n_genes), NaN where
    a landmark gene is absent; the caller decides how to fill.

    The first column is the gene identifier, in Entrez ids, gene symbols or
    Ensembl ids. Row order in the file is irrelevant -- values are reindexed
    onto `genes`, which is the step that silently ruins a query if done by hand.
    """
    import pandas as pd

    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] < 2:
        raise SystemExit(
            f"{path} has {df.shape[1]} column(s); expected a gene id column "
            "followed by at least one value column"
        )
    key = df.columns[0]
    names = [str(c) for c in df.columns[1:]]

    ids = df[key].astype(str).str.strip()
    alias = gene_aliases(geneinfo)
    wanted = set(genes)
    matched = ids.where(ids.isin(wanted), ids.str.upper().map(alias).fillna(""))
    hit = matched.isin(wanted)
    if not hit.any():
        kind = "Entrez ids" if not alias else "Entrez ids, symbols or Ensembl ids"
        raise SystemExit(
            f"none of the {len(ids):,} identifiers in column {key!r} match the "
            f"model's landmark genes. Expected {kind}"
            + ("" if alias else "; pass --geneinfo to map symbols or Ensembl ids")
        )

    vals = df.loc[:, df.columns[1:]].apply(pd.to_numeric, errors="coerce")
    frame = vals[hit.to_numpy()].copy()
    frame.index = matched[hit.to_numpy()].to_numpy()
    # A gene listed twice -- common after a symbol->Entrez collapse -- is
    # averaged rather than arbitrarily last-wins.
    dup = int(frame.index.duplicated().sum())
    if dup:
        frame = frame.groupby(level=0).mean()

    X = frame.reindex(genes).to_numpy(dtype=np.float64).T       # (cols, genes)
    report = {"file": str(path), "n_queries": len(names),
              "n_rows_read": int(len(df)), "n_matched": int(hit.sum()),
              "n_landmarks": len(genes), "n_duplicate_genes": dup}
    return X, names, report


def load_query(path: str, genes: list[str],
               geneinfo: "str | Mapping[str, str] | None" = None,
               standardize: str = "none") -> tuple[np.ndarray, list[str], dict]:
    """Read the query file and reindex it onto `genes`. Returns (X, names, report).

    Missing landmarks are filled with 0.0 -- the mean of a z-scored gene, i.e.
    "no change" -- which is the least-assuming fill available. It is still a
    fabricated value, so the coverage is reported and a query below
    `MIN_COVERAGE` is refused rather than quietly answered.
    """
    X, names, rep = _read(path, genes, geneinfo)
    present = ~np.isnan(X).any(axis=0)
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    # Guard the commonest mistake: raw expression instead of a differential.
    # Expression is almost all positive, and standardizing it across genes
    # encodes gene abundance rather than response -- which looks fine and is
    # not what the encoder was trained on.
    neg = float((X[:, present] < 0).mean()) if present.any() else 0.0
    if neg < MIN_NEGATIVE_FRAC:
        raise SystemExit(
            f"only {neg:.1%} of the values in {path} are negative. A LINCS "
            "Level 5 signature is a differential -- centered near zero, roughly "
            "half negative -- so this looks like raw expression. Pass "
            "--control to compute a differential, or supply an already-"
            "differential signature."
        )

    report = {
        **rep,
        "n_present": int(present.sum()),
        "coverage": float(present.mean()),
        "standardize": standardize,
        "per_query": [{"name": n, "mean": float(X[i].mean()),
                       "std": float(X[i].std()),
                       "absmax": float(np.abs(X[i]).max())}
                      for i, n in enumerate(names)],
    }
    if report["coverage"] < MIN_COVERAGE:
        raise SystemExit(
            f"only {report['n_present']}/{len(genes)} landmark genes "
            f"({report['coverage']:.0%}) are present in {path}. Below "
            f"{MIN_COVERAGE:.0%} the zero-fill dominates the query; fix the "
            "identifier mapping or use a more complete signature"
        )
    if standardize == "zscore":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        X = (X - mu) / np.maximum(sd, 1e-6)
    return np.ascontiguousarray(X, dtype=np.float32), names, report


def rank_compounds(model, bank, X: np.ndarray, device, k: int = 25):
    """Top-k bank positions and cosine scores for each query row.

    The bank is encoded once. `truth` never appears -- this is the half of
    `evaluate._rank_chunk` that does not need to know the answer.
    """
    # Imported here, not at module scope: `app.py` and any numpy-only serving
    # path use the readers and guards in this file but never `rank_compounds`,
    # and a top-level torch import would put a ~200 MB wheel back into a
    # deployment that was built to avoid it.
    import torch

    model.eval()
    with torch.no_grad():
        keys = model.encode_bank(bank.features_tensor(device))
        q = model.signature_encoder(torch.as_tensor(X, device=device))
        scores = q @ keys.t()                       # cosine: both sides unit
        top = scores.topk(min(k, scores.shape[1]), dim=1)
    return top.indices.cpu().numpy(), top.values.cpu().numpy()


def format_hits(bank, idx: np.ndarray, score: np.ndarray, name: str,
                width: int = 118) -> str:
    lines = [f"\n=== {name} ===",
             f"{'#':>3}  {'compound':<16}{'cosine':>9}  {'MOA':<34}{'target':<12}SMILES"]
    for r, (i, v) in enumerate(zip(idx.tolist(), score.tolist()), 1):
        moa = (bank.moa[i] or "-")[:33]
        tgt = ",".join(bank.targets[i])[:11] if bank.targets[i] else "-"
        smi = (bank.smiles[i] or "")[:width - 78]
        lines.append(f"{r:>3}  {bank.ids[i]:<16}{v:>9.4f}  {moa:<34}{tgt:<12}{smi}")
    return "\n".join(lines)


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="a finished run directory")
    ap.add_argument("--query", required=True,
                    help="CSV/TSV of the TREATED samples, or of an already-"
                         "differential signature; see the module docstring")
    ap.add_argument("--control", default=None,
                    help="CSV/TSV of control samples. Replicates on both sides "
                         "are averaged in log space, then the contrast is put "
                         "on a robust z scale. Without this, --query must "
                         "already be a differential")
    ap.add_argument("--dispersion", default="pooled",
                    choices=("pooled", "per-gene"),
                    help="how to scale the contrast. 'pooled' uses one MAD "
                         "over all genes and is the measured default; "
                         "'per-gene' is LINCS Level 4 proper and was worse at "
                         "every control-group size tested from 2 to 300")
    ap.add_argument("--log", action="store_true",
                    help="log2(x+1) both files first; use when your values are "
                         "linear (TPM, CPM, counts) rather than already log")
    ap.add_argument("--geneinfo", default=None,
                    help="geneinfo_beta.txt, to accept gene symbols or Ensembl ids")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--standardize", default="none", choices=("none", "zscore"),
                    help="'zscore' rescales each query across genes; it cannot "
                         "fix a wrong sign convention")
    ap.add_argument("--mol-features", default=None,
                    help="the SAME file the run trained with")
    ap.add_argument("--mol-embeddings", default=None)
    ap.add_argument("--mol-embeddings-mode", default="concat",
                    choices=("concat", "replace", "only"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the hits to this CSV")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    genes = json.loads(
        (Path(args.data) / "meta.json").read_text(encoding="utf-8"))["gene_ids"]
    if args.control:
        X, names, rep = differential(args.query, args.control, genes,
                                     args.geneinfo, args.log, args.dispersion)
    else:
        if args.log:
            raise SystemExit("--log only applies with --control")
        X, names, rep = load_query(args.query, genes, args.geneinfo,
                                   args.standardize)

    print(f"\nquery: {rep['file']}")
    print(f"  {rep['n_queries']} quer{'y' if rep['n_queries'] == 1 else 'ies'}, "
          f"{rep['n_matched']:,}/{rep['n_rows_read']:,} identifiers matched, "
          f"{rep['n_present']}/{rep['n_landmarks']} landmark genes present "
          f"({rep['coverage']:.0%})")
    if rep["n_duplicate_genes"]:
        print(f"  {rep['n_duplicate_genes']} duplicate genes averaged")
    if rep["coverage"] < 0.9:
        print(f"  WARNING: {rep['n_landmarks'] - rep['n_present']} landmark genes "
              "were filled with 0.0 (no change). Hits are less reliable.")
    if rep.get("n_treated"):
        print(f"  {rep['n_treated']} treated ({', '.join(rep['treated'])}) vs "
              f"{rep['n_control']} control ({', '.join(rep['control'])})")
    if rep.get("dispersion") == "per-gene":
        print(f"\n  WARNING: --dispersion per-gene, {rep.get('n_control', 0)} "
              "controls. On held-out signatures this lost to pooling at every "
              "group size tested: 2.0% against 25.3% recall@10 at n=2, and "
              "still 29% against 69% at n=300. It rescales the contrast by "
              "your noise structure, where the model was trained on LINCS's "
              "own per-gene scale. Compare against --dispersion pooled before "
              "trusting these hits.")
    relq = rep.get("reliability")
    if relq:
        r1, sb = relq["single_contrast_r"], relq["averaged_r_spearman_brown"]
        print(f"\n  replicate reliability: single contrast rho = {r1:+.3f} "
              f"(mean over {relq['n_pairings']} disjoint pairings)")
        print(f"  averaged over {relq['k']} replicates, Spearman-Brown "
              f"estimate = {sb:+.3f}")
        print(f"  -> {describe_reliability(sb)}")
    elif args.control:
        print("\n  only one replicate per side: no reliability estimate is "
              "possible, and no replicate averaging was done. LINCS consensus "
              "signatures score 0.449; an unreplicated contrast is below the "
              "shallowest bin measured (0.168).")
    for q in rep["per_query"]:
        print(f"  {q['name']:<24} mean {q['mean']:+.3f}  sd {q['std']:.3f}  "
              f"|max| {q['absmax']:.2f}")
        if not args.control and not 0.2 < q["std"] < 12:
            print("    WARNING: LINCS Level 5 z-scores typically have sd ~1-3. "
                  "This query is far outside that; consider --standardize zscore")

    from .runio import load_run

    run = load_run(args.data, args.run, args.mol_features, args.mol_embeddings,
                   args.mol_embeddings_mode, args.device)
    if X.shape[1] != run.test_ds.signatures.shape[1]:
        raise SystemExit(
            f"query has {X.shape[1]} genes but the model expects "
            f"{run.test_ds.signatures.shape[1]}"
        )
    idx, score = rank_compounds(run.model, run.bank, X, run.device, args.k)

    for i, name in enumerate(names):
        print(format_hits(run.bank, idx[i], score[i], name))

    if args.out:
        import csv

        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["query", "rank", "compound_id", "cosine", "moa",
                        "target", "smiles"])
            for i, name in enumerate(names):
                for r, (j, v) in enumerate(zip(idx[i].tolist(),
                                               score[i].tolist()), 1):
                    w.writerow([name, r, run.bank.ids[j], f"{v:.6f}",
                                run.bank.moa[j] or "",
                                ",".join(run.bank.targets[j]),
                                run.bank.smiles[j] or ""])
        print(f"\nwrote {args.out}")

    print("\nCosine is a similarity, not a probability. Read the ranking, and "
          "check MOA agreement across the top hits -- on held-out data a "
          "same-mechanism compound appears in the top 10 far more often than "
          "the exact one does.")


def differential(treated_path: str, control_path: str, genes: list[str],
                 geneinfo: str | None = None, log: bool = False,
                 dispersion: str = "pooled"):
    """Level-4-style differential from treated and control expression files.

    Replicates are averaged in log space first -- the stand-in for MODZ, and the
    single biggest thing a second replicate buys. The averaged contrast is then
    put on a robust z scale with the dispersion pooled across genes.

    Returns (X, names, report) with X shaped (1, n_genes).
    """
    T, t_names, t_rep = _read(treated_path, genes, geneinfo)
    C, c_names, c_rep = _read(control_path, genes, geneinfo)

    if log:
        if min(np.nanmin(T), np.nanmin(C)) < 0:
            raise SystemExit(
                "--log was given but the input contains negative values, so it "
                "is already on a log scale (or is already a differential)"
            )
        T, C = np.log2(T + 1.0), np.log2(C + 1.0)

    keep = ~(np.isnan(T).any(axis=0) | np.isnan(C).any(axis=0))
    cov = float(keep.mean())
    if cov < MIN_COVERAGE:
        raise SystemExit(
            f"only {int(keep.sum())}/{len(genes)} landmark genes ({cov:.0%}) are "
            "present in BOTH files; fix the identifier mapping"
        )
    rel = split_half_reliability(T[:, keep], C[:, keep])

    z = np.zeros(len(genes), dtype=np.float32)
    if dispersion == "per-gene":
        z[keep] = per_gene_z(T[:, keep].mean(axis=0), C[:, keep]).astype(np.float32)
    else:
        delta = T[:, keep].mean(axis=0) - C[:, keep].mean(axis=0)
        z[keep] = robust_z(delta).astype(np.float32)

    report = {
        "file": f"{treated_path}  vs  {control_path}",
        "n_queries": 1, "n_treated": len(t_names), "n_control": len(c_names),
        "treated": t_names, "control": c_names,
        "n_rows_read": t_rep["n_rows_read"], "n_matched": t_rep["n_matched"],
        "n_landmarks": len(genes), "n_present": int(keep.sum()),
        "coverage": cov, "n_duplicate_genes": t_rep["n_duplicate_genes"],
        "standardize": ("per-gene MAD (LINCS Level 4)" if dispersion == "per-gene"
                        else "robust-z, dispersion pooled across genes"),
        "dispersion": dispersion,
        "reliability": rel,
        "per_query": [{"name": "treated - control", "mean": float(z[keep].mean()),
                       "std": float(z[keep].std()),
                       "absmax": float(np.abs(z[keep]).max())}],
    }
    return z[None, :], ["treated - control"], report


if __name__ == "__main__":
    main()

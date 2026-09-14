"""Metadata-only census of a LINCS release. No GCTX required.

Run this on `siginfo_beta.txt` + `compoundinfo_beta.txt` *before* downloading
the ~35 GB level-5 matrix. It answers the question that constrains the whole
project -- how many compounds survive filtering with enough signatures to
train on -- for about 100 MB of download instead of 35 GB.

    python -m genetomol.census \\
        --siginfo data/siginfo_beta.txt \\
        --compoundinfo data/compoundinfo_beta.txt \\
        --out data/census.json

What to look at, in order:

1. The **bank size** after filtering. Recall@k against a 2,000-compound bank is
   a different (much easier) claim than against 20,000. Chance rates are
   printed so you can calibrate.
2. The **scaffold split preview**. If the test split holds 40 compounds, no
   held-out number you compute will be stable, and you should widen the split
   or reconsider the filters.
3. **MOA coverage**. False-negative masking and MOA hit rate both degrade
   quietly to no-ops when annotations are sparse -- the metrics still print,
   they just stop meaning anything.
4. The **TAS sweep**, which reports how many compounds each `tas_min`
   threshold costs. Choose the threshold from your own release rather than
   inheriting the default.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# LINCS 2020 beta names first, then the GSE92742 / GSE70138 equivalents.
_ALIASES = {
    "cell": ["cell_iname", "cell_id"],
    "dose": ["pert_dose", "pert_idose"],
    "time": ["pert_time", "pert_itime"],
    "tas": ["tas"],
    "exemplar": ["is_exemplar_sig", "is_exemplar"],
}

_TRUTHY = {"1", "1.0", "true", "True", "TRUE", "t", "y", "yes"}


def _resolve(df, key: str) -> str | None:
    for name in _ALIASES[key]:
        if name in df.columns:
            return name
    return None


def load_metadata(siginfo_path: str, compoundinfo_path: str | None):
    import pandas as pd

    logger.info("reading %s", siginfo_path)
    sig = pd.read_csv(siginfo_path, sep="\t", low_memory=False)
    logger.info("  %d rows, %d columns", len(sig), sig.shape[1])

    cmpd = None
    if compoundinfo_path:
        logger.info("reading %s", compoundinfo_path)
        cmpd = pd.read_csv(compoundinfo_path, sep="\t", low_memory=False)
        cmpd = cmpd.drop_duplicates("pert_id")
        logger.info("  %d unique pert_ids", len(cmpd))
    return sig, cmpd


def filter_cascade(sig, cmpd, tas_min: float, exemplar_only: bool, min_sigs: int,
                   validate_smiles: bool = True):
    """Apply the same filters as `data.load_l1000`, counting at each stage."""
    import pandas as pd

    stages: list[dict] = []

    def record(name: str, df, note: str = ""):
        stages.append(
            {
                "stage": name,
                "signatures": int(len(df)),
                "compounds": int(df["pert_id"].nunique()) if len(df) else 0,
                "note": note,
            }
        )

    record("all signatures", sig)

    sig = sig[sig["pert_type"] == "trt_cp"]
    record("pert_type == trt_cp", sig)

    ex_col = _resolve(sig, "exemplar")
    if exemplar_only and ex_col:
        sig = sig[sig[ex_col].astype(str).isin(_TRUTHY)]
        record(f"{ex_col} is true", sig)
    elif exemplar_only:
        stages.append({"stage": "exemplar filter", "signatures": int(len(sig)),
                       "compounds": int(sig["pert_id"].nunique()),
                       "note": "NO exemplar column found -- filter skipped"})

    tas_col = _resolve(sig, "tas")
    if tas_col:
        sig = sig[pd.to_numeric(sig[tas_col], errors="coerce") >= tas_min]
        record(f"tas >= {tas_min}", sig)
    else:
        stages.append({"stage": "tas filter", "signatures": int(len(sig)),
                       "compounds": int(sig["pert_id"].nunique()),
                       "note": "NO tas column -- signatures unfiltered for activity. "
                               "Derive TAS yourself or fall back to distil_cc_q75."})

    smiles_map: dict[str, str] = {}
    if cmpd is not None and "canonical_smiles" in cmpd:
        col = cmpd["canonical_smiles"].astype(str).str.strip()
        ok = cmpd["canonical_smiles"].notna() & (col != "")
        smiles_map = dict(zip(cmpd.loc[ok, "pert_id"], cmpd.loc[ok, "canonical_smiles"]))
        sig = sig[sig["pert_id"].isin(smiles_map)]
        record("compound has SMILES", sig)

        if validate_smiles:
            # A non-empty string is not a structure. LINCS uses the literal
            # placeholder "restricted" where the structure is withheld, which
            # passes an emptiness test and then silently becomes an all-zero
            # feature row -- and zero rows collapse into one spurious cluster.
            # Only RDKit tells you the truth.
            from rdkit import Chem, RDLogger

            RDLogger.DisableLog("rdApp.*")
            bad = {p for p, s in smiles_map.items() if Chem.MolFromSmiles(s) is None}
            if bad:
                for p in bad:
                    smiles_map.pop(p)
                sig = sig[sig["pert_id"].isin(smiles_map)]
                examples = sorted({str(s)[:24] for s in
                                   (cmpd.set_index("pert_id")["canonical_smiles"].get(p)
                                    for p in list(bad)[:20])})[:3]
                record("SMILES parses in RDKit", sig,
                       f"{len(bad)} unparsable, e.g. {examples}")
            else:
                record("SMILES parses in RDKit", sig)
    else:
        stages.append({"stage": "SMILES join", "signatures": int(len(sig)),
                       "compounds": int(sig["pert_id"].nunique()),
                       "note": "no compoundinfo given -- cannot check SMILES coverage"})

    counts = sig["pert_id"].value_counts()
    keep = counts[counts >= min_sigs].index
    sig = sig[sig["pert_id"].isin(keep)]
    record(f">= {min_sigs} signatures per compound", sig)

    return sig, smiles_map, stages


def sig_count_table(sig, thresholds=(1, 2, 3, 5, 10, 20, 50)):
    counts = sig["pert_id"].value_counts()
    arr = counts.to_numpy()
    return {
        "quantiles": {
            q: float(np.quantile(arr, q / 100)) for q in (10, 25, 50, 75, 90, 99)
        },
        "max": int(arr.max()) if len(arr) else 0,
        "compounds_at_least": {int(t): int((arr >= t).sum()) for t in thresholds},
    }


def annotation_coverage(pert_ids, cmpd):
    if cmpd is None:
        return {}
    sub = cmpd[cmpd["pert_id"].isin(set(pert_ids))]
    out: dict[str, float] = {"n_compounds": float(len(sub))}
    for col in ("moa", "target"):
        if col in sub:
            has = sub[col].notna() & (sub[col].astype(str).str.strip() != "")
            out[f"{col}_annotated"] = float(has.sum())
            out[f"{col}_frac"] = float(has.mean()) if len(sub) else 0.0
            out[f"{col}_distinct"] = float(sub.loc[has, col].nunique())
    return out


def scaffold_preview(pert_ids, smiles_map, frac_train=0.8, frac_valid=0.1, seed=0,
                     weights=None):
    """Run the *actual* split function so the preview cannot drift from the run."""
    from .data import scaffold_split
    from .featurize import bemis_murcko_scaffolds

    smiles = [smiles_map[p] for p in pert_ids]
    scaf = bemis_murcko_scaffolds(smiles)
    tr, va, te = scaffold_split(scaf, frac_train, frac_valid, seed, weights)

    arr = np.array(scaf)
    sizes = np.array([np.sum(arr == s) for s in set(arr)]) if len(arr) else np.zeros(0)
    leak = set(arr[tr]) & set(arr[te]) if len(te) else set()
    return {
        "n_compounds": int(len(pert_ids)),
        "n_scaffolds": int(len(set(scaf))),
        "singleton_scaffold_frac": float((sizes == 1).mean()) if len(sizes) else 0.0,
        "largest_scaffold": int(sizes.max()) if len(sizes) else 0,
        "train": int(len(tr)),
        "valid": int(len(va)),
        "test": int(len(te)),
        "scaffold_leak": int(len(leak)),
        "train_idx": tr, "valid_idx": va, "test_idx": te,
    }


def _fmt_int(n) -> str:
    return f"{int(n):,}"


def report(result: dict) -> None:
    p = print
    p("\n" + "=" * 78)
    p("FILTER CASCADE")
    p("=" * 78)
    p(f"{'stage':<38}{'signatures':>14}{'compounds':>12}")
    p("-" * 78)
    for s in result["cascade"]:
        p(f"{s['stage']:<38}{_fmt_int(s['signatures']):>14}{_fmt_int(s['compounds']):>12}")
        if s["note"]:
            p(f"    !! {s['note']}")

    st = result["sig_counts"]
    p("\n" + "=" * 78)
    p("SIGNATURES PER SURVIVING COMPOUND")
    p("=" * 78)
    p("  percentile " + "".join(f"{q:>8}%" for q in st["quantiles"]))
    p("  count      " + "".join(f"{v:>9.0f}" for v in st["quantiles"].values())
      + f"   (max {st['max']})")
    p("\n  compounds retaining at least N signatures:")
    p("    N        " + "".join(f"{k:>9}" for k in st["compounds_at_least"]))
    p("    count    " + "".join(f"{v:>9,}" for v in st["compounds_at_least"].values()))

    ann = result.get("annotation", {})
    if ann:
        p("\n" + "=" * 78)
        p("ANNOTATION COVERAGE (surviving compounds)")
        p("=" * 78)
        for col in ("moa", "target"):
            if f"{col}_frac" in ann:
                p(f"  {col:<8} {ann[f'{col}_annotated']:>8,.0f} / {ann['n_compounds']:,.0f} "
                  f"({100*ann[f'{col}_frac']:5.1f}%)   {ann[f'{col}_distinct']:,.0f} distinct")
        if ann.get("moa_frac", 1.0) < 0.3:
            p("    !! MOA coverage is low. False-negative masking and MOA hit rate")
            p("       both quietly become no-ops -- they will still print numbers.")

    sc = result.get("scaffold")
    if sc:
        p("\n" + "=" * 78)
        p("SCAFFOLD SPLIT PREVIEW (the real scaffold_split, same seed)")
        p("=" * 78)
        p(f"  {sc['n_compounds']:,} compounds over {sc['n_scaffolds']:,} scaffolds "
          f"({100*sc['singleton_scaffold_frac']:.1f}% singletons, "
          f"largest holds {sc['largest_scaffold']:,})")
        p(f"  train {sc['train']:,}  /  valid {sc['valid']:,}  /  test {sc['test']:,} compounds")
        p(f"  signatures: train {sc['train_sigs']:,} / valid {sc['valid_sigs']:,} "
          f"/ test {sc['test_sigs']:,}")
        p(f"  scaffold leak train<->test: {sc['scaffold_leak']}")
        if sc["test"] < 200:
            p(f"    !! only {sc['test']} held-out compounds -- recall@k will be noisy.")
            p("       Widen the split or loosen the filters before trusting a headline.")
        if sc["valid"] == 0:
            p("    !! the VALIDATION set is empty. scaffold_split assigns whole")
            p("       scaffold groups greedily: a group too large for the train")
            p("       budget falls to valid, and if it is too large for valid too it")
            p("       goes to test permanently. With few large scaffolds valid")
            p("       starves. train() silently skips validation when this happens,")
            p("       so you would not notice. Raise frac_valid, or accept that you")
            p("       are training without a validation curve.")
        want_test = 1.0 - 0.7 - 0.1
        got_test = sc["test"] / max(sc["n_compounds"], 1)
        if abs(got_test - want_test) > 0.1:
            p(f"    !! test split is {100*got_test:.0f}% of compounds, not the "
              f"{100*want_test:.0f}% requested -- scaffold groups do not divide evenly.")
        if sc["n_scaffolds"] < 50:
            p(f"    !! only {sc['n_scaffolds']} distinct scaffolds. A scaffold split")
            p("       over this few groups is coarse and its fractions are unstable.")

    ch = result["chance"]
    p("\n" + "=" * 78)
    p(f"CHANCE RECALL against a bank of {result['bank_size']:,} compounds")
    p("=" * 78)
    p("  " + "".join(f"{k:>14}" for k in ch))
    p("  " + "".join(f"{v:>14.5f}" for v in ch.values()))

    tas = result.get("tas_sweep")
    if tas:
        p("\n" + "=" * 78)
        p("TAS THRESHOLD SWEEP (after trt_cp + exemplar + SMILES)")
        p("=" * 78)
        p(f"{'tas_min':>10}{'signatures':>14}{'compounds':>12}")
        for row in tas:
            p(f"{row['tas_min']:>10.2f}{_fmt_int(row['signatures']):>14}"
              f"{_fmt_int(row['compounds']):>12}")

    mem = result["estimates"]
    p("\n" + "=" * 78)
    p("SIZE ESTIMATES for the prepared artifacts")
    p("=" * 78)
    p(f"  signatures kept (cap {mem['max_sigs_per_pert']}/compound): "
      f"{mem['n_signatures_capped']:,}")
    p(f"  signature matrix  {mem['n_signatures_capped']:,} x {mem['n_genes']} float32"
      f"  = {mem['signatures_gb']:.2f} GB")
    p(f"  ECFP bank         {result['bank_size']:,} x {mem['n_features']} float32"
      f"  = {mem['features_gb']:.2f} GB")
    p(f"  peak RAM is roughly 2-3x the signature matrix during prepare.")
    if mem["signatures_gb"] > 2.0:
        p("    !! large for an 8 GB machine -- lower --max-sigs-per-pert.")
    p("")


def run_census(
    siginfo_path: str,
    compoundinfo_path: str | None,
    tas_min: float = 0.1,
    exemplar_only: bool = False,
    min_sigs: int = 1,
    max_sigs_per_pert: int = 60,
    n_genes: int = 978,
    n_bits: int = 2048,
    n_descriptors: int = 15,
    skip_scaffold: bool = False,
    validate_smiles: bool = True,
    seed: int = 0,
) -> dict:
    import pandas as pd

    sig_all, cmpd = load_metadata(siginfo_path, compoundinfo_path)
    sig, smiles_map, cascade = filter_cascade(
        sig_all, cmpd, tas_min, exemplar_only, min_sigs, validate_smiles
    )

    pert_ids = sorted(sig["pert_id"].unique().tolist())
    bank_size = len(pert_ids)
    result: dict = {
        "params": {
            "tas_min": tas_min, "exemplar_only": exemplar_only,
            "min_sigs": min_sigs, "max_sigs_per_pert": max_sigs_per_pert, "seed": seed,
        },
        "cascade": cascade,
        "bank_size": bank_size,
        "n_signatures": int(len(sig)),
        "sig_counts": sig_count_table(sig),
        "annotation": annotation_coverage(pert_ids, cmpd),
        "chance": {f"recall@{k}": (min(k, bank_size) / bank_size if bank_size else 0.0)
                   for k in (1, 5, 10, 20, 50)},
    }

    # TAS sweep, recomputed from the pre-TAS population so the rows are comparable.
    tas_col = _resolve(sig_all, "tas")
    if tas_col:
        base = sig_all[sig_all["pert_type"] == "trt_cp"]
        ex_col = _resolve(base, "exemplar")
        if exemplar_only and ex_col:
            base = base[base[ex_col].astype(str).isin(_TRUTHY)]
        if smiles_map:
            base = base[base["pert_id"].isin(smiles_map)]
        tas_vals = pd.to_numeric(base[tas_col], errors="coerce")
        result["tas_sweep"] = [
            {
                "tas_min": t,
                "signatures": int((tas_vals >= t).sum()),
                "compounds": int(base.loc[tas_vals >= t, "pert_id"].nunique()),
            }
            for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        ]

    if not skip_scaffold and smiles_map and bank_size:
        logger.info("computing Bemis-Murcko scaffolds for %d compounds", bank_size)
        counts = sig["pert_id"].value_counts()
        w = np.array([counts.get(p, 0) for p in pert_ids], dtype=np.float64)
        sc = scaffold_preview(pert_ids, smiles_map, seed=seed, weights=w)
        pos = {p: i for i, p in enumerate(pert_ids)}
        sig_pos = sig["pert_id"].map(pos).to_numpy()
        for name in ("train", "valid", "test"):
            idx = set(sc.pop(f"{name}_idx").tolist())
            sc[f"{name}_sigs"] = int(np.isin(sig_pos, list(idx)).sum())
        result["scaffold"] = sc

    capped = int(
        sig["pert_id"].value_counts().clip(upper=max_sigs_per_pert).sum()
    )
    n_features = n_bits + n_descriptors
    result["estimates"] = {
        "max_sigs_per_pert": max_sigs_per_pert,
        "n_signatures_capped": capped,
        "n_genes": n_genes,
        "n_features": n_features,
        "signatures_gb": capped * n_genes * 4 / 1e9,
        "features_gb": bank_size * n_features * 4 / 1e9,
    }
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--siginfo", required=True, help="siginfo_beta.txt")
    ap.add_argument("--compoundinfo", default=None, help="compoundinfo_beta.txt")
    ap.add_argument("--tas-min", type=float, default=0.1,
                    help="minimum Transcriptional Activity Score. 0.1 by default: "
                         "on LINCS 2020 it costs ~10%% of compounds while 0.2 costs "
                         "~46%%, and the 0.1-0.2 band is weakly-but-reproducibly "
                         "active compounds, not noise")
    ap.add_argument("--exemplar-only", action="store_true",
                    help="keep only is_exemplar_sig rows. OFF by default: the flag "
                         "removes 81%% of signatures but only 12%% of compounds, i.e. "
                         "extra views of the same compound, which is exactly what "
                         "the encoder needs now that it gets no cell line or dose")
    ap.add_argument("--min-sigs", type=int, default=1,
                    help="compounds with fewer surviving signatures are dropped. "
                         "1 by default: a single signature is still a usable "
                         "(query, positive) pair, and the bank does not have to "
                         "equal the training set anyway")
    ap.add_argument("--max-sigs-per-pert", type=int, default=60,
                    help="only affects the size estimate, not the counts")
    ap.add_argument("--n-genes", type=int, default=978)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--skip-scaffold", action="store_true",
                    help="skip the RDKit scaffold pass (the slow part)")
    ap.add_argument("--no-validate-smiles", action="store_true",
                    help="trust non-empty SMILES strings instead of parsing them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = run_census(
        args.siginfo,
        args.compoundinfo,
        tas_min=args.tas_min,
        exemplar_only=args.exemplar_only,
        min_sigs=args.min_sigs,
        max_sigs_per_pert=args.max_sigs_per_pert,
        n_genes=args.n_genes,
        n_bits=args.n_bits,
        skip_scaffold=args.skip_scaffold,
        validate_smiles=not args.no_validate_smiles,
        seed=args.seed,
    )
    report(result)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

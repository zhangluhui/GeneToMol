"""Recall stratified by how many times a compound was profiled.

Splits held-out compounds by profiling depth and reports retrieval in each
stratum, scoring ECFP-NN in the same strata as a control.

**The bank is NOT filtered.** Every query still ranks all bank compounds, so the
strata stay comparable to an unstratified number. `--restrict-bank` additionally
shrinks the candidate set, which is an easier task and is reported separately;
do not quote it beside a full-bank number.

Reads a finished run directory; no retraining.

    python -m genetomol.depth --data ../lincs/prepared --run runs/cc_all_seed0 \\
        --mol-features ../lincs/prepared/ecfp_chiral.npz \\
        --mol-embeddings ../lincs/prepared/cc_all.npz --mol-embeddings-mode only
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .baselines import ECFPNearestNeighbor
from .evaluate import rank_queries
from .runio import load_run
from .tanimoto import breakdown, per_compound_hits
from .train import rank_bank

logger = logging.getLogger(__name__)

# Chosen to line up with the depth rows in `signal.py`'s reliability table, so a
# stratum here can be read against the noise ceiling for the same stratum.
DEFAULT_BINS = (1, 2, 4, 8, 20, 60)


def _edges(bins, depth: np.ndarray) -> tuple[float, ...]:
    """Close the top bin at the observed maximum.

    `breakdown` treats `bins[-1]` as the inclusive right edge, so an open top
    stratum has to be given a finite one. Using the data's own maximum keeps it
    honest and keeps the JSON free of `Infinity`, which is not valid JSON.
    """
    return tuple(float(b) for b in bins) + (float(max(depth.max(), bins[-1])),)


def restricted_bank_recall(run, ranking_truth, min_depth: float, k: int) -> dict:
    """Recall when the candidate set is also limited to well-profiled compounds.

    A different, easier task: shrinking the bank raises chance retrieval by the
    same factor it shrinks by, so this is reported with its own chance baseline
    and never mixed into the full-bank table.
    """
    depth = run.sigs_per_compound
    keep = np.flatnonzero(depth >= min_depth)
    if len(keep) == 0:
        return {}
    # Queries are only scoreable if their true compound survived the restriction.
    remap = np.full(len(depth), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    truth_small = remap[ranking_truth]
    rows = np.flatnonzero(truth_small >= 0)
    if len(rows) == 0:
        return {}

    sigs = run.test_ds.signatures[rows]
    mol = run.bank.features_tensor(run.device)[torch.as_tensor(keep)]
    r = rank_bank(run.model, sigs, mol, truth_small[rows], run.device)
    ids, rec = per_compound_hits(r, truth_small[rows], k)
    return {
        "min_depth": float(min_depth),
        "n_bank": int(len(keep)),
        "n_compounds": int(len(ids)),
        "macro_recall": float(rec.mean()),
        "chance": float(min(k, len(keep)) / len(keep)),
    }


def analyze(data_dir: str, run_dir: str, k: int = 10, bins=DEFAULT_BINS,
            mol_features: str | None = None, mol_embeddings: str | None = None,
            mol_embeddings_mode: str = "concat", csls_r: int = 0,
            restrict_bank: float | None = None, device: str = "cpu") -> dict:
    run = load_run(data_dir, run_dir, mol_features, mol_embeddings,
                   mol_embeddings_mode, device, want_train=True)

    truth = run.test_ds.bank_index
    mol_tensor = run.bank.features_tensor(run.device)
    model_ranking = rank_bank(run.model, run.test_ds.signatures, mol_tensor, truth,
                              run.device, csls_r=csls_r)

    ecfp = ECFPNearestNeighbor(n_bits=run.n_bits).fit(
        run.train_ds.signatures, run.train_ds.bank_index, run.bank, run.train_compounds
    )
    ecfp_ranking = rank_queries(
        lambda lo, hi: ecfp.score(run.test_ds.signatures[lo:hi]), truth, len(run.bank)
    )
    run.bank.mol_features = run.raw_features

    ids_m, rec_m = per_compound_hits(model_ranking, truth, k)
    ids_e, rec_e = per_compound_hits(ecfp_ranking, truth, k)
    assert np.array_equal(ids_m, ids_e), "compound ordering diverged"
    depth = run.sigs_per_compound[ids_m]

    rows = breakdown(rec_m, rec_e, depth, _edges(bins, depth))
    report = {
        "run": str(run_dir), "k": k, "split": run.split,
        "n_test_compounds": int(len(ids_m)),
        "n_bank": int(len(run.bank)),
        "chance": float(min(k, len(run.bank)) / len(run.bank)),
        "median_depth": float(np.median(depth)),
        "csls_r": csls_r,
        "overall": {
            "model": float(rec_m.mean()), "ecfp": float(rec_e.mean()),
            "ratio": float(rec_m.mean() / rec_e.mean()) if rec_e.mean() > 0 else float("nan"),
        },
        "bins": rows,
        "per_compound": {
            "depth": depth.tolist(), "model": rec_m.tolist(), "ecfp": rec_e.tolist(),
        },
    }
    if restrict_bank is not None:
        report["restricted_bank"] = restricted_bank_recall(run, truth, restrict_bank, k)
    return report


def format_report(rep: dict) -> str:
    k = rep["k"]
    out = [
        f"\ndepth breakdown -- {rep['run']}, {rep['split']} split, "
        f"{rep['n_test_compounds']} held-out compounds",
        f"bank {rep['n_bank']:,} compounds (NOT filtered); "
        f"chance macro recall@{k} = {rep['chance']:.5f}",
        f"median profiling depth {rep['median_depth']:.0f} signatures\n",
        f"{'depth':>12} {'n':>6} {'model':>9} {'ECFP-NN':>9} {'ratio':>8} {'vs chance':>10}",
    ]
    for r in rep["bins"]:
        lo, hi = int(r["lo"]), int(r["hi"])
        out.append(f"{f'{lo}-{hi}':>12}{r['n']:>7}{r['model']:>10.4f}"
                   f"{r['ecfp']:>10.4f}{r['ratio']:>8.2f}x"
                   f"{r['model'] / rep['chance']:>9.0f}x")
    o = rep["overall"]
    out.append(f"{'overall':>12}{rep['n_test_compounds']:>7}{o['model']:>10.4f}"
               f"{o['ecfp']:>10.4f}{o['ratio']:>8.2f}x"
               f"{o['model'] / rep['chance']:>9.0f}x")

    rb = rep.get("restricted_bank")
    if rb:
        out += [
            f"\nSEPARATE, EASIER TASK -- candidate set also cut to depth >= "
            f"{rb['min_depth']:.0f}:",
            f"  bank {rb['n_bank']:,} compounds, {rb['n_compounds']} queries scored",
            f"  macro recall@{k} {rb['macro_recall']:.4f} against chance "
            f"{rb['chance']:.4f} ({rb['macro_recall'] / rb['chance']:.0f}x)",
            "  Do not quote this beside a full-bank number.",
        ]
    out.append(
        "\nRead the ratio column, not the model column. Model recall rising with\n"
        "depth while the ratio stays flat means well-profiled compounds are just\n"
        "easier for everyone -- it is not evidence that noise was the limit."
    )
    return "\n".join(out)


def plot(report: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["bins"]
    if not rows:
        raise ValueError("no populated depth bins to plot")
    labels = [f"{int(r['lo'])}-{int(r['hi'])}" for r in rows]
    x = np.arange(len(rows))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax.bar(x - 0.2, [r["model"] for r in rows], 0.4, label="model")
    ax.bar(x + 0.2, [r["ecfp"] for r in rows], 0.4, label="ECFP-NN")
    ax.axhline(report["chance"], ls=":", c="k", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("profiling depth (signatures per compound)")
    ax.set_ylabel(f"macro recall@{report['k']}")
    ax.set_title("recall by depth (full bank)")
    ax.legend(fontsize=8)

    ax2.plot(x, [r["ratio"] for r in rows], "o-")
    ax2.axhline(1.0, ls="--", c="k", lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_xlabel("profiling depth")
    ax2.set_ylabel("model / ECFP-NN")
    ax2.set_title("the control: does the ADVANTAGE grow?")

    for a in (ax, ax2):
        a.grid(alpha=0.3)
        for n, xi in zip([r["n"] for r in rows], x):
            a.annotate(f"n={n}", (xi, a.get_ylim()[0]), fontsize=7,
                       ha="center", va="bottom", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="a finished run directory")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mol-features", default=None,
                    help="the SAME file the run trained with")
    ap.add_argument("--mol-embeddings", default=None)
    ap.add_argument("--mol-embeddings-mode", default="concat",
                    choices=("concat", "replace", "only"))
    ap.add_argument("--csls-r", type=int, default=0,
                    help="hubness correction on the model only; see hubness.py")
    ap.add_argument("--restrict-bank", type=float, default=None,
                    help="ALSO report recall with the candidate set cut to this "
                         "minimum depth. A different, easier task")
    ap.add_argument("--bins", default=",".join(str(b) for b in DEFAULT_BINS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    bins = tuple(float(b) for b in args.bins.split(","))
    rep = analyze(args.data, args.run, args.k, bins, args.mol_features,
                  args.mol_embeddings, args.mol_embeddings_mode, args.csls_r,
                  args.restrict_bank, args.device)
    print(format_report(rep))

    out = Path(rep["run"])
    (out / "depth.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    try:
        print(f"\nwrote {plot(rep, out / 'depth.png')}")
    except ImportError:
        logger.warning("matplotlib not installed; skipping the figure")
    print(f"wrote {out / 'depth.json'}")


if __name__ == "__main__":
    main()

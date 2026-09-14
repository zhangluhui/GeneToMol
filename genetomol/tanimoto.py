"""Model-vs-ECFP-NN recall, broken down by donor fingerprint similarity.

`ECFPNearestNeighbor` transfers a training compound's consensus signature to a
held-out one by fingerprint similarity, so it is strongest where some training
compound is a close analogue. This bins held-out compounds by donor Tanimoto --
how similar the nearest training compound is -- and reports both methods in each
bin, separating an advantage that is concentrated where chemistry runs out from
one that is flat across bins.

Reads a finished run directory; no retraining.

    python -m genetomol.tanimoto --data ../lincs/prepared --run runs/chiral
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .baselines import ECFPNearestNeighbor
from .evaluate import DEFAULT_KS, rank_queries
from .runio import load_run
from .train import rank_bank

logger = logging.getLogger(__name__)

DEFAULT_BINS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def per_compound_hits(ranking, truth: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-compound recall@k. Returns (compound ids, recall), aligned.

    Averages within each compound first, exactly as `macro_recall_at_k` does,
    so a compound profiled 600 times counts the same as one profiled twice.
    """
    hits = (ranking.ranks < k).astype(np.float64)
    order = np.argsort(truth, kind="stable")
    t_sorted, h_sorted = truth[order], hits[order]
    starts = np.flatnonzero(np.r_[True, t_sorted[1:] != t_sorted[:-1]])
    counts = np.diff(np.r_[starts, len(t_sorted)])
    return t_sorted[starts], np.add.reduceat(h_sorted, starts) / counts


def breakdown(
    model_recall: np.ndarray,
    ecfp_recall: np.ndarray,
    similarity: np.ndarray,
    bins: tuple[float, ...] = DEFAULT_BINS,
) -> list[dict]:
    """Group per-compound recall by donor Tanimoto."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        # Closed on the right for the last bin so Tanimoto 1.0 is not dropped.
        sel = (similarity >= lo) & ((similarity < hi) if hi < bins[-1]
                                    else (similarity <= hi))
        if not sel.any():
            continue
        m, e = float(model_recall[sel].mean()), float(ecfp_recall[sel].mean())
        out.append({
            "lo": lo, "hi": hi, "n": int(sel.sum()),
            "model": m, "ecfp": e,
            "ratio": (m / e) if e > 0 else float("inf") if m > 0 else float("nan"),
        })
    return out


def analyze(data_dir: str, run_dir: str, k: int = 10, bins=DEFAULT_BINS,
            mol_features: str | None = None, mol_embeddings: str | None = None,
            mol_embeddings_mode: str = "concat", csls_r: int = 0) -> dict:
    run = load_run(data_dir, run_dir, mol_features, mol_embeddings,
                   mol_embeddings_mode, "cpu", want_train=True)
    bank, split, n_bits = run.bank, run.split, run.n_bits
    train_ds, test_ds, tr_c = run.train_ds, run.test_ds, run.train_compounds
    model, device, raw_features = run.model, run.device, run.raw_features

    truth = test_ds.bank_index
    mol_tensor = bank.features_tensor(device)
    model_ranking = rank_bank(model, test_ds.signatures, mol_tensor, truth, device,
                              csls_r=csls_r)

    ecfp = ECFPNearestNeighbor(n_bits=n_bits).fit(
        train_ds.signatures, train_ds.bank_index, bank, tr_c
    )
    ecfp_ranking = rank_queries(
        lambda lo, hi: ecfp.score(test_ds.signatures[lo:hi]), truth, len(bank)
    )
    similarity_by_position = ecfp.max_similarity_
    bank.mol_features = raw_features

    ids_m, rec_m = per_compound_hits(model_ranking, truth, k)
    ids_e, rec_e = per_compound_hits(ecfp_ranking, truth, k)
    assert np.array_equal(ids_m, ids_e), "compound ordering diverged"
    sim = similarity_by_position[ids_m]

    keep = np.isfinite(sim)
    if not keep.all():
        # A test compound with no donor was never transferred; ECFP-NN cannot
        # score it at all, so it belongs in neither bin.
        logger.warning("%d/%d test compounds have no donor similarity; dropped",
                       int((~keep).sum()), len(keep))
    rows = breakdown(rec_m[keep], rec_e[keep], sim[keep], bins)
    return {
        "run": str(run_dir), "k": k, "split": split,
        "n_test_compounds": int(len(ids_m)),
        "median_donor_tanimoto": float(np.median(sim[keep])),
        "overall": {
            "model": float(rec_m[keep].mean()),
            "ecfp": float(rec_e[keep].mean()),
            "ratio": float(rec_m[keep].mean() / rec_e[keep].mean()),
        },
        "bins": rows,
        "per_compound": {
            "similarity": sim[keep].tolist(),
            "model": rec_m[keep].tolist(),
            "ecfp": rec_e[keep].tolist(),
        },
    }


def _cfg(d: dict):
    """Kept as a re-export; the implementation moved to `runio.config_of`."""
    from .runio import config_of

    return config_of(d)


def plot(report: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["bins"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(rows))
    labels = [f"{r['lo']:.1f}-{r['hi']:.1f}\nn={r['n']}" for r in rows]
    ax.bar(x - 0.2, [r["model"] for r in rows], 0.4, label="model")
    ax.bar(x + 0.2, [r["ecfp"] for r in rows], 0.4, label="ECFP-NN")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set(xlabel="donor Tanimoto (nearest training compound)",
           ylabel=f"macro recall@{report['k']}",
           title=f"where the advantage lives -- {Path(report['run']).name}")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ratios = [r["ratio"] for r in rows]
    ax2.bar(x, ratios, 0.6, color=["tab:green" if r > 1 else "tab:red" for r in ratios])
    ax2.axhline(1.0, color="k", lw=1, ls="--")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set(xlabel="donor Tanimoto", ylabel="model / ECFP-NN",
            title=f"ratio by bin (overall {report['overall']['ratio']:.2f}x)")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="a finished run directory")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mol-features", default=None,
                    help="MUST match what the run trained on -- a chiral rebuild "
                         "is the same width as the achiral original, so a "
                         "mismatch is silent")
    ap.add_argument("--mol-embeddings", default=None)
    ap.add_argument("--mol-embeddings-mode", default="concat",
                    choices=("concat", "replace", "only"))
    ap.add_argument("--csls-r", type=int, default=0,
                    help="hubness correction on the model's ranking; 10 is the "
                         "measured setting, 0 is off. The ECFP-NN baseline is "
                         "left uncorrected either way")
    ap.add_argument("--bins", default=",".join(str(b) for b in DEFAULT_BINS))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bins = tuple(float(b) for b in args.bins.split(","))
    report = analyze(args.data, args.run, args.k, bins,
                     args.mol_features, args.mol_embeddings, args.mol_embeddings_mode,
                     args.csls_r)

    print(f"\ndonor Tanimoto breakdown -- {report['run']}, {report['split']} split, "
          f"{report['n_test_compounds']} held-out compounds")
    print(f"median donor Tanimoto {report['median_donor_tanimoto']:.3f}\n")
    print(f"{'Tanimoto':>12} {'n':>6} {'model':>9} {'ECFP-NN':>9} {'ratio':>8}")
    for r in report["bins"]:
        print(f"{r['lo']:.1f}-{r['hi']:.1f}".rjust(12)
              + f"{r['n']:>7}{r['model']:>10.4f}{r['ecfp']:>10.4f}{r['ratio']:>8.2f}x")
    o = report["overall"]
    print(f"{'overall':>12} {report['n_test_compounds']:>6} {o['model']:>9.4f} "
          f"{o['ecfp']:>9.4f} {o['ratio']:>7.2f}x")

    out = Path(report["run"])
    (out / "tanimoto.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        print(f"\nwrote {plot(report, out / 'tanimoto.png')}")
    except ImportError:
        logger.warning("matplotlib not installed; skipping the figure")
    print(f"wrote {out / 'tanimoto.json'}")


if __name__ == "__main__":
    main()

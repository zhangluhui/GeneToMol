"""Plot what a training run produced: `history.json` and `results.json`.

Training writes both files and nothing else; this module is the only reader.
It is deliberately import-light -- matplotlib is not a runtime dependency of
the model, so it is imported here and nowhere else (`pip install -e .[plot]`).

Two figures, because they answer two different questions:

  training.png   did the run converge, and did held-out retrieval move while
                 the loss fell? The loss is per-epoch; the validation curves
                 exist only on eval epochs (`eval_every`), so they are drawn
                 as marked points on a sparse x, not resampled.
  baselines.png  is the final model actually better than looking the compound
                 up by fingerprint? Macro metrics only -- one compound, one
                 vote -- since the micro numbers are weighted by how often
                 each compound was screened.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_KS = (1, 5, 10, 20, 50)
# Subset drawn on the train-vs-validation panel, where every k costs two lines.
PLOT_KS = (1, 10, 50)

# Baseline order is the reading order of the table `print_result` emits.
BASELINE_ORDER = ("model", "ECFP-NN transfer", "cosine consensus",
                  "gene-set overlap", "chance")


def _series(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    """Epochs and values for `key`, skipping rows that lack it.

    Validation keys are present only on eval epochs, so the two axes have to
    be built together or they desynchronize.
    """
    xs, ys = [], []
    for row in history:
        if key in row:
            xs.append(row["epoch"])
            ys.append(row[key])
    return xs, ys


def _xmin(history: list[dict]) -> float:
    """Left edge of every x-axis, with room for the epoch-0 marker.

    Epoch 0 is the untrained model and epoch N means N epochs of training are
    done, so the axis starts just left of 0 rather than at it -- otherwise the
    baseline point sits on the spine and is easy to miss.
    """
    eps = [r["epoch"] for r in history if "epoch" in r]
    return min(min(eps, default=0.0), 0.0) - 0.5


def _panel(ax, history, prefix, metric, ks, label, colors):
    """Draw one split's curves for one metric family into one axes."""
    drawn = False
    for i, k in enumerate(ks):
        xs, ys = _series(history, f"{prefix}_{metric}@{k}")
        if xs:
            ax.plot(xs, ys, marker="o", ms=4, color=colors[i], label=f"@{k}")
            drawn = True
    ax.set(xlabel="epoch", ylabel=label)
    ax.set_xlim(left=_xmin(history))
    ax.grid(alpha=0.3)
    return drawn


def plot_training(history: list[dict], path: Path) -> Path:
    """Four rows: loss/schedule, then macro, micro and MOA-target retrieval.

    Train is the left column and validation the right, **sharing a y-axis per
    row** -- that shared scale is the whole point. Overlaying the two in one
    axes hides the gap once train sits near 1.0 and validation near 0.02;
    side by side on one scale, the gap is the first thing you see.

    Micro gets its own row rather than sharing with macro because the two
    differ by 2-3x on this data (micro weights by how often CMap screened each
    compound) and a shared axis would squash macro flat.

    Every x-axis starts at 0. Training evaluates at epoch 0, so the curves
    show where the run began, not where `eval_every` first fired.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(13, 17))
    ax_loss, ax_opt = axes[0]

    # --- loss -------------------------------------------------------------
    ep, nce = _series(history, "nce")
    ax_loss.plot(ep, nce, color="tab:blue", label="train (objective)")
    vx, vy = _series(history, "val_nce")
    if vx:
        ax_loss.plot(vx, vy, color="tab:red", marker="o", ms=4, label="validation")
    xs, ys = _series(history, "nce_inbatch")
    # With bank negatives off these are the same quantity up to the logit_scale
    # clamp applied between the two calls, so drawing both is a thicker line.
    if xs and not np.allclose(ys, nce, rtol=0.05):
        ax_loss.plot(xs, ys, color="tab:orange", label="train (in-batch ref)")
        n_neg = history[-1].get("n_negatives")
        if n_neg:
            ax_loss.set_title(f"loss ({int(n_neg):,} negatives/step)")
    ax_loss.legend(fontsize=8)
    ax_loss.set(xlabel="epoch", ylabel="InfoNCE loss")
    ax_loss.set_xlim(left=_xmin(history))
    if not ax_loss.get_title():
        ax_loss.set_title("loss" if vx else "training loss")
    ax_loss.grid(alpha=0.3)
    if vx and len(vy) > 2 and vy[-1] > min(vy):
        # The rise is the learned temperature, not degradation: it correlates
        # with logit_scale at +0.85 and with val recall at +0.73. Saying so on
        # the figure stops the curve being read as overfitting.
        ax_loss.annotate("rising val loss tracks the\nlearned temperature,\n"
                         "not degradation -- read recall",
                         xy=(0.97, 0.03), xycoords="axes fraction", fontsize=7,
                         ha="right", va="bottom", alpha=0.65)

    # --- schedule ---------------------------------------------------------
    # Each series carries its own x. The untrained baseline row has `lr` and
    # `temperature` but no `nce`, so reusing the loss curve's epochs here
    # silently mismatched lengths and matplotlib raised.
    lx, lr = _series(history, "lr")
    ax_opt.plot(lx, lr, color="tab:green", label="learning rate")
    ax_opt.set(xlabel="epoch", ylabel="learning rate")
    ax_opt.set_yscale("log")
    ax_opt.set_xlim(left=_xmin(history))
    ax_temp = ax_opt.twinx()
    tx, temp = _series(history, "temperature")
    ax_temp.plot(tx, temp, color="tab:red", label="temperature")
    ax_temp.set_ylabel("temperature")
    ax_opt.set_title("schedule")
    lines = ax_opt.get_lines() + ax_temp.get_lines()
    ax_opt.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax_opt.grid(alpha=0.3)

    # --- retrieval: one row per metric, train left, validation right -------
    colors = [f"C{i}" for i in range(len(PLOT_KS))]
    rows = [
        ("macro_recall", "macro recall", PLOT_KS, "per compound"),
        ("recall", "micro recall", PLOT_KS, "per signature"),
    ]
    for r, (metric, label, ks, note) in enumerate(rows, start=1):
        left, right = axes[r]
        for ax, prefix, who in ((left, "train", "train probe"),
                                (right, "val", "validation")):
            drawn = _panel(ax, history, prefix, metric, ks, label, colors)
            ax.set_title(f"{who}: exact-compound retrieval ({note})")
            if drawn:
                ax.legend(fontsize=8, ncol=len(ks))
            else:
                ax.text(0.5, 0.5, f"no {prefix}_{metric} in history",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, alpha=0.6)
        _share_y(left, right)
        _annotate_range(right, history, "val", metric, ks)
        # `newsig` is held-out signatures of TRAIN compounds -- neither probe,
        # and only present on compound-disjoint runs. Drawn on the train panel
        # because the compound was seen; dotted because the signature was not.
        for i, k in enumerate(ks):
            xs, ys = _series(history, f"newsig_{metric}@{k}")
            if xs:
                left.plot(xs, ys, ls=":", marker="s", ms=3, color=colors[i],
                          alpha=0.8, label=f"newsig @{k}")
        if any(f"newsig_{metric}@{k}" in r_ for r_ in history for k in ks):
            left.legend(fontsize=7, ncol=2)
            left.set_title(left.get_title()
                           + "\ndotted = newsig (seen compound, new signature)")

    # --- MOA / target -----------------------------------------------------
    left, right = axes[3]
    for ax, prefix, who in ((left, "train", "train probe"),
                            (right, "val", "validation")):
        drawn = False
        for j, (fam, style) in enumerate((("moa", "-"), ("target", "--"))):
            for i, k in enumerate((1, 10, 50)):
                xs, ys = _series(history, f"{prefix}_{fam}_hit@{k}")
                if xs:
                    ax.plot(xs, ys, style, marker="o", ms=4, color=f"C{i}",
                            alpha=1.0 if fam == "moa" else 0.6,
                            label=f"{fam} @{k}")
                    drawn = True
        ax.set(xlabel="epoch", ylabel="hit rate")
        ax.set_xlim(left=_xmin(history))
        ax.set_title(f"{who}: MOA (solid) / target (dashed) hit rate")
        ax.grid(alpha=0.3)
        if drawn:
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.text(0.5, 0.5, f"no {prefix}_moa_hit in history",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, alpha=0.6)
    _share_y(left, right)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _annotate_range(ax, history, prefix, metric, ks):
    """Spell out first -> last on a panel the shared y-scale has flattened.

    The shared scale is what makes the train/validation gap visible, and it is
    also what can squash a low-valued curve into a flat line at zero. Both
    matter, so the movement is written out rather than rescaled away.
    """
    parts = []
    # Trained epochs only. Including the untrained baseline would report a
    # ratio against chance (hundreds of x) and read as a training improvement.
    trained = [r for r in history if r.get("epoch", 1) >= 1]
    for k in ks:
        xs, ys = _series(trained, f"{prefix}_{metric}@{k}")
        if len(ys) > 1 and ys[0] > 0:
            parts.append(f"@{k}: {ys[0]:.4f} -> {ys[-1]:.4f} ({ys[-1]/ys[0]:.2f}x)")
        elif ys:
            parts.append(f"@{k}: {ys[0]:.4f} -> {ys[-1]:.4f}")
    if parts:
        ax.annotate("\n".join(parts), xy=(0.03, 0.97), xycoords="axes fraction",
                    fontsize=7, ha="left", va="top", alpha=0.75,
                    family="monospace")


def _share_y(*axes):
    """One y-scale across a row, so train and validation are comparable.

    Without this the two panels autoscale independently and a train curve at
    0.95 looks the same shape as a validation curve at 0.02 -- which is the
    exact comparison the split into columns exists to make.
    """
    lims = [ax.get_ylim() for ax in axes if ax.has_data()]
    if len(lims) < 2:
        return
    lo, hi = min(l for l, _ in lims), max(h for _, h in lims)
    for ax in axes:
        ax.set_ylim(lo, hi)


def plot_baselines(result: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = result["rows"]
    names = [n for n in BASELINE_ORDER if n in rows]
    names += [n for n in rows if n not in names]

    fig, (ax_recall, ax_hit) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(DEFAULT_KS))
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        vals = [rows[name].get(f"macro_recall@{k}", float("nan")) for k in DEFAULT_KS]
        ax_recall.bar(x + i * width, vals, width, label=name)
    ax_recall.set_xticks(x + width * (len(names) - 1) / 2)
    ax_recall.set_xticklabels([f"@{k}" for k in DEFAULT_KS])
    ax_recall.set(ylabel="macro recall",
                  title=f"held-out compound retrieval ({result['split']} split, "
                        f"bank of {result['n_bank']:,})")
    ax_recall.legend(fontsize=8)
    ax_recall.grid(axis="y", alpha=0.3)

    # Only rows that were scored for annotation get this panel. The baselines
    # currently emit rank metrics alone, so in practice this is the model's
    # MOA rate beside its target rate -- the pair evaluate.py calls the honest
    # headline, since retrieving *a* compound with the right mechanism is the
    # thing a screen actually wants.
    hit_kinds = [(n, kind) for n in names for kind in ("moa", "target")
                 if f"{kind}_hit@{DEFAULT_KS[0]}" in rows[n]]
    hit_width = 0.8 / max(len(hit_kinds), 1)
    for i, (name, kind) in enumerate(hit_kinds):
        vals = [rows[name].get(f"{kind}_hit@{k}", float("nan")) for k in DEFAULT_KS]
        label = kind if len(names) == 1 else f"{name} -- {kind}"
        ax_hit.bar(x + i * hit_width, vals, hit_width, label=label)
    ax_hit.set_xticks(x + hit_width * (max(len(hit_kinds), 1) - 1) / 2)
    ax_hit.set_xticklabels([f"@{k}" for k in DEFAULT_KS])
    n_scored = rows["model"].get("moa_n_scored")
    title = "held-out annotation hit rate"
    if n_scored:
        title += f" ({int(n_scored):,} annotated queries)"
    ax_hit.set(ylabel="hit rate", title=title)
    ax_hit.legend(fontsize=8)
    ax_hit.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_run(run_dir: Path, out_dir: Path | None = None) -> list[Path]:
    """Draw every figure a run directory has the data for."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    history_path = run_dir / "history.json"
    results_path = run_dir / "results.json"
    if not history_path.exists() and not results_path.exists():
        raise FileNotFoundError(f"no history.json or results.json under {run_dir}")

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        written.append(plot_training(history, out_dir / "training.png"))
    if results_path.exists():
        result = json.loads(results_path.read_text(encoding="utf-8"))
        written.append(plot_baselines(result, out_dir / "baselines.png"))
    return written


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="a run directory, e.g. runs/scaffold")
    ap.add_argument("--out", default=None, help="where to write PNGs (default: run_dir)")
    args = ap.parse_args()

    for path in plot_run(Path(args.run_dir), args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

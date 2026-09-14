"""Data-side diagnostics, computed without a trained model.

Two measurements that bound any retrieval number:

**Noise ceiling.** Split a compound's signatures into two halves, build a
consensus from each, and take the cosine between them. That is the most any two
consensus vectors could agree given L1000 replicate noise, in the same units as
every cross-compound similarity. Reported overall and by profiling depth.

**Structure-response correlation.** Correlate Tanimoto similarity with
consensus-signature similarity across compound pairs, with and without a
correction for that noise ceiling. Near zero bounds what any structure-only
encoder can reach.

Also provides the treatment-quality filters `train.py` exposes as `--min-dose`,
`--min-time` and `--drop-outlier-frac`.

    python -m genetomol.signal --data ../lincs/prepared
    python -m genetomol.signal --data ../lincs/prepared --run runs/chiral \\
        --mol-features ../lincs/prepared/ecfp_chiral.npz
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .data import (
    load_artifacts,
    signature_mask,
    split_cache_dir,
    subset_dataset,
    subset_to_memmap,
)
from .featurize import tanimoto_matrix

logger = logging.getLogger(__name__)

CHUNK = 4096          # rows per streamed block; 4096 x 978 float32 is 16 MB
DEPTH_BINS = ((4, 6), (6, 10), (10, 20), (20, 60), (60, 10 ** 9))


# --------------------------------------------------------------------------
# per-signature quality
# --------------------------------------------------------------------------

def quality_scores(ds, slot: np.ndarray, n_candidates: int):
    """Per-signature L2 norm and leave-one-out agreement with its compound.

    The LOO score is the cosine between a signature and the consensus of its
    compound's *other* signatures. Comparing against a consensus the signature
    helped define would flatter every borderline signature, which matters
    exactly where the score is used -- at the cutoff.

    Two streamed passes so nothing larger than a block is ever resident.
    """
    n = len(ds)
    genes = ds.signatures.shape[1]
    bank_index = ds.bank_index

    total = np.zeros((n_candidates, genes), dtype=np.float32)
    counts = np.zeros(n_candidates, dtype=np.int64)
    norm = np.zeros(n, dtype=np.float32)
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        block = np.asarray(ds.signatures[lo:hi], dtype=np.float32)
        norm[lo:hi] = np.linalg.norm(block, axis=1)
        s = slot[bank_index[lo:hi]]
        take = s >= 0
        if take.any():
            np.add.at(total, s[take], block[take])
            np.add.at(counts, s[take], 1)

    loo = np.full(n, np.nan, dtype=np.float32)
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        s = slot[bank_index[lo:hi]]
        take = s >= 0
        if not take.any():
            continue
        block = np.asarray(ds.signatures[lo:hi][take], dtype=np.float32)
        sel = s[take]
        ref = (total[sel] - block) / np.maximum(counts[sel] - 1, 1)[:, None]
        num = (block * ref).sum(1)
        den = np.linalg.norm(block, axis=1) * np.linalg.norm(ref, axis=1)
        out = np.full(hi - lo, np.nan, dtype=np.float32)
        out[take] = num / np.maximum(den, 1e-9)
        loo[lo:hi] = out
    return norm, loo


def _row_groups(ds, slot, keep: np.ndarray) -> dict[int, np.ndarray]:
    """Row indices per candidate compound, among rows passing `keep`."""
    idx = np.flatnonzero(keep)
    s = slot[ds.bank_index[idx]]
    order = np.argsort(s, kind="stable")
    idx, s = idx[order], s[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return dict(zip(s[starts], np.split(idx, starts[1:])))


def _halves(ds, rows: np.ndarray, k: int, rng):
    """Two disjoint k-signature consensus vectors from `rows`."""
    pick = rng.choice(rows, 2 * k, replace=False)
    a = np.asarray(ds.signatures[np.sort(pick[:k])], dtype=np.float32).mean(0)
    b = np.asarray(ds.signatures[np.sort(pick[k:])], dtype=np.float32).mean(0)
    return a, b


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan


# --------------------------------------------------------------------------
# the noise ceiling
# --------------------------------------------------------------------------

def noise_ceiling(ds, min_signatures: int = 4, seed: int = 0) -> dict:
    """Split-half self-cosine per compound, and how it varies with depth.

    Uses every available signature rather than a fixed count, so the result is
    the ceiling as the data actually is -- which is why it must be read
    alongside the depth breakdown rather than as a single number.
    """
    counts = np.bincount(ds.bank_index)
    cand = np.flatnonzero(counts >= min_signatures)
    slot = np.full(counts.shape[0], -1, dtype=np.int64)
    slot[cand] = np.arange(len(cand))
    genes = ds.signatures.shape[1]

    rng = np.random.default_rng(seed)
    half = rng.integers(0, 2, len(ds))
    acc = np.zeros((2, len(cand), genes), dtype=np.float32)
    n = np.zeros((2, len(cand)), dtype=np.int64)
    for lo in range(0, len(ds), CHUNK):
        hi = min(lo + CHUNK, len(ds))
        s = slot[ds.bank_index[lo:hi]]
        take = s >= 0
        if not take.any():
            continue
        block = np.asarray(ds.signatures[lo:hi][take], dtype=np.float32)
        h = half[lo:hi][take]
        for side in (0, 1):
            m = h == side
            if m.any():
                np.add.at(acc[side], s[take][m], block[m])
                np.add.at(n[side], s[take][m], 1)

    ok = (n[0] >= 2) & (n[1] >= 2)
    a = acc[0][ok] / n[0][ok][:, None]
    b = acc[1][ok] / n[1][ok][:, None]
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    self_cos = (a * b).sum(1)
    depth = counts[cand][ok]

    by_depth = []
    for lo_d, hi_d in DEPTH_BINS:
        m = (depth >= lo_d) & (depth < hi_d)
        if m.sum() > 20:
            label = f">={lo_d}" if hi_d > 10 ** 8 else f"{lo_d}-{hi_d - 1}"
            by_depth.append({"depth": label, "n": int(m.sum()),
                             "self_cosine": float(self_cos[m].mean())})

    sample = np.random.default_rng(seed).choice(len(a), min(2000, len(a)), replace=False)
    x = a[sample]
    cross = (x @ x.T)[np.triu_indices(len(sample), 1)]
    return {
        "n_compounds": int(ok.sum()),
        "self_cosine_mean": float(self_cos.mean()),
        "self_cosine_median": float(np.median(self_cos)),
        "random_pair_cosine": float(cross.mean()),
        "by_depth": by_depth,
    }


# --------------------------------------------------------------------------
# do quality filters raise it?
# --------------------------------------------------------------------------

def treatment_dose(ds) -> np.ndarray:
    """Dose in uM, undoing `prepare.py`'s log storage.

    A signature with no recorded dose was stored as 0 before the log, so it
    comes back as 0.0 uM and fails any positive `min_dose`. That is inherited
    from `prepare.py` and is what the measured filter table in the README was
    computed with -- but it means a dose filter silently also drops
    unknown-dose signatures, so `signature_filter` counts them separately.
    """
    return (10.0 ** np.asarray(ds.log_dose) - 1e-3).astype(np.float32)


def treatment_time(ds) -> np.ndarray:
    """Duration in hours. Missing duration was filled with 24 h, so unlike dose
    it *passes* a `>= 24` filter rather than failing it. The asymmetry is
    `prepare.py`'s; it is documented here because it surprises people."""
    return (10.0 ** np.asarray(ds.log_time) - 1e-3).astype(np.float32)


def signature_filter(ds, min_dose: float | None = None, min_time: float | None = None,
                     drop_outlier_frac: float = 0.0, n_compounds: int | None = None,
                     restrict: np.ndarray | None = None,
                     min_compound_sigs: int | None = None) -> tuple[np.ndarray, dict]:
    """Keep-mask over signatures, by treatment quality. Returns (keep, stats).

    Four criteria. Three drop individual signatures: `min_dose`, `min_time`,
    and `drop_outlier_frac`, the last by leave-one-out agreement with the
    compound's other replicates.

    `min_compound_sigs` is different in kind and needs its own warning: it drops
    whole compounds rather than individual measurements. Because a minority of
    compounds carry most of the signatures, it leaves the signature encoder
    nearly as much data while sharply cutting the distinct compounds the
    molecule encoder sees.

    `restrict` limits *which* signatures are eligible to be dropped -- pass the
    training mask to filter training data while leaving the benchmark alone.
    Signatures outside it are kept untouched.

    The outlier threshold is a quantile of the LOO score taken over all
    eligible signatures independently of dose and time, then intersected, to
    match how `build_filters` composes them. Compounds with fewer than
    two signatures have no leave-one-out consensus and are **exempt** rather
    than scored as outliers -- without that they all score ~0 and the filter
    would quietly delete every singleton compound from training.
    """
    n = len(ds)
    eligible = np.ones(n, dtype=bool) if restrict is None else np.asarray(restrict, bool)
    keep = np.ones(n, dtype=bool)
    stats: dict = {"n_total": int(n), "n_eligible": int(eligible.sum())}
    if n_compounds is None:
        n_compounds = int(ds.bank_index.max()) + 1

    if min_compound_sigs is not None:
        # Profiling depth over the WHOLE dataset, not within the split. The
        # splits are compound-disjoint, so a compound's total depth is also its
        # depth wherever it lives, and taking it globally means the criterion
        # does not silently change meaning when `restrict` changes.
        depth = np.bincount(ds.bank_index, minlength=n_compounds)
        keep &= (depth[ds.bank_index] >= min_compound_sigs) | ~eligible
        stats["after_compound_depth"] = int((keep & eligible).sum())

    if min_dose is not None:
        dose = treatment_dose(ds)
        stats["n_no_dose"] = int((~np.asarray(ds.has_dose, bool) & eligible).sum())
        keep &= (dose >= min_dose) | ~eligible
        stats["after_dose"] = int((keep & eligible).sum())
    if min_time is not None:
        keep &= (treatment_time(ds) >= min_time) | ~eligible
        stats["after_time"] = int((keep & eligible).sum())

    if drop_outlier_frac > 0.0:
        counts = np.bincount(ds.bank_index, minlength=n_compounds)
        scorable = counts[ds.bank_index] >= 2
        _, loo = quality_scores(ds, np.arange(n_compounds), n_compounds)
        pool = eligible & scorable & np.isfinite(loo)
        if pool.any():
            thr = float(np.nanquantile(loo[pool], drop_outlier_frac))
            keep &= (loo >= thr) | ~pool
            stats["loo_threshold"] = thr
            stats["n_exempt_singleton"] = int((eligible & ~scorable).sum())
        stats["after_outliers"] = int((keep & eligible).sum())

    stats["n_kept"] = int((keep & eligible).sum())
    stats["frac_kept"] = (stats["n_kept"] / max(stats["n_eligible"], 1))
    return keep, stats


def build_filters(dose, time, norm, loo, live, drop_quantile: float = 0.25) -> dict:
    q_norm = np.quantile(norm[live], drop_quantile)
    q_loo = np.nanquantile(loo[live], drop_quantile)
    return {
        "none (baseline)": np.ones(len(dose), dtype=bool),
        "dose >= 1 uM": dose >= 1.0,
        "dose >= 10 uM": dose >= 10.0,
        "time >= 24 h": time >= 24.0,
        "dose>=1 & time>=24": (dose >= 1.0) & (time >= 24.0),
        "drop weakest 25% (L2)": norm >= q_norm,
        "drop 25% outliers (LOO)": loo >= q_loo,
        "LOO + dose>=1 + t>=24": (loo >= q_loo) & (dose >= 1.0) & (time >= 24.0),
    }


def filter_comparison(ds, slot, n_candidates, filters: dict, k: int = 4,
                      seed: int = 0) -> list[dict]:
    """Self-cosine under each filter, at matched depth and matched compounds.

    Both controls are load-bearing. Self-cosine rises steeply with depth, so a
    filter that drops signatures would look worse for that reason alone; and a
    strict filter that happens to retain only well-behaved compounds would look
    better for the wrong reason. Fixing k per half and intersecting the
    compound sets removes both, leaving only signature quality.
    """
    live = slot[ds.bank_index] >= 0
    groups, eligible = {}, np.ones(n_candidates, dtype=bool)
    for name, mask in filters.items():
        groups[name] = _row_groups(ds, slot, mask & live)
        have = np.zeros(n_candidates, dtype=np.int64)
        for cid, rows in groups[name].items():
            have[cid] = len(rows)
        eligible &= have >= 2 * k
    common = np.flatnonzero(eligible)

    rng = np.random.default_rng(seed)
    out, base = [], None
    for name, mask in filters.items():
        vals = [_cos(*_halves(ds, groups[name][c], k, rng)) for c in common]
        m = float(np.nanmean(vals))
        base = m if base is None else base
        out.append({"filter": name, "kept": float((mask & live).sum() / live.sum()),
                    "self_cosine": m, "delta": m - base, "n_compounds": len(common)})
    return out


# --------------------------------------------------------------------------
# does structure predict the response?
# --------------------------------------------------------------------------

def chem_bio_correlation(ds, bank, slot, n_candidates, filters: dict,
                         n_bits: int, k: int = 6, seed: int = 0) -> list[dict]:
    """corr(Tanimoto, consensus-signature cosine), raw and disattenuated.

    Noise in the signature drags any correlation toward zero, so the raw number
    confounds "no relationship" with "relationship buried in noise". The
    disattenuated column divides by sqrt(reliability), where reliability is the
    split-half correlation stepped up by Spearman-Brown -- what the correlation
    would be against a perfectly reproducible signature. Tanimoto itself is
    measured without error, so only the biology side needs correcting.

    A filter defined by agreement-with-replicates (the LOO one) inflates
    reliability by construction without adding signal, so its disattenuated
    figure is an underestimate. Read the dose/time arm, whose criteria are
    independent of the metric.
    """
    live = slot[ds.bank_index] >= 0
    groups, eligible = {}, np.ones(n_candidates, dtype=bool)
    for name, mask in filters.items():
        groups[name] = _row_groups(ds, slot, mask & live)
        have = np.zeros(n_candidates, dtype=np.int64)
        for cid, rows in groups[name].items():
            have[cid] = len(rows)
        eligible &= have >= 2 * k
    common = np.flatnonzero(eligible)

    cand = np.flatnonzero(slot >= 0)
    order = np.argsort(slot[cand])
    positions = cand[order][common]
    bits = np.ascontiguousarray(bank.mol_features[positions, :n_bits])
    iu = np.triu_indices(len(common), 1)
    chem = tanimoto_matrix(bits, bits)[iu]

    rng = np.random.default_rng(seed)
    genes = ds.signatures.shape[1]
    out = []
    for name in filters:
        a = np.zeros((len(common), genes), dtype=np.float32)
        b = np.zeros((len(common), genes), dtype=np.float32)
        for j, c in enumerate(common):
            a[j], b[j] = _halves(ds, groups[name][c], k, rng)
        an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
        bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        half = float((an * bn).sum(1).mean())
        reliability = 2 * half / (1 + half)          # Spearman-Brown to depth 2k
        full = (a + b) / 2
        fn = full / np.maximum(np.linalg.norm(full, axis=1, keepdims=True), 1e-9)
        bio = (fn @ fn.T)[iu]
        r = float(np.corrcoef(chem, bio)[0, 1])
        out.append({"filter": name, "reliability": reliability, "corr": r,
                    "disattenuated": r / np.sqrt(max(reliability, 1e-9)),
                    "n_compounds": len(common)})
    return out


def annotation_similarity(ds, bank, slot, n_candidates, n_bits: int,
                          k: int = 6, seed: int = 0) -> dict:
    """Do compounds sharing a TARGET have more similar signatures than random?

    Structure is a proxy for what a compound does; the target is the thing
    itself. Scored against the same split-half ceiling as the structural
    comparison, so "fraction of achievable range" is directly comparable
    between the two.

    Caveat worth carrying: annotated compounds are the well-studied ones, which
    are also the heavily-profiled ones, so this subset is easier than the bank.
    """
    live = slot[ds.bank_index] >= 0
    groups = _row_groups(ds, slot, live)
    have = np.zeros(n_candidates, dtype=np.int64)
    for cid, rows in groups.items():
        have[cid] = len(rows)

    cand = np.flatnonzero(slot >= 0)
    cand = cand[np.argsort(slot[cand])]
    moa = [bank.moa[i] for i in cand]
    tgt = [set(bank.targets[i]) for i in cand]
    annotated = np.array([bool(m) and bool(t) for m, t in zip(moa, tgt)])
    common = np.flatnonzero((have >= 2 * k) & annotated)
    if len(common) < 50:
        return {"n_compounds": int(len(common)), "note": "too few annotated compounds"}

    rng = np.random.default_rng(seed)
    genes = ds.signatures.shape[1]
    a = np.zeros((len(common), genes), dtype=np.float32)
    b = np.zeros((len(common), genes), dtype=np.float32)
    for j, c in enumerate(common):
        a[j], b[j] = _halves(ds, groups[c], k, rng)
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    ceiling = float((an * bn).sum(1).mean())
    full = (a + b) / 2
    fn = full / np.maximum(np.linalg.norm(full, axis=1, keepdims=True), 1e-9)

    iu = np.triu_indices(len(common), 1)
    bio = (fn @ fn.T)[iu]
    pos = cand[common]
    bits = np.ascontiguousarray(bank.mol_features[pos, :n_bits])
    chem = tanimoto_matrix(bits, bits)[iu]
    m_arr = np.array([moa[c] for c in common], dtype=object)
    same_moa = (m_arr[:, None] == m_arr[None, :])[iu]
    t_list = [tgt[c] for c in common]
    same_tgt = np.array([bool(t_list[i] & t_list[j]) for i, j in zip(*iu)])

    base = float(bio[~same_moa & ~same_tgt & (chem < 0.4)].mean())

    def row(name, mask):
        if mask.sum() < 20:
            return None
        v = float(bio[mask].mean())
        return {"group": name, "n_pairs": int(mask.sum()), "bio_cosine": v,
                "fraction_of_ceiling": (v - base) / max(ceiling - base, 1e-9)}

    rows = [r for r in (
        row("unrelated (baseline)", ~same_moa & ~same_tgt & (chem < 0.4)),
        row("Tanimoto >= 0.7", chem >= 0.7),
        row("same target", same_tgt),
        row("same MOA", same_moa),
        row("same target, Tanimoto < 0.4", same_tgt & (chem < 0.4)),
    ) if r]
    return {"n_compounds": int(len(common)), "ceiling": ceiling,
            "baseline": base, "rows": rows}


def descriptor_correlation(ds, bank, slot, n_candidates, path: str,
                          k: int = 6, seed: int = 0, n_sample: int = 1200) -> dict:
    """corr(descriptor similarity, consensus-signature similarity), per block.

    Answers two things at once. Which blocks of a descriptor file actually
    predict the response -- so a subset can be chosen rather than throwing every
    column at an encoder that already memorizes. And whether any block is
    **circular**: a descriptor derived from the same L1000 data would correlate
    far above every other block, which is the tell. Chemical Checker's D level
    is "cells" and may include a transcriptomics space; a D-inclusive run would
    otherwise look excellent for the wrong reason.

    Honest blocks land near 0.10 disattenuated on this data, about 1% of
    variance. Anything several times that deserves suspicion before celebration.
    """
    from .molembed import load_blocks, load_embeddings, select_blocks

    live = slot[ds.bank_index] >= 0
    groups = _row_groups(ds, slot, live)
    have = np.zeros(n_candidates, dtype=np.int64)
    for cid, rows in groups.items():
        have[cid] = len(rows)
    common = np.flatnonzero(have >= 2 * k)
    rng = np.random.default_rng(seed)
    if len(common) > n_sample:
        common = np.sort(rng.choice(common, n_sample, replace=False))

    genes = ds.signatures.shape[1]
    a = np.zeros((len(common), genes), dtype=np.float32)
    b = np.zeros((len(common), genes), dtype=np.float32)
    for j, c in enumerate(common):
        a[j], b[j] = _halves(ds, groups[c], k, rng)

    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)

    half = float((unit(a) * unit(b)).sum(1).mean())
    reliability = 2 * half / (1 + half)
    full = unit((a + b) / 2)
    iu = np.triu_indices(len(common), 1)
    bio = (full @ full.T)[iu]

    cand = np.flatnonzero(slot >= 0)
    positions = cand[np.argsort(slot[cand])][common]
    blocks = load_blocks(path)
    # Slice to the sampled compounds immediately; the full matrix is 3,200
    # columns for all 25 CC spaces and only these rows are ever needed.
    emb = load_embeddings(path, bank.ids)[positions]

    rows = []
    for name in (blocks or ["<whole file>"]):
        x = unit(select_blocks(emb, blocks, [name]) if blocks else emb)
        r = float(np.corrcoef((x @ x.T)[iu], bio)[0, 1])
        rows.append({"block": name, "corr": r,
                     "disattenuated": r / np.sqrt(max(reliability, 1e-9))})
    rows.sort(key=lambda d: -abs(d["disattenuated"]))
    return {"path": path, "n_compounds": int(len(common)),
            "reliability": reliability, "blocks": rows}


# --------------------------------------------------------------------------

def main():
    import argparse

    from .train import make_split

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mol-features", default=None,
                    help="use the same fingerprints a run was trained on")
    ap.add_argument("--split", default="scaffold", choices=("scaffold", "random", "moa"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-total", type=int, default=24,
                    help="candidate compounds need this many signatures, so a "
                         "filter can still leave enough for a matched-depth split")
    ap.add_argument("--k-filter", type=int, default=4, help="signatures per half")
    ap.add_argument("--k-corr", type=int, default=6)
    ap.add_argument("--descriptor", default=None,
                    help="an embeddings .npz; report per-block correlation with "
                         "the response, and flag any block that looks circular")
    ap.add_argument("--out", default=None, help="write the report as JSON")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bank, ds, meta = load_artifacts(args.data)
    n_bits = meta.get("n_bits", 2048)
    if args.mol_features:
        from .molembed import load_embeddings

        bank.mol_features = load_embeddings(args.mol_features, bank.ids)

    sigs_per = np.bincount(ds.bank_index, minlength=len(bank)).astype(np.float64)
    tr, _, _ = make_split(bank, args.split, args.seed, 0.8, 0.1, weights=sigs_per)
    mask = signature_mask(ds.bank_index, tr)
    cache = split_cache_dir(ds)
    tag = f"{args.split}_{args.seed}_{len(bank)}"
    train_ds = (subset_to_memmap(ds, mask, cache / f"{tag}_train.npy")
                if cache is not None else subset_dataset(ds, mask))
    dose = (10 ** ds.log_dose[mask] - 1e-3).astype(np.float32)
    time = (10 ** ds.log_time[mask] - 1e-3).astype(np.float32)
    ds.release()

    report = {"data": args.data, "split": args.split, "seed": args.seed}

    ceiling = noise_ceiling(train_ds, seed=args.seed)
    report["noise_ceiling"] = ceiling
    print(f"\n=== noise ceiling: same compound, disjoint halves ===")
    print(f"  {ceiling['n_compounds']:,} compounds | self-cosine "
          f"mean {ceiling['self_cosine_mean']:+.3f} median "
          f"{ceiling['self_cosine_median']:+.3f} | random pairs "
          f"{ceiling['random_pair_cosine']:+.3f}")
    for row in ceiling["by_depth"]:
        print(f"    {row['depth']:>8} sigs  n={row['n']:>6,}  {row['self_cosine']:+.3f}")

    counts = np.bincount(train_ds.bank_index, minlength=len(bank))
    cand = np.flatnonzero(counts >= args.min_total)
    slot = np.full(len(bank), -1, dtype=np.int64)
    slot[cand] = np.arange(len(cand))
    norm, loo = quality_scores(train_ds, slot, len(cand))
    live = slot[train_ds.bank_index] >= 0
    filters = build_filters(dose, time, norm, loo, live)

    rows = filter_comparison(train_ds, slot, len(cand), filters, k=args.k_filter,
                             seed=args.seed)
    report["filters"] = rows
    print(f"\n=== does filtering raise it? (depth fixed at {args.k_filter}/half, "
          f"{rows[0]['n_compounds']:,} identical compounds) ===")
    print(f"  {'filter':<26}{'kept':>7}{'self-cosine':>13}{'vs base':>10}")
    for r in rows:
        print(f"  {r['filter']:<26}{r['kept']:>6.0%}{r['self_cosine']:>13.3f}"
              f"{r['delta']:>+10.3f}")

    rows = chem_bio_correlation(train_ds, bank, slot, len(cand),
                                {k: filters[k] for k in
                                 ("none (baseline)", "dose>=1 & time>=24",
                                  "LOO + dose>=1 + t>=24")},
                                n_bits, k=args.k_corr, seed=args.seed)
    report["chem_bio"] = rows
    print(f"\n=== does structure predict the response? "
          f"({rows[0]['n_compounds']:,} compounds, depth {2 * args.k_corr}) ===")
    print(f"  {'arm':<26}{'reliability':>12}{'corr':>9}{'disattenuated':>15}")
    for r in rows:
        print(f"  {r['filter']:<26}{r['reliability']:>12.3f}{r['corr']:>9.3f}"
              f"{r['disattenuated']:>15.3f}")

    ann = annotation_similarity(train_ds, bank, slot, len(cand), n_bits,
                                k=args.k_corr, seed=args.seed)
    report["annotation"] = ann
    if ann.get("rows"):
        print(f"\n=== does a shared TARGET predict the response? "
              f"({ann['n_compounds']:,} annotated compounds, ceiling "
              f"{ann['ceiling']:.3f}) ===")
        print(f"  {'group':<30}{'pairs':>10}{'bio cosine':>12}{'% of ceiling':>14}")
        for r in ann["rows"]:
            print(f"  {r['group']:<30}{r['n_pairs']:>10,}{r['bio_cosine']:>12.3f}"
                  f"{r['fraction_of_ceiling']:>13.1%}")

    if args.descriptor:
        d = descriptor_correlation(train_ds, bank, slot, len(cand), args.descriptor,
                                   k=args.k_corr, seed=args.seed)
        report["descriptor"] = d
        print(f"\n=== which blocks of {args.descriptor} predict the response? "
              f"({d['n_compounds']:,} compounds) ===")
        print(f"  {'block':>10} {'corr':>9} {'disattenuated':>15} {'var explained':>15}")
        for r in d["blocks"]:
            print(f"  {r['block']:>10} {r['corr']:>9.3f} {r['disattenuated']:>15.3f}"
                  f"{100 * r['disattenuated'] ** 2:>14.2f}%")
        top = d["blocks"][0]
        rest = np.median([abs(r["disattenuated"]) for r in d["blocks"][1:]]) if len(d["blocks"]) > 1 else 0
        if rest and abs(top["disattenuated"]) > 3 * rest:
            print(f"\n  WARNING: {top['block']} is {abs(top['disattenuated'])/rest:.1f}x "
                  f"the median block. A descriptor derived from this same L1000 "
                  f"data would look exactly like this -- check its provenance "
                  f"before training on it.")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

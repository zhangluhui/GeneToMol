"""Is retrieval being crowded out by hub compounds, and does correcting it help?

High-dimensional cosine retrieval has a well-documented pathology: a few items
sit near everything and occupy top-k lists regardless of the query. The true
compound is then pushed out not because the model ranks it badly but because the
same handful of compounds always rank well. If that is happening, it is fixable
at inference with no retraining.

Two things are reported.

**k-occurrence** -- how many of the N queries put each compound in their top-k.
Uniform would be `k*N/P`. What matters is the tail: if the top 1% of compounds
occupy far more than 1% of all slots, the space is hub-dominated, and the count
of compounds that never appear says how much of the bank is unreachable.

**CSLS** -- rank by `2*cos(q,k) - mean_top-r cos(q',k)` instead of `cos(q,k)`,
subtracting each compound's average similarity to its r nearest queries. Hubs
are penalized, rare compounds promoted. Standard hubness correction from
cross-lingual retrieval.

Reads a finished run directory; no retraining.

    python -m genetomol.hubness --data ../lincs/prepared --run runs/cc_all_seed0 \\
        --mol-features ../lincs/prepared/ecfp_chiral.npz \\
        --mol-embeddings ../lincs/prepared/cc_all.npz --mol-embeddings-mode only
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .data import (
    load_artifacts,
    signature_mask,
    split_cache_dir,
    subset_dataset,
    subset_to_memmap,
)
from .evaluate import DEFAULT_KS, hub_penalty as _bank_hub_penalty
from .featurize import DescriptorScaler
from .train import build_model, make_split

logger = logging.getLogger(__name__)


def _macro(hits: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(truth, kind="stable")
    t, h = truth[order], hits[order].astype(np.float64)
    starts = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    counts = np.diff(np.r_[starts, len(t)])
    return float((np.add.reduceat(h, starts) / counts).mean())


def query_embeddings(model, ds, device, chunk: int = 1024) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for lo in range(0, len(ds), chunk):
            block = np.asarray(ds.signatures[lo:lo + chunk], dtype=np.float32)
            out.append(model.signature_encoder(torch.as_tensor(block, device=device)))
    return torch.cat(out)


def hub_penalty(keys: torch.Tensor, queries: torch.Tensor, r: int = 10,
                chunk: int = 256) -> torch.Tensor:
    """Mean cosine from each compound to its `r` nearest queries.

    Chunked over compounds: the full (P, N) similarity is 24k x 37k, which is
    3.5 GB as float32 and the reason this is the memory-critical step.

    This takes already-encoded queries because the diagnostic needs them
    anyway. `evaluate.hub_penalty` is the same quantity computed from raw
    signatures with nothing but a (P, r) buffer held between blocks; that is
    the one `rank_bank --csls-r` uses, and `_agrees_with_evaluate` checks the
    two against each other.
    """
    pen = torch.zeros(keys.shape[0], device=keys.device)
    with torch.no_grad():
        for lo in range(0, keys.shape[0], chunk):
            s = keys[lo:lo + chunk] @ queries.t()
            pen[lo:lo + chunk] = s.topk(min(r, s.shape[1]), dim=1).values.mean(1)
    return pen


def _agrees_with_evaluate(model, ds, keys, device, r, penalty) -> float:
    """Max absolute disagreement with the streaming penalty `rank_bank` uses.

    Same quantity, different loop order, so this should be at float32 noise.
    Cheap enough to always run, and it is the only thing tying the diagnostic
    to what `--csls-r` will actually do in production.
    """
    other = _bank_hub_penalty(model, ds.signatures, keys, device, r)
    return float((other - penalty).abs().max())


def rank_with(keys, queries, truth, penalty=None, ks=DEFAULT_KS, chunk: int = 1024):
    """Ranks and top-k ids, optionally with the CSLS correction applied."""
    n, max_k = len(truth), max(ks)
    ranks = np.empty(n, dtype=np.int64)
    topk = np.empty((n, max_k), dtype=np.int64)
    t_all = torch.as_tensor(truth, device=keys.device)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            s = queries[lo:hi] @ keys.t()
            if penalty is not None:
                s = 2 * s - penalty.unsqueeze(0)
            t = t_all[lo:hi].unsqueeze(1)
            ranks[lo:hi] = (s > s.gather(1, t)).sum(1).cpu().numpy()
            topk[lo:hi] = s.topk(max_k, dim=1).indices.cpu().numpy()
    return ranks, topk


def analyze(data_dir: str, run_dir: str, mol_features=None, mol_embeddings=None,
            mol_embeddings_mode="concat", csls_r: int = 10, device="cpu",
            ks=DEFAULT_KS) -> dict:
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    cfg_d, split = result["config"], result["split"]
    dev = torch.device(device)

    bank, ds, meta = load_artifacts(data_dir)
    n_bits = meta.get("n_bits", 2048)
    if mol_features:
        from .molembed import load_embeddings

        bank.mol_features = load_embeddings(mol_features, bank.ids)
    bank.fingerprints = np.ascontiguousarray(bank.mol_features[:, :n_bits])
    if mol_embeddings:
        from .molembed import attach

        attach(bank, mol_embeddings, mol_embeddings_mode, n_bits)

    expected = torch.load(run_dir / "model.pt", map_location="cpu")["model"][
        "molecule_encoder.net.0.weight"].shape[1]
    if bank.n_features != expected:
        raise SystemExit(
            f"{run_dir} trained on {expected} molecule features but this data "
            f"gives {bank.n_features}; pass the same --mol-features / "
            "--mol-embeddings the run used"
        )

    sigs_per = np.bincount(ds.bank_index, minlength=len(bank)).astype(np.float64)
    tr, _, te = make_split(bank, split, cfg_d["seed"], 0.8, 0.1, weights=sigs_per)
    bank.mol_features = DescriptorScaler(n_bits).fit_transform(bank.mol_features, tr)
    cache = split_cache_dir(ds)
    tag = f"{split}_{cfg_d['seed']}_{len(bank)}"
    mask = signature_mask(ds.bank_index, te)
    test_ds = (subset_to_memmap(ds, mask, cache / f"{tag}_test.npy")
               if cache is not None else subset_dataset(ds, mask))
    ds.release()

    from .train import TrainConfig

    known = {f: cfg_d[f] for f in TrainConfig.__dataclass_fields__ if f in cfg_d}
    for f in ("sig_hidden", "mol_hidden"):
        if isinstance(known.get(f), list):
            known[f] = tuple(known[f])
    model = build_model(bank, test_ds.signatures.shape[1], TrainConfig(**known),
                        dev, n_bits=n_bits)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu")["model"])
    model.eval()

    with torch.no_grad():
        keys = model.encode_bank(bank.features_tensor(dev))
    queries = query_embeddings(model, test_ds, dev)
    truth = test_ds.bank_index
    n, p = len(truth), keys.shape[0]
    logger.info("%d test queries against %d compounds", n, p)

    plain_ranks, plain_topk = rank_with(keys, queries, truth, None, ks)
    penalty = hub_penalty(keys, queries, csls_r)
    agree = _agrees_with_evaluate(model, test_ds, keys, dev, csls_r, penalty)
    logger.info("penalty agrees with evaluate.hub_penalty to %.2e", agree)
    csls_ranks, _ = rank_with(keys, queries, truth, penalty, ks)

    occ = np.bincount(plain_topk[:, :max(ks)].ravel(), minlength=p)
    uniform = max(ks) * n / p
    top1pct = int(max(1, 0.01 * p))
    return {
        "run": str(run_dir), "n_queries": int(n), "n_bank": int(p),
        "csls_r": csls_r, "penalty_agreement": agree,
        "cosine": {f"macro_recall@{k}": _macro(plain_ranks < k, truth) for k in ks},
        "csls": {f"macro_recall@{k}": _macro(csls_ranks < k, truth) for k in ks},
        "occupancy": {
            "uniform": uniform,
            "max": int(occ.max()),
            "p99": float(np.percentile(occ, 99)),
            "median": float(np.median(occ)),
            "never_retrieved": int((occ == 0).sum()),
            "never_retrieved_frac": float((occ == 0).mean()),
            "top1pct_share": float(np.sort(occ)[::-1][:top1pct].sum() / max(occ.sum(), 1)),
        },
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--mol-features", default=None)
    ap.add_argument("--mol-embeddings", default=None)
    ap.add_argument("--mol-embeddings-mode", default="concat",
                    choices=("concat", "replace", "only"))
    ap.add_argument("--csls-r", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rep = analyze(args.data, args.run, args.mol_features, args.mol_embeddings,
                  args.mol_embeddings_mode, args.csls_r, args.device)

    print(f"\n{rep['n_queries']:,} test queries against {rep['n_bank']:,} compounds\n")
    ks = [k for k in DEFAULT_KS]
    print(f"{'':<10}" + "".join(f"{'macro@' + str(k):>12}" for k in ks))
    for label in ("cosine", "csls"):
        print(f"{label:<10}" + "".join(
            f"{rep[label][f'macro_recall@{k}']:>12.5f}" for k in ks))
    delta = (rep["csls"]["macro_recall@10"] / rep["cosine"]["macro_recall@10"] - 1)
    print(f"\nCSLS (r={rep['csls_r']}) changes macro recall@10 by {delta:+.1%}")

    o = rep["occupancy"]
    print(f"\nk-occurrence in top-{max(ks)} (uniform would be {o['uniform']:.1f}):")
    print(f"  max {o['max']:,}   p99 {o['p99']:.0f}   median {o['median']:.0f}")
    print(f"  never retrieved: {o['never_retrieved']:,} compounds "
          f"({o['never_retrieved_frac']:.0%} of the bank)")
    print(f"  top 1% of compounds hold {o['top1pct_share']:.1%} of all slots "
          f"(1.0% means no hubness)")

    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

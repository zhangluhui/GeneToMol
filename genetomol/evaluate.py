"""Retrieval metrics and the diagnostics that decide whether this worked.

Report instance-level Recall@k and MRR so the numbers are comparable to
GeneSpeak-FP, but treat MOA-level and target-level hit rate as the honest
headline: exact-compound retrieval from a 30k bank is punishing and slightly
meaningless when 27 compounds can share a mechanism.

The signature encoder takes the expression vector alone, so every number here
is already the deployment number: a disease-vs-healthy differential arrives
with no cell line, no dose and no timepoint, and the model never had them
either.

**Nothing here materializes a (queries x bank) score matrix.** At LINCS scale
that matrix is 37k x 24k: 3.5 GB as float32, and `argpartition` over it wants
another 6.6 GB for its int64 output. Every metric in this module is a function
of just two per-query quantities -- the rank of the true compound, and the
identity of the top-k compounds -- so `Ranking` carries those and the score
matrix is consumed in chunks and thrown away. Pass a `Ranking` (via
`rank_queries`) on real data; passing a dense array still works and is the
convenient thing at smoke-test scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

DEFAULT_KS = (1, 5, 10, 20, 50)

# Queries per chunk when streaming. The transient cost is dominated by
# argpartition's int64 output, ~8 * chunk * n_bank bytes: 190 MB per chunk at
# 1024 queries against a 24k bank.
CHUNK_SIZE = 1024


@dataclass
class Ranking:
    """What every metric below actually needs, for one evaluation set.

    `ranks[i]` is the 0-based rank of the true compound for query i, and
    `topk[i]` holds the best-first column indices of its top compounds. This
    is a few MB where the score matrix it came from is a few GB.
    """

    ranks: np.ndarray      # (N,) int64
    topk: np.ndarray       # (N, <= max_k) int32, best-first
    n_bank: int

    @property
    def max_k(self) -> int:
        return int(self.topk.shape[1])


def _rank_chunk(scores: np.ndarray, truth: np.ndarray, max_k: int):
    """Ranks and top-k ids for one block of queries.

    Ties count against you: an item scoring exactly equal to the truth does not
    increment the rank, so this is the optimistic tie-break. With continuous
    cosine scores ties are vanishingly rare; with `GeneSetOverlap` they are not,
    which is one more reason that baseline reads differently.
    """
    n_rows, n_cols = scores.shape
    truth_score = scores[np.arange(n_rows), truth]
    ranks = np.count_nonzero(scores > truth_score[:, None], axis=1)

    k = min(max_k, n_cols)
    if k == n_cols:
        part = np.tile(np.arange(n_cols), (n_rows, 1))
    else:
        # Partition from the ascending side and keep the tail. `argpartition`
        # on `-scores` would be the obvious spelling and costs a full extra
        # copy of the block for nothing.
        part = np.argpartition(scores, n_cols - k, axis=1)[:, n_cols - k:]
    sel = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-sel, axis=1, kind="stable")
    topk = np.take_along_axis(part, order, axis=1)
    return ranks.astype(np.int64), topk.astype(np.int32)


def rank_queries(
    score_fn: Callable[[int, int], np.ndarray],
    truth: np.ndarray,
    n_bank: int,
    max_k: int = max(DEFAULT_KS),
    chunk_size: int = CHUNK_SIZE,
) -> Ranking:
    """Stream queries through `score_fn` and keep only the ranking.

    `score_fn(lo, hi)` returns the (hi - lo, n_bank) scores for that block of
    queries, as a NumPy array or a torch tensor. It is called once per block
    and its result is discarded before the next one, which is the whole point.
    """
    truth = np.asarray(truth, dtype=np.int64)
    n = len(truth)
    if n == 0:
        return Ranking(
            np.zeros(0, np.int64),
            np.zeros((0, min(max_k, n_bank)), np.int32),
            n_bank,
        )

    ranks, topk = [], []
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        s = score_fn(lo, hi)
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        r, t = _rank_chunk(np.asarray(s), truth[lo:hi], max_k)
        ranks.append(r)
        topk.append(t)
        del s
    return Ranking(np.concatenate(ranks), np.concatenate(topk, axis=0), n_bank)


def as_ranking(x, truth: np.ndarray, max_k: int = max(DEFAULT_KS)) -> Ranking:
    """Accept either a `Ranking` or a dense score matrix.

    The dense path is for small banks -- the smoke test, a notebook. It builds
    the arrays the streaming path would have built, in one block.
    """
    if isinstance(x, Ranking):
        return x
    s = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    truth = np.asarray(truth, dtype=np.int64)
    if not len(truth):
        return Ranking(np.zeros(0, np.int64), np.zeros((0, 0), np.int32), s.shape[1])
    ranks, topk = _rank_chunk(s, truth, max_k)
    return Ranking(ranks, topk, s.shape[1])


def _per_compound_mean(values: np.ndarray, truth: np.ndarray) -> float:
    """Average within each true compound, then across compounds."""
    order = np.argsort(truth, kind="stable")
    t_sorted, v_sorted = truth[order], values[order]
    starts = np.flatnonzero(np.r_[True, t_sorted[1:] != t_sorted[:-1]])
    counts = np.diff(np.r_[starts, len(t_sorted)])
    return float((np.add.reduceat(v_sorted.astype(np.float64), starts) / counts).mean())


def recall_at_k(ranking, truth: np.ndarray, ks=DEFAULT_KS) -> dict:
    """Micro-averaged: every *signature* counts once.

    Weighted by how often each compound was profiled, so heavily-profiled
    compounds dominate. Use `macro_recall_at_k` for a per-compound number.
    """
    r = as_ranking(ranking, truth).ranks
    return {f"recall@{k}": float((r < k).mean()) for k in ks}


def macro_recall_at_k(ranking, truth: np.ndarray, ks=DEFAULT_KS) -> dict:
    """Macro-averaged: every *compound* counts once, however often it was profiled.

    Each compound's own hit rate is computed over its signatures, then those
    per-compound rates are averaged. This is the number that matches the claim
    "given a transcriptional state, we can identify the compound behind it",
    because it asks about compounds rather than about assay wells.

    The difference is not cosmetic. Profiling depth in LINCS reflects how
    screening effort was allocated -- approved drugs and tool compounds were run
    hundreds of times -- and those are also the compounds with the strongest,
    cleanest, most retrievable signatures. Micro-averaging therefore weights the
    metric toward the easy cases, and does so for a reason that has nothing to
    do with biology.

    Note this is a separate issue from split balance. A signature-weighted split
    equalises signatures-per-compound *across* train/valid/test; it does nothing
    about the skew *within* the test set, which is what this fixes.
    """
    truth = np.asarray(truth, dtype=np.int64)
    r = as_ranking(ranking, truth).ranks
    out = {f"macro_recall@{k}": _per_compound_mean(r < k, truth) for k in ks}
    out["n_compounds_scored"] = float(len(np.unique(truth)))
    return out


def mean_reciprocal_rank(ranking, truth: np.ndarray) -> float:
    r = as_ranking(ranking, truth).ranks
    return float((1.0 / (r + 1)).mean())


def macro_mean_reciprocal_rank(ranking, truth: np.ndarray) -> float:
    """MRR with every compound weighted equally. See `macro_recall_at_k`."""
    truth = np.asarray(truth, dtype=np.int64)
    r = as_ranking(ranking, truth).ranks
    return _per_compound_mean(1.0 / (r + 1), truth)


def group_hit_at_k(
    ranking,
    truth: np.ndarray,
    group_of: list[str | None],
    ks=DEFAULT_KS,
    prefix: str = "group",
) -> dict:
    """Does the top-k contain *any* compound from the true one's group?

    Pass MOA labels for MOA hit rate, or target-gene labels for target hit
    rate. Queries whose true compound has no annotation are skipped, and the
    count of skipped queries is returned so the denominator is auditable.

    The true compound is itself a member of its group, so this is bounded below
    by Recall@k by construction. That is the conventional definition; do not
    read a high MOA hit rate as evidence of anything beyond it without also
    reporting Recall@k.
    """
    truth = np.asarray(truth, dtype=np.int64)
    r = as_ranking(ranking, truth, max_k=max(ks))

    codes: dict[str, int] = {}
    gid = np.full(len(group_of), -1, dtype=np.int64)
    for i, g in enumerate(group_of):
        if g:
            gid[i] = codes.setdefault(g, len(codes))

    truth_gid = gid[truth]
    keep = truth_gid >= 0
    n_skipped = int((~keep).sum())

    res: dict[str, float] = {}
    if keep.any():
        # Group membership is an equality test on the group id, which is what
        # the set intersection this replaces was computing -- one array
        # comparison instead of a Python set per query.
        hit = gid[r.topk[keep]] == truth_gid[keep][:, None]
        for k in ks:
            res[f"{prefix}_hit@{k}"] = float(
                hit[:, : min(k, r.max_k)].any(axis=1).mean()
            )
    else:
        for k in ks:
            res[f"{prefix}_hit@{k}"] = float("nan")

    res[f"{prefix}_n_skipped"] = float(n_skipped)
    res[f"{prefix}_n_scored"] = float(len(truth) - n_skipped)
    return res


def summarize(ranking, truth: np.ndarray, bank) -> dict:
    """Full metric block for one evaluation set.

    `ranking` is a `Ranking` from `rank_queries`, or a dense score matrix.
    """
    truth = np.asarray(truth, dtype=np.int64)
    if len(truth) == 0:
        return {"n_queries": 0.0}
    r = as_ranking(ranking, truth)

    out: dict[str, float] = {"n_queries": float(len(truth))}
    out.update(macro_recall_at_k(r, truth))          # lead with the per-compound number
    out["macro_mrr"] = macro_mean_reciprocal_rank(r, truth)
    out.update(recall_at_k(r, truth))
    out["mrr"] = mean_reciprocal_rank(r, truth)
    out.update(group_hit_at_k(r, truth, bank.moa, prefix="moa"))
    out.update(group_hit_at_k(r, truth, bank.primary_targets(), prefix="target"))
    return out


@torch.no_grad()
def as_writable_tensor(block: np.ndarray, device=None) -> torch.Tensor:
    """Tensor sharing memory with `block`, copying only if it is read-only.

    A memmap opened `mode="r"` yields non-writable slices, and `torch.as_tensor`
    on one produces a tensor torch believes it may write to -- it warns, and any
    in-place op would then fault against a read-only mapping. Nothing writes to
    these today, but the obvious next feature (adding noise to the query as
    train-time augmentation) is spelled `x.add_(...)`, so this closes it now
    rather than after the segfault. The copy is skipped entirely for in-RAM
    arrays, and costs 4 MB per 1024-query block when it happens.
    """
    if not block.flags.writeable:
        block = np.array(block)
    return torch.as_tensor(block, device=device)


def hub_penalty(
    model,
    signatures: np.ndarray,
    keys: torch.Tensor,
    device,
    r: int = 10,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor | None:
    """Each compound's mean cosine to its `r` nearest queries -- the CSLS term.

    A hub is a compound that sits close to many queries at once; subtracting
    this quantity penalizes exactly that. See `genetomol.hubness` for
    the measurement that motivates it.

    The full (compounds x queries) similarity is 3.5 GB on the LINCS test
    split, so it is never held: queries stream in blocks and only a running
    (compounds, r) top-r buffer survives between them, which is under 1 MB.
    Peak residency matches the ranking loop's own (compounds x block).

    Returns `None` when there is nothing to correct, so callers can pass the
    result straight through as an optional penalty.
    """
    n = len(signatures)
    if r <= 0 or n == 0:
        return None
    buf = None
    with torch.no_grad():
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            q = model.signature_encoder(as_writable_tensor(signatures[lo:hi], device))
            s = keys @ q.t()
            if buf is not None:
                s = torch.cat([buf, s], dim=1)
            buf = s.topk(min(r, s.shape[1]), dim=1).values
            del s, q
    return buf.mean(1)


def rank_bank(
    model,
    signatures: np.ndarray,
    mol_features: torch.Tensor,
    truth: np.ndarray,
    device,
    batch_size: int = CHUNK_SIZE,
    max_k: int = max(DEFAULT_KS),
    csls_r: int = 0,
) -> Ranking:
    """Rank the full bank for every query, one block at a time.

    The bank is encoded once; only the query block and its scores are ever
    resident.

    `csls_r > 0` ranks by `2*cos(q, k) - hub_penalty(k)` instead of `cos(q, k)`,
    which demotes compounds that are close to many queries at once. It costs one
    extra streaming pass over the queries and needs the whole query set up front,
    so it is a batch-retrieval correction, not a per-query one. Off by default.
    """
    model.eval()
    keys = model.encode_bank(mol_features)
    penalty = hub_penalty(model, signatures, keys, device, csls_r, batch_size)

    def score_fn(lo: int, hi: int):
        q = model.signature_encoder(as_writable_tensor(signatures[lo:hi], device))
        s = q @ keys.t()
        return s if penalty is None else 2 * s - penalty.unsqueeze(0)

    return rank_queries(
        score_fn, truth, int(keys.shape[0]), max_k=max_k, chunk_size=batch_size
    )


def chance_baseline(n_bank: int, ks=DEFAULT_KS) -> dict:
    """Expected Recall@k for a uniformly random ranking of `n_bank` compounds."""
    return {f"recall@{k}": min(k, n_bank) / n_bank for k in ks}


def format_comparison(
    rows: dict[str, dict], ks=DEFAULT_KS, prefix: str = ""
) -> str:
    """Render {name: metric_dict} as the table you paste into the writeup.

    `prefix="macro_"` selects the per-compound numbers.

    Rows carrying nothing at this prefix are skipped rather than rendered as a
    line of `nan` -- the duplicate-ceiling row is macro-only by definition, so
    it belongs in the MACRO table and nowhere else. The column width still
    comes from every row, so the two tables stay aligned with each other.
    """
    keys = [f"{prefix}recall@{k}" for k in ks]
    width = max((len(n) for n in rows), default=10) + 2
    lines = [f"{'':<{width}}" + "".join(f"{'recall@' + str(k):>12}" for k in ks)]
    for name, r in rows.items():
        if not any(k in r for k in keys):
            continue
        lines.append(
            f"{name:<{width}}"
            + "".join(f"{r.get(k, float('nan')):>12.4f}" for k in keys)
        )
    return "\n".join(lines)

"""Reference baselines to score a run against.

`ECFPNearestNeighbor` assigns each held-out compound the consensus signature of
its nearest training compound by Tanimoto, then ranks by cosine.

`CosineConsensus` is the L1000CDS2-style baseline; `GeneSetOverlap` is the
L2S2-style one, ranking by overlap of up/down gene sets rather than by cosine.
"""

from __future__ import annotations

import logging

import numpy as np

from .featurize import tanimoto_matrix

logger = logging.getLogger(__name__)


def consensus_signatures(
    signatures: np.ndarray,
    bank_index: np.ndarray,
    n_compounds: int,
    chunk_rows: int = 32768,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean signature per compound. Returns (P, G) and a coverage mask.

    A plain mean, not the MODZ weighted average CMap uses.

    Accumulated `chunk_rows` at a time, a block of whole compounds per pass, so
    the full signature matrix is never promoted to float64 at once.
    """
    g = signatures.shape[1]
    out = np.zeros((n_compounds, g), dtype=np.float32)
    covered = np.zeros(n_compounds, dtype=bool)
    if not len(signatures):
        return out, covered

    order = np.argsort(bank_index, kind="stable")
    sorted_idx = bank_index[order]
    starts = np.flatnonzero(np.r_[True, sorted_idx[1:] != sorted_idx[:-1]])
    counts = np.diff(np.r_[starts, len(sorted_idx)])
    groups = sorted_idx[starts]

    # Blocks are cut on compound boundaries, so each compound is summed inside
    # exactly one block and no cross-block accumulation is needed.
    lo_g = 0
    while lo_g < len(starts):
        hi_g = lo_g + 1
        row_lo = starts[lo_g]
        while hi_g < len(starts) and starts[hi_g] - row_lo < chunk_rows:
            hi_g += 1
        row_hi = starts[hi_g] if hi_g < len(starts) else len(sorted_idx)

        block = signatures[order[row_lo:row_hi]].astype(np.float64)
        sums = np.add.reduceat(block, starts[lo_g:hi_g] - row_lo, axis=0)
        out[groups[lo_g:hi_g]] = (sums / counts[lo_g:hi_g, None]).astype(np.float32)
        del block, sums
        lo_g = hi_g

    covered[groups] = True
    return out, covered


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (_unit_rows(a) @ _unit_rows(b).T).astype(np.float32)


class CosineConsensus:
    """Rank compounds by cosine between the query and their consensus signature.

    Cannot score a compound with no training signature: those columns are -inf.
    That is the honest representation of the limitation your model is meant to
    remove, so do not quietly impute them.
    """

    def __init__(self):
        self.consensus_: np.ndarray | None = None
        self.covered_: np.ndarray | None = None

    def fit(self, signatures, bank_index, n_compounds):
        self.consensus_, self.covered_ = consensus_signatures(
            signatures, bank_index, n_compounds
        )
        # Normalized once at fit time: `score` is called per query block when
        # the caller streams, and re-normalizing a 24k x 978 bank on every
        # block is pure waste.
        self.keys_ = _unit_rows(self.consensus_)
        return self

    def score(self, queries: np.ndarray) -> np.ndarray:
        s = _unit_rows(queries) @ self.keys_.T
        s[:, ~self.covered_] = -np.inf
        return s


class ECFPNearestNeighbor:
    """Transfer a training compound's consensus signature by fingerprint similarity.

    For every bank compound without a training signature, borrow the consensus
    signature of its most Tanimoto-similar training compound. Chemistry alone,
    no learning.

    `n_bits` restricts the Tanimoto to the leading fingerprint block; `k`
    averages over that many donors; `chunk_size` bounds the similarity matrix
    held in memory at once.

    After `fit`, `max_similarity_` holds the donor Tanimoto per transferred
    compound.
    """

    def __init__(self, n_bits: int = 2048, k: int = 1, chunk_size: int = 512):
        self.n_bits = n_bits
        self.k = k
        self.chunk_size = chunk_size
        self.max_similarity_: np.ndarray | None = None

    def fit(self, signatures, bank_index, bank, train_compound_positions):
        self.consensus_, covered = consensus_signatures(
            signatures, bank_index, len(bank)
        )
        train_pos = np.asarray(train_compound_positions, dtype=np.int64)
        train_pos = train_pos[covered[train_pos]]
        if not len(train_pos):
            raise ValueError("no covered training compounds to transfer from")

        fp_source = (bank.mol_features if getattr(bank, "fingerprints", None) is None
                     else bank.fingerprints)
        train_fp = fp_source[train_pos][:, : self.n_bits]
        need = np.flatnonzero(~covered)
        self.max_similarity_ = np.full(len(bank), np.nan, dtype=np.float32)

        if len(need):
            # The full (needed x train) Tanimoto matrix is ~370 MB on the LINCS
            # scaffold split and `tanimoto_matrix` holds several arrays that
            # size at once, so donors are resolved a block of compounds at a
            # time. Only the donor index and the donor similarity survive.
            need_fp = fp_source[need][:, : self.n_bits]
            transferred = np.empty(
                (len(need), self.consensus_.shape[1]), dtype=self.consensus_.dtype
            )
            for lo in range(0, len(need), self.chunk_size):
                hi = min(lo + self.chunk_size, len(need))
                sim = tanimoto_matrix(need_fp[lo:hi], train_fp)
                if self.k == 1:
                    transferred[lo:hi] = self.consensus_[train_pos[sim.argmax(1)]]
                else:
                    top = np.argsort(-sim, axis=1)[:, : self.k]
                    w = np.take_along_axis(sim, top, 1)
                    w = w / (w.sum(1, keepdims=True) + 1e-9)
                    transferred[lo:hi] = np.einsum(
                        "nk,nkg->ng", w, self.consensus_[train_pos[top]]
                    )
                self.max_similarity_[need[lo:hi]] = sim.max(1)
                del sim
            self.consensus_[need] = transferred
            del transferred
            logger.info(
                "ECFP-NN transferred %d compounds; donor Tanimoto median %.3f",
                len(need), float(np.median(self.max_similarity_[need])),
            )
            covered = covered.copy()
            covered[need] = True
        self.covered_ = covered
        self.keys_ = _unit_rows(self.consensus_)  # see CosineConsensus.fit
        return self

    def score(self, queries: np.ndarray) -> np.ndarray:
        s = _unit_rows(queries) @ self.keys_.T
        s[:, ~self.covered_] = -np.inf
        return s


class GeneSetOverlap:
    """L2S2-style ranking by signed overlap of up/down gene sets.

    Take the top and bottom `n_set` genes of the query and of each consensus
    signature, then score as
        |up n up| + |dn n dn| - |up n dn| - |dn n up|
    normalized by set size. For a reversal query, negate the query first rather
    than flipping the sign of the score, so that ties break the same way.

    Implemented as a single matmul over signed indicator matrices, which is
    exactly equivalent to the set arithmetic and roughly four orders of
    magnitude faster than the nested loop at 30k compounds.
    """

    def __init__(self, n_set: int = 100):
        self.n_set = n_set

    @staticmethod
    def _signed_indicator(x: np.ndarray, n: int) -> np.ndarray:
        order = np.argsort(x, axis=1)
        ind = np.zeros(x.shape, dtype=np.float32)
        rows = np.arange(len(x))[:, None]
        ind[rows, order[:, -n:]] = 1.0
        ind[rows, order[:, :n]] = -1.0
        return ind

    def fit(self, signatures, bank_index, n_compounds):
        cons, self.covered_ = consensus_signatures(signatures, bank_index, n_compounds)
        self.indicator_ = self._signed_indicator(cons, self.n_set)
        return self

    def score(self, queries: np.ndarray) -> np.ndarray:
        q = self._signed_indicator(queries, self.n_set)
        out = (q @ self.indicator_.T) / self.n_set
        out[:, ~self.covered_] = -np.inf
        return out


"""Encoders and loss for compound retrieval from a transcriptomic state transition.

One objective:

  L = InfoNCE(signature, compound)

with same-MOA pairs masked out of the denominator.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes: list[int], dropout: float = 0.1, norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SignatureEncoder(nn.Module):
    """MLP over landmark z-scores. Nothing else.

    Cell line, dose and timepoint conditioning was removed on purpose. The
    query at deployment is a disease-vs-healthy differential that carries none
    of them, so the context-free path was the only one whose number could ever
    be quoted; training a conditioned path alongside it bought a number that
    was not reportable and risked letting the encoder retrieve by "which
    compounds were run on MCF7" rather than by biology.

    `SignatureDataset` still carries `cell_index`, `log_dose` and `log_time` --
    they are needed for splitting and for diagnostics such as "is retrieval
    carried by a single cell line". The model simply does not see them.
    """

    def __init__(
        self,
        n_genes: int = 978,
        embed_dim: int = 256,
        hidden: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trunk = mlp([n_genes, *hidden, embed_dim], dropout=dropout)

    def forward(self, signature: torch.Tensor) -> torch.Tensor:  # (B, n_genes)
        return F.normalize(self.trunk(signature), dim=-1)


class MoleculeEncoder(nn.Module):
    """ECFP bits + standardized descriptors -> shared space.

    This is the only thing that lets the model score a compound it never saw in
    training. Keep it a function of structure alone.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 256,
        hidden: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
        bit_dropout: float = 0.0,
        n_bits: int | None = None,
    ):
        super().__init__()
        self.net = mlp([in_dim, *hidden, embed_dim], dropout=dropout)
        self.embed_dim = embed_dim
        self.bit_dropout = bit_dropout
        self.n_bits = n_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.bit_dropout > 0.0:
            # Drop fingerprint *bits* before the first weight matrix. `mlp`
            # puts its dropout after each hidden layer, so without this the raw
            # fingerprint reaches Linear(2063->1024) intact on every step -- and
            # ECFP bits are near-unique per compound, so a 2.8M-parameter
            # encoder can learn a near-injective bits->embedding hash. That is a
            # lookup table reached by a path `LookupMoleculeEncoder` does not
            # catch. Corrupting the key each step makes memorizing it useless
            # while leaving the redundant substructure signal intact.
            #
            # The descriptor block is deliberately excluded: those columns are
            # continuous and standardized, so zeroing one means "average
            # molecule", not "absent substructure".
            cut = self.n_bits if self.n_bits is not None else x.shape[1]
            bits = F.dropout(x[:, :cut], p=self.bit_dropout, training=True)
            x = torch.cat([bits, x[:, cut:]], dim=1) if cut < x.shape[1] else bits
        return self.net(x)


class LookupMoleculeEncoder(nn.Module):
    """Ablation: a free embedding per compound, keyed by index.

    Structurally incapable of scoring a held-out compound, since an unseen
    molecule has no row in the table. Select it with `--lookup-ablation`.
    """

    def __init__(self, n_compounds: int, embed_dim: int = 256):
        super().__init__()
        self.table = nn.Embedding(n_compounds, embed_dim)
        self.embed_dim = embed_dim
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """`positions` are bank positions -- structure is ignored by design."""
        return self.table(positions)


class CompoundRetrievalModel(nn.Module):
    """Signature encoder + compound encoder in one cosine space."""

    def __init__(
        self,
        n_genes: int,
        mol_in_dim: int,
        embed_dim: int = 256,
        dropout: float = 0.1,
        init_temperature: float = 0.07,
        lookup_ablation_n: int | None = None,
        bit_dropout: float = 0.0,
        n_bits: int | None = None,
        sig_hidden: tuple[int, ...] = (1024, 512),
        mol_hidden: tuple[int, ...] = (1024, 512),
        sig_dropout: float | None = None,
        mol_dropout: float | None = None,
    ):
        """`embed_dim` is shared -- both encoders write into one cosine space --
        but the trunks are sized independently, and they face very different
        amounts of data. The signature encoder sees ~298k training signatures;
        the molecule encoder sees ~19k compounds. At the shared default that is
        5.6 parameters per training signature against 144.7 per compound, and
        ECFP bits are near-unique per compound, so the molecule side has ample
        capacity to learn a bits->embedding hash instead of chemistry. Shrink
        `mol_hidden` before anything else.
        """
        super().__init__()
        self.signature_encoder = SignatureEncoder(
            n_genes=n_genes, embed_dim=embed_dim, hidden=sig_hidden,
            dropout=dropout if sig_dropout is None else sig_dropout,
        )
        if lookup_ablation_n is not None:
            self.molecule_encoder = LookupMoleculeEncoder(lookup_ablation_n, embed_dim)
            self.lookup_ablation = True
        else:
            self.molecule_encoder = MoleculeEncoder(
                mol_in_dim, embed_dim, hidden=mol_hidden,
                dropout=dropout if mol_dropout is None else mol_dropout,
                bit_dropout=bit_dropout, n_bits=n_bits,
            )
            self.lookup_ablation = False
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_bank(
        self, mol_features: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Embed compounds. Returns (n, D), L2-normalized.

        Pass a *subset* of the feature matrix during training -- only the
        compounds in the batch are ever needed as in-batch negatives, so there
        is no reason to push all 30k rows through the encoder on every step.
        `positions` gives the corresponding bank positions and is required only
        by the lookup ablation, which has nothing but positions to work with.
        """
        if self.lookup_ablation:
            if positions is None:
                positions = torch.arange(
                    mol_features.shape[0], device=mol_features.device
                )
            z = self.molecule_encoder(positions)
        else:
            z = self.molecule_encoder(mol_features)
        return F.normalize(z, dim=-1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        mol_features: torch.Tensor,
        positions: torch.Tensor | None = None,
    ):
        q = self.signature_encoder(batch["signature"])
        k = self.encode_bank(mol_features, positions)
        return q, k


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

# Masked logits use a large finite value, not -inf. Cosine similarity times a
# logit scale capped at 100 lives in [-100, 100], so -1e4 is numerically an
# exact mask after the softmax -- and unlike -inf it cannot produce a 0 * -inf
# NaN in the multi-positive branch, where masked columns carry zero weight.
MASK_LOGIT = -1e4


def _directional_loss(
    logits: torch.Tensor,          # (B, C); column i is the exact positive for row i
    same_moa: torch.Tensor | None, # (B, C) True where row i and column j share an annotation
    moa_positive_weight: float,
) -> torch.Tensor:
    """One direction of the contrastive loss.

    `moa_positive_weight == 0` reproduces the original behavior: same-MOA
    pairs are neither positives nor negatives, so they are deleted from the
    denominator. Above zero they become *positives* carrying that weight
    relative to the exact compound, which optimizes MOA hit rate -- the metric
    `evaluate.py` calls the honest headline -- instead of merely declining to
    punish it.
    """
    b = logits.shape[0]
    labels = torch.arange(b, device=logits.device)
    positive = torch.zeros_like(logits, dtype=torch.bool)
    positive[labels, labels] = True

    if moa_positive_weight <= 0.0:
        if same_moa is not None:
            logits = logits.masked_fill(same_moa & ~positive, MASK_LOGIT)
        return F.cross_entropy(logits, labels)

    live = logits > MASK_LOGIT / 2      # a masked column is not a candidate
    targets = positive.float()
    if same_moa is not None:
        siblings = same_moa & ~positive & live
        # `moa_positive_weight` is the mass given to the same-MOA SET, shared
        # among its members -- not the mass given to each member. Per-member
        # weighting looks equivalent and is not: MOA class size runs from 2 to
        # several hundred, so it would hand the exact compound 1/(1+w*n) of the
        # target and let a large class drown the very thing being retrieved.
        # Sharing keeps the exact positive at a fixed 1/(1+w) for every query,
        # which is what makes this knob mean the same thing across compounds.
        n = siblings.sum(1, keepdim=True).clamp(min=1)
        targets = targets + moa_positive_weight * siblings.float() / n
    targets = targets.masked_fill(~live, 0.0)
    # Every row keeps its exact positive, so the row sum is >= 1.
    targets = targets / targets.sum(1, keepdim=True)
    return -(targets * F.log_softmax(logits, dim=1)).sum(1).mean()


def info_nce(
    query: torch.Tensor,            # (B, D) normalized
    keys: torch.Tensor,             # (B, D) normalized, keys[i] positive for query[i]
    logit_scale: torch.Tensor,
    same_moa: torch.Tensor | None = None,             # (B, B) True = shares an annotation
    *,
    memory: torch.Tensor | None = None,               # (M, D) normalized, detached
    same_moa_memory: torch.Tensor | None = None,      # (B, M)
    memory_is_batch: torch.Tensor | None = None,      # (M,) True = duplicated in `keys`
    moa_positive_weight: float = 0.0,
) -> torch.Tensor:
    """Symmetric CLIP-style loss, optionally against a cross-batch memory.

    Without `memory` this is the in-batch loss: each query is contrasted
    against the other B-1 compounds in its batch.

    `memory` is a table of molecule embeddings for *every training compound*,
    refreshed as batches go by (see `train`). Passing it makes the query->
    compound denominator run over the whole training bank instead of the
    batch. Gradients flow through `query` for those columns but not through
    `memory`, which is detached -- this is cross-batch memory (XBM), not a
    second forward pass, so the extra cost is one (B, M) matmul.

    Two details that are easy to get wrong:

    - `memory_is_batch` marks the memory rows belonging to compounds already
      present in `keys`. Those columns are dropped, because their memory copy
      is a *stale duplicate* of a column that is already in the loss with a
      live gradient -- and for the row's own positive it would be a second,
      wrong positive.
    - The compound->signature direction still runs over the batch alone. There
      is no memory of signature embeddings to contrast against, so that term
      keeps its original (B, B) shape and the two directions are averaged as
      before.
    """
    scale = logit_scale.exp().clamp(max=100.0)
    batch_logits = scale * query @ keys.t()                       # (B, B), grad

    forward_logits, forward_moa = batch_logits, same_moa
    if memory is not None and memory.numel():
        memory_logits = scale * query @ memory.t()                # (B, M)
        if memory_is_batch is not None:
            memory_logits = memory_logits.masked_fill(
                memory_is_batch.unsqueeze(0), MASK_LOGIT
            )
            if same_moa_memory is not None:
                # A dropped duplicate is neither negative nor positive here;
                # that compound's annotation is already handled in the batch
                # block, where it sits with a live gradient.
                same_moa_memory = same_moa_memory & ~memory_is_batch.unsqueeze(0)
        forward_logits = torch.cat([batch_logits, memory_logits], dim=1)
        if same_moa is not None or same_moa_memory is not None:
            b, m = query.shape[0], memory.shape[0]
            zeros_b = torch.zeros((b, b), dtype=torch.bool, device=query.device)
            zeros_m = torch.zeros((b, m), dtype=torch.bool, device=query.device)
            forward_moa = torch.cat(
                [same_moa if same_moa is not None else zeros_b,
                 same_moa_memory if same_moa_memory is not None else zeros_m],
                dim=1,
            )

    forward = _directional_loss(forward_logits, forward_moa, moa_positive_weight)
    backward = _directional_loss(
        batch_logits.t(),
        None if same_moa is None else same_moa.t(),
        moa_positive_weight,
    )
    return 0.5 * (forward + backward)


def build_false_negative_mask(
    moa_ids: torch.Tensor,
    target_ids: torch.Tensor | None = None,
    duplicate_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """(B, B) mask of pairs that are not meaningfully negatives."""
    pairs = [(moa_ids, moa_ids)]
    if target_ids is not None:
        pairs.append((target_ids, target_ids))
    if duplicate_ids is not None:
        pairs.append((duplicate_ids, duplicate_ids))
    return same_annotation(*pairs)


def same_annotation(*id_pairs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Mask of (row, col) pairs matching on ANY of the given label arrays.

    Each argument is a `(row_ids, col_ids)` pair of integer labels. Rectangular
    so the same rule covers the in-batch block and the (B, M) block against the
    cross-batch memory. -1 means unlabelled and never matches, including
    against another -1.

    Callers pass MOA, target and *duplicate-structure* group ids. The last one
    matters: the molecule encoder is a function of structure alone, so a
    feature-identical twin has an identical embedding by construction and
    cannot be ranked below the query. See `featurize.duplicate_group_ids`.
    """
    mask = None
    for row_ids, col_ids in id_pairs:
        if row_ids is None or col_ids is None:
            continue
        m = (
            (row_ids.unsqueeze(1) == col_ids.unsqueeze(0))
            & (row_ids >= 0).unsqueeze(1)
            & (col_ids >= 0).unsqueeze(0)
        )
        mask = m if mask is None else (mask | m)
    if mask is None:
        raise ValueError("same_annotation needs at least one non-None id pair")
    return mask

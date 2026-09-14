"""Pretrained molecular embeddings for `MoleculeEncoder`.

`MoleculeEncoder` is an MLP over ECFP bits plus RDKit descriptors. On a
scaffold split that is close to a smoothed version of the ECFP nearest-
neighbor baseline it is supposed to beat, which is the structural reason the
margin over that baseline is narrow: both see the molecule through the same
substructure-count lens, so both fail on the same off-scaffold compounds.

A chemical language model has seen tens of millions of molecules and encodes
things a 2048-bit fold does not -- ring systems, conjugation, and the fact that
two different scaffolds can be bioisosteres. Precompute those embeddings once
and concatenate them onto the feature matrix; the descriptor block of
`DescriptorScaler` then standardizes them on the train compounds along with
everything else, including its zero-variance and clipping guards.

    pip install transformers
    python -m genetomol.molembed --data ../lincs/prepared --out ../lincs/prepared/chemberta.npz
    python -m genetomol.train --data ../lincs/prepared --mol-embeddings ../lincs/prepared/chemberta.npz --out runs/chemberta

**This is the slow step.** It is a one-time CPU forward pass over every
compound in the bank -- budget roughly half an hour to two hours for ~24k
molecules on four cores, and it is cached afterwards. Nothing else in the
project needs `transformers`, which is why it is an optional extra.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 6 layers, 384 hidden, trained on 77M PubChem molecules. Small enough to run
# on CPU in reasonable time, which the larger chemical LMs are not.
DEFAULT_MODEL = "DeepChem/ChemBERTa-77M-MLM"


def encode_smiles(
    smiles: list[str | None],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    max_length: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """Mean-pooled last hidden state per molecule, (N, H) float32.

    Rows whose SMILES is missing are left at zero rather than dropped, so the
    result stays aligned 1:1 with the bank. After standardization a zero row is
    simply the train mean, which is the least-committal thing it could be.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "pretrained molecule embeddings need `pip install transformers`"
        ) from exc

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    hidden = model.config.hidden_size

    out = np.zeros((len(smiles), hidden), dtype=np.float32)
    present = [i for i, s in enumerate(smiles) if s]
    if len(present) < len(smiles):
        logger.warning(
            "%d/%d compounds have no SMILES; their embedding stays zero",
            len(smiles) - len(present), len(smiles),
        )

    with torch.no_grad():
        for start in range(0, len(present), batch_size):
            rows = present[start:start + batch_size]
            enc = tok(
                [smiles[i] for i in rows],
                padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(device)
            states = model(**enc).last_hidden_state          # (B, T, H)
            # Mean over real tokens only -- padding would otherwise pull every
            # short molecule toward the pad embedding, and SMILES lengths in a
            # screening library vary by an order of magnitude.
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (states * m).sum(1) / m.sum(1).clamp(min=1.0)
            out[rows] = pooled.cpu().numpy().astype(np.float32)
            if start % (batch_size * 20) == 0:
                logger.info("embedded %d/%d", start + len(rows), len(present))
    return out


def save_embeddings(path, ids: list[str], embeddings: np.ndarray,
                    blocks: list[str] | None = None) -> Path:
    """Store ids alongside the matrix so alignment can be checked, not assumed.

    `blocks` names equal-width column groups -- the Chemical Checker spaces, for
    instance -- so a consumer can select a subset by name later. Encoding is the
    expensive step and selection is the cheap one; without this, trying five
    spaces out of twenty-five means encoding five times.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if blocks:
        if embeddings.shape[1] % len(blocks):
            raise ValueError(
                f"{embeddings.shape[1]} columns do not divide into {len(blocks)} blocks"
            )
        extra["blocks"] = np.array(blocks, dtype=object)
    np.savez_compressed(path, ids=np.array(ids, dtype=object),
                        embeddings=embeddings, **extra)
    return path


def load_blocks(path) -> list[str] | None:
    """Column-group names stored with the matrix, or None for an older file."""
    path = Path(path)
    if path.suffix == ".npy":
        return None
    with np.load(path, allow_pickle=True) as f:
        return [str(x) for x in f["blocks"]] if "blocks" in f.files else None


def select_blocks(features: np.ndarray, blocks: list[str] | None,
                  wanted: list[str]) -> np.ndarray:
    """Keep only the named column groups, in the order requested."""
    if not blocks:
        raise ValueError(
            "this file has no block names; rebuild it with `ccsign`, or use "
            "the whole matrix"
        )
    width = features.shape[1] // len(blocks)
    missing = [w for w in wanted if w not in blocks]
    if missing:
        raise ValueError(f"{missing} not in this file; it has {blocks}")
    cols = np.concatenate([
        np.arange(blocks.index(w) * width, (blocks.index(w) + 1) * width)
        for w in wanted
    ])
    return np.ascontiguousarray(features[:, cols])


def load_embeddings(path, ids: list[str]) -> np.ndarray:
    """Load embeddings and put them in `ids` order.

    A silently mis-ordered feature matrix trains to a plausible-looking loss and
    a meaningless model, so the id check is not optional. `.npy` is accepted for
    convenience but can only be checked on row count.
    """
    path = Path(path)
    if path.suffix == ".npy":
        emb = np.load(path)
        if len(emb) != len(ids):
            raise ValueError(
                f"{path} has {len(emb)} rows for {len(ids)} compounds; "
                "use the .npz form, which carries ids and can be reordered"
            )
        logger.warning("%s carries no ids -- assuming bank order", path)
        return emb.astype(np.float32)

    with np.load(path, allow_pickle=True) as f:
        stored_ids = [str(x) for x in f["ids"]]
        emb = f["embeddings"].astype(np.float32)
    if stored_ids == list(ids):
        return emb
    where = {pid: i for i, pid in enumerate(stored_ids)}
    missing = [pid for pid in ids if pid not in where]
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} of {len(ids)} bank compounds "
            f"(e.g. {missing[:3]}); re-run molembed against this bank"
        )
    logger.info("reordering %d embeddings to bank order", len(ids))
    return emb[[where[pid] for pid in ids]]


def attach(bank, path, mode: str = "concat", n_bits: int = 2048,
           blocks: list[str] | None = None) -> int:
    """Fold pretrained embeddings into `bank.mol_features`. Returns added width.

    `concat` keeps the fingerprint and descriptor blocks. `replace` drops the
    descriptor block but never the bits -- the ECFP-NN baseline slices
    `[:, :n_bits]` off this same matrix, so removing the bit block would change
    what the model is being compared against and quietly invalidate the
    headline ratio.
    """
    emb = load_embeddings(path, bank.ids)
    if blocks:
        emb = select_blocks(emb, load_blocks(path), blocks)
        logger.info("selected blocks %s -> %d columns", blocks, emb.shape[1])
    if mode == "concat":
        keep = bank.mol_features
    elif mode == "replace":
        keep = bank.mol_features[:, :n_bits]
    elif mode == "only":
        # Nothing but the embedding. `concat` lets the encoder ignore the new
        # columns and keep using the bits, so it cannot answer "does this
        # representation carry signal the fingerprint does not". This can.
        # Set `bank.fingerprints` first or the baselines lose their bits.
        bank.mol_features = emb.astype(np.float32)
        return emb.shape[1]
    else:
        raise ValueError(
            f"unknown mode: {mode!r} (expected 'concat', 'replace' or 'only')"
        )
    bank.mol_features = np.hstack([keep, emb]).astype(np.float32)
    return emb.shape[1]


def main():
    import argparse

    from .data import load_artifacts

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory holding artifacts.npz")
    ap.add_argument("--out", required=True, help="destination .npz")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bank, _, _ = load_artifacts(args.data)
    logger.info("embedding %d compounds with %s", len(bank), args.model)
    emb = encode_smiles(
        bank.smiles, args.model, args.batch_size, args.max_length, args.device
    )
    path = save_embeddings(args.out, bank.ids, emb)
    logger.info("wrote %s  shape=%s", path, emb.shape)
    print(f"wrote {path}  shape={emb.shape}")


if __name__ == "__main__":
    main()

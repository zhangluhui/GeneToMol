"""Chemical Checker bioactivity signatures, as a drop-in molecule descriptor.

An alternative to ECFP for the molecule encoder, describing what a compound
engages rather than what it resembles.

The Chemical Checker (Duran-Frigola et al., Nat Biotechnol 2020) organizes
bioactivity into 25 spaces across five levels -- A chemistry, B targets, C
networks, D cells, E clinics -- each a 128-dimensional signature. `signaturizer`
predicts them from SMILES alone, so they exist for every compound in the bank
rather than only those carrying a curated annotation.

    pip install signaturizer
    python -m genetomol.ccsign --data ../lincs/prepared --out ../lincs/prepared/cc_B.npz --spaces B1,B2,B4,B5
    python -m genetomol.train --data ../lincs/prepared --mol-features ../lincs/prepared/ecfp_chiral.npz --mol-embeddings ../lincs/prepared/cc_B.npz --mol-embeddings-mode only --out runs/cc

**Start with the B spaces.** They are the target spaces. The A spaces are
chemistry -- 2D/3D fingerprints and scaffolds -- which overlaps what ECFP
already provides. `--spaces all` exists for completeness, not as a default.

Output is the same ids-plus-matrix `.npz` that `molembed` writes, so
`--mol-embeddings` consumes it unchanged. Pair it with
`--mol-embeddings-mode only`: under `concat` the encoder can ignore the new
columns and keep using the fingerprint bits, leaving the comparison unreadable.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .molembed import save_embeddings

logger = logging.getLogger(__name__)

LEVELS = {"A": "chemistry", "B": "targets", "C": "networks",
          "D": "cells", "E": "clinics"}
ALL_SPACES = [f"{lvl}{i}" for lvl in LEVELS for i in range(1, 6)]
TARGET_SPACES = ["B1", "B2", "B3", "B4", "B5"]
SIGNATURE_DIM = 128


def encode_spaces(
    smiles: list[str | None],
    spaces: list[str],
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted CC signatures, concatenated across `spaces`.

    Returns (features, valid) where features is (n, 128 * len(spaces)) float32
    and `valid` marks rows the predictor could handle. Rows it could not are
    left at zero and flagged, exactly as `featurize_molecules` does -- after
    `DescriptorScaler` a zero row becomes the train mean, which is the
    least-committal thing an unusable molecule can be.

    One predictor is loaded at a time. Each is a separate model and holding all
    25 at once is a large amount of memory for no reason.
    """
    # TF Hub's `KerasLayer` -- which is how signaturizer loads its models --
    # only works with Keras 2, and TensorFlow 2.16+ ships Keras 3 by default.
    # The symptom is "exception encountered when calling layer keras_layer",
    # which names neither TensorFlow nor Keras nor a version. TF reads this at
    # import time, so it has to be set before `signaturizer` pulls TF in, and
    # setting it after import is silently too late.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    try:
        from signaturizer import Signaturizer
    except ImportError as exc:  # pragma: no cover - optional extra
        # Report the module that actually failed. `signaturizer` imports h5py,
        # tensorflow and pkg_resources at module load, so a missing dependency
        # of ITS OWN raises ImportError too -- and a handler that assumes the
        # top-level package is absent then tells you to install something you
        # already have. `pkg_resources` is the common one: it ships with
        # setuptools, which recent Pythons no longer install by default.
        missing = getattr(exc, "name", None)
        if missing and missing != "signaturizer":
            hint = "pip install setuptools" if missing == "pkg_resources" else \
                f"pip install {missing}"
            raise ImportError(
                f"signaturizer is installed but cannot import: no module "
                f"named {missing!r}. Try `{hint}`."
            ) from exc
        raise ImportError(
            "Chemical Checker signatures need `pip install signaturizer` "
            "(not `pip install -e .[bioactivity]`, which re-resolves the core "
            "dependencies -- see the README)"
        ) from exc

    n = len(smiles)
    usable = [i for i, s in enumerate(smiles) if s]
    if len(usable) < n:
        logger.warning("%d/%d compounds have no SMILES; left at zero", n - len(usable), n)

    out = np.zeros((n, SIGNATURE_DIM * len(spaces)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    valid[usable] = True

    for col, space in enumerate(spaces):
        logger.info("space %s (%s) -- loading predictor", space, LEVELS.get(space[0], "?"))
        model = Signaturizer(space)
        lo_col = col * SIGNATURE_DIM
        for start in range(0, len(usable), batch_size):
            rows = usable[start:start + batch_size]
            result = model.predict([smiles[i] for i in rows])
            # The package has moved this attribute around between releases;
            # accept whichever shape it hands back rather than pinning a version.
            block = getattr(result, "signature", result)
            block = np.asarray(block, dtype=np.float32)
            if block.shape[1] != SIGNATURE_DIM:
                raise RuntimeError(
                    f"space {space} returned width {block.shape[1]}, expected "
                    f"{SIGNATURE_DIM}; the signaturizer API has changed"
                )
            failed = getattr(result, "failed", None)
            if failed is not None:
                failed = np.asarray(failed, dtype=bool)
                valid[np.asarray(rows)[failed]] = False
            out[rows, lo_col:lo_col + SIGNATURE_DIM] = block
            if start % (batch_size * 10) == 0:
                logger.info("  %s: %d/%d", space, start + len(rows), len(usable))
        del model

    n_bad = int((~valid).sum())
    if n_bad:
        logger.warning("%d/%d compounds produced no usable signature", n_bad, n)
    return out, valid


def parse_spaces(text: str) -> list[str]:
    """`B1,B4` | `B` (a whole level) | `targets` | `all`."""
    text = text.strip()
    if text.lower() == "all":
        return list(ALL_SPACES)
    if text.lower() in ("targets", "target"):
        return list(TARGET_SPACES)
    out: list[str] = []
    for part in text.split(","):
        part = part.strip().upper()
        if part in LEVELS:
            out.extend(f"{part}{i}" for i in range(1, 6))
        elif part in ALL_SPACES:
            out.append(part)
        else:
            raise ValueError(
                f"unknown space {part!r}; expected one of {ALL_SPACES}, "
                "a level letter A-E, 'targets', or 'all'"
            )
    if not out:
        raise ValueError("no spaces selected")
    return out


def main():
    import argparse

    from .data import load_artifacts

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory holding artifacts.npz")
    ap.add_argument("--out", required=True, help="destination .npz")
    ap.add_argument("--spaces", default="targets",
                    help="B1,B4 | a level letter A-E | 'targets' (B1-B5) | 'all'")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spaces = parse_spaces(args.spaces)
    bank, _, _ = load_artifacts(args.data)
    logger.info("%d compounds x %d spaces -> %d columns",
                len(bank), len(spaces), len(spaces) * SIGNATURE_DIM)

    features, valid = encode_spaces(bank.smiles, spaces, args.batch_size)
    path = save_embeddings(args.out, bank.ids, features, blocks=spaces)
    print(f"\nwrote {path}  shape={features.shape}  "
          f"({int(valid.sum()):,}/{len(valid):,} compounds usable)")
    print(f"spaces: {' '.join(spaces)}")
    print(f"\nuse it with:\n  --mol-embeddings {path} --mol-embeddings-mode only")


if __name__ == "__main__":
    main()

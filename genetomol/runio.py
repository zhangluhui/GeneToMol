"""Rebuild a finished run: model, bank, and the same splits it trained on.

Three post-hoc analyses -- `tanimoto`, `depth`, `hubness` -- all need the same
forty lines: read `results.json`, reattach the exact feature matrix, rebuild the
split from the recorded seed, fit `DescriptorScaler` on the training compounds
*only*, and load the checkpoint. Getting any one of those subtly wrong scores a
model against molecules it never saw and fails silently, so it lives here once.

The feature-width check is necessary but NOT sufficient: a chirality-aware
rebuild is exactly as wide as the achiral original. Pass the same
`--mol-features` / `--mol-embeddings` the run trained with.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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
from .featurize import DescriptorScaler

logger = logging.getLogger(__name__)


@dataclass
class LoadedRun:
    """Everything a post-hoc analysis needs, with nothing left to rederive."""

    model: torch.nn.Module
    bank: object
    test_ds: object
    train_ds: object | None          # None unless want_train
    train_compounds: np.ndarray
    test_compounds: np.ndarray
    sigs_per_compound: np.ndarray    # over the FULL dataset, i.e. profiling depth
    raw_features: np.ndarray         # pre-scaler, for baselines that want them
    n_bits: int
    split: str
    config: dict
    device: torch.device


def config_of(d: dict):
    """`TrainConfig` from a serialized config, coercing what JSON flattened."""
    from .train import TrainConfig

    known = {f: d[f] for f in TrainConfig.__dataclass_fields__ if f in d}
    for f in ("sig_hidden", "mol_hidden"):
        if isinstance(known.get(f), list):
            known[f] = tuple(known[f])
    return TrainConfig(**known)


def load_run(
    data_dir: str,
    run_dir: str | Path,
    mol_features: str | None = None,
    mol_embeddings: str | None = None,
    mol_embeddings_mode: str = "concat",
    device: str | torch.device = "cpu",
    want_train: bool = False,
) -> LoadedRun:
    """Rebuild the run at `run_dir` against the data in `data_dir`.

    `want_train` also materializes the training split, which the ECFP-NN
    baseline needs and the hubness diagnostic does not.
    """
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    cfg_dict, split = result["config"], result["split"]
    device = torch.device(device)

    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    bank, ds, meta = load_artifacts(data_dir)
    n_bits = meta.get("n_bits", 2048)

    if mol_features:
        from .molembed import load_embeddings

        bank.mol_features = load_embeddings(mol_features, bank.ids)
        logger.info("mol features replaced from %s", mol_features)
    bank.fingerprints = np.ascontiguousarray(bank.mol_features[:, :n_bits])

    if mol_embeddings:
        from .molembed import attach

        attach(bank, mol_embeddings, mol_embeddings_mode, n_bits)
        logger.info("mol embeddings attached from %s", mol_embeddings)

    expected = ckpt["model"]["molecule_encoder.net.0.weight"].shape[1]
    if bank.n_features != expected:
        raise SystemExit(
            f"{run_dir} was trained on {expected} molecule features but "
            f"{data_dir} has {bank.n_features}. Re-run with the same "
            "--mol-features / --mol-embeddings this run used."
        )

    # Profiling depth is a property of the full dataset, so it has to be taken
    # before any subsetting. The split is compound-disjoint, so for a test
    # compound this also equals its signature count within the test split.
    sigs_per = np.bincount(ds.bank_index, minlength=len(bank)).astype(np.float64)
    cache = split_cache_dir(ds)
    tag = f"{split}_{cfg_dict['seed']}_{len(bank)}"
    take = (lambda m, name: subset_to_memmap(ds, m, cache / f"{tag}_{name}.npy")) \
        if cache is not None else (lambda m, name: subset_dataset(ds, m))

    if split == "signature":
        # A signature-level split divides MEASUREMENTS, not compounds, so there
        # is no compound partition to rebuild and `make_split` does not know
        # this kind -- it would raise. Rebuild the same masks the run used and
        # derive the compound lists from them. Every compound is in training by
        # construction, so fitting the scaler on `tr_c` sees the whole bank and
        # leaks nothing training did not already have.
        from .train import make_signature_split

        tr_m, _, te_m = make_signature_split(ds, cfg_dict["seed"], 0.8, 0.1, len(bank))
        tr_c, te_c = np.unique(ds.bank_index[tr_m]), np.unique(ds.bank_index[te_m])
        raw_features = bank.mol_features
        bank.mol_features = DescriptorScaler(n_bits).fit_transform(raw_features, tr_c)
        train_ds = take(tr_m, "train") if want_train else None
        test_ds = take(te_m, "test")
    else:
        tr_c, _, te_c = make_split_like(bank, split, cfg_dict["seed"], sigs_per)
        raw_features = bank.mol_features
        bank.mol_features = DescriptorScaler(n_bits).fit_transform(raw_features, tr_c)
        train_ds = take(signature_mask(ds.bank_index, tr_c), "train") if want_train else None
        test_ds = take(signature_mask(ds.bank_index, te_c), "test")
    ds.release()

    from .train import build_model

    model = build_model(bank, test_ds.signatures.shape[1], config_of(cfg_dict),
                        device, n_bits=n_bits)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return LoadedRun(
        model=model, bank=bank, test_ds=test_ds, train_ds=train_ds,
        train_compounds=tr_c, test_compounds=te_c, sigs_per_compound=sigs_per,
        raw_features=raw_features, n_bits=n_bits, split=split,
        config=cfg_dict, device=device,
    )


def make_split_like(bank, split: str, seed: int, sigs_per: np.ndarray):
    """The same 80/10/10 the training run used. Kept as one call so no analysis
    can drift from the split it is meant to be scoring."""
    from .train import make_split

    return make_split(bank, split, seed, 0.8, 0.1, weights=sigs_per)

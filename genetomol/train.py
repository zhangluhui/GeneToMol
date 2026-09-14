"""Training loop and CLI.

`run_experiment` splits compounds by Bemis-Murcko scaffold, trains, and scores
the model against ECFP nearest-neighbor transfer, cosine consensus, gene-set
overlap and chance on the same held-out queries and the same bank.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import CosineConsensus, ECFPNearestNeighbor, GeneSetOverlap
from .data import (
    SignatureDataset,
    UniquePertBatchSampler,
    load_artifacts,
    moa_disjoint_split,
    random_split,
    scaffold_split,
    signature_mask,
    consume_subset_dataset,
    split_cache_dir,
    subset_to_memmap,
    subset_dataset,
)
from .evaluate import (
    chance_baseline,
    format_comparison,
    macro_recall_at_k,
    rank_bank,
    rank_queries,
    recall_at_k,
    summarize,
)
from .featurize import (
    DescriptorScaler,
    bemis_murcko_scaffolds,
    duplicate_group_ids,
    recall_ceiling,
)
from .model import (
    CompoundRetrievalModel,
    build_false_negative_mask,
    info_nce,
    same_annotation,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    embed_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.07
    # Floor on the learned temperature, i.e. a cap on how sharp the softmax may
    # get. The gradient pushes `logit_scale` up whenever training is confidently
    # correct; a sharp softmax then multiplies the penalty on every held-out
    # query ranked imperfectly, and rewards memorizing an exact match over
    # learning a broad neighborhood. None keeps CLIP's 0.01 (scale 100).
    min_temperature: float | None = None
    # Rank the final test split by `2*cos - hub_penalty` rather than plain
    # cosine, correcting for hub compounds that fill top-k lists regardless of
    # the query. Applies to the test evaluation only -- per-epoch validation
    # stays on plain cosine, so early stopping selects on an unchanged metric.
    csls_r: int = 0
    # Treatment-quality filters on the TRAINING signatures; see `signal.py` for
    # what each one selects. All off by default.
    min_dose: float | None = None
    min_time: float | None = None
    drop_outlier_frac: float = 0.0
    # Drop whole COMPOUNDS below this profiling depth. Asymmetric: it cuts the
    # distinct compounds the molecule encoder sees far more than it cuts the
    # signatures the signature encoder sees.
    min_compound_sigs: int | None = None
    # Evaluate once before training starts, recorded as epoch -1. One extra
    # evaluation, and it is the only control that can catch a leaking split:
    # an untrained encoder must score at chance.
    baseline_eval: bool = True
    # Warm-start the encoders from a finished run's model.pt. Weights only:
    # the optimizer and the LR schedule start fresh, sized to this run's
    # --epochs. See `load_weights_into` for why that is the right thing.
    resume_from: str | None = None
    # Re-initialize the learned temperature from --temperature instead of
    # inheriting the checkpoint's, which is confident about a ranking the new
    # task has never seen.
    reset_temperature: bool = False
    # Where the memmapped splits land. None puts them beside artifacts.npz.
    # A separate folder keeps one split regime's caches from sitting next to
    # another's; the filenames already carry split kind, seed and filter, so
    # this is for tidiness and disk placement, not correctness.
    split_cache_dir: str | None = None
    # "train" leaves valid/test untouched, so the benchmark stays comparable to
    # every existing run and only the training data changes. "all" filters the
    # queries too, which is a different experiment and NOT comparable.
    filter_splits: str = "train"
    batch_size: int = 256
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    mask_false_negatives: bool = True
    grad_clip: float = 1.0
    seed: int = 0
    lookup_ablation: bool = False
    num_workers: int = 0
    eval_every: int = 5
    max_sigs_per_epoch: int | None = 60
    # Contrast against every training compound, not just the batch. See
    # `model.info_nce`. Costs n_train * embed_dim * 4 bytes for the table and
    # one extra (batch, n_train) matmul per step. With a full-bank denominator
    # every fingerprint-identical twin of the positive is present at every
    # step, so leave `mask_duplicates` on when enabling this.
    bank_negatives: bool = False
    bank_negatives_k: int | None = None   # sample this many per step; None = all
    mask_duplicates: bool = True
    # A fixed subsample of the train split, scored with the same code as the
    # valid split so the two curves are comparable. Defaults to the size of the
    # valid split. 0 disables.
    train_probe_size: int | None = None
    # "auto" shows a bar only on a terminal. A backgrounded run redirected to a
    # log file would otherwise fill it with carriage returns.
    progress: str = "auto"
    # --- generalization ---------------------------------------------------
    # Fraction of each train compound's signatures withheld from training and
    # scored as `newsig_*`. Separates "memorized this compound" from
    # "memorized these signatures": the model has seen the compound but not
    # the query. 0 disables.
    train_holdout_frac: float = 0.05
    # Keep the train split's signatures resident on the accelerator and index
    # them directly, skipping the DataLoader. Costs (train signatures x genes x
    # 4) bytes of device memory and removes the host-side collate and the
    # per-step host->device copy. Ignored on CPU, and pointless on unified
    # memory, where device RAM is host RAM.
    signatures_on_device: bool = False
    # Trunk widths, sized independently. `embed_dim` stays shared because both
    # encoders write into one cosine space.
    sig_hidden: tuple[int, ...] = (1024, 512)
    mol_hidden: tuple[int, ...] = (1024, 512)
    # Weight decay for the molecule encoder alone. None follows `weight_decay`.
    mol_weight_decay: float | None = None
    # Per-encoder dropout, to regularize one trunk without the other.
    # `dropout` alone hits both. None follows `dropout`.
    sig_dropout: float | None = None
    mol_dropout: float | None = None
    # Drop fingerprint bits before the first weight matrix. See MoleculeEncoder.
    bit_dropout: float = 0.0
    # Evaluate the test set with the best-validation weights, not the last
    # epoch's, selected by `early_stop_metric`.
    restore_best: bool = True
    early_stop_metric: str = "val_macro_recall@10"
    early_stop_patience: int | None = None   # evals without improvement; None = never stop
    memory_refresh_every: int = 1         # epochs between a full re-encode
    # > 0 makes same-MOA compounds positives rather than merely masked.
    moa_positive_weight: float = 0.0


def load_weights_into(model, path: str, reset_temperature: bool = False,
                      init_temperature: float = 0.07) -> dict:
    """Warm-start `model` from a finished run's `model.pt`. Returns its config.

    **This is warm-starting, not exact continuation**, and the difference is
    deliberate. The optimizer's moment estimates and the LR schedule are NOT
    restored: a cosine schedule has decayed to ~1e-8 by the end of a run, so
    continuing it learns nothing, and restarting it with stale Adam moments
    shocks converged weights. A new phase -- a different split, filtered data,
    another feature arm -- wants a fresh optimizer and a fresh schedule sized
    to the new budget. That is what you get.

    Shapes must match exactly. A mismatch here means the checkpoint was trained
    on different features or a different trunk, and loading it anyway would
    score a model against molecules it never saw.
    """
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"]
    if reset_temperature:
        # A finished run ends near scale 60-77. Handing that to a new task
        # starts it maximally confident about a ranking it has never seen.
        state = dict(state)
        state["logit_scale"] = torch.tensor(math.log(1.0 / init_temperature))
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise SystemExit(
            f"cannot warm-start from {path}: {exc}\n"
            "The checkpoint's architecture or feature width differs from this "
            "run. Pass the same --mol-features / --mol-embeddings / "
            "--sig-hidden / --mol-hidden / --embed-dim the checkpoint used."
        ) from exc
    scale = float(torch.as_tensor(state["logit_scale"]).exp())
    logger.info("warm-started from %s (temperature %.4f, scale %.1f); "
                "optimizer and LR schedule start fresh", path, 1 / scale, scale)
    return ckpt.get("config", {})


def build_model(bank, n_genes: int, cfg: TrainConfig, device, n_bits: int | None = None):
    return CompoundRetrievalModel(
        n_genes=n_genes,
        mol_in_dim=bank.n_features,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
        init_temperature=cfg.temperature,
        lookup_ablation_n=len(bank) if cfg.lookup_ablation else None,
        bit_dropout=cfg.bit_dropout,
        n_bits=n_bits,
        sig_hidden=tuple(cfg.sig_hidden),
        mol_hidden=tuple(cfg.mol_hidden),
        sig_dropout=cfg.sig_dropout,
        mol_dropout=cfg.mol_dropout,
    ).to(device)


def _lr_lambda(step: int, total_steps: int, warmup: int) -> float:
    """Linear warmup then cosine decay, clamped so an over-run cannot crash.

    `len(sampler)` is exact for this sampler, but a scheduler that raises when
    the step count is off by one is a bad trade for a schedule shape nobody
    tunes. This one saturates instead.
    """
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min((step - warmup) / max(total_steps - warmup, 1), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def device_batches(sampler, signatures, bank_index, device):
    """Yield batches straight from device-resident tensors.

    Batch *composition* is unchanged -- it comes from `sampler`, which owns its
    own RNG seeded by `seed + epoch`. What changes is that no `DataLoader`
    iterator is created, and torch draws a base seed from the global RNG every
    time one is. So a run with this flag will not be bit-identical to one
    without: the dropout stream is offset. Batches, split and schedule are the
    same; only the noise differs, exactly as a different seed would.
    """
    for rows in sampler:
        idx = torch.as_tensor(rows, device=device, dtype=torch.long)
        yield {"signature": signatures[idx], "bank_index": bank_index[idx]}


def progress_bar(desc: str, total: int, mode: str = "auto"):
    """Return a tqdm bar to `update()` by hand, or None.

    Deliberately NOT `tqdm(loader)`. Wrapping a DataLoader makes tqdm call
    `DataLoader.__iter__` twice -- once to probe, once to iterate -- and torch
    draws a fresh base seed from the *global* RNG on every iterator creation.
    That extra draw shifts every dropout mask for the rest of training: with
    the bar wrapped around the loader, epoch-1 batch-0 loss moved from 5.2128
    to 5.4437 while the batch contents were byte-identical. A progress bar that
    silently changes your loss curve is worse than no progress bar, so this one
    never touches the iterable.

    Off unless stderr is a terminal: the common case here is a long run
    redirected to a file, where a bar is thousands of carriage returns and no
    information. A missing tqdm is not an error -- it is an optional extra, and
    training must not depend on a display library.
    """
    if mode == "off" or (mode == "auto" and not sys.stderr.isatty()):
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        if mode == "on":
            logger.warning("--progress on, but tqdm is not installed "
                           "(pip install -e .[progress])")
        return None
    return tqdm(total=total, desc=desc, unit="batch", leave=False,
                dynamic_ncols=True)


@torch.no_grad()
def encode_positions(model, mol_features, positions, chunk: int = 4096):
    """Embed the given bank positions in eval mode, chunked.

    Used to build and refresh the cross-batch memory. Chunked because the
    molecule encoder widens 2048 fingerprint bits to 1024 hidden units and a
    single 19k-row forward would allocate ~80 MB of activations for no reason
    on a box that has 8 GB total.
    """
    was_training = model.training
    model.eval()
    out = [
        model.encode_bank(mol_features[positions[s:s + chunk]],
                          positions=positions[s:s + chunk])
        for s in range(0, len(positions), chunk)
    ]
    if was_training:
        model.train()
    return torch.cat(out, dim=0)


@torch.no_grad()
def eval_loss(model, mol_features, ds, cfg, device, moa_t=None, dup_t=None):
    """In-batch InfoNCE over `ds`, shaped exactly like the training reference.

    Same batch size and same unique-compound batching as training, in eval mode
    and without the cross-batch memory, so `val_nce` is directly comparable to
    `nce_inbatch` and the gap between them is the overfitting signal. Comparing
    against `nce` instead would be meaningless whenever bank negatives are on,
    since that objective has a different candidate count.
    """
    was_training = model.training
    model.eval()
    # The sampler needs `batch_size` distinct compounds to emit a batch, so a
    # split with fewer than that yields none at all -- and averaging over zero
    # batches would report a loss of 0.0, which reads as a perfect score rather
    # than as missing data. Shrink the batch instead, and say so, because the
    # chance level is log(batch) and a shrunk batch is no longer comparable to
    # `nce_inbatch`.
    n_compounds = int(len(np.unique(ds.bank_index)))
    batch_size = min(cfg.batch_size, n_compounds)
    if batch_size < cfg.batch_size:
        logger.warning(
            "loss split has %d compounds but batch_size is %d; using batches of "
            "%d, so this number is not comparable to nce_inbatch",
            n_compounds, cfg.batch_size, batch_size,
        )
    if batch_size < 2:
        return float("nan")
    sampler = UniquePertBatchSampler(
        ds.bank_index, batch_size, seed=cfg.seed, max_per_epoch=None
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.num_workers)
    total, seen = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        idx = batch["bank_index"]
        q = model.signature_encoder(batch["signature"])
        k = model.encode_bank(mol_features[idx], positions=idx)
        mask = None
        if moa_t is not None or dup_t is not None:
            mask = build_false_negative_mask(
                moa_t[idx] if moa_t is not None else torch.full_like(idx, -1),
                duplicate_ids=None if dup_t is None else dup_t[idx],
            )
        total += info_nce(q, k, model.logit_scale, mask).item()
        seen += 1
    if was_training:
        model.train()
    return total / seen if seen else float("nan")


def train(
    model,
    bank,
    mol_features: torch.Tensor,
    train_ds: SignatureDataset,
    valid_ds: SignatureDataset | None,
    cfg: TrainConfig,
    device,
    moa_ids: np.ndarray | None = None,
    duplicate_ids: np.ndarray | None = None,
    newsig_ds: SignatureDataset | None = None,
    out_dir: Path | None = None,
):
    torch.manual_seed(cfg.seed)
    sampler = UniquePertBatchSampler(
        train_ds.bank_index, cfg.batch_size, seed=cfg.seed,
        max_per_epoch=cfg.max_sigs_per_epoch,
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # The molecule encoder gets its own group: it is the side with 26x more
    # parameters per training example, so it is the side worth regularizing
    # harder without also damping the signature encoder.
    decay, mol_decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if p.ndim <= 1 or "logit_scale" in name:
            no_decay.append(p)
        elif name.startswith("molecule_encoder"):
            mol_decay.append(p)
        else:
            decay.append(p)
    mol_wd = cfg.weight_decay if cfg.mol_weight_decay is None else cfg.mol_weight_decay
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": mol_decay, "weight_decay": mol_wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    steps_per_epoch = max(len(sampler), 1)
    total_steps = max(cfg.epochs * steps_per_epoch, 1)
    warmup = int(cfg.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, total_steps, warmup)
    )

    moa_t = None if moa_ids is None else torch.as_tensor(moa_ids, device=device)
    max_logit_scale = math.log(
        100.0 if cfg.min_temperature is None else 1.0 / cfg.min_temperature
    )
    if cfg.min_temperature is not None:
        logger.info("temperature floored at %.4f (logit_scale <= %.1f)",
                    cfg.min_temperature, math.exp(max_logit_scale))
    # The same-MOA matrix is needed both to mask (weight 0) and to score
    # positives (weight > 0), so build it whenever either is switched on.
    use_moa = moa_t is not None and (cfg.mask_false_negatives or cfg.moa_positive_weight > 0)
    dup_t = None
    if cfg.mask_duplicates and duplicate_ids is not None and (duplicate_ids >= 0).any():
        dup_t = torch.as_tensor(duplicate_ids, device=device)
        logger.info(
            "masking %d compounds that share a fingerprint with another (%.1f%% of bank)",
            int((duplicate_ids >= 0).sum()), 100 * float((duplicate_ids >= 0).mean()),
        )

    # --- cross-batch memory ------------------------------------------------
    # Restricted to TRAIN compounds. Held-out compounds must not appear in the
    # denominator: nothing would leak a label, but the encoder would be shaped
    # against the structures it is about to be scored on, which is exactly the
    # transductive shortcut the scaffold split exists to forbid.
    memory = memory_moa = memory_dup = train_positions = pos_to_row = None
    if cfg.bank_negatives:
        train_positions = torch.as_tensor(
            np.unique(train_ds.bank_index), device=device, dtype=torch.long
        )
        pos_to_row = torch.full((len(bank),), -1, dtype=torch.long, device=device)
        pos_to_row[train_positions] = torch.arange(len(train_positions), device=device)
        memory = encode_positions(model, mol_features, train_positions)
        if moa_t is not None:
            memory_moa = moa_t[train_positions]
        memory_dup = None if dup_t is None else dup_t[train_positions]
        logger.info(
            "cross-batch memory: %d train compounds x %d dims (%.1f MB)",
            len(train_positions), memory.shape[1], memory.numel() * 4 / 1e6,
        )

    # Fixed across epochs so the train curve moves only because the model did.
    probe_ds = None
    if (valid_ds is not None and len(valid_ds) and len(train_ds)
            and cfg.train_probe_size != 0):
        n_probe = min(cfg.train_probe_size or len(valid_ds), len(train_ds))
        keep = np.zeros(len(train_ds), dtype=bool)
        keep[np.random.default_rng(cfg.seed).choice(
            len(train_ds), n_probe, replace=False)] = True
        probe_ds = subset_dataset(train_ds, keep)
        logger.info("train probe: %d signatures scored alongside %d valid",
                    len(probe_ds), len(valid_ds))

    device_sigs = device_bank = None
    if cfg.signatures_on_device:
        if device.type == "cpu":
            logger.warning("--signatures-on-device does nothing on CPU; ignoring")
        else:
            nbytes = train_ds.signatures.size * 4
            logger.info(
                "pinning %d x %d train signatures to %s (%.2f GB device memory)",
                len(train_ds), train_ds.signatures.shape[1], device, nbytes / 1e9,
            )
            device_sigs = torch.as_tensor(
                np.ascontiguousarray(train_ds.signatures), device=device
            )
            device_bank = torch.as_tensor(train_ds.bank_index, device=device)

    history = []
    best_score, best_epoch, best_state, stale = -float("inf"), -1, None, 0

    # Epoch 0 is the model before any training, so epoch N means "N epochs of
    # training have happened" and the first trained point is 1. It is the
    # control every curve should be read against: retrieval here must sit at
    # chance (k / n_bank); anything above it
    # means the held-out queries are reachable without learning, which would be
    # a leak in the split rather than a result. Deliberately excluded from
    # early-stopping bookkeeping: an untrained model must never become
    # `best_state`, however badly training goes.
    if cfg.baseline_eval and valid_ds is not None and len(valid_ds):
        model.eval()
        t_eval = time.time()
        row = {"epoch": 0.0, "secs": 0.0,
               "temperature": float(1.0 / model.logit_scale.exp().item()),
               "lr": float(sched.get_last_lr()[0])}
        row.update(eval_all(model, bank, mol_features, valid_ds, probe_ds,
                            newsig_ds, cfg, device, moa_t, dup_t))
        row["eval_secs"] = time.time() - t_eval
        history.append(row)
        chance = 10.0 / max(len(bank), 1)      # expected macro recall@10
        got = row.get("val_macro_recall@10")
        logger.info("untrained baseline: val_macro_recall@10 %.5f (chance %.5f)",
                    got if got is not None else float("nan"), chance)
        if got is not None and got > 10 * chance:
            logger.warning(
                "untrained model retrieves %.1fx chance at k=10. An untrained "
                "encoder should not -- check the split for leakage before "
                "trusting anything this run reports", got / max(chance, 1e-12),
            )

    for epoch in range(cfg.epochs):
        model.train()
        sampler.set_epoch(epoch)
        if memory is not None and epoch and epoch % max(cfg.memory_refresh_every, 1) == 0:
            # Per-step writes keep most rows fresh, but a compound the sampler
            # skipped this epoch would otherwise drift arbitrarily stale.
            memory = encode_positions(model, mol_features, train_positions)
        t0, agg = time.time(), []
        bar = progress_bar(
            f"epoch {epoch + 1}/{cfg.epochs}", len(sampler), cfg.progress
        )
        source = (
            loader if device_sigs is None
            else device_batches(sampler, device_sigs, device_bank, device)
        )
        for step, batch in enumerate(source):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            idx = batch["bank_index"]

            q = model.signature_encoder(batch["signature"])
            # Only the compounds in this batch are ever used as in-batch
            # negatives, so encode those rows and not the whole bank.
            k = model.encode_bank(mol_features[idx], positions=idx)

            fn_mask = None
            if use_moa or dup_t is not None:
                fn_mask = build_false_negative_mask(
                    moa_t[idx] if use_moa else torch.full_like(idx, -1),
                    duplicate_ids=None if dup_t is None else dup_t[idx],
                )

            mem = mem_moa_mask = mem_is_batch = None
            if memory is not None:
                rows = pos_to_row[idx]
                in_batch = torch.zeros(
                    len(memory), dtype=torch.bool, device=device
                )
                in_batch[rows] = True
                sel = None
                if cfg.bank_negatives_k and cfg.bank_negatives_k < len(memory):
                    sel = torch.randint(
                        len(memory), (cfg.bank_negatives_k,), device=device
                    )
                mem = memory if sel is None else memory[sel]
                mem_is_batch = in_batch if sel is None else in_batch[sel]
                if use_moa or dup_t is not None:
                    pairs = []
                    if use_moa:
                        cols = memory_moa if sel is None else memory_moa[sel]
                        pairs.append((moa_t[idx], cols))
                    if dup_t is not None:
                        dcols = memory_dup if sel is None else memory_dup[sel]
                        pairs.append((dup_t[idx], dcols))
                    mem_moa_mask = same_annotation(*pairs)

            loss = info_nce(
                q, k, model.logit_scale, fn_mask,
                memory=mem,
                same_moa_memory=mem_moa_mask,
                memory_is_batch=mem_is_batch,
                moa_positive_weight=cfg.moa_positive_weight,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, max_logit_scale)
                if memory is not None:
                    # These were encoded in train mode, so they carry dropout
                    # noise the periodic eval-mode refresh does not. That is
                    # tolerable precisely because memory rows are only ever
                    # negatives, and detached ones: noise on a negative is
                    # regularization, not a wrong gradient.
                    memory[pos_to_row[idx]] = k.detach()
            # `nce` is not comparable across configurations: its chance level
            # is log(n_candidates), so switching bank negatives on moves it from
            # log(256)=5.5 to log(19164)=9.9, and it can *rise* across epochs as
            # the memory refresh replaces random-init negatives with trained
            # ones. Record a fixed in-batch, weight-0 reference alongside it so
            # the training curve stays readable and comparable to older runs.
            # One extra (B, B) matmul under no_grad, against a (B, 19k) one.
            with torch.no_grad():
                reference = info_nce(q, k, model.logit_scale, fn_mask)
            agg.append({
                "nce": loss.detach().item(),
                "nce_inbatch": reference.item(),
            })
            if bar is not None:
                bar.update(1)
                if step % 20 == 0:
                    bar.set_postfix(nce=f"{agg[-1]['nce']:.3f}", refresh=False)

        if bar is not None:
            bar.close()

        # 1-indexed: epoch 0 is the untrained baseline, so the first trained
        # row is 1 and the number always means "epochs of training completed".
        # This also matches what the progress bar prints.
        row = {"epoch": float(epoch + 1), "secs": time.time() - t0}
        for key in ("nce", "nce_inbatch"):
            row[key] = float(np.mean([a[key] for a in agg])) if agg else float("nan")
        row["n_negatives"] = float(cfg.batch_size - 1 + (len(memory) if memory is not None else 0))
        row["lr"] = float(sched.get_last_lr()[0])
        row["temperature"] = float(1.0 / model.logit_scale.exp().item())

        if valid_ds is not None and len(valid_ds) and (
            epoch % cfg.eval_every == cfg.eval_every - 1
            or epoch == cfg.epochs - 1
        ):
            t_eval = time.time()
            row.update(eval_all(model, bank, mol_features, valid_ds, probe_ds,
                                newsig_ds, cfg, device, moa_t, dup_t))
            # `secs` covers the training loop only, so without this the eval
            # cost is invisible in the history.
            row["eval_secs"] = time.time() - t_eval

            score = row.get(cfg.early_stop_metric)
            if score is not None and score > best_score:
                best_score, best_epoch, stale = score, epoch + 1, 0
                # Copied to CPU so the snapshot does not pin device memory.
                best_state = {
                    k: v.detach().to("cpu", copy=True)
                    for k, v in model.state_dict().items()
                }
            elif score is not None:
                stale += 1
        history.append(row)
        logger.info("epoch %d %s", epoch + 1, {k: round(v, 4) for k, v in row.items()})

        if cfg.early_stop_patience is not None and stale >= cfg.early_stop_patience:
            logger.info(
                "early stop at epoch %d: %s has not improved on %.5f (epoch %d) "
                "for %d evals",
                epoch + 1, cfg.early_stop_metric, best_score, best_epoch, stale,
            )
            break

    # Compare epoch numbers, not row indices: `len(history) - 1` is off by one
    # whenever the untrained baseline row is present and correct when it is
    # not, so it silently mis-fires depending on --no-baseline-eval.
    final_epoch = history[-1].get("epoch") if history else None
    if best_state is not None and cfg.restore_best and best_epoch != final_epoch:
        # Without this the test numbers come from whatever the last epoch
        # happened to be, which on an overfitting run is strictly worse than
        # the best checkpoint you already paid to find.
        logger.info(
            "restoring epoch %d (%s = %.5f) over the final epoch's weights",
            best_epoch, cfg.early_stop_metric, best_score,
        )
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": asdict(cfg),
                    "best_epoch": best_epoch, "best_score": best_score},
                   out_dir / "model.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


@torch.no_grad()
def eval_all(model, bank, mol_features, valid_ds, probe_ds, newsig_ds,
             cfg, device, moa_t=None, dup_t=None) -> dict:
    """Every held-out metric for one moment in training, as one row.

    **Restores the RNG.** `eval_loss` batches the validation split, which draws
    from the global torch stream, so without the guard the *number* of
    evaluations changes training: two runs at one seed with different
    `--eval-every` diverge by epoch 2. Evaluation has to be a pure observer or
    the schedule becomes a hyperparameter.

    Shared by the in-loop evaluation and the untrained baseline so the two
    cannot measure different things.
    """
    rng_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)
    row: dict = {}
    row.update(evaluate_split(model, bank, mol_features, valid_ds, device, "val"))
    row["val_nce"] = eval_loss(model, mol_features, valid_ds, cfg, device,
                               moa_t, dup_t)
    if probe_ds is not None:
        row.update(evaluate_split(model, bank, mol_features, probe_ds, device, "train"))
    # Train compounds, signatures never trained on. `train_*` is the
    # memorization ceiling and `val_*` the generalization floor; this sits
    # between them and says which of the two encoders is at fault. High here
    # means the signature encoder generalizes and the failure is compound
    # novelty alone; low means both are memorizing.
    if newsig_ds is not None and len(newsig_ds):
        row.update(evaluate_split(model, bank, mol_features, newsig_ds, device, "newsig"))
    torch.set_rng_state(rng_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return row


def evaluate_split(
    model,
    bank,
    mol_features: torch.Tensor,
    ds: SignatureDataset,
    device,
    prefix: str = "val",
    csls_r: int = 0,
) -> dict:
    """Metrics against the full bank.

    The encoder sees the signature and nothing else, so this is the deployment
    condition already -- there is no separate null-context number.

    Queries are streamed: the (queries x bank) score matrix is 3.5 GB on the
    LINCS test split and is never held whole. See `evaluate.Ranking`.

    `csls_r > 0` applies the hubness correction; see `evaluate.rank_bank`.
    """
    ranking = rank_bank(model, ds.signatures, mol_features, ds.bank_index, device,
                        csls_r=csls_r)
    return {
        f"{prefix}_{k}": v
        for k, v in summarize(ranking, ds.bank_index, bank).items()
        if isinstance(v, float)
    }


# --------------------------------------------------------------------------
# Phase 1 experiment
# --------------------------------------------------------------------------

def make_split(bank, kind: str, seed: int, frac_train: float, frac_valid: float,
               weights: np.ndarray | None = None):
    """`weights` is the signature count per compound; it keeps signatures
    balanced across splits, not just compound counts."""
    if kind == "scaffold":
        scaf = bemis_murcko_scaffolds([s or "" for s in bank.smiles])
        tr, va, te = scaffold_split(scaf, frac_train, frac_valid, seed, weights)
        scaf_arr = np.array(scaf)
        leak = set(scaf_arr[tr]) & set(scaf_arr[te])
        if leak:
            raise AssertionError(f"scaffold leak across train/test: {len(leak)} scaffolds")
        return tr, va, te
    if kind == "random":
        return random_split(len(bank), frac_train, frac_valid, seed)
    if kind == "moa":
        return moa_disjoint_split(bank.moa, frac_train, frac_valid, seed, weights)
    raise ValueError(f"unknown split kind: {kind!r}")


def make_signature_split(ds, seed: int, frac_train: float, frac_valid: float,
                         n_compounds: int):
    """Split SIGNATURES, not compounds: every compound appears in training.

    The compound-disjoint splits ask "can you retrieve a molecule you have never
    seen". This asks the other question -- "given a new measurement of a
    compound already in your library, can you identify it" -- which is the
    Connectivity Map task and the one a fixed screening library actually poses.

    Stratified per compound, so the eval splits are not dominated by whichever
    compounds happen to be profiled hundreds of times, and every compound keeps
    **at least one** training signature. A compound profiled once therefore
    trains and is never queried: it stays a retrieval candidate, so it still
    makes the problem harder, it just contributes no questions. Every compound
    with two or more signatures *is* queried -- exactly one of valid or test
    gets it when that is all it can spare.

    Returns three boolean masks over signatures.
    """
    rng = np.random.default_rng(seed)
    n = len(ds.bank_index)
    tr = np.zeros(n, dtype=bool)
    va = np.zeros(n, dtype=bool)
    te = np.zeros(n, dtype=bool)
    frac_test = max(0.0, 1.0 - frac_train - frac_valid)

    order = np.argsort(ds.bank_index, kind="stable")
    starts = np.unique(ds.bank_index[order], return_index=True)[1][1:]
    for rows in np.split(order, starts):
        m = len(rows)
        if m < 2:
            tr[rows] = True                 # cannot spare one; trains only
            continue
        rows = rng.permutation(rows)

        eval_frac = frac_valid + frac_test
        n_eval = min(m - 1, max(1, int(round(m * eval_frac))))
        if n_eval == 1:
            # One query to give: send it to valid or test by a weighted coin,
            # so both splits see shallow compounds in proportion rather than
            # one split monopolising them.
            to_test = rng.random() < (frac_test / max(eval_frac, 1e-9))
            n_te, n_va = (1, 0) if to_test else (0, 1)
        else:
            n_te = int(round(n_eval * frac_test / max(eval_frac, 1e-9)))
            n_te = min(max(n_te, 0), n_eval)
            n_va = n_eval - n_te
        te[rows[:n_te]] = True
        va[rows[n_te:n_te + n_va]] = True
        tr[rows[n_te + n_va:]] = True
    return tr, va, te


def apply_signature_filter(ds, masks: dict, cfg: TrainConfig, n_compounds: int):
    """Narrow `masks` in place by treatment quality. Returns (stats, cache tag).

    Filtering the training split only is the comparable experiment: the model
    sees cleaner data while the held-out queries stay exactly what every
    previous run was scored on. `--filter-splits all` filters the queries too,
    which measures something else entirely and is flagged as such.

    The returned tag goes into the split cache filename. A filtered and an
    unfiltered run of the same seed produce different training data, and
    without the tag the second would silently inherit the first's cache.
    """
    if (cfg.min_dose is None and cfg.min_time is None
            and cfg.drop_outlier_frac <= 0 and cfg.min_compound_sigs is None):
        return None, ""

    from .signal import signature_filter

    which = ("train",) if cfg.filter_splits == "train" else tuple(masks)
    restrict = np.zeros(len(ds.bank_index), dtype=bool)
    for name in which:
        restrict |= masks[name]

    keep, stats = signature_filter(
        ds, cfg.min_dose, cfg.min_time, cfg.drop_outlier_frac,
        n_compounds=n_compounds, restrict=restrict,
        min_compound_sigs=cfg.min_compound_sigs,
    )
    # Keep the masks themselves, not just their counts: the compound-level
    # before/after below has to index by them.
    before = {name: masks[name].copy() for name in which}
    for name in which:
        masks[name] &= keep

    # A compound can lose every one of its signatures. On the train split that
    # means it contributes nothing to training while remaining a retrieval
    # candidate. On valid or test it is worse: the compound drops out of the
    # metric entirely, so macro recall is then averaged over a smaller and
    # differently-composed set than the run you are comparing against. Counted
    # per split, and warned about loudly for the evaluation splits.
    stats["splits_filtered"] = list(which)
    stats["per_split"] = {}
    for name in which:
        b, a = int(before[name].sum()), int(masks[name].sum())
        c_before = len(np.unique(ds.bank_index[before[name]])) if b else 0
        c_after = len(np.unique(ds.bank_index[masks[name]])) if a else 0
        stats["per_split"][name] = {
            "signatures_before": b, "signatures_after": a,
            "compounds_before": int(c_before), "compounds_after": int(c_after),
        }
        logger.info("filter: %s split %d -> %d signatures (%.0f%% kept), "
                    "%d -> %d compounds", name, b, a, 100 * a / max(b, 1),
                    c_before, c_after)
        if a == 0:
            raise ValueError(
                f"the quality filter removed every {name} signature; "
                "loosen --min-dose / --min-time / --drop-outlier-frac"
            )
        if name in ("valid", "test") and c_after < c_before:
            logger.warning(
                "filter dropped %d of %d %s compounds entirely. %s_macro_recall "
                "is now averaged over a DIFFERENT compound set and is not "
                "comparable to an unfiltered run -- report it as its own "
                "benchmark, not beside the headline number",
                c_before - c_after, c_before, name, name,
            )
    stats["n_train_compounds_after"] = (
        stats["per_split"]["train"]["compounds_after"] if "train" in which else None
    )

    tag = "_f" + "".join(
        s for s in (
            f"s{cfg.min_compound_sigs:g}" if cfg.min_compound_sigs is not None else "",
            f"d{cfg.min_dose:g}" if cfg.min_dose is not None else "",
            f"t{cfg.min_time:g}" if cfg.min_time is not None else "",
            f"o{cfg.drop_outlier_frac:g}" if cfg.drop_outlier_frac > 0 else "",
            "" if cfg.filter_splits == "train" else "ALL",
        ) if s
    )
    return stats, tag


def run_experiment(
    bank,
    ds: SignatureDataset,
    cfg: TrainConfig,
    device,
    split: str = "scaffold",
    n_bits: int = 2048,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    out_dir: Path | None = None,
    release_source: bool = False,
) -> dict:
    """Train, then compare against every baseline on the same held-out queries.

    `release_source` frees `ds`'s arrays once the split has been taken; see
    `SignatureDataset.release`. Leave it False if you intend to reuse `ds`,
    as the smoke test does for the lookup ablation.
    """
    # Seed here, not just inside `train`: `build_model` draws the initial
    # weights below and `train`'s own manual_seed comes too late to cover them.
    # Without this, two runs at the same seed do not reproduce each other.
    torch.manual_seed(cfg.seed)

    sigs_per_compound = np.bincount(ds.bank_index, minlength=len(bank)).astype(np.float64)
    signature_level = split == "signature"
    if signature_level:
        # Every compound trains; the splits divide its signatures instead. The
        # bank is untouched, so a query still ranks all of them and the metric
        # stays comparable in difficulty to the compound-disjoint runs.
        masks = dict(zip(
            ("train", "valid", "test"),
            make_signature_split(ds, cfg.seed, frac_train, frac_valid, len(bank)),
        ))
        tr_c, va_c, te_c = (np.unique(ds.bank_index[masks[k]])
                            for k in ("train", "valid", "test"))
        logger.info(
            "signature split: %d/%d/%d signatures over %d/%d/%d compounds "
            "(bank still %d)",
            *(int(masks[k].sum()) for k in ("train", "valid", "test")),
            len(tr_c), len(va_c), len(te_c), len(bank),
        )
    else:
        tr_c, va_c, te_c = make_split(
            bank, split, cfg.seed, frac_train, frac_valid, weights=sigs_per_compound
        )
        logger.info(
            "%s split: %d train / %d valid / %d test compounds",
            split, len(tr_c), len(va_c), len(te_c),
        )
    if not len(te_c):
        raise ValueError("split produced no test compounds")

    scaler = DescriptorScaler(n_bits)
    scaled = scaler.fit_transform(bank.mol_features, tr_c)
    raw_features = bank.mol_features
    bank.mol_features = scaled

    if not signature_level:
        masks = {
            "train": signature_mask(ds.bank_index, tr_c),
            "valid": signature_mask(ds.bank_index, va_c),
            "test": signature_mask(ds.bank_index, te_c),
        }
    filter_stats, filter_tag = apply_signature_filter(ds, masks, cfg, len(bank))

    cache = (Path(cfg.split_cache_dir) if cfg.split_cache_dir
             else split_cache_dir(ds))
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    if cache is not None:
        tag = f"{split}_{cfg.seed}_{len(bank)}{filter_tag}"
        splits = {
            name: subset_to_memmap(ds, m, cache / f"{tag}_{name}.npy")
            for name, m in masks.items()
        }
    else:
        splits = {name: subset_dataset(ds, m) for name, m in masks.items()}
    train_ds, valid_ds, test_ds = splits["train"], splits["valid"], splits["test"]
    if release_source:
        # Before the holdout carve, not after: every split above is an
        # independent copy in both the in-RAM and the memmapped path, so the
        # source is dead weight from here on. Gating this on the memmap path
        # was a bug -- with `--signatures ram` it left the full 1.45 GB source
        # resident for the whole run, roughly doubling host memory.
        ds.release()

    # Carve held-out signatures out of the TRAIN split: same compounds, queries
    # the model never sees. Stratified per compound so every compound with more
    # than one signature contributes, and compounds with exactly one keep it --
    # a compound with no training signature would silently become a different
    # kind of test case.
    newsig_ds = None
    
    if cfg.train_holdout_frac and len(train_ds) and not signature_level:
        rng = np.random.default_rng(cfg.seed)
        held = np.zeros(len(train_ds), dtype=bool)
        order = np.argsort(train_ds.bank_index, kind="stable")
        groups = np.split(order, np.unique(train_ds.bank_index[order],
                                           return_index=True)[1][1:])
        for rows in groups:
            # At least one signature from every compound that can spare one.
            # `int(len * frac)` rounds to zero below 1/frac signatures, which
            # silently restricted the probe to heavily-profiled compounds --
            # exactly the easy ones -- so `newsig` was not comparable to the
            # train and val probes it sits between.
            if len(rows) < 2:
                continue
            n_hold = min(max(1, round(len(rows) * cfg.train_holdout_frac)),
                         len(rows) - 1)
            held[rng.choice(rows, n_hold, replace=False)] = True
        if held.any():
            newsig_ds = subset_dataset(train_ds, held)
            kept = subset_dataset(train_ds, ~held)
            train_ds.release()      # drop the pre-carve copy before rebinding
            train_ds = kept
            logger.info(
                "held out %d signatures from %d train compounds as `newsig` "
                "(%d signatures left for training)",
                len(newsig_ds), len(np.unique(newsig_ds.bank_index)), len(train_ds),
            )
    logger.info(
        "signatures: %d train / %d valid / %d test", len(train_ds), len(valid_ds), len(test_ds)
    )
    if not len(train_ds) or not len(test_ds):
        raise ValueError("split produced an empty signature set")

    n_genes = train_ds.signatures.shape[1]
    model = build_model(bank, n_genes, cfg, device, n_bits=n_bits)
    if cfg.resume_from:
        prev = load_weights_into(model, cfg.resume_from, cfg.reset_temperature,
                                 cfg.temperature)

        for field in ("split", "mol_embeddings_mode", "n_bits"):
            was, now = prev.get(field), getattr(cfg, field, None)
            if was is not None and now is not None and was != now:
                logger.warning("warm-start: checkpoint had %s=%r, this run uses %r",
                               field, was, now)
    mol_features = bank.features_tensor(device)

    # Grouped on the full raw feature matrix -- everything the encoder sees --
    # before standardization touches anything.
    dup_ids = duplicate_group_ids(raw_features)
    n_dup = int((dup_ids >= 0).sum())
    if n_dup:
        logger.warning(
            "%d/%d compounds (%.1f%%) share a fingerprint with another compound; "
            "a structure-only encoder cannot separate them",
            n_dup, len(bank), 100 * n_dup / len(bank),
        )

    history = train(
        model, bank, mol_features, train_ds, valid_ds, cfg, device,
        moa_ids=bank.moa_ids(), duplicate_ids=dup_ids,
        newsig_ds=newsig_ds, out_dir=out_dir,
    )

    # --- held-out retrieval against the FULL bank --------------------------
    test_metrics = evaluate_split(model, bank, mol_features, test_ds, device, "test",
                                  csls_r=cfg.csls_r)
    model_row = {k.removeprefix("test_"): v for k, v in test_metrics.items()}

    queries = test_ds.signatures
    truth = test_ds.bank_index

    def baseline_row(scorer) -> dict:
        """Score a baseline in query blocks and keep only the ranking.

        Each of these returns a (queries x bank) matrix if asked for one at
        once -- 3.5 GB here -- so they are streamed exactly like the model, and
        the fitted scorer is dropped afterwards because the next one is about
        to allocate its own consensus matrix.
        """
        ranking = rank_queries(
            lambda lo, hi: scorer.score(queries[lo:hi]), truth, len(bank)
        )
        return {**macro_recall_at_k(ranking, truth), **recall_at_k(ranking, truth)}

    ecfp = ECFPNearestNeighbor(n_bits=n_bits).fit(
        train_ds.signatures, train_ds.bank_index, bank, tr_c
    )
    ecfp_row = baseline_row(ecfp)
    del ecfp

    cos = CosineConsensus().fit(train_ds.signatures, train_ds.bank_index, len(bank))
    cos_row = baseline_row(cos)
    del cos

    gso = GeneSetOverlap().fit(train_ds.signatures, train_ds.bank_index, len(bank))
    gso_row = baseline_row(gso)
    del gso

    rows = {
        "ceiling (fingerprint duplicates)": recall_ceiling(dup_ids, te_c),
        "model": model_row,
        "ECFP-NN transfer": ecfp_row,
        "cosine consensus": cos_row,
        "gene-set overlap": gso_row,
        "chance": {**{f"macro_{k}": v for k, v in chance_baseline(len(bank)).items()},
                   **chance_baseline(len(bank))},
    }

    bank.mol_features = raw_features  # leave the bank as we found it
    result = {
        "split": split,
        "n_bank": len(bank),
        "n_train_compounds": int(len(tr_c)),
        "n_valid_compounds": int(len(va_c)),
        "n_test_compounds": int(len(te_c)),
        "n_test_signatures": int(len(test_ds)),
        "config": asdict(cfg),
        "signature_filter": filter_stats,
        "n_duplicate_compounds": n_dup,
        "n_distinct_fingerprints": int(len(bank) - n_dup + len(set(dup_ids[dup_ids >= 0].tolist()))),
        "rows": rows,
        "test_metrics": test_metrics,
        "history": history,
    }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return result


def print_result(result: dict) -> None:
    print(
        f"\nheld-out compound retrieval -- {result['split']} split, "
        f"bank of {result['n_bank']} compounds, "
        f"{result['n_test_signatures']} test signatures "
        f"over {result['n_test_compounds']} unseen compounds\n"
    )
    print("MACRO -- every compound counts once (lead with this one)\n")
    print(format_comparison(result["rows"], prefix="macro_"))
    print("\n\nMICRO -- every signature counts once; weighted by how often each")
    print("compound was profiled, which reflects screening effort, not biology\n")
    print(format_comparison(result["rows"]))

    m = result["rows"]["model"].get("macro_recall@10", float("nan"))
    e = result["rows"]["ECFP-NN transfer"].get("macro_recall@10", float("nan"))
    print(f"\nmacro recall@10   model {m:.4f}  vs  ECFP-NN {e:.4f}"
          + (f"  ->  ratio {m / e:.2f}x" if e else "   (ECFP-NN is zero)"))
    mi = result["rows"]["model"].get("recall@10", float("nan"))
    ei = result["rows"]["ECFP-NN transfer"].get("recall@10", float("nan"))
    print(f"micro recall@10   model {mi:.4f}  vs  ECFP-NN {ei:.4f}"
          + (f"  ->  ratio {mi / ei:.2f}x" if ei else "   (ECFP-NN is zero)"))
    n_c = result["rows"]["model"].get("n_compounds_scored")
    if n_c:
        print(f"\n{int(n_c):,} held-out compounds / "
              f"{int(result['rows']['model'].get('n_queries', 0)):,} query signatures")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory holding artifacts.npz")
    ap.add_argument("--out", default="runs/genetomol")
    ap.add_argument("--split", default="scaffold",
                    choices=("scaffold", "random", "moa", "signature"),
                    help="scaffold/random/moa hold whole COMPOUNDS out. "
                         "`signature` holds out measurements instead, so every "
                         "compound is in training -- the library-matching task, "
                         "not generalization to new chemistry")
    ap.add_argument("--split-cache-dir", default=None,
                    help="where to write the memmapped splits (default: a "
                         "`splits/` folder beside artifacts.npz). Point a new "
                         "split regime at its own folder to keep it separate")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bits", type=int, default=None, help="defaults to meta.json")
    ap.add_argument("--lookup-ablation", action="store_true",
                    help="free embedding per compound; should collapse to chance")
    ap.add_argument("--no-mask-false-negatives", action="store_true")
    ap.add_argument("--bank-negatives", action="store_true",
                    help="contrast against the whole train bank, not just the batch")
    ap.add_argument("--no-mask-duplicates", action="store_true",
                    help="do not mask fingerprint-identical compounds; this is what "
                         "broke bank negatives, so leave it alone unless ablating")
    ap.add_argument("--bank-negatives-k", type=int, default=None,
                    help="sample this many bank negatives per step (default: all)")
    ap.add_argument("--memory-refresh-every", type=int, default=1,
                    help="epochs between a full re-encode of the negative memory")
    ap.add_argument("--moa-positive-weight", type=float, default=0.0,
                    help="> 0 scores same-MOA compounds as positives at this "
                         "weight instead of masking them out")
    ap.add_argument("--mol-features", default=None,
                    help="npz from `python -m genetomol.featurize`; "
                         "replaces the bank's fingerprint matrix")
    ap.add_argument("--mol-embeddings", default=None,
                    help="npz from `python -m genetomol.molembed`")
    ap.add_argument("--min-temperature", type=float, default=None,
                    help="floor on the learned temperature, e.g. 0.03. Unset "
                         "lets it reach CLIP's 0.01, which it does -- and the "
                         "validation loss inflates with it")
    ap.add_argument("--min-compound-sigs", type=int, default=None,
                    help="use only compounds profiled at least this many times. "
                         "On LINCS >=20 keeps 76%% of signatures but only 14.5%% "
                         "of compounds, so it mostly cuts chemical diversity")
    ap.add_argument("--min-dose", type=float, default=None,
                    help="drop training signatures dosed below this many uM. "
                         "Signatures with NO recorded dose are stored as 0 and "
                         "so are dropped too")
    ap.add_argument("--min-time", type=float, default=None,
                    help="drop training signatures treated for fewer hours. "
                         "Missing durations were filled with 24 h and survive")
    ap.add_argument("--drop-outlier-frac", type=float, default=0.0,
                    help="drop this fraction of training signatures by "
                         "leave-one-out agreement with their compound's other "
                         "replicates, e.g. 0.25. Compounds with one signature "
                         "are exempt")
    ap.add_argument("--filter-splits", default="train", choices=("train", "all"),
                    help="'train' keeps the benchmark identical to every "
                         "existing run; 'all' filters the held-out queries too "
                         "and is NOT comparable to them")
    ap.add_argument("--resume-from", default=None, metavar="MODEL_PT",
                    help="warm-start the encoders from a finished run's "
                         "model.pt. Weights only -- the optimizer and LR "
                         "schedule start fresh, sized to this run's --epochs")
    ap.add_argument("--reset-temperature", action="store_true",
                    help="with --resume-from, re-initialize the learned "
                         "temperature from --temperature instead of inheriting "
                         "the checkpoint's (which ends around scale 60-77)")
    ap.add_argument("--no-baseline-eval", action="store_true",
                    help="skip the untrained (epoch 0) evaluation")
    ap.add_argument("--csls-r", type=int, default=0,
                    help="hubness correction on the final test split: rank by "
                         "2*cos - mean cosine to the compound's r nearest "
                         "queries. 10 is the measured setting; 0 is off")
    ap.add_argument("--mol-embeddings-blocks", default=None,
                    help="select named column groups from the embeddings file, "
                         "e.g. B1,B4 -- encode all 25 CC spaces once, then sweep "
                         "subsets without re-encoding")
    ap.add_argument("--mol-embeddings-mode", default="concat",
                    choices=("concat", "replace", "only"),
                    help="concat keeps bits + descriptors; replace drops the "
                         "descriptors; only gives the encoder the embedding "
                         "alone, which is the sharper test -- concat lets it "
                         "ignore the new columns and keep using the bits")
    ap.add_argument("--mol-hidden", default=None,
                    help="molecule encoder trunk, e.g. 512,256 (default 1024,512). "
                         "The first thing to shrink: it carries 26x more "
                         "parameters per training example than the signature side")
    ap.add_argument("--sig-hidden", default=None,
                    help="signature encoder trunk, e.g. 1024,512")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="dropout for BOTH trunks; --mol-dropout / --sig-dropout "
                         "override it per encoder")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.07,
                    help="initial temperature; it is learned from here")
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="epochs between validation passes")
    ap.add_argument("--max-sigs-per-epoch", type=int, default=60,
                    help="cap on signatures per compound per epoch; 0 for no cap")
    ap.add_argument("--mol-dropout", type=float, default=None,
                    help="dropout for the molecule encoder alone -- the half "
                         "that fails to generalize. Defaults to --dropout")
    ap.add_argument("--sig-dropout", type=float, default=None,
                    help="dropout for the signature encoder alone. Defaults to "
                         "--dropout")
    ap.add_argument("--mol-weight-decay", type=float, default=None,
                    help="weight decay for the molecule encoder alone "
                         "(default: follow --weight-decay)")
    ap.add_argument("--signatures-on-device", action="store_true",
                    help="keep the train split on the GPU and skip the DataLoader; "
                         "costs ~1.2 GB VRAM on LINCS, removes the per-step host "
                         "copy. CUDA only -- pointless on CPU or unified memory")
    ap.add_argument("--bit-dropout", type=float, default=0.0,
                    help="drop this fraction of ECFP bits before the first layer; "
                         "0.1-0.2 is the targeted fix for compound memorization")
    ap.add_argument("--train-holdout-frac", type=float, default=0.05,
                    help="fraction of each train compound's signatures withheld "
                         "and scored as `newsig_*`; 0 disables")
    ap.add_argument("--early-stop-patience", type=int, default=None,
                    help="stop after this many evals without improving "
                         "--early-stop-metric (default: never stop)")
    ap.add_argument("--early-stop-metric", default="val_macro_recall@10")
    ap.add_argument("--no-restore-best", action="store_true",
                    help="score the test set with the last epoch instead of the "
                         "best-validation checkpoint")
    ap.add_argument("--progress", default="auto", choices=("auto", "on", "off"),
                    help="per-epoch tqdm bar; auto shows it only on a terminal")
    ap.add_argument("--train-probe", type=int, default=None,
                    help="signatures from the train split scored alongside valid, "
                         "so the plots show train vs validation; 0 disables. "
                         "Defaults to the valid split's size (~10 s per eval)")
    ap.add_argument("--signatures", default="auto", choices=("auto", "ram", "mmap"),
                    help="where the 1.45 GB signature matrix lives: ram is ~2x "
                         "faster per epoch and peaks near 2.9 GB; mmap keeps it "
                         "on disk and survives on an 8 GB box")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bank, ds, meta = load_artifacts(args.data, signatures=args.signatures)
    n_bits = args.n_bits or meta.get("n_bits", 2048)
    logger.info(
        "loaded %d compounds / %d signatures / %d genes",
        len(bank), len(ds), ds.signatures.shape[1],
    )

    if args.mol_features:
        from .molembed import load_embeddings

        replacement = load_embeddings(args.mol_features, bank.ids)
        if replacement.shape[1] <= n_bits:
            raise SystemExit(
                f"--mol-features has {replacement.shape[1]} columns but n_bits is "
                f"{n_bits}; the descriptor block is missing"
            )
        logger.info("mol features: replaced %d columns with %d from %s",
                    bank.n_features, replacement.shape[1], args.mol_features)
        bank.mol_features = replacement

    # Snapshot the ECFP bits for the baselines before any embedding touches
    # `mol_features` -- after `--mol-embeddings-mode only` there are no bits
    # left in there, and the baseline would binarize embedding floats instead.
    # Taken after `--mol-features`, so the bar uses whichever fingerprints the
    # run is built on.
    bank.fingerprints = np.ascontiguousarray(bank.mol_features[:, :n_bits])

    if args.mol_embeddings:
        from .molembed import attach

        blocks = ([b.strip() for b in args.mol_embeddings_blocks.split(",")]
                  if args.mol_embeddings_blocks else None)
        added = attach(bank, args.mol_embeddings, args.mol_embeddings_mode,
                       n_bits, blocks=blocks)
        if args.bit_dropout > 0 and args.mol_embeddings_mode == "only":
            # `MoleculeEncoder` drops the first `n_bits` columns, trusting them
            # to be fingerprint bits. Under `only` there are no bits left --
            # those columns are embedding dimensions, and zeroing them means
            # "average molecule", not "absent substructure". Silently the wrong
            # experiment, so refuse it.
            raise SystemExit(
                "--bit-dropout has nothing to drop under --mol-embeddings-mode "
                f"only: the feature matrix is {bank.n_features} embedding "
                f"columns, and bit dropout would corrupt the first {n_bits} of "
                "them. Use --mol-dropout instead."
            )
        logger.info(
            "mol features: %s -> %d columns (%s %d pretrained dims)",
            args.mol_embeddings_mode, bank.n_features,
            "added" if args.mol_embeddings_mode == "concat" else "swapped in", added,
        )

    cfg = TrainConfig(
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        warmup_frac=args.warmup_frac,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        max_sigs_per_epoch=args.max_sigs_per_epoch or None,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        lookup_ablation=args.lookup_ablation,
        mask_false_negatives=not args.no_mask_false_negatives,
        bank_negatives=args.bank_negatives,
        mask_duplicates=not args.no_mask_duplicates,
        train_probe_size=args.train_probe,
        progress=args.progress,
        bit_dropout=args.bit_dropout,
        signatures_on_device=args.signatures_on_device,
        mol_weight_decay=args.mol_weight_decay,
        sig_dropout=args.sig_dropout,
        mol_dropout=args.mol_dropout,
        min_temperature=args.min_temperature,
        csls_r=args.csls_r,
        baseline_eval=not args.no_baseline_eval,
        resume_from=args.resume_from,
        reset_temperature=args.reset_temperature,
        min_compound_sigs=args.min_compound_sigs,
        split_cache_dir=args.split_cache_dir,
        min_dose=args.min_dose,
        min_time=args.min_time,
        drop_outlier_frac=args.drop_outlier_frac,
        filter_splits=args.filter_splits,
        **{k: tuple(int(x) for x in v.split(","))
           for k, v in (("sig_hidden", args.sig_hidden),
                        ("mol_hidden", args.mol_hidden)) if v},
        train_holdout_frac=args.train_holdout_frac,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
        restore_best=not args.no_restore_best,
        bank_negatives_k=args.bank_negatives_k,
        memory_refresh_every=args.memory_refresh_every,
        moa_positive_weight=args.moa_positive_weight,
        num_workers=args.num_workers,
    )
    result = run_experiment(
        bank, ds, cfg, torch.device(args.device),
        split=args.split,
        n_bits=n_bits,
        out_dir=Path(args.out),
        release_source=True,  # the CLI splits once and never reuses `ds`
    )
    print_result(result)
    print(f"\nwrote {Path(args.out) / 'results.json'}")


if __name__ == "__main__":
    main()

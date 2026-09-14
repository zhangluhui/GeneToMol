"""End-to-end check on synthetic data.

Builds a small fake bank with real RDKit-parsable SMILES and a planted
signature-structure relationship, then runs the phase-1 experiment: scaffold
split, train, and compare against ECFP nearest-neighbor transfer. This does
not validate any biology -- it validates that the shapes, the sampler, the
split, the bank indexing and the loss all connect, and that the comparison
harness reports what you think it reports. Run it once after any refactor.

    python -m genetomol.smoke_test
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from .data import CompoundBank, SignatureDataset
from .featurize import featurize_molecules
from .train import TrainConfig, print_result, run_experiment

N_GENES = 978
N_BITS = 512  # small for speed; use 2048 for real runs


def _make_smiles(n: int, seed: int = 0) -> list[str]:
    # Concatenating a scaffold with a substituent string produces plenty of
    # chemically invalid candidates (broken aromaticity, over-valent halogens).
    # That is fine -- they are filtered below -- but RDKit shouts about each
    # one, so silence it here as well as in featurize._require_rdkit.
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    scaffolds = [
        "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "C1CCNCC1", "c1cc2ccccc2s1",
        "C1CCOC1", "c1cnc2ccccc2n1", "C1CCCCC1", "c1ccc(-c2ccccc2)cc1", "c1ccsc1",
    ]
    subs = ["C", "CC", "CCC", "O", "OC", "N", "NC", "F", "Cl",
            "C(=O)O", "C(=O)N", "S(=O)(=O)N"]
    rng = np.random.default_rng(seed)
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < n:
        core = scaffolds[rng.integers(len(scaffolds))]
        tail = "".join(subs[rng.integers(len(subs))] for _ in range(rng.integers(1, 3)))
        cand = core + tail
        if cand in seen:
            continue
        if Chem.MolFromSmiles(cand) is not None:
            seen.add(cand)
            out.append(cand)
    return out


def build_synthetic(n_compounds: int = 200, sigs_per_pert: int = 8, seed: int = 0):
    """A bank whose signatures are a linear function of the ECFP bits.

    A structure-aware encoder can recover that map and generalize to unseen
    scaffolds; a lookup table cannot, and fingerprint-nearest-neighbor
    transfer can only approximate it. That is the gap the real experiment is
    trying to measure, in miniature.

    `sigs_per_pert` sets the *mean* profiling depth; the actual per-compound
    counts are Pareto-skewed, so macro and micro metrics genuinely differ here
    rather than coinciding by construction.
    """
    rng = np.random.default_rng(seed)
    smiles = _make_smiles(n_compounds, seed)
    mol_feats, valid = featurize_molecules(smiles, n_bits=N_BITS)
    assert valid.all(), "synthetic SMILES should all parse"

    moa = [f"moa{i % 20}" for i in range(n_compounds)]
    targets = [(f"GENE{i % 40}",) for i in range(n_compounds)]
    bank = CompoundBank(
        ids=list(smiles),
        mol_features=mol_feats,
        smiles=list(smiles),
        moa=moa,
        targets=targets,
    )

    # Planted map from the *bit block only*, so the relationship the model has
    # to learn is the same one ECFP-NN gets to approximate -- a fair contest.
    w = (rng.normal(0, 1, (N_BITS, N_GENES)) / 30).astype(np.float32)
    profile = mol_feats[:, :N_BITS] @ w

    # Skewed profiling depth, like the real thing: LINCS runs from 1 to ~6,000
    # signatures per compound with a median of 3. Uniform depth would make the
    # macro and micro metrics identical by construction and quietly stop the
    # smoke test from exercising either that difference or the sampler's cap.
    counts = np.clip(
        (rng.pareto(1.2, n_compounds) * sigs_per_pert * 0.6 + 1).astype(int), 1, 400
    )
    bank_index = np.repeat(np.arange(n_compounds), counts)
    n_sig = len(bank_index)
    sigs = profile[bank_index] + rng.normal(0, 0.35, (n_sig, N_GENES)).astype(np.float32)
    ds = SignatureDataset(
        signatures=sigs,
        bank_index=bank_index,
        cell_index=rng.integers(0, 8, n_sig),
        log_dose=rng.normal(0, 0.5, n_sig).astype(np.float32),
        log_time=np.full(n_sig, 1.38, dtype=np.float32),
        has_dose=np.ones(n_sig, dtype=np.float32),
    )
    return bank, ds


def check_descriptor_scaler():
    """Regression guard: a descriptor that is constant on the *train* split.

    This is not hypothetical. On a scaffold split a descriptor can easily be
    constant across the training compounds and vary on the held-out ones. When
    the scaler divided by `std + 1e-6`, those held-out values became features of
    ~1e6, every unseen compound collapsed into one corner of the embedding
    space, and recall went to exactly zero -- while the training loss converged
    beautifully. Silent, split-dependent, and fatal. Keep this check.
    """
    from .featurize import DescriptorScaler

    n_bits = 4
    x = np.zeros((6, n_bits + 3), dtype=np.float32)
    x[:, :n_bits] = np.random.default_rng(0).integers(0, 2, (6, n_bits))
    train_rows = np.array([0, 1, 2, 3])
    x[:, n_bits + 0] = [1.0, 2.0, 3.0, 4.0, 2.5, 3.5]   # varies on train
    x[:, n_bits + 1] = [7.0, 7.0, 7.0, 7.0, 9.0, 5.0]   # CONSTANT on train
    x[:, n_bits + 2] = [0.0, 1.0, 0.0, 1.0, 400.0, -9.0]  # extreme on test

    scaled = DescriptorScaler(n_bits).fit_transform(x, train_rows)
    assert np.isfinite(scaled).all(), "scaler produced non-finite features"
    biggest = float(np.abs(scaled[:, n_bits:]).max())
    assert biggest <= 10.0 + 1e-6, (
        f"held-out descriptor blew up to {biggest:.1f}; the zero-variance guard "
        "or the clip has regressed"
    )
    assert np.allclose(scaled[:, :n_bits], x[:, :n_bits]), "bit block was modified"
    print(f"descriptor scaler guard OK (max |scaled descriptor| = {biggest:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device(args.device)

    check_descriptor_scaler()

    bank, ds = build_synthetic()
    cfg = TrainConfig(
        epochs=args.epochs, batch_size=64, lr=1e-3, embed_dim=128,
        num_workers=0, eval_every=10,
    )
    result = run_experiment(bank, ds, cfg, device, split="scaffold", n_bits=N_BITS)
    print_result(result)

    rows = result["rows"]
    model_r10 = rows["model"]["recall@10"]
    ecfp_r10 = rows["ECFP-NN transfer"]["recall@10"]
    chance_r10 = rows["chance"]["recall@10"]

    assert model_r10 > 5 * chance_r10, (
        f"model recall@10 {model_r10:.4f} is not clearly above chance {chance_r10:.4f}"
    )

    # NOT asserted: that the model beats ECFP-NN. The planted profile here is
    # exactly linear in the ECFP bits, which is the best case there is for
    # fingerprint transfer -- a Tanimoto-near neighbor has near-identical bits
    # and therefore a near-identical profile, which NN transfer gets for free
    # while the model has to learn the map from ~130 compounds. Asserting a win
    # on a task rigged for the baseline would only produce flaky failures. The
    # comparison that decides the project is the one on real L1000, not this.
    print(f"\nECFP-NN recall@10 {ecfp_r10:.4f} vs model {model_r10:.4f} "
          "(informational -- see the note in smoke_test.main)")

    if not args.skip_ablation:
        print("\n--- lookup ablation: should collapse toward chance on unseen compounds ---")
        abl_cfg = TrainConfig(
            epochs=args.epochs, batch_size=64, lr=1e-3, embed_dim=128,
            num_workers=0, eval_every=10, lookup_ablation=True,
        )
        abl = run_experiment(bank, ds, abl_cfg, device, split="scaffold",
                             n_bits=N_BITS)
        abl_r10 = abl["rows"]["model"]["recall@10"]
        print(f"lookup-ablation recall@10 {abl_r10:.4f} "
              f"vs structure model {model_r10:.4f} vs chance {chance_r10:.4f}")
        assert abl_r10 < model_r10, (
            "the lookup ablation matched the structure encoder -- the held-out "
            "compounds are leaking, or the split is not doing what you think"
        )

    print("\nsmoke test passed")


if __name__ == "__main__":
    main()

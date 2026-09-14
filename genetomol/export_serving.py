"""Freeze a trained run into a small, torch-free serving bundle.

Analysis reads 2.8 GB -- `artifacts.npz` plus the signature matrix -- and
rebuilds the train/test split. A query needs none of that. Once the model is
trained the bank embeddings are FIXED, so they are computed once here and the
molecule encoder is discarded entirely:

    signature encoder weights     2.8 MB   kept, this is the only model left
    encoded bank (23,966 x 256)  24.5 MB   kept, precomputed
    compound metadata             ~3 MB    kept, for display
    molecule encoder             11.1 MB   dropped, bank already encoded
    artifacts.npz + signatures  2806 MB    dropped

About 30 MB, against 2.8 GB. That fits any free hosting tier, starts in
seconds rather than minutes, and -- because `serve.py` reimplements the
encoder in numpy -- removes torch from the deployment.

    python -m genetomol.export_serving --data ../lincs/prepared \\
        --run runs/genetomol \\
        --mol-features ../lincs/prepared/ecfp_chiral.npz \\
        --out serving_bundle.npz
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 1


def landmark_aliases(genes, geneinfo: str | None):
    """symbol/Ensembl -> Entrez, for the landmark genes ONLY.

    A user's file may hold 20,000 genes, but anything outside the landmark set
    is discarded regardless, so only these 978 need to be recognizable. That
    turns a 1 MB external lookup into a few KB that travels inside the bundle,
    which is what lets the app map gene symbols on a machine that has no LINCS
    data at all.
    """
    if not geneinfo or not Path(geneinfo).exists():
        return {}
    import pandas as pd

    df = pd.read_csv(geneinfo, sep="	", dtype=str)
    wanted = set(genes)
    out = {}
    for col in ("gene_symbol", "ensembl_id"):
        if col not in df:
            continue
        for alias, gid in zip(df[col], df["gene_id"]):
            gid = str(gid).strip()
            if gid in wanted and isinstance(alias, str) and alias.strip() not in ("", "-"):
                out.setdefault(alias.strip().upper(), gid)
    return out


def build(data_dir: str, run_dir: str, mol_features: str | None = None,
          mol_embeddings: str | None = None, mol_embeddings_mode: str = "concat",
          task: str | None = None, geneinfo: str | None = None) -> dict:
    """Encode the bank and collect everything a query needs. Torch runs here."""
    import torch

    from .runio import load_run

    run = load_run(data_dir, run_dir, mol_features, mol_embeddings,
                   mol_embeddings_mode, "cpu")
    with torch.no_grad():
        keys = run.model.encode_bank(run.bank.features_tensor(run.device))
    keys = keys.cpu().numpy().astype(np.float32)

    state = run.model.state_dict()
    weights = {k[len("signature_encoder."):]: v.cpu().numpy().astype(np.float32)
               for k, v in state.items() if k.startswith("signature_encoder.")}

    genes = json.loads(
        (Path(data_dir) / "meta.json").read_text(encoding="utf-8"))["gene_ids"]
    if keys.shape[0] != len(run.bank):
        raise SystemExit("bank/keys mismatch; refusing to write a bundle")

    # The split decides what the numbers mean, so it travels with the weights
    # rather than being remembered by whoever deploys it.
    alias = landmark_aliases(genes, geneinfo)
    logger.info("%d gene aliases embedded (symbols + Ensembl for %d landmarks)",
                len(alias), len(genes))
    split = run.split
    if task is None:
        task = ("library-matching" if split == "signature"
                else "novel-compound")
    bundle = {
        "version": BUNDLE_VERSION,
        "run": str(run_dir),
        "split": split,
        "task": task,
        "genes": np.array(genes, dtype=np.str_),
        "keys": keys,
        "ids": np.array(run.bank.ids, dtype=np.str_),
        "smiles": np.array([s or "" for s in run.bank.smiles], dtype=np.str_),
        "moa": np.array([m or "" for m in run.bank.moa], dtype=np.str_),
        "targets": np.array([",".join(t) if t else ""
                             for t in run.bank.targets], dtype=np.str_),
        "alias_key": np.array(list(alias), dtype=np.str_),
        "alias_val": np.array([alias[k] for k in alias], dtype=np.str_),
        **{f"w__{k}": v for k, v in weights.items()},
    }
    logger.info("encoded %d compounds into %d dims from %s (%s split)",
                keys.shape[0], keys.shape[1], run_dir, split)
    return bundle


def save(bundle: dict, out: str | Path) -> Path:
    out = Path(out)
    meta = {k: v for k, v in bundle.items() if isinstance(v, (str, int))}
    arrays = {k: v for k, v in bundle.items() if not isinstance(v, (str, int))}
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--mol-features", default=None,
                    help="the SAME file the run trained with")
    ap.add_argument("--mol-embeddings", default=None)
    ap.add_argument("--mol-embeddings-mode", default="concat",
                    choices=("concat", "replace", "only"))
    ap.add_argument("--geneinfo", default=None,
                    help="geneinfo_beta.txt. Its symbol/Ensembl aliases for the "
                         "landmark genes are copied INTO the bundle, so the "
                         "served app needs no LINCS files")
    ap.add_argument("--task", default=None,
                    help="override the task label shown in the UI")
    ap.add_argument("--out", default="serving_bundle.npz")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    b = build(args.data, args.run, args.mol_features, args.mol_embeddings,
              args.mol_embeddings_mode, args.task, args.geneinfo)
    path = save(b, args.out)
    mb = path.stat().st_size / 1e6
    print(f"\nwrote {path}  ({mb:.1f} MB)")
    print(f"  run     {b['run']}")
    print(f"  split   {b['split']}   task: {b['task']}")
    print(f"  bank    {len(b['ids']):,} compounds x {b['keys'].shape[1]} dims")
    print(f"  genes   {len(b['genes'])} landmarks, "
          f"{len(b['alias_key']):,} symbol/Ensembl aliases embedded")
    if len(b["alias_key"]) == 0:
        print("  WARNING: no --geneinfo given, so the app will accept Entrez "
              "ids only. Pass it unless your users key on Entrez.")
    print("\nserve it with:  streamlit run app.py")


if __name__ == "__main__":
    main()

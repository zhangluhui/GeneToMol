"""Query a serving bundle. NumPy only -- torch is never imported.

The signature encoder is three Linear layers with LayerNorm and GELU between
them, then an L2 normalize. That is a few lines of numpy, and reimplementing it
here removes a ~200 MB wheel from the deployment and most of the cold start.

`selftest()` compares this encoder against the torch module it replaces, and
runs wherever torch happens to be installed.

Dropout is absent because it is identity at eval time. No other layer differs
between train and eval in this trunk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_ERF = np.frompyfunc(math.erf, 1, 1)          # stdlib erf, exact to double


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU, matching `nn.GELU()`'s default (erf, not the tanh approx).

    Use `math.erf`, which is exact to double precision. Neither the tanh
    approximation nor a rational erf is close enough here: small errors in the
    activation move compounds several ranks.
    """
    return 0.5 * x * (1.0 + _ERF(x / math.sqrt(2)).astype(np.float64))


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class Bundle:
    """A frozen run: the signature encoder, the encoded bank, and its metadata."""

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.genes = [str(g) for g in z["genes"]]
        self.keys = z["keys"].astype(np.float32)          # (P, D), unit rows
        self.ids = [str(s) for s in z["ids"]]
        self.smiles = [str(s) for s in z["smiles"]]
        self.moa = [str(s) for s in z["moa"]]
        self.targets = [str(s) for s in z["targets"]]
        # symbol/Ensembl -> Entrez for the landmarks, so a served machine needs
        # no LINCS metadata to accept gene symbols.
        self.aliases = ({str(k): str(v) for k, v in
                         zip(z["alias_key"], z["alias_val"])}
                        if "alias_key" in z.files else {})
        self.w = {k[len("w__"):]: z[k].astype(np.float64)
                  for k in z.files if k.startswith("w__")}
        # trunk.0 / trunk.4 / trunk.8 are the Linears; 1 and 5 the LayerNorms.
        self._linear = sorted(
            int(k.split(".")[1]) for k in self.w if k.endswith(".weight")
            and self.w[k].ndim == 2)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def task(self) -> str:
        return self.meta.get("task", "unknown")

    @property
    def run(self) -> str:
        return self.meta.get("run", "unknown")

    def encode(self, X: np.ndarray) -> np.ndarray:
        """(Q, n_genes) -> (Q, D) unit vectors, same arithmetic as the trunk."""
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != len(self.genes):
            raise ValueError(
                f"query has {x.shape[1]} genes, bundle expects {len(self.genes)}")
        last = self._linear[-1]
        for i in self._linear:
            x = x @ self.w[f"trunk.{i}.weight"].T + self.w[f"trunk.{i}.bias"]
            if i != last:                     # no norm/activation on the output
                x = layer_norm(x, self.w[f"trunk.{i + 1}.weight"],
                               self.w[f"trunk.{i + 1}.bias"])
                x = gelu(x)
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return (x / np.maximum(n, 1e-12)).astype(np.float32)

    def rank(self, X: np.ndarray, k: int = 25):
        """Top-k bank positions and cosine scores. No ground truth involved."""
        q = self.encode(X)
        s = q @ self.keys.T                   # both unit -> cosine
        k = int(min(k, s.shape[1]))
        idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
        take = np.take_along_axis(s, idx, axis=1)
        order = np.argsort(-take, axis=1, kind="stable")
        idx = np.take_along_axis(idx, order, axis=1)
        return idx, np.take_along_axis(s, idx, axis=1)

    def hits(self, X: np.ndarray, k: int = 25) -> list[list[dict]]:
        idx, score = self.rank(X, k)
        return [[{"rank": r + 1, "compound_id": self.ids[j],
                  "cosine": float(v), "moa": self.moa[j],
                  "target": self.targets[j], "smiles": self.smiles[j]}
                 for r, (j, v) in enumerate(zip(row.tolist(), sc.tolist()))]
                for row, sc in zip(idx, score)]


def selftest(bundle_path: str | Path, run_dir: str, n: int = 256) -> dict:
    """Compare the numpy encoder against the torch module it replaces.

    Needs torch, so it runs where the bundle is BUILT, not where it is served.
    """
    import torch

    from .model import SignatureEncoder

    b = Bundle(bundle_path)
    ck = torch.load(Path(run_dir) / "model.pt", map_location="cpu")["model"]
    w = {k[len("signature_encoder."):]: v for k, v in ck.items()
         if k.startswith("signature_encoder.")}
    hidden = tuple(w[f"trunk.{i}.weight"].shape[0] for i in b._linear[:-1])
    enc = SignatureEncoder(n_genes=len(b.genes),
                           embed_dim=w[f"trunk.{b._linear[-1]}.weight"].shape[0],
                           hidden=hidden)
    enc.load_state_dict(w)
    enc.eval()
    X = np.random.default_rng(0).normal(size=(n, len(b.genes))).astype(np.float32)
    with torch.no_grad():
        ref = enc(torch.as_tensor(X)).numpy()
    got = b.encode(X)
    return {"max_abs_diff": float(np.abs(ref - got).max()),
            "mean_cosine": float((ref * got).sum(1).mean()),
            "n": n}

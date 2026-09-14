"""Compound retrieval from transcriptomic state transitions.

Submodules are imported lazily: `featurize` needs RDKit and `data`/`model`/
`train` need torch, so a bare `import genetomol` stays cheap and does
not fail on a machine that only has the preprocessing dependencies installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

_SUBMODULES = (
    "baselines", "census", "data", "download", "evaluate", "featurize", "model",
    "prepare", "train",
)

if TYPE_CHECKING:  # pragma: no cover
    from . import (
        baselines, census, data, download, evaluate, featurize, model, prepare, train,
    )


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_SUBMODULES])

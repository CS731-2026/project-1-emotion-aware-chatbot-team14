"""tutorial — heavily commented reference model. Read in this order:
    __init__.py (here) → model.py → augment.py → data.py → train_loop.py

This module is both a working model (trains on synthetic_smoke in
seconds) AND the source `make new-model` copies from.
"""

from __future__ import annotations

# Three framework types — you receive Context + DatasetSpec from the
# driver and return TrainedModel back to it. Never construct them yourself.
from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

# Sibling files. Re-export PREPROCESS so the live model_service can use the
# same eval transforms at inference (convention every model module follows).
from .augment import VAL_TF as PREPROCESS
from .model import build                  # noqa: F401 — re-exported on purpose
from .train_loop import run as _run


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    # The driver imports this module and calls train(ctx, dataset). That's it.
    # Splitting train()/_run() lets you swap training implementations
    # (sweep, multi-stage, etc) without touching the driver contract.
    return _run(ctx, dataset, model=build(dataset.num_classes))

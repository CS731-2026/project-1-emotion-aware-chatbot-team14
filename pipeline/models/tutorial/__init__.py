"""tutorial, read in this order: __init__ → model → augment → data → train_loop.

Working model (trains on synthetic_smoke in seconds) AND the source
`make new-model` copies from.
"""

from __future__ import annotations

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

from .augment import VAL_TF as PREPROCESS    # re-exported for the live model_service
from .model import build                      # noqa: F401, re-exported on purpose
from .train_loop import run as _run


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    # The driver imports this module and calls train(ctx, dataset). That's it.
    return _run(ctx, dataset, model=build(dataset.num_classes))

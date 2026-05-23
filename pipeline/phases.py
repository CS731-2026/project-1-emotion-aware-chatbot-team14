"""Phase functions. Each has signature (ctx: Context) -> None and is
registered in driver.py's PHASES dict; the driver iterates Context.config.phases."""

from __future__ import annotations

import logging
import random

from .framework import keys as K
from .framework.context import Context
from .framework.specs import DatasetSpec, TrainedModel

logger = logging.getLogger(__name__)


def setup(ctx: Context) -> None:
    """Seed RNGs + drop a breadcrumb. Bookkeeping only."""
    seed = ctx.config.seed
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.info("setup: run_dir=%s seed=%d", ctx.run_dir, seed)
    ctx.save_text("setup", f"slug: {ctx.config.slug()}\nseed: {seed}\n")


def prepare_dataset(ctx: Context) -> None:
    """Delegate to ctx.dataset_module.prepare(ctx) — the USER's function in
    pipeline/datasets/<name>/__init__.py."""
    # ↓ user function — every dataset module exports this (see protocols.DatasetModule).
    spec = ctx.dataset_module.prepare(ctx)
    if not isinstance(spec, DatasetSpec):
        raise TypeError(
            f"{ctx.dataset_module.__name__}.prepare(ctx) returned "
            f"{type(spec).__name__}, expected DatasetSpec"
        )
    ctx.store.put(K.DATASET, spec)
    ctx.save_json("dataset_used", spec.to_manifest())


def train(ctx: Context) -> None:
    """Delegate to ctx.model_module.train(ctx, dataset) — the USER's function
    in pipeline/models/<name>/__init__.py.

    Two functions called `train` are involved: this one (framework phase)
    vs. the user's train (the one whose contract the model tutorial documents).
    """
    ds = ctx.store.get(K.DATASET, DatasetSpec)
    # ↓ user function — every model module exports this (see protocols.ModelModule).
    trained = ctx.model_module.train(ctx, ds)
    if not isinstance(trained, TrainedModel):
        raise TypeError(
            f"{ctx.model_module.__name__}.train(ctx, dataset) returned "
            f"{type(trained).__name__}, expected TrainedModel"
        )
    ctx.store.put(K.TRAINED_MODEL, trained)

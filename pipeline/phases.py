"""Phase functions.

Composition file in the utility/state/composition taxonomy. Each
phase has signature `(ctx: Context) -> None` and is registered in
pipeline/driver.py's PHASES dict. Order of execution is set by the
phase list on Context.config; the driver iterates it.

What each phase does:

  setup            seed RNGs, drop a breadcrumb
  prepare_dataset  delegate to ctx.dataset_module.prepare(ctx) → DatasetSpec
  train            build the model, train + eval, save checkpoint + TrainedModel
"""

from __future__ import annotations

import logging
import random

from .framework import keys as K
from .framework.context import Context
from .framework.specs import DatasetSpec, TrainedModel

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------


def setup(ctx: Context) -> None:
    """Seed RNGs and drop a breadcrumb in the run dir. Bookkeeping only."""
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


# ----------------------------------------------------------------------------


def prepare_dataset(ctx: Context) -> None:
    """Delegate to the registered dataset module's prepare() function.

    Each dataset module knows how to fetch its own source (Kaggle,
    synthetic, local, …) and uses the helpers in pipeline.ingest to
    walk, remap, split, and persist a DatasetSpec. The phase is just
    glue — store the spec + drop a self-describing artifact.
    """
    spec = ctx.dataset_module.prepare(ctx)
    if not isinstance(spec, DatasetSpec):
        raise TypeError(
            f"{ctx.dataset_module.__name__}.prepare(ctx) returned "
            f"{type(spec).__name__}, expected DatasetSpec"
        )
    ctx.store.put(K.DATASET, spec)
    ctx.save_json("dataset_used", spec.to_manifest())


# ----------------------------------------------------------------------------


def train(ctx: Context) -> None:
    """Delegate to the registered model module's train() function.

    The model module owns its training loop — vanilla classifiers
    typically call pipeline.training.standard.train_classifier; custom
    architectures (multi-stage, paper-specific losses, GANs, etc.)
    write their own loop directly here. The phase is just glue:
    invoke train(), assert the return type, put the TrainedModel
    in the store.
    """
    ds = ctx.store.get(K.DATASET, DatasetSpec)
    trained = ctx.model_module.train(ctx, ds)
    if not isinstance(trained, TrainedModel):
        raise TypeError(
            f"{ctx.model_module.__name__}.train(ctx, dataset) returned "
            f"{type(trained).__name__}, expected TrainedModel"
        )
    ctx.store.put(K.TRAINED_MODEL, trained)

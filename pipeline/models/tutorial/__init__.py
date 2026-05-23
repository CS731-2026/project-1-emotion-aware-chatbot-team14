"""tutorial — heavily commented reference model. READ THIS FIRST.

This module is a working model end-to-end (it trains on synthetic_smoke
in a few seconds) AND the template used by `make new-model`. Every
file in this directory is annotated to explain WHY each piece exists,
not just what it does. Read top to bottom in this order:

    1. __init__.py     ← you are here. The pipeline-facing surface.
    2. model.py        ← architecture. Your nn.Module.
    3. augment.py      ← transforms. What images look like going in.
    4. data.py         ← dataset class. How the CSV becomes batches.
    5. train_loop.py   ← the actual training. CFG + loop + reporting.

A model module is just a Python package. The pipeline driver imports
it and calls one function: `train(ctx, dataset)`. Everything else is
your call — how you build the architecture, what loss you use, what
optimiser, how many epochs, how you log. The framework provides
helpers (see pipeline/training/) but doesn't force a shape on the
training loop itself.

## What the driver expects

When `runs.yaml` has `model: tutorial`, the driver does roughly:

    import pipeline.models.tutorial as model_module
    trained = model_module.train(ctx, dataset_spec)

Two attributes must exist on this module:
  - `train(ctx, dataset) -> TrainedModel`  ← what the driver calls
  - `build(num_classes) -> nn.Module`      ← reusable factory for the
                                              architecture (also used
                                              by deploy + tests)

Some models additionally export `PREPROCESS` — a torchvision transform
the live model_service uses at inference time. We export it below.

## Running this model

    # smoke-test on synthetic data (no Kaggle needed)
    make new-model ID=my_first --template tutorial   # copies this whole dir
    # then edit runs.yaml to add: { dataset: synthetic_smoke, model: my_first, config: fast }
    make train RUN=my_first

The tutorial itself isn't in runs.yaml by default (it's a reference,
not a real model to ship). But you CAN run it directly to see the
end-to-end flow before scaffolding your own:

    python -c "
    import logging; logging.basicConfig(level=logging.INFO, format='%(message)s')
    import configs.fast as cfg
    from pipeline.datasets import synthetic_smoke
    from pipeline.models import tutorial
    from pipeline.driver import sweep
    sweep([(synthetic_smoke, tutorial, cfg)], fail_fast=True)
    "
"""

from __future__ import annotations

# Type-only imports — Context + DatasetSpec + TrainedModel are the three
# framework types your train() function interacts with. You don't
# construct them yourself; the driver hands them to you (Context,
# DatasetSpec) or expects them back (TrainedModel).
from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

# Sibling files. Keep imports narrow — re-exporting the augment is a
# convention (model_service uses PREPROCESS at inference, so it has to
# be importable from the model module's top level).
from .augment import VAL_TF as PREPROCESS
from .model import build                  # noqa: F401 — re-exported on purpose
from .train_loop import run as _run


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    """Driver entry point. The whole module exists for this one function.

    The split between `train()` here and `_run()` in train_loop.py is
    intentional: this file is the thin pipeline-facing surface, the
    training loop is the implementation. Keeping them separate means
    you can swap implementations (different train_loop variants) by
    changing one import here, without the driver caring.
    """
    return _run(ctx, dataset, model=build(dataset.num_classes))

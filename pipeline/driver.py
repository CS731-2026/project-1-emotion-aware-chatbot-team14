"""Pipeline driver.

Composition file — orchestrates phases and the dataset×model×config
sweep. Reads module references registered in pipeline/train.py (the
entry point) and walks the cross-product, one run per cell.

The PHASES dict is the registry of phase functions. Adding a new
phase = one function in phases.py + one entry here.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Callable, Sequence

from . import phases
from .framework import Config, Context


# A single training run — one (dataset_module, model_module, config_module)
# triple. The entry point (pipeline/train.py) declares a list of these;
# the sweep iterates them in order.
Run = tuple[ModuleType, ModuleType, ModuleType]

logger = logging.getLogger(__name__)

PhaseFn = Callable[[Context], None]

# Phase registry. Adding a new phase = one import + one dict entry.
PHASES: dict[str, PhaseFn] = {
    "setup":           phases.setup,
    "prepare_dataset": phases.prepare_dataset,
    "train":           phases.train,
    # evaluate lands later; for now train does inline val/test eval
}


def run_one(
    dataset_module: ModuleType,
    model_module:   ModuleType,
    config_module:  ModuleType,
    *,
    seed: int = 42,
    phases_to_run: list[str] | None = None,
) -> Context:
    """Run one (dataset, model, config) combination end-to-end.

    The three modules are everything we need — names come from each
    module's NAME attribute; the train hyperparam dict comes from
    config_module.CONFIG; the dataset module's prepare() is called by
    the prepare_dataset phase; the model module's build()/PREPROCESS
    are called by the train phase.
    """
    phases_to_run = phases_to_run or ["setup", "prepare_dataset", "train"]
    cfg = Config(
        dataset    = dataset_module.NAME,
        model      = _model_name(model_module),
        config     = config_module.NAME,
        seed       = seed,
        train_cfg  = dict(config_module.CONFIG),
        phases     = list(phases_to_run),
    )
    ctx = Context.create(cfg, dataset_module=dataset_module, model_module=model_module)
    try:
        for name in cfg.phases:
            phase_fn = PHASES.get(name)
            if phase_fn is None:
                raise KeyError(
                    f"unknown phase {name!r}. registered: {sorted(PHASES.keys())}"
                )
            logger.info("→ phase: %s", name)
            phase_fn(ctx)
        logger.info("✓ experiment complete: %s", ctx.run_dir)
    finally:
        ctx.close()
    return ctx


def sweep(
    runs:       Sequence[Run],
    *,
    seed:       int = 42,
    fail_fast:  bool = False,
) -> list[Context]:
    """Run each (dataset, model, config) triple in `runs`, in order. One
    triple = one run, one run dir.

    Explicit triples (rather than a cross-product of three lists) because
    not every model belongs with every dataset / config — the entry
    point should declare the specific pairings worth training.

    A run that raises is logged and the sweep continues by default —
    the leaderboard reflects partial results. `fail_fast=True` flips
    to "any failure stops the sweep" (CI-style).
    """
    logger.info("sweep: %d run(s) queued", len(runs))
    contexts: list[Context] = []
    for i, (ds, m, c) in enumerate(runs, 1):
        logger.info("─" * 60)
        logger.info("sweep [%d/%d]: %s × %s × %s",
                    i, len(runs), ds.NAME, _model_name(m), c.NAME)
        try:
            contexts.append(run_one(ds, m, c, seed=seed))
        except Exception:
            logger.exception("sweep cell failed: %s × %s × %s",
                             ds.NAME, _model_name(m), c.NAME)
            if fail_fast:
                raise
    return contexts


def _model_name(module: ModuleType) -> str:
    """Models don't carry a NAME constant — derive it from the module
    path. `models.tiny_cnn` → `tiny_cnn`."""
    return module.__name__.rsplit(".", 1)[-1]

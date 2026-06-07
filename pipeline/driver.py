"""Pipeline driver — orchestrates phases and the sweep over runs.yaml entries.

Call chain for one run:

    make train
      → pipeline/train.py::main()
          → pipeline.runs_loader.load_runs("runs.yaml")
              # importlib resolves dataset/model/config names → modules
          → pipeline.driver.sweep(runs)
              → for each: pipeline.driver.run_one(ds, m, c)
                  → pipeline.framework.context.Context.create(...)   # builds run dir
                  → for each phase in PHASES:
                      → pipeline.phases.<phase>(ctx)
                          # prepare_dataset → pipeline.datasets.<name>.prepare(ctx)
                          # train           → pipeline.models.<name>.train(ctx, dataset)
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Callable, Sequence

from . import phases
from .framework import Config, Context


# One run = one (dataset_module, model_module, config_module) triple.
Run = tuple[ModuleType, ModuleType, ModuleType]

logger = logging.getLogger(__name__)

PhaseFn = Callable[[Context], None]

# Phase registry. Adding a new phase = one function in phases.py + one entry here.
PHASES: dict[str, PhaseFn] = {
    "setup":           phases.setup,
    "prepare_dataset": phases.prepare_dataset,
    "train":           phases.train,
    "evaluate":        phases.evaluate,
}

# Default phase list when a run doesn't specify one. evaluate runs after
# train so the freshly-saved best checkpoint gets fed through the same
# eval pipeline every other run uses — apples-to-apples leaderboard
# results fall out automatically.
DEFAULT_PHASES: list[str] = ["setup", "prepare_dataset", "train", "evaluate"]


def run_one(
    dataset_module: ModuleType,
    model_module:   ModuleType,
    config_module:  ModuleType,
    *,
    seed: int = 42,
    phases_to_run: list[str] | None = None,
) -> Context:
    """Run one (dataset, model, config) combination end-to-end."""
    phases_to_run = phases_to_run or list(DEFAULT_PHASES)
    cfg = Config(
        dataset_name = dataset_module.NAME,
        model_name   = _short_module_name(model_module),
        config_name  = config_module.NAME,
        seed         = seed,
        train_cfg    = dict(config_module.CONFIG),
        phases       = list(phases_to_run),
    )
    ctx = Context.create(cfg, dataset_module=dataset_module, model_module=model_module)
    try:
        for name in cfg.phases:
            phase_fn = PHASES.get(name)
            if phase_fn is None:
                raise KeyError(f"unknown phase {name!r}. registered: {sorted(PHASES)}")
            logger.info("→ phase: %s", name)
            phase_fn(ctx)
        logger.info("✓ experiment complete: %s", ctx.run_dir)
    finally:
        ctx.close()
    return ctx


def sweep(runs: Sequence[Run], *, seed: int = 42, fail_fast: bool = False) -> list[Context]:
    """Run each (dataset, model, config) triple in order.

    A failing run is logged and the sweep continues — leaderboard reflects
    partial results. fail_fast=True flips to "stop on first failure".
    """
    logger.info("sweep: %d run(s) queued", len(runs))
    contexts: list[Context] = []
    for i, (ds, m, c) in enumerate(runs, 1):
        logger.info("─" * 60)
        logger.info("sweep [%d/%d]: %s × %s × %s",
                    i, len(runs), ds.NAME, _short_module_name(m), c.NAME)
        try:
            contexts.append(run_one(ds, m, c, seed=seed))
        except Exception:
            logger.exception("sweep cell failed: %s × %s × %s",
                             ds.NAME, _short_module_name(m), c.NAME)
            if fail_fast:
                raise
    return contexts


def _short_module_name(module: ModuleType) -> str:
    # "pipeline.models.tiny_cnn" → "tiny_cnn"
    return module.__name__.rsplit(".", 1)[-1]

"""Pipeline driver.

Loads an experiment yaml into a Config, builds a Context, then walks
the phases listed in cfg.phases in order. Each phase is looked up in
the PHASES registry — adding a new phase = a new function in
phases.py + a new entry in this dict.

A phase that raises propagates the exception (we want the run to fail
loudly mid-pipeline); the caller decides whether to keep going across
a sweep of experiments. Context's metrics handle is closed in a
finally so the file isn't left open even on failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from . import phases
from .framework import Config, Context, load_experiment

logger = logging.getLogger(__name__)

PhaseFn = Callable[[Context], None]

# Phase registry. Adding a new phase = one import + one dict entry.
PHASES: dict[str, PhaseFn] = {
    "setup":           phases.setup,
    "prepare_dataset": phases.prepare_dataset,
    "train":           phases.train,
    # evaluate lands in a later commit; for now train does inline val/test eval
}


def run_experiment(cfg: Config) -> Context:
    """Walk cfg.phases against the registry. Returns the populated
    Context so callers can inspect ctx.store / ctx.run_dir after."""
    ctx = Context.create(cfg)
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


def run_experiment_file(path: str | Path) -> Context:
    """Convenience wrapper — load yaml + run_experiment in one call."""
    return run_experiment(load_experiment(path))

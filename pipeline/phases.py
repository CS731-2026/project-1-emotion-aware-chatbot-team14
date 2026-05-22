"""Phase functions.

Each phase has the same signature — `(ctx: Context) -> None` — and
participates in the pipeline by being registered in `pipeline.driver.PHASES`.
A phase reads what it needs from `ctx.store` (typed `get`s), does its
work, drops artifacts via `ctx.save_*`, and `put`s any produced
objects back into the store for later phases.

Order of execution is controlled by `cfg.phases` (a list of phase
names from the experiment yaml), so an experiment can skip phases for
fast iteration or insert new ones without code changes to the driver.

Real phase bodies (prepare_dataset, train, evaluate) land in later
commits — this file ships with `setup` so the driver has something to
exercise end-to-end.
"""

from __future__ import annotations

import logging
import random

from .context import Context

logger = logging.getLogger(__name__)


def setup(ctx: Context) -> None:
    """First phase. Seed RNGs, log run identity, leave a breadcrumb in
    the run dir. No store puts — setup is bookkeeping only.

    Importing torch / numpy lives inside the function so the framework
    module itself stays import-time cheap; phases that don't need them
    (e.g. a notebook-driven dry run) won't pay the cost.
    """
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

"""Pipeline plumbing — the never-touch core.

Everything in this package is framework: the typed handoff store, the
config loader, the run-dir-owning Context, the phase driver. Nothing
here knows about specific models / datasets / training mechanics —
adding any of those is a one-file change OUTSIDE this package.

Public surface re-exported for convenience:
    from pipeline.framework import Config, Context, Store, run_experiment
"""

from .config import Config, load_experiment
from .context import Context
from .driver import PHASES, run_experiment, run_experiment_file
from .store import Store

__all__ = [
    "Config",
    "Context",
    "Store",
    "PHASES",
    "load_experiment",
    "run_experiment",
    "run_experiment_file",
]

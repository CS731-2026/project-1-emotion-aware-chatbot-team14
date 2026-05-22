"""Framework — the plumbing the pipeline composes on.

Everything in this package is **state** (in the utility/state/composition
taxonomy): primitive types and the immutable handoff objects. Nothing
here knows about specific phases, models, or datasets.

  store.py     phase-to-phase handoff bag (typed put/get)
  keys.py      the keys phases agree on when using the store
  config.py    the resolved experiment plan (frozen dataclass)
  context.py   what every phase receives — config + store + save_*
  specs.py     DatasetSpec, TrainedModel — typed objects passed via store

Composition (phases.py, driver.py at the parent level) imports from
here. The reverse is forbidden.
"""

from .config import Config, load_experiment
from .context import Context
from .specs import DatasetSpec, TrainedModel
from .store import Store

__all__ = [
    "Config",
    "Context",
    "DatasetSpec",
    "Store",
    "TrainedModel",
    "load_experiment",
]

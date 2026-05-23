"""Framework — the plumbing the pipeline composes on.

State category in the utility/state/composition taxonomy: primitive
types + immutable handoff objects. Nothing here knows about specific
phases, models, or datasets.

  store.py     phase-to-phase handoff bag (typed put/get)
  keys.py      store keys phases agree on
  config.py    resolved experiment plan (frozen)
  context.py   what every phase receives — config + store + module
               refs + save_* artifact methods
  specs.py     DatasetSpec, TrainedModel — typed handoff objects
"""

from .config import Config
from .context import Context
from .protocols import ConfigModule, DatasetModule, ModelModule
from .specs import DatasetSpec, TrainedModel
from .store import Store

__all__ = [
    "Config", "ConfigModule", "Context", "DatasetModule", "DatasetSpec",
    "ModelModule", "Store", "TrainedModel",
]

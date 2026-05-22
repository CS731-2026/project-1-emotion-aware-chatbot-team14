"""Experiment config — the immutable input plan for one run.

A Config is built by the driver from three Python modules (dataset,
model, config) registered in pipeline/train.py. The names go into the
run slug; train_cfg (the dict from the config module's CONFIG export)
goes to the train phase.

There's no yaml-loader here anymore — datasets and configs are Python,
not yaml, so the driver constructs Config directly from module
references without parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Config:
    # Three names — for slug + leaderboard
    dataset:  str
    model:    str
    config:   str

    seed:     int

    # Loaded from the config module's CONFIG dict
    train_cfg:  dict[str, Any]

    # Phases the driver will run, in order
    phases:   list[str] = field(default_factory=list)

    def slug(self) -> str:
        """Filesystem-safe identifier for the (dataset × model × config)
        triple. The timestamp suffix is added by Context.create."""
        return f"{self.dataset}__{self.model}__{self.config}"

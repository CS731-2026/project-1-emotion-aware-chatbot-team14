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
    # Three names — for slug + leaderboard. Renamed from `config` to
    # `config_name` so ctx.config.config_name reads cleanly instead of
    # the recursive ctx.config.config that previously looked like a typo.
    dataset_name:  str
    model_name:    str
    config_name:   str

    seed:     int

    # Loaded from the config module's CONFIG dict
    train_cfg:  dict[str, Any]

    # Phases the driver will run, in order
    phases:   list[str] = field(default_factory=list)

    def slug(self) -> str:
        """Filesystem-safe identifier for the (dataset × model × config)
        triple. The timestamp suffix is added by Context.create."""
        return f"{self.dataset_name}__{self.model_name}__{self.config_name}"

    # Back-compat shims — older call sites use ctx.config.dataset /
    # .model / .config. Keep these reading through so we don't have
    # to touch every train_loop in this same commit.
    @property
    def dataset(self) -> str:
        return self.dataset_name

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def config(self) -> str:
        return self.config_name

"""Resolved experiment plan (frozen). Built by the driver from the three modules in runs.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Config:
    dataset_name:  str         # e.g. "fer2013"
    model_name:    str         # e.g. "empathbot_final"
    config_name:   str         # e.g. "thorough"
    seed:          int
    train_cfg:     dict[str, Any]                     # from configs.<name>.CONFIG (+ per-run overrides)
    phases:        list[str] = field(default_factory=list)

    def slug(self) -> str:
        # <dataset>__<model>__<config>; Context.create appends a timestamp.
        return f"{self.dataset_name}__{self.model_name}__{self.config_name}"

    # Back-compat shims — older call sites use ctx.config.dataset / .model / .config.
    # Drop these once all model train_loops migrate to the *_name fields.
    @property
    def dataset(self) -> str:    return self.dataset_name
    @property
    def model(self) -> str:      return self.model_name
    @property
    def config(self) -> str:     return self.config_name

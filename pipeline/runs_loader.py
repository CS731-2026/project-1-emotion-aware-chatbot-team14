"""Load runs.yaml into the (dataset_module, model_module, config_module)
tuples the driver expects.

YAML schema:

    runs:
      - dataset: <name>            # → pipeline.datasets.<name>
        model:   <name>            # → pipeline.models.<name>
        config:  <name>            # → configs.<name>          (CONFIG dict)
        enabled: true              # optional, defaults to true
        train_cfg:                 # optional per-run override layer
          epochs: 5
          backbone_freeze_epochs: 3

The optional `train_cfg` is shallow-merged over the named config's
CONFIG dict so a single run can tweak hyperparameters without forking
a whole config file.

Errors are explicit — an unknown name or a missing required field
raises with a clear message naming the offending run index.
"""

from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


@dataclass
class ResolvedRun:
    """One run after name → module resolution. The driver's `sweep()`
    accepts these as 3-tuples (dataset_module, model_module, config_like).
    `config_like` is a SimpleNamespace exposing the same `NAME` + `CONFIG`
    that real config modules do, so per-run train_cfg overrides slot in
    without changing the driver."""
    dataset: ModuleType
    model:   ModuleType
    config:  Any   # module-like — has .NAME and .CONFIG attributes

    def as_tuple(self) -> tuple[ModuleType, ModuleType, Any]:
        return (self.dataset, self.model, self.config)


def _import(parent: str, name: str, run_index: int) -> ModuleType:
    """Import `parent.name`, with a friendly error if it doesn't exist."""
    try:
        return importlib.import_module(f"{parent}.{name}")
    except ImportError as e:
        raise ValueError(
            f"runs.yaml entry #{run_index}: can't import {parent}.{name} "
            f"({e}). Check the spelling against the directories under "
            f"{parent.replace('.', '/')}/."
        ) from e


def _make_overridden_config(base_module: ModuleType, overrides: dict[str, Any],
                              run_index: int) -> Any:
    """Return a config object that exposes NAME + a merged CONFIG dict.

    When `overrides` is empty, returns the base module unchanged so the
    sweep slug stays the named config. When non-empty, wraps it in a
    SimpleNamespace with a `+overrides` suffix on the name so two runs
    sharing the same named config but different overrides land in
    distinct run dirs.
    """
    if not overrides:
        return base_module

    if not hasattr(base_module, "CONFIG"):
        raise ValueError(
            f"runs.yaml entry #{run_index}: config '{base_module.__name__}' "
            "has no CONFIG dict — can't apply train_cfg overrides."
        )

    from types import SimpleNamespace
    merged = deepcopy(getattr(base_module, "CONFIG", {}))
    merged.update(overrides)
    base_name = getattr(base_module, "NAME", base_module.__name__.split(".")[-1])
    suffix = "+" + "-".join(sorted(overrides.keys()))[:40]
    return SimpleNamespace(NAME=f"{base_name}{suffix}", CONFIG=merged)


def load_runs(runs_yaml: Path | str = "runs.yaml") -> list[ResolvedRun]:
    """Parse `runs_yaml` and return one ResolvedRun per enabled entry."""
    path = Path(runs_yaml)
    if not path.exists():
        raise FileNotFoundError(
            f"runs file not found at {path.resolve()}. "
            f"Create it (see pipeline/MIGRATING_NOTEBOOKS.md) or pass --runs."
        )

    with path.open() as f:
        data = yaml.safe_load(f) or {}
    raw_runs = data.get("runs") or []
    if not isinstance(raw_runs, list):
        raise ValueError(f"runs.yaml: top-level `runs:` must be a list, got {type(raw_runs).__name__}")

    resolved: list[ResolvedRun] = []
    for idx, entry in enumerate(raw_runs):
        if not isinstance(entry, dict):
            raise ValueError(f"runs.yaml entry #{idx}: each run must be a mapping, got {type(entry).__name__}")
        if entry.get("enabled", True) is False:
            continue

        for required in ("dataset", "model", "config"):
            if required not in entry:
                raise ValueError(f"runs.yaml entry #{idx}: missing required field '{required}'")

        dataset_mod = _import("pipeline.datasets", entry["dataset"], idx)
        model_mod   = _import("pipeline.models",   entry["model"],   idx)
        config_mod  = _import("configs",           entry["config"],  idx)

        overrides = entry.get("train_cfg") or {}
        if overrides and not isinstance(overrides, dict):
            raise ValueError(f"runs.yaml entry #{idx}: train_cfg must be a mapping")

        cfg_obj = _make_overridden_config(config_mod, overrides, idx)
        resolved.append(ResolvedRun(dataset=dataset_mod, model=model_mod, config=cfg_obj))

    return resolved

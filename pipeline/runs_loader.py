"""Load runs.yaml → ResolvedRun list the driver expects.

YAML schema:

    runs:
      - dataset: <name>            # importlib → pipeline.datasets.<name>
        model:   <name>            # importlib → pipeline.models.<name>
        config:  <name>            # importlib → configs.<name>
        enabled: true              # optional, defaults to true
        train_cfg:                 # optional per-run override layer
          epochs: 5
          backbone_freeze_epochs: 3

What each field's module must expose (also in framework.protocols):

    dataset    NAME, CLASS_NAMES, prepare(ctx) -> DatasetSpec
    model      train(ctx, dataset) -> TrainedModel
    config     NAME, CONFIG (dict)

train_cfg is shallow-merged over the named config's CONFIG. When
overrides are present the run dir slug gets a `+key1-key2` suffix so
distinct variants don't collide. Errors name the offending run index.
"""

from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import yaml


@dataclass
class ResolvedRun:
    """One run after name → module resolution. .as_tuple() = what sweep() takes."""
    dataset: ModuleType
    model:   ModuleType
    config:  Any   # module or SimpleNamespace — has .NAME and .CONFIG

    def as_tuple(self) -> tuple[ModuleType, ModuleType, Any]:
        return (self.dataset, self.model, self.config)


def _import(parent: str, name: str, run_index: int) -> ModuleType:
    try:
        return importlib.import_module(f"{parent}.{name}")
    except ImportError as e:
        raise ValueError(
            f"runs.yaml entry #{run_index}: can't import {parent}.{name} "
            f"({e}). Check spelling vs the directories under "
            f"{parent.replace('.', '/')}/."
        ) from e


def _make_overridden_config(base_module: ModuleType, overrides: dict[str, Any],
                             run_index: int) -> Any:
    """No overrides → return module unchanged (keeps slug clean). Otherwise
    wrap in a SimpleNamespace with a `+keys` suffix so two runs with the
    same base config but different overrides land in distinct run dirs."""
    if not overrides:
        return base_module
    if not hasattr(base_module, "CONFIG"):
        raise ValueError(
            f"runs.yaml entry #{run_index}: config '{base_module.__name__}' "
            "has no CONFIG dict — can't apply train_cfg overrides."
        )
    merged = deepcopy(base_module.CONFIG)
    merged.update(overrides)
    base_name = getattr(base_module, "NAME", base_module.__name__.split(".")[-1])
    suffix = "+" + "-".join(sorted(overrides))[:40]
    return SimpleNamespace(NAME=f"{base_name}{suffix}", CONFIG=merged)


def load_runs(runs_yaml: Path | str = "runs.yaml") -> list[ResolvedRun]:
    """Parse runs_yaml, return one ResolvedRun per enabled entry."""
    path = Path(runs_yaml)
    if not path.exists():
        raise FileNotFoundError(
            f"runs file not found at {path.resolve()}. "
            f"Create it (see pipeline/MIGRATING_NOTEBOOKS.md) or pass --runs."
        )

    data = yaml.safe_load(path.read_text()) or {}
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

        overrides = entry.get("train_cfg") or {}
        if overrides and not isinstance(overrides, dict):
            raise ValueError(f"runs.yaml entry #{idx}: train_cfg must be a mapping")

        resolved.append(ResolvedRun(
            dataset=_import("pipeline.datasets", entry["dataset"], idx),
            model  =_import("pipeline.models",   entry["model"],   idx),
            config =_make_overridden_config(
                _import("configs", entry["config"], idx), overrides, idx),
        ))
    return resolved

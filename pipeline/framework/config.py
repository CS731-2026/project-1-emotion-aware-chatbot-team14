"""Experiment config — the immutable input plan for one run.

An experiment yaml is intentionally tiny: it names three other yamls
(a dataset, a model, a train config) plus a seed and the phases to run.
The driver looks each name up and produces the final resolved Config
that phases receive via Context.

  # experiments/<name>.yaml
  dataset:  fer2013                    # → datasets/fer2013.yaml
  model:    tiny_cnn                   # → models/tiny_cnn (importable module)
  config:   baseline                   # → configs/baseline.yaml
  seed:     42
  phases:   [setup, prepare_dataset, train]

Each name resolves to a sub-config dict (yaml-loaded) that the phases
read by section. Sub-configs aren't typed here — they're free-form
dicts that the phase that owns them validates. That keeps this module
boring and means a new training knob doesn't need a Config edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project-relative roots. Phases find their config yamls under here.
_REPO_ROOT       = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR  = _REPO_ROOT / "experiments"
DATASETS_DIR     = _REPO_ROOT / "datasets"
CONFIGS_DIR      = _REPO_ROOT / "configs"


@dataclass(frozen=True)
class Config:
    """The resolved experiment plan. All four name-references have been
    looked up and their yaml contents loaded into the dict fields below."""

    # The three names — for slug + leaderboard
    dataset:  str
    model:    str
    config:   str

    seed:     int

    # Loaded sub-config dicts
    dataset_cfg:  dict[str, Any]
    train_cfg:    dict[str, Any]

    # Phases the driver will run, in order
    phases:   list[str] = field(default_factory=list)

    def slug(self) -> str:
        """Filesystem-safe identifier for the (dataset × model × config)
        triple. The timestamp suffix is added by Context.create."""
        return f"{self.dataset}__{self.model}__{self.config}"


def load_experiment(path: str | Path) -> Config:
    """Load `experiments/<name>.yaml` and resolve all its name-refs.

    Raises FileNotFoundError with a clear path if any referenced yaml
    is missing — better to fail at load time than mid-training.
    """
    exp_path = Path(path)
    exp = _yaml(exp_path)
    return Config(
        dataset    = exp["dataset"],
        model      = exp["model"],
        config     = exp["config"],
        seed       = int(exp.get("seed", 42)),
        phases     = list(exp.get("phases", ["setup", "prepare_dataset", "train"])),
        dataset_cfg = _yaml(DATASETS_DIR / f"{exp['dataset']}.yaml"),
        train_cfg   = _yaml(CONFIGS_DIR  / f"{exp['config']}.yaml"),
    )


def _yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config yaml not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}

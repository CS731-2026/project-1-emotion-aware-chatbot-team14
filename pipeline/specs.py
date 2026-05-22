"""Typed handoff objects passed phase-to-phase via Context.store.

Each spec is a frozen dataclass — pure data, no behaviour beyond
serialization. They're small enough to live together; splitting per
type would just spread three imports across three files.

  - DatasetSpec   produced by `prepare_dataset` from a dataset yaml
  - TrainedModel  produced by `train` from a DatasetSpec + model module
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Points at split CSVs on disk + carries the metadata the train loop
    needs (class names + weights). Cache lives under output/data/<name>/
    so the same .gitignore rule covers it as the run dirs."""

    name:           str
    cache_dir:      Path
    splits:         dict[str, Path]     # split name → CSV path ("train", "val", "test")
    num_classes:    int
    class_names:    list[str]
    class_weights:  list[float] | None  # None = uniform; floats are inverse-frequency
    source_md5:     str                  # of the source dir; detects re-prep need

    def to_manifest(self) -> dict:
        """JSON-safe dict — Paths become strings."""
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        d["splits"] = {k: str(v) for k, v in self.splits.items()}
        return d

    @classmethod
    def from_manifest(cls, path: Path) -> "DatasetSpec":
        raw = json.loads(path.read_text())
        return cls(
            name=raw["name"],
            cache_dir=Path(raw["cache_dir"]),
            splits={k: Path(v) for k, v in raw["splits"].items()},
            num_classes=raw["num_classes"],
            class_names=raw["class_names"],
            class_weights=raw["class_weights"],
            source_md5=raw["source_md5"],
        )


@dataclass(frozen=True)
class TrainedModel:
    """Bookkeeping for a finished training run — enough for a future
    dedicated evaluate phase to reload the weights and re-run rich
    analysis without retraining."""

    model_name:       str                                          # for importlib.import_module
    num_classes:      int
    checkpoint_path:  Path
    history:          list[dict] = field(default_factory=list)     # per-epoch metrics
    final_val:        dict = field(default_factory=dict)           # {"loss":..., "acc":...}
    final_test:       dict = field(default_factory=dict)           # filled when test ran

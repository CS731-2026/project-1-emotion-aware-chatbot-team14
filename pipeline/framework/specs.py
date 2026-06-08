"""Typed handoff objects (frozen dataclasses) passed phase → phase via Store."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Produced by prepare_dataset. Points at CSVs + carries class metadata."""

    name:           str
    cache_dir:      Path
    splits:         dict[str, Path]     # "train" / "val" / "test" → CSV path
    num_classes:    int
    class_names:    list[str]
    class_weights:  list[float] | None  # inverse-frequency from train split; None = uniform
    source_md5:     str                  # hash of source dir; detects re-prep need

    def to_manifest(self) -> dict:
        # JSON-safe, Paths → strings.
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
    """Produced by train. Enough for a later evaluate phase to reload without retraining."""

    model_name:       str
    num_classes:      int
    checkpoint_path:  Path
    history:          list[dict] = field(default_factory=list)   # per-epoch metrics
    final_val:        dict = field(default_factory=dict)         # {"loss":..., "acc":...}
    final_test:       dict = field(default_factory=dict)         # filled when test ran

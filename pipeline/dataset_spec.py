"""DatasetSpec — the typed handoff produced by prepare_dataset.

A DatasetSpec is what later phases (train, evaluate) read from the
store. It points at split CSVs on disk (columns: path, label) plus a
small bag of metadata the training loop needs (class names, weights).

The CSVs and manifest live under output/data/<name>/ — the same
output/ tree everything else writes to, so a single .gitignore rule
covers it and clearing the disk is `rm -rf output/`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name:           str
    cache_dir:      Path
    splits:         dict[str, Path]    # split name → CSV path (e.g. "train", "val", "test")
    num_classes:    int
    class_names:    list[str]
    class_weights:  list[float] | None  # None = uniform; floats are inverse-frequency
    source_md5:     str                 # of the source archive; used to detect re-download need

    def to_manifest(self) -> dict:
        """JSON-safe dict for writing manifest.json — Paths become strings."""
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        d["splits"] = {k: str(v) for k, v in self.splits.items()}
        return d

    @classmethod
    def from_manifest(cls, path: Path) -> "DatasetSpec":
        """Inverse of to_manifest. Used by prepare_dataset to skip work
        when a previous run already wrote a usable manifest."""
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

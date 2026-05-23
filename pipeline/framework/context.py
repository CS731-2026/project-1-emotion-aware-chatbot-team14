"""Context — the single arg every phase receives.

Bundles read-only Config, typed Store, the imported dataset+model
modules, and artifact save_* methods that write into the run dir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import IO, Any

import yaml

from .config import Config
from .store import Store


# Hard-coded so a stray config can't redirect run output elsewhere.
_OUTPUT_RUN_ROOT = Path("output/run")


@dataclass
class Context:
    config:           Config
    store:            Store
    run_dir:          Path
    dataset_module:   ModuleType    # satisfies framework.protocols.DatasetModule
    model_module:     ModuleType    # satisfies framework.protocols.ModelModule
    _metrics_fh:      IO[str] = field(repr=False)

    @classmethod
    def create(
        cls,
        config: Config,
        *,
        dataset_module: ModuleType,
        model_module:   ModuleType,
    ) -> "Context":
        # Build output/run/<slug>__<ts>/ — append _v2/v3/... on collision so
        # two runs in the same second don't clobber each other.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = _OUTPUT_RUN_ROOT / f"{config.slug()}__{ts}"
        run_dir = base
        suffix = 2
        while run_dir.exists():
            run_dir = base.with_name(f"{base.name}_v{suffix}")
            suffix += 1
        (run_dir / "artifacts").mkdir(parents=True)
        (run_dir / "checkpoints").mkdir()

        # Snapshot the resolved config first — first thing in the dir.
        (run_dir / "config.yaml").write_text(yaml.safe_dump({
            "dataset":    config.dataset_name,
            "model":      config.model_name,
            "config":     config.config_name,
            "seed":       config.seed,
            "phases":     config.phases,
            "train_cfg":  config.train_cfg,
        }, sort_keys=False))

        return cls(
            config=config,
            store=Store(),
            run_dir=run_dir,
            dataset_module=dataset_module,
            model_module=model_module,
            _metrics_fh=(run_dir / "metrics.jsonl").open("a"),
        )

    def close(self) -> None:
        self._metrics_fh.close()

    # ---- artifact API (all return the destination Path) -------------------

    def save_image(self, name: str, fig: Any) -> Path:
        # matplotlib Figure → artifacts/<name>.png. Subdirs in name are auto-created.
        dest = self._artifact_path(name, ".png")
        fig.savefig(dest, bbox_inches="tight", dpi=120)
        return dest

    def save_json(self, name: str, obj: Any) -> Path:
        dest = self._artifact_path(name, ".json")
        dest.write_text(json.dumps(obj, indent=2, default=str))
        return dest

    def save_text(self, name: str, text: str) -> Path:
        dest = self._artifact_path(name, ".txt")
        dest.write_text(text)
        return dest

    def save_scalar(self, name: str, value: float, step: int | None = None) -> None:
        # One JSONL line → metrics.jsonl. Cheap; safe per-batch.
        row: dict[str, Any] = {"name": name, "value": float(value)}
        if step is not None:
            row["step"] = step
        self._metrics_fh.write(json.dumps(row) + "\n")
        self._metrics_fh.flush()

    def save_checkpoint(self, name: str, state_dict: dict) -> Path:
        # → checkpoints/<name>.pth. Torch imported lazily to keep import-time cheap.
        import torch
        dest = self.run_dir / "checkpoints" / f"{name}.pth"
        torch.save(state_dict, dest)
        return dest

    def _artifact_path(self, name: str, default_ext: str) -> Path:
        # Resolves name under artifacts/, auto-creates parent dirs, adds
        # default_ext when name has no extension.
        rel = Path(name)
        if rel.suffix == "":
            rel = rel.with_suffix(default_ext)
        dest = self.run_dir / "artifacts" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

"""The single arg every phase receives.

Context bundles three things every phase needs:
  - the read-only Config
  - the typed Store for phase-to-phase handoff
  - artifact save methods (image / json / text / scalar / checkpoint)
    that write into the run dir

The run dir path is hard-coded to `output/run/<slug>__<timestamp>/` —
phases can't override it, so every run lands in the same predictable
tree. `output/` is gitignored, so nothing the pipeline writes leaks
into version control.

Context.create() is the only public constructor. It builds the run
dir, writes a snapshot of config.yaml for reproducibility, opens the
metrics jsonl handle, and returns a ready-to-use Context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import yaml

from .config import Config
from .store import Store


# Hard-coded so a stray config can't redirect run output elsewhere.
_OUTPUT_RUN_ROOT = Path("output/run")


@dataclass
class Context:
    config:  Config
    store:   Store
    run_dir: Path
    _metrics_fh: IO[str] = field(repr=False)

    # ---- construction ---------------------------------------------------

    @classmethod
    def create(cls, config: Config) -> "Context":
        """Build the run dir under output/run/<slug>__<ts>/ and open the
        metrics jsonl handle. If the slug+timestamp dir already exists
        (unlikely but possible — two runs in the same second), append
        _v2 / _v3 / … so the prior run dir isn't clobbered."""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = _OUTPUT_RUN_ROOT / f"{config.slug()}__{ts}"
        run_dir = base
        suffix = 2
        while run_dir.exists():
            run_dir = base.with_name(f"{base.name}_v{suffix}")
            suffix += 1
        (run_dir / "artifacts").mkdir(parents=True)
        (run_dir / "checkpoints").mkdir()

        # Snapshot the resolved config — first thing in the dir.
        (run_dir / "config.yaml").write_text(yaml.safe_dump({
            "dataset":     config.dataset,
            "model":       config.model,
            "config":      config.config,
            "seed":        config.seed,
            "phases":      config.phases,
            "dataset_cfg": config.dataset_cfg,
            "train_cfg":   config.train_cfg,
        }, sort_keys=False))

        metrics_fh = (run_dir / "metrics.jsonl").open("a")
        return cls(config=config, store=Store(), run_dir=run_dir, _metrics_fh=metrics_fh)

    def close(self) -> None:
        """Flush + close the metrics handle. The driver calls this."""
        self._metrics_fh.close()

    # ---- artifact API ---------------------------------------------------
    # All save_* methods return the destination Path so the caller can
    # log it / hand it back in a result struct if it wants.

    def save_image(self, name: str, fig: Any) -> Path:
        """Save a matplotlib Figure as PNG under artifacts/<name>.png.
        Subdirectories in `name` are created automatically — e.g.
        'epoch_3/predictions' → artifacts/epoch_3/predictions.png."""
        dest = self._artifact_path(name, ".png")
        fig.savefig(dest, bbox_inches="tight", dpi=120)
        return dest

    def save_json(self, name: str, obj: Any) -> Path:
        """Pretty-printed JSON under artifacts/<name>.json."""
        dest = self._artifact_path(name, ".json")
        dest.write_text(json.dumps(obj, indent=2, default=str))
        return dest

    def save_text(self, name: str, text: str) -> Path:
        """Plain text under artifacts/<name>.txt (or whatever ext is in name)."""
        dest = self._artifact_path(name, ".txt")
        dest.write_text(text)
        return dest

    def save_scalar(self, name: str, value: float, step: int | None = None) -> None:
        """Append one JSONL line to metrics.jsonl. Cheap; safe to call
        every batch — the file grows linearly but is plain text and
        easy to grep / pandas.read_json(lines=True)."""
        row = {"name": name, "value": float(value)}
        if step is not None:
            row["step"] = step
        self._metrics_fh.write(json.dumps(row) + "\n")
        self._metrics_fh.flush()

    def save_checkpoint(self, name: str, state_dict: dict) -> Path:
        """Torch state dict under checkpoints/<name>.pth. Torch is imported
        lazily so the framework module itself stays import-time cheap."""
        import torch
        dest = self.run_dir / "checkpoints" / f"{name}.pth"
        torch.save(state_dict, dest)
        return dest

    # ---- internals ------------------------------------------------------

    def _artifact_path(self, name: str, default_ext: str) -> Path:
        """Resolve `name` against artifacts/, creating parent dirs and
        adding `default_ext` if `name` has no extension."""
        rel = Path(name)
        if rel.suffix == "":
            rel = rel.with_suffix(default_ext)
        dest = self.run_dir / "artifacts" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

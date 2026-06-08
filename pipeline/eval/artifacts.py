"""Write the standard eval artifact bundle to disk.

Four files per (model, dataset) pair:
  summary.json      aggregate acc, macro_f1, weighted_f1, n_samples (+ meta if passed)
  per_class.json    per-class precision / recall / F1 / support
  confusion.json    raw NxN matrix + class_names (for downstream tooling)
  confusion.png     row-normalised heatmap (for the report)

The PNG is best-effort, a missing matplotlib install logs a warning
and skips, but the JSON artifacts always land. summary.json is the file
`make compare` reads, so it's the most important one to guarantee.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_eval_artifacts(
    metrics: dict[str, Any],
    out_dir: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write the four-file bundle into `out_dir`.

    `meta` (optional) is merged into summary.json, used to record
    which checkpoint / dataset / timestamp this eval came from.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_samples":   metrics["n_samples"],
        "acc":         metrics["acc"],
        "macro_f1":    metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }
    if meta:
        summary["meta"] = meta
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    (out_dir / "per_class.json").write_text(json.dumps({
        "per_class": metrics["per_class"],
    }, indent=2))

    (out_dir / "confusion.json").write_text(json.dumps({
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names":      metrics["class_names"],
    }, indent=2))

    # PNG is best-effort, eval still counts as "done" if matplotlib
    # isn't installed or the headless backend fails.
    try:
        _save_confusion_png(metrics, out_dir / "confusion.png")
    except Exception as e:  # noqa: BLE001, best-effort artifact
        logger.warning("eval: confusion.png skipped (%s)", e)


def _save_confusion_png(metrics: dict[str, Any], dest: Path) -> None:
    """Single-panel row-normalised heatmap. Matches the visual language
    of pipeline.training.reporting._save_confusion_matrix but trimmed
    to one panel for the eval report (the training reporter already
    produces the two-panel version)."""
    if not metrics["confusion_matrix"]:
        return

    import numpy as np
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    names = metrics["class_names"]
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm, dtype=float)
    nz = row_sum.flatten() != 0
    cm_norm[nz] = cm[nz].astype(float) / row_sum[nz]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    try:
        import seaborn as sns
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=names, yticklabels=names, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
    except ImportError:
        im = ax.imshow(cm_norm, cmap="Blues")
        n = len(names)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(names, rotation=0, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
        for i in range(n):
            for j in range(n):
                v = cm_norm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(dest, dpi=120, bbox_inches="tight")
    plt.close(fig)

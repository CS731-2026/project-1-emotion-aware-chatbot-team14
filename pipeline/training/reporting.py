"""Shared post-training artifact writer.

Reproduces the artifacts the source notebooks produced at the end of a
training run, so every model in the pipeline emits the same shape of
outputs into its run dir:

  artifacts/training_curves.png    train/val loss + acc per epoch (+ lr when present)
  artifacts/confusion_matrix.png   raw counts + row-normalised, side-by-side
  artifacts/classification_report.txt
                                   sklearn classification_report (per-class P/R/F1)
  artifacts/per_class_metrics.json structured per-class metrics + confusion matrix
  artifacts/final.json             top-level summary (best_epoch, best_val, test_acc, …)

The PNG generation is split out so it can be skipped headlessly
(matplotlib backend = "Agg" is set automatically so this works on
servers without a display).

Usage from a model's train_loop, right before the `return TrainedModel(...)`:

    from pipeline.training.reporting import write_standard_artifacts

    write_standard_artifacts(
        ctx,
        history=history,
        test_preds=all_preds,
        test_labels=all_labels,
        num_classes=num_classes,
        class_names=dataset.class_names,
        final_summary={"best_epoch": best_epoch, ...},
    )
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_agg() -> None:
    """Force a non-interactive matplotlib backend before pyplot import
    so this works on headless boxes (no $DISPLAY)."""
    import matplotlib
    matplotlib.use("Agg", force=True)


def _coerce_history(history: list[dict] | dict) -> dict[str, list]:
    """Accept either a list-of-dicts (one row per epoch) or a dict-of-lists.
    Returns a dict-of-lists keyed by metric name."""
    if isinstance(history, dict):
        return {k: list(v) for k, v in history.items()}
    if not history:
        return {}
    out: dict[str, list] = {}
    for row in history:
        for k, v in row.items():
            out.setdefault(k, []).append(v)
    return out


def _save_training_curves(ctx, history: list[dict] | dict) -> None:
    """Three-panel matplotlib figure: accuracy, loss, learning rate.

    Matches the layout in notebook 2's cell 22 (and the similar variants
    in notebooks 4, 5, 6, 6b). Skips silently if a metric isn't present
    (e.g. lr only appears in some notebooks' histories).
    """
    h = _coerce_history(history)
    if not h or "epoch" not in h:
        logger.info("reporting: no epoch history, skipping training_curves.png")
        return

    _ensure_agg()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    epochs = h["epoch"]
    has_lr = "lr" in h and any(x is not None for x in h["lr"])
    n_panels = 3 if has_lr else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # ── accuracy panel
    ax = axes[0]
    if "train_acc" in h:
        ax.plot(epochs, [a * 100 for a in h["train_acc"]], label="Train", color="steelblue")
    if "val_acc" in h:
        ax.plot(epochs, [a * 100 for a in h["val_acc"]], label="Val",
                color="tomato", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.grid(True, alpha=0.3); ax.legend()

    # ── loss panel
    ax = axes[1]
    if "train_loss" in h:
        ax.plot(epochs, h["train_loss"], label="Train", color="steelblue")
    if "val_loss" in h:
        ax.plot(epochs, h["val_loss"], label="Val", color="tomato", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.grid(True, alpha=0.3); ax.legend()

    # ── lr panel (optional)
    if has_lr:
        ax = axes[2]
        ax.plot(epochs, h["lr"], color="purple")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); ax.set_title("LR schedule")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"{ctx.config.model}, Training Curves", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ctx.save_image("training_curves", fig)
    plt.close(fig)


def _save_confusion_matrix(ctx, preds: np.ndarray, labels: np.ndarray,
                            num_classes: int, class_names: list[str]) -> np.ndarray:
    """Two-panel heatmap (raw counts + row-normalised), notebook 2 cell 26
    layout. Returns the raw confusion matrix so the caller can include
    it in the JSON summary."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm, dtype=float)
    nz = row_sum.flatten() != 0
    cm_norm[nz] = cm[nz].astype(float) / row_sum[nz]

    _ensure_agg()
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        have_sns = True
    except ImportError:
        have_sns = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"{ctx.config.model}, Confusion matrix",
                 fontsize=12, fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw counts",  "d"),
        (axes[1], cm_norm, "Row-normalised", ".2f"),
    ]:
        if have_sns:
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={"shrink": 0.8},
            )
        else:
            im = ax.imshow(data, cmap="Blues")
            ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
            ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
            ax.set_yticklabels(class_names, rotation=0, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
            for i in range(num_classes):
                for j in range(num_classes):
                    val = data[i, j]
                    txt = f"{int(val)}" if fmt == "d" else f"{val:.2f}"
                    ax.text(j, i, txt, ha="center", va="center",
                            color="white" if (val > data.max() / 2) else "black",
                            fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)

    fig.tight_layout()
    ctx.save_image("confusion_matrix", fig)
    plt.close(fig)
    return cm


def _save_classification_report(ctx, preds: np.ndarray, labels: np.ndarray,
                                 num_classes: int, class_names: list[str]) -> dict:
    """sklearn classification_report saved as plain text + a structured
    per-class dict ready for the JSON summary."""
    from sklearn.metrics import (
        classification_report, f1_score, precision_score, recall_score,
    )

    target_names = [class_names[i] if i < len(class_names) else str(i)
                    for i in range(num_classes)]
    report_txt = classification_report(
        labels, preds, target_names=target_names,
        labels=list(range(num_classes)), digits=3, zero_division=0,
    )
    ctx.save_text("classification_report", report_txt)

    per_prec = precision_score(labels, preds, average=None, zero_division=0,
                                labels=list(range(num_classes)))
    per_rec = recall_score(labels, preds, average=None, zero_division=0,
                            labels=list(range(num_classes)))
    per_f1 = f1_score(labels, preds, average=None, zero_division=0,
                       labels=list(range(num_classes)))

    per_class: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(target_names):
        per_class[name] = {
            "precision": round(float(per_prec[i]), 4),
            "recall":    round(float(per_rec[i]),  4),
            "f1":        round(float(per_f1[i]),   4),
            "support":   int((labels == i).sum()),
        }
    return per_class


def write_standard_artifacts(
    ctx,
    *,
    history: list[dict] | dict,
    test_preds: Iterable[int] | np.ndarray,
    test_labels: Iterable[int] | np.ndarray,
    num_classes: int,
    class_names: list[str],
    final_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit the standard post-training artifacts into ctx.run_dir.

    Returns the per-class metrics dict so the caller can include them
    in its own logging / TrainedModel result if it wants.

    Failures in any sub-step are logged but don't propagate, a missing
    matplotlib install shouldn't kill an otherwise-successful run.
    """
    preds_arr = np.asarray(list(test_preds), dtype=np.int64)
    labels_arr = np.asarray(list(test_labels), dtype=np.int64)

    try:
        _save_training_curves(ctx, history)
    except Exception as e:
        logger.warning("reporting: training_curves failed (%s)", e)

    cm = None
    try:
        cm = _save_confusion_matrix(ctx, preds_arr, labels_arr,
                                      num_classes, class_names)
    except Exception as e:
        logger.warning("reporting: confusion_matrix failed (%s)", e)

    per_class: dict = {}
    try:
        per_class = _save_classification_report(ctx, preds_arr, labels_arr,
                                                  num_classes, class_names)
    except Exception as e:
        logger.warning("reporting: classification_report failed (%s)", e)

    summary = dict(final_summary or {})
    summary["per_class"] = per_class
    if cm is not None:
        summary["confusion_matrix"] = cm.tolist()
    ctx.save_json("per_class_metrics", {"per_class": per_class,
                                         "confusion_matrix": cm.tolist() if cm is not None else None})
    if final_summary is not None:
        ctx.save_json("final", summary)

    return per_class

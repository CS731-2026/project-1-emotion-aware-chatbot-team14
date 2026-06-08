"""Eval-phase metrics, single function: model + loader → metrics dict.

The dict is the contract every eval artifact downstream consumes:
  summary.json gets (acc, macro_f1, weighted_f1, n_samples)
  per_class.json gets per_class
  confusion.json + confusion.png both get confusion_matrix
  make compare reads the summary

Returning one dict from one function (rather than scattering side-effects)
keeps the eval phase tiny and lets baselines.py call this directly
without going through the phase machinery.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.training.loop import collect_predictions


def compute_eval(
    model: nn.Module,
    loader: DataLoader,
    device: Any,                       # torch.device, annotated Any to avoid stub friction
    *,
    num_classes: int,
    class_names: list[str],
) -> dict[str, Any]:
    """Run inference over `loader` once and produce the full metrics dict.

    Imported lazily: sklearn (precision_recall_fscore_support /
    confusion_matrix). Heavy imports stay out of import-time cost.

    Reuses pipeline.training.loop.collect_predictions so the eval and
    training paths walk the test loader the same way, no surprise
    drift between "what the train phase reported as test_acc" and what
    a follow-up eval phase reports.
    """
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
    )

    preds_list, labels_list = collect_predictions(model, loader, device)
    preds = np.asarray(preds_list, dtype=np.int64)
    labels = np.asarray(labels_list, dtype=np.int64)
    n = int(labels.size)

    if n == 0:
        # Empty test split, make the failure obvious in the artifact
        # without crashing the whole eval pass.
        return {
            "n_samples": 0,
            "acc": None,
            "macro_f1": None,
            "weighted_f1": None,
            "per_class": [],
            "confusion_matrix": [],
            "class_names": class_names,
        }

    acc = float((preds == labels).mean())

    label_idx = list(range(num_classes))
    # average=None returns four per-class arrays. Cast explicitly so
    # downstream indexing / arithmetic doesn't trip Pyright's union view.
    prfs = precision_recall_fscore_support(
        labels, preds, average=None, labels=label_idx, zero_division=0,  # type: ignore[arg-type]
    )
    p_arr      = np.asarray(prfs[0], dtype=np.float64)
    r_arr      = np.asarray(prfs[1], dtype=np.float64)
    f1_arr     = np.asarray(prfs[2], dtype=np.float64)
    support_arr = np.asarray(prfs[3], dtype=np.int64)

    # macro_f1 = unweighted mean across classes; weighted_f1 weights by support.
    # Macro is the right metric when minority classes matter (FER has rare
    # fear_anxiety); weighted is closer to overall accuracy.
    macro_f1 = float(f1_arr.mean()) if f1_arr.size else 0.0
    total_support = int(support_arr.sum()) or 1
    weighted_f1 = float((f1_arr * support_arr).sum() / total_support)

    per_class = []
    for i, name in enumerate(class_names[:num_classes]):
        per_class.append({
            "class": name,
            "index": i,
            "precision": round(float(p_arr[i]),  4),
            "recall":    round(float(r_arr[i]),  4),
            "f1":        round(float(f1_arr[i]), 4),
            "support":   int(support_arr[i]),
        })

    cm = confusion_matrix(labels, preds, labels=label_idx).tolist()

    return {
        "n_samples": n,
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": class_names[:num_classes],
    }

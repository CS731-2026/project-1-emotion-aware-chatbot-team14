"""Faithful inference benchmark — ported from
Notebooks/3_benchmark_posterplus.ipynb (cells 11, 13, 15, 21).

The notebook is inference-only: it loads POSTER++'s published RAF-DB
checkpoint and reports accuracy + per-class metrics against the test
split. This module mirrors that loop.

Resolution order for the published checkpoint:
  1. ctx.config.train_cfg["checkpoint_path"]
  2. $POSTER_CHECKPOINT_PATH
  3. output/models/posterv2_rafdb.pth (matches notebook cell 4)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.loop import auto_device

logger = logging.getLogger(__name__)


PUBLISHED_RAFDB_ACC = 92.21  # Mao et al., TPAMI 2023


class _CsvDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])


def _resolve_checkpoint(cfg: dict[str, Any]) -> Path | None:
    for src in (cfg.get("checkpoint_path"),
                os.environ.get("POSTER_CHECKPOINT_PATH"),
                "output/models/posterv2_rafdb.pth"):
        if not src:
            continue
        p = Path(src)
        if p.exists():
            return p
    return None


def _load_checkpoint(model: nn.Module, ckpt_path: Path, device) -> None:
    """Verbatim from notebook cell 11 — POSTER_V2 ckpts use several keys."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict):
        found = next((k for k in ("model_state_dict", "state_dict", "model")
                      if k in ck), None)
        state = ck[found] if found else ck
        logger.info("posterplus checkpoint format: dict, key=%s", found or "flat")
    else:
        state = ck
        logger.info("posterplus checkpoint format: raw state dict")
    model.load_state_dict(state, strict=True)


def run(ctx: Context, dataset: DatasetSpec, model: nn.Module,
        preprocess) -> TrainedModel:
    cfg = dict(ctx.config.train_cfg)
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes

    ckpt_path = _resolve_checkpoint(cfg)
    if ckpt_path is None:
        raise FileNotFoundError(
            "posterplus inference requires the published RAF-DB checkpoint. "
            "Place it at output/models/posterv2_rafdb.pth, set "
            "POSTER_CHECKPOINT_PATH, or pass checkpoint_path in train_cfg. "
            "Download URL is in Notebooks/3_benchmark_posterplus.ipynb cell 9."
        )
    logger.info("posterplus: loading published checkpoint from %s", ckpt_path)
    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    test_ds = _CsvDataset(dataset.splits["test"], preprocess)
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            all_preds.extend(out.argmax(1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    test_acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    per_prec = precision_score(labels, preds, average=None, zero_division=0,
                                labels=list(range(num_classes)))
    per_rec = recall_score(labels, preds, average=None, zero_division=0,
                            labels=list(range(num_classes)))
    per_f1 = f1_score(labels, preds, average=None, zero_division=0,
                       labels=list(range(num_classes)))
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    per_class: dict[str, dict[str, float | int]] = {}
    for i in range(num_classes):
        name = dataset.class_names[i] if i < len(dataset.class_names) else str(i)
        per_class[name] = {
            "precision": round(float(per_prec[i]), 4),
            "recall":    round(float(per_rec[i]),  4),
            "f1":        round(float(per_f1[i]),   4),
            "support":   int((labels == i).sum()),
        }

    ctx.save_scalar("test/acc", test_acc)
    ctx.save_scalar("test/macro_f1", macro_f1)
    ctx.save_scalar("test/weighted_f1", weighted_f1)
    ctx.save_json("results", {
        "model": "POSTER++ (Mao et al., IEEE TPAMI 2023)",
        "checkpoint": str(ckpt_path),
        "test_size": int(len(labels)),
        "test_accuracy": round(test_acc * 100, 4),
        "published_rafdb_accuracy": PUBLISHED_RAFDB_ACC,
        "gap_from_published": round(test_acc * 100 - PUBLISHED_RAFDB_ACC, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    })

    target_names = [dataset.class_names[i] if i < len(dataset.class_names)
                    else str(i) for i in range(num_classes)]
    report = classification_report(labels, preds, target_names=target_names,
                                    labels=list(range(num_classes)),
                                    digits=3, zero_division=0)
    ctx.save_text("classification_report", report)

    logger.info("posterplus: test_acc=%.4f macro_f1=%.4f (published RAF-DB=%.2f%%)",
                test_acc, macro_f1, PUBLISHED_RAFDB_ACC)

    return TrainedModel(
        model_name=ctx.config.model, num_classes=num_classes,
        checkpoint_path=ckpt_path, history=[],
        final_val={"acc": float("nan")},
        final_test={"acc": test_acc, "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1},
    )

"""Training loop — ported from notebook 6b cells 13 + 15.

Reproduces the notebook's training discipline as closely as the
pipeline contract allows:

  - AdamW with split LR: backbone @ CFG['backbone_lr'], head @ CFG['head_lr']
  - LambdaLR: linear warmup over CFG['warmup_epochs'] then cosine decay
    to CFG['min_lr']
  - Per-epoch backbone freeze for the first CFG['freeze_epochs'] epochs
  - Gradient clipping at CFG['grad_clip']
  - WeightedRandomSampler with inverse-frequency weights, with extra
    1.3x bias toward HARD_LABEL_IDS (same as notebook cell 7)
  - FocalLoss(class_weights, gamma, label_smoothing)
  - Early stopping on val_acc with CFG['patience']
  - Best-checkpoint save in the notebook's envelope format
    (model_state, val_acc, per_cls_recall, class_names, cfg) — same
    shape that application/model_service/core/emotion/empathbot.py
    knows how to load.

The pipeline phase calls `run(ctx, dataset, model)` which orchestrates
everything; `__init__.py` is the thin pipeline-facing surface.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.loop import auto_device, collect_predictions
from pipeline.training.reporting import write_standard_artifacts

from .augment import HARD_LABEL_IDS
from .data import EmpathBotDataset
from .loss import FocalLoss

logger = logging.getLogger(__name__)


CFG = dict(
    # Architecture
    backbone        = "efficientnet_b2",
    use_timm        = True,
    se_reduction    = 16,

    # Training
    epochs          = 50,
    batch_size      = 64,
    img_size        = 224,
    freeze_epochs   = 8,

    backbone_lr     = 5e-5,
    head_lr         = 5e-4,
    weight_decay    = 1e-4,

    warmup_epochs   = 4,
    min_lr          = 1e-7,

    # Loss
    label_smoothing = 0.1,
    focal_gamma     = 2.0,

    # Misc
    mixup_alpha     = 0.0,   # off per notebook (hurts subtle classes)
    grad_clip       = 1.0,
    patience        = 10,
    priority_classes = [2, 3],
    num_workers     = 2,
)


def _config_overrides(ctx_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge ctx.config.train_cfg over the default CFG. The pipeline's
    fast/baseline/thorough configs can override individual keys
    (epochs, batch_size) without forcing the team to maintain a
    fully-specified empathbot-specific yaml."""
    out = dict(CFG)
    for key in ("epochs", "batch_size", "num_workers", "patience",
                "backbone_lr", "head_lr", "weight_decay", "focal_gamma",
                "label_smoothing", "freeze_epochs", "grad_clip"):
        if key in ctx_cfg:
            out[key] = ctx_cfg[key]
    return out


def _compute_class_weights(train_csv: str, num_classes: int) -> np.ndarray:
    """Inverse-frequency with the notebook's 1.3x bias toward HARD_LABEL_IDS."""
    df = pd.read_csv(train_csv)
    counts = np.bincount(df["label"].astype(int).values, minlength=num_classes).astype(float)
    weights = 1.0 / np.where(counts == 0, 1.0, counts)
    for hid in HARD_LABEL_IDS:
        if hid < num_classes:
            weights[hid] *= 1.3
    return weights


def _make_scheduler(optimizer, steps_per_epoch: int, cfg: dict[str, Any]):
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps  = cfg["epochs"] * steps_per_epoch
    head_lr      = cfg["head_lr"]
    min_lr       = cfg["min_lr"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = min_lr / head_lr
        return floor + (1.0 - floor) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler,
                 criterion: nn.Module, device, freeze_backbone: bool,
                 grad_clip: float) -> tuple[float, float]:
    """Verbatim from notebook 6b cell 15."""
    model.train()
    for p in model.backbone_params():
        p.requires_grad_(not freeze_backbone)

    loss_sum, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += int((logits.argmax(1) == labels).sum().item())
        n += int(labels.size(0))
    return loss_sum / max(1, n), correct / max(1, n)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device,
              num_classes: int) -> tuple[float, list[float]]:
    """Returns (overall_acc, per_class_recall). Simplified vs the
    notebook's TTA-capable evaluator — same per-class recall via
    confusion matrix."""
    model.eval()
    preds_all, labels_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds_all.extend(model(imgs).argmax(1).cpu().tolist())
        labels_all.extend(labels.tolist())
    preds_arr = np.array(preds_all)
    labels_arr = np.array(labels_all)
    acc = float((preds_arr == labels_arr).mean()) if len(preds_arr) else 0.0
    per_cls = []
    for c in range(num_classes):
        mask = labels_arr == c
        per_cls.append(float((preds_arr[mask] == c).mean()) if mask.any() else 0.0)
    return acc, per_cls


def run(ctx: Context, dataset: DatasetSpec, model: nn.Module) -> TrainedModel:
    """Run the notebook-6b training procedure against the prepared dataset."""
    cfg = _config_overrides(ctx.config.train_cfg)
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes

    # ── Class weights + WeightedRandomSampler (notebook cell 7) ──────────
    cls_weights = _compute_class_weights(str(dataset.splits["train"]), num_classes)
    class_weights_t = torch.tensor(cls_weights, dtype=torch.float32).to(device)
    train_labels = pd.read_csv(dataset.splits["train"])["label"].astype(int).values
    sample_w = cls_weights[train_labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_ds = EmpathBotDataset(dataset.splits["train"], is_train=True)
    val_ds   = EmpathBotDataset(dataset.splits["val"],   is_train=False)
    test_ds  = EmpathBotDataset(dataset.splits["test"],  is_train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    # ── Loss + optimizer + scheduler (notebook cells 11 + 13) ────────────
    criterion = FocalLoss(
        weight=class_weights_t,
        gamma=cfg["focal_gamma"],
        label_smoothing=cfg["label_smoothing"],
    )
    optimizer = optim.AdamW(
        [
            {"params": model.backbone_params(), "lr": cfg["backbone_lr"]},
            {"params": model.head_params(),     "lr": cfg["head_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = _make_scheduler(optimizer, steps_per_epoch=len(train_loader), cfg=cfg)

    logger.info(
        "empathbot_v1: device=%s epochs=%d batch=%d backbone_lr=%.1e head_lr=%.1e "
        "freeze=%d patience=%d",
        device, cfg["epochs"], cfg["batch_size"], cfg["backbone_lr"],
        cfg["head_lr"], cfg["freeze_epochs"], cfg["patience"],
    )

    # ── Training loop ────────────────────────────────────────────────────
    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    patience_cnt = 0

    for epoch in range(1, cfg["epochs"] + 1):
        frozen = epoch <= cfg["freeze_epochs"]
        tr_loss, tr_acc = _train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, frozen, cfg["grad_clip"],
        )
        val_acc, per_cls = _evaluate(model, val_loader, device, num_classes)

        ctx.save_scalar("train/loss",  tr_loss, step=epoch - 1)
        ctx.save_scalar("train/acc",   tr_acc,  step=epoch - 1)
        ctx.save_scalar("val/acc",     val_acc, step=epoch - 1)
        for c, recall in enumerate(per_cls):
            ctx.save_scalar(f"val/recall_class_{c}", recall, step=epoch - 1)

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_acc": val_acc, "per_class_recall": per_cls, "frozen": frozen,
        })
        logger.info(
            "epoch %d/%d: train_loss=%.4f train_acc=%.3f val_acc=%.3f%s",
            epoch, cfg["epochs"], tr_loss, tr_acc, val_acc,
            " [frozen]" if frozen else "",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_cnt = 0
            # Notebook's checkpoint envelope — model_service can load this.
            ctx.save_checkpoint("best", {
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "val_acc":        val_acc,
                "per_cls_recall": per_cls,
                "class_names":    {i: n for i, n in enumerate(dataset.class_names)},
                "cfg":            cfg,
            })
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                logger.info("early stop at epoch %d (patience=%d)", epoch, cfg["patience"])
                break

    # ── Final test eval on best ──────────────────────────────────────────
    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    test_acc, test_per_cls = _evaluate(model, test_loader, device, num_classes)
    test_preds, test_labels = collect_predictions(model, test_loader, device)
    ctx.save_scalar("test/acc", test_acc)
    for c, recall in enumerate(test_per_cls):
        ctx.save_scalar(f"test/recall_class_{c}", recall)

    ctx.save_json("history", history)
    write_standard_artifacts(
        ctx, history=history,
        test_preds=test_preds, test_labels=test_labels,
        num_classes=num_classes, class_names=dataset.class_names,
        final_summary={
            "best_epoch":     best_epoch,
            "best_val_acc":   best_val_acc,
            "test_acc":       test_acc,
            "test_per_class": test_per_cls,
        },
    )

    logger.info(
        "empathbot_v1: complete. best_val_acc=%.4f@epoch%d  test_acc=%.4f",
        best_val_acc, best_epoch, test_acc,
    )
    return TrainedModel(
        model_name=ctx.config.model,
        num_classes=num_classes,
        checkpoint_path=best_ckpt,
        history=history,
        final_val={"acc": best_val_acc},
        final_test={"acc": test_acc, "per_class": test_per_cls},
    )

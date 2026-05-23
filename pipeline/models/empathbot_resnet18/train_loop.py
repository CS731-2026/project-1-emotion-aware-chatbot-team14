"""Training loop — ported from notebook 6 cells 13 + 15.

Differs from empathbot_v1 (6b) — uses CrossEntropyLoss with label
smoothing instead of FocalLoss, and MixUp alpha=0.2 (6b disables
MixUp). Otherwise the schedule is the same: split-LR AdamW, linear-
warmup + cosine LambdaLR, freeze the backbone for the first N epochs,
weighted-CE with class weights, early stopping on val_acc.

Checkpoint envelope matches notebook cell 15:
  epoch, model_state, val_acc, per_cls_recall, class_names, cfg
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
from torch.utils.data import DataLoader

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.loop import auto_device, collect_predictions, merge_cfg
from pipeline.training.reporting import write_standard_artifacts

from .augment import HARD_LABEL_IDS
from .data import EmpathBotDataset

logger = logging.getLogger(__name__)


CFG = dict(
    backbone        = "resnet18",
    se_reduction    = 16,
    epochs          = 25,
    batch_size      = 64,
    img_size        = 224,
    freeze_epochs   = 5,
    backbone_lr     = 1.0e-4,
    head_lr         = 1.0e-3,
    weight_decay    = 1.0e-4,
    warmup_epochs   = 3,
    min_lr          = 1.0e-6,
    label_smoothing = 0.05,
    mixup_alpha     = 0.2,
    grad_clip       = 1.0,
    patience        = 8,
    num_workers     = 2,
)


def _compute_class_weights(train_csv, num_classes: int) -> np.ndarray:
    """Inverse-frequency with 1.3x bias toward HARD_LABEL_IDS, matching
    notebook 6's approach (lifted from cell 9)."""
    df = pd.read_csv(train_csv)
    counts = np.bincount(df["label"].astype(int).values, minlength=num_classes).astype(float)
    weights = 1.0 / np.where(counts == 0, 1.0, counts)
    for hid in HARD_LABEL_IDS:
        if hid < num_classes:
            weights[hid] *= 1.3
    return weights


def _mixup(x, y, alpha):
    """Notebook cell 15."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _train_epoch(model, loader, optimizer, scheduler, criterion, device,
                 freeze_backbone, cfg):
    """Verbatim from notebook cell 15."""
    model.train()
    for p in model.backbone_params():
        p.requires_grad_(not freeze_backbone)

    loss_sum, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)
        if cfg["mixup_alpha"] > 0:
            imgs, ya, yb, lam = _mixup(imgs, labels, cfg["mixup_alpha"])
            logits = model(imgs)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += int((logits.argmax(1) == labels).sum().item())
        n += int(labels.size(0))
    return loss_sum / max(1, n), correct / max(1, n)


@torch.no_grad()
def _evaluate(model, loader, device, num_classes):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        preds.extend(model(imgs.to(device)).argmax(1).cpu().tolist())
        labels.extend(lbls.tolist())
    pa, la = np.array(preds), np.array(labels)
    acc = float((pa == la).mean()) if len(pa) else 0.0
    per_cls = []
    for c in range(num_classes):
        mask = la == c
        per_cls.append(float((pa[mask] == c).mean()) if mask.any() else 0.0)
    return acc, per_cls


def run(ctx: Context, dataset: DatasetSpec, model: nn.Module) -> TrainedModel:
    cfg = merge_cfg(CFG, ctx.config.train_cfg)
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes

    cls_w = torch.tensor(
        _compute_class_weights(str(dataset.splits["train"]), num_classes),
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=cfg["label_smoothing"])

    optimizer = optim.AdamW(
        [
            {"params": model.backbone_params(), "lr": cfg["backbone_lr"]},
            {"params": model.head_params(),     "lr": cfg["head_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )

    train_ds = EmpathBotDataset(dataset.splits["train"], is_train=True)
    val_ds   = EmpathBotDataset(dataset.splits["val"],   is_train=False)
    test_ds  = EmpathBotDataset(dataset.splits["test"],  is_train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    warmup_steps = cfg["warmup_epochs"] * len(train_loader)
    total_steps  = cfg["epochs"] * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = cfg["min_lr"] / cfg["head_lr"]
        return floor + (1.0 - floor) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(
        "empathbot_resnet18: device=%s epochs=%d batch=%d backbone_lr=%.1e head_lr=%.1e "
        "freeze=%d mixup=%.2f patience=%d",
        device, cfg["epochs"], cfg["batch_size"], cfg["backbone_lr"],
        cfg["head_lr"], cfg["freeze_epochs"], cfg["mixup_alpha"], cfg["patience"],
    )

    history = []
    best_val = -1.0
    best_epoch = 0
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        frozen = epoch <= cfg["freeze_epochs"]
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, scheduler,
                                        criterion, device, frozen, cfg)
        val_acc, per_cls = _evaluate(model, val_loader, device, num_classes)

        ctx.save_scalar("train/loss", tr_loss, step=epoch - 1)
        ctx.save_scalar("train/acc",  tr_acc,  step=epoch - 1)
        ctx.save_scalar("val/acc",    val_acc, step=epoch - 1)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_acc": val_acc, "per_class_recall": per_cls,
                        "frozen": frozen})
        logger.info("epoch %d/%d: tr_loss=%.4f tr_acc=%.3f val_acc=%.3f%s",
                    epoch, cfg["epochs"], tr_loss, tr_acc, val_acc,
                    " [frozen]" if frozen else "")

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            patience = 0
            ctx.save_checkpoint("best", {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "per_cls_recall": per_cls,
                "class_names": {i: n for i, n in enumerate(dataset.class_names)},
                "cfg": cfg,
            })
        else:
            patience += 1
            if patience >= cfg["patience"]:
                logger.info("early stop at epoch %d", epoch)
                break

    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
    test_acc, test_per_cls = _evaluate(model, test_loader, device, num_classes)
    test_preds, test_labels = collect_predictions(model, test_loader, device)
    ctx.save_scalar("test/acc", test_acc)
    ctx.save_json("history", history)
    write_standard_artifacts(
        ctx, history=history,
        test_preds=test_preds, test_labels=test_labels,
        num_classes=num_classes, class_names=dataset.class_names,
        final_summary={
            "best_epoch": best_epoch, "best_val_acc": best_val,
            "test_acc": test_acc, "test_per_class": test_per_cls,
        },
    )

    logger.info("empathbot_resnet18: complete. best_val=%.4f@%d test=%.4f",
                best_val, best_epoch, test_acc)
    return TrainedModel(
        model_name=ctx.config.model, num_classes=num_classes,
        checkpoint_path=best_ckpt, history=history,
        final_val={"acc": best_val},
        final_test={"acc": test_acc, "per_class": test_per_cls},
    )

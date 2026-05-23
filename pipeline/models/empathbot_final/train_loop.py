"""Training loop — ported from notebook 5_v4 cells 20, 22, 24, 25.

Differs from empathbot_v1 (6b):
  - MixUp augmentation from epoch 1 (alpha=0.2)
  - AdamW split-LR: backbone @ 1e-4, head/classifier @ 1e-3
  - LambdaLR: 3-epoch linear warmup, cosine to 0 over remaining epochs
  - Per-class augmentation routing (STD vs NEG transforms)
  - Backbone freeze for first 5 epochs, then unfreeze + reset both LRs
  - CrossEntropyLoss (not focal) with class weights + label_smoothing=0.05
  - Checkpoint envelope: epoch, val_acc, model_state_dict,
    optimizer_state_dict, empathbot_classes, architecture
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.loop import auto_device, collect_predictions
from pipeline.training.reporting import write_standard_artifacts

from .augment import NEGATIVE_LABEL_IDS
from .data import EmpathBotDataset

logger = logging.getLogger(__name__)


CFG = dict(
    backbone        = "efficientnet_b2",
    lr_head         = 1.0e-3,
    lr_backbone     = 1.0e-4,
    weight_decay    = 1.0e-4,
    epochs          = 40,
    batch_size      = 32,
    num_workers     = 2,
    mixup_alpha     = 0.2,
    mixup_start_epoch = 0,             # MixUp from epoch 1 (idx 0)
    backbone_freeze_epochs = 5,
    label_smoothing = 0.05,
    grad_clip       = 1.0,
    neg_boost       = 1.2,             # class_weights boost for NEGATIVE classes
    warmup_epochs   = 3,
)


def _config_overrides(ctx_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(CFG)
    for k in ("epochs", "batch_size", "num_workers", "lr_head", "lr_backbone",
              "weight_decay", "mixup_alpha", "backbone_freeze_epochs",
              "label_smoothing", "grad_clip", "neg_boost", "warmup_epochs"):
        if k in ctx_cfg:
            out[k] = ctx_cfg[k]
    return out


def _compute_class_weights(train_ds: EmpathBotDataset, num_classes: int,
                            neg_boost: float) -> torch.Tensor:
    """Notebook cell 20 — inverse-frequency * neg_boost for NEGATIVE classes."""
    counts = train_ds.class_counts(num_classes)
    total = sum(counts)
    w = [total / (num_classes * max(c, 1)) for c in counts]
    for i in NEGATIVE_LABEL_IDS:
        if i < num_classes:
            w[i] *= neg_boost
    wt = torch.tensor(w, dtype=torch.float32)
    wt = wt / wt.sum() * num_classes   # mean ≈ 1
    return wt


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Notebook cell 24."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _mixup_loss(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


def _train_epoch(model, loader, optimizer, criterion, device,
                 use_mixup: bool, alpha: float, grad_clip: float):
    """Verbatim from notebook cell 24."""
    model.train()
    total_loss = correct = total = 0
    y_a = y_b = None
    lam = 1.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_mixup:
            imgs, y_a, y_b, lam = _mixup_batch(imgs, labels, alpha)
        optimizer.zero_grad()
        out = model(imgs)
        loss = _mixup_loss(criterion, out, y_a, y_b, lam) if use_mixup \
               else criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(1)
        ref = y_a if use_mixup else labels
        correct += int(preds.eq(ref).sum().item())
        total += int(labels.size(0))
    return total_loss / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    """Notebook cell 24 eval_epoch — returns (loss, acc)."""
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        correct += int(out.argmax(1).eq(labels).sum().item())
        total += int(labels.size(0))
    return total_loss / max(1, len(loader)), correct / max(1, total)


def run(ctx: Context, dataset: DatasetSpec, model: nn.Module) -> TrainedModel:
    cfg = _config_overrides(ctx.config.train_cfg)
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes

    train_ds = EmpathBotDataset(dataset.splits["train"], is_train=True)
    val_ds   = EmpathBotDataset(dataset.splits["val"],   is_train=False)
    test_ds  = EmpathBotDataset(dataset.splits["test"],  is_train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))

    # Class weights + loss (cell 20)
    class_w = _compute_class_weights(train_ds, num_classes, cfg["neg_boost"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=cfg["label_smoothing"])

    # Freeze backbone, split-LR AdamW (cell 22)
    model.freeze_backbone()
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": cfg["lr_backbone"]},
        {"params": model.classifier.parameters(), "lr": cfg["lr_head"]},
    ], weight_decay=cfg["weight_decay"])

    epochs = cfg["epochs"]
    warmup = cfg["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(epochs - warmup, 1)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(
        "empathbot_final: device=%s epochs=%d batch=%d head_lr=%.1e bb_lr=%.1e "
        "freeze=%d mixup_alpha=%.2f",
        device, epochs, cfg["batch_size"], cfg["lr_head"], cfg["lr_backbone"],
        cfg["backbone_freeze_epochs"], cfg["mixup_alpha"],
    )

    history: list[dict] = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        if epoch == cfg["backbone_freeze_epochs"] + 1:
            model.unfreeze_backbone()
            # Reset both LRs after unfreeze (notebook cell 25)
            for g in optimizer.param_groups:
                first_param = next(iter(g["params"]), None)
                bb_first = next(model.backbone.parameters(), None)
                g["lr"] = cfg["lr_backbone"] if first_param is bb_first else cfg["lr_head"]
            logger.info("epoch %d: backbone unfrozen", epoch)

        use_mixup = epoch > cfg["mixup_start_epoch"]
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion,
                                        device, use_mixup, cfg["mixup_alpha"], cfg["grad_clip"])
        vl_loss, vl_acc = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        ctx.save_scalar("train/loss", tr_loss, step=epoch - 1)
        ctx.save_scalar("train/acc",  tr_acc,  step=epoch - 1)
        ctx.save_scalar("val/loss",   vl_loss, step=epoch - 1)
        ctx.save_scalar("val/acc",    vl_acc,  step=epoch - 1)
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc,
            "frozen": epoch <= cfg["backbone_freeze_epochs"],
            "mixup": use_mixup,
        })
        logger.info(
            "epoch %d/%d  bb=%s mix=%s  tr_loss=%.4f tr_acc=%.3f  vl_loss=%.4f vl_acc=%.3f",
            epoch, epochs,
            "frozen" if epoch <= cfg["backbone_freeze_epochs"] else "thawed",
            "on" if use_mixup else "off",
            tr_loss, tr_acc, vl_loss, vl_acc,
        )

        if vl_acc > best_val_acc:
            best_val_acc, best_epoch = vl_acc, epoch
            ctx.save_checkpoint("best", {
                "epoch": epoch,
                "val_acc": vl_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "empathbot_classes": {i: n for i, n in enumerate(dataset.class_names)},
                "architecture": f"EmpathBotV1-{cfg['backbone']}",
            })

    # Final test eval on best
    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    test_loss, test_acc = _eval_epoch(model, test_loader, criterion, device)
    test_preds, test_labels = collect_predictions(model, test_loader, device)
    ctx.save_scalar("test/loss", test_loss)
    ctx.save_scalar("test/acc",  test_acc)
    ctx.save_json("history", history)
    write_standard_artifacts(
        ctx, history=history,
        test_preds=test_preds, test_labels=test_labels,
        num_classes=num_classes, class_names=dataset.class_names,
        final_summary={
            "best_epoch": best_epoch, "best_val_acc": best_val_acc,
            "test_acc": test_acc, "test_loss": test_loss,
        },
    )

    logger.info("empathbot_final: complete. best_val_acc=%.4f@epoch%d test_acc=%.4f",
                best_val_acc, best_epoch, test_acc)
    return TrainedModel(
        model_name=ctx.config.model,
        num_classes=num_classes,
        checkpoint_path=best_ckpt,
        history=history,
        final_val={"acc": best_val_acc},
        final_test={"acc": test_acc, "loss": test_loss},
    )

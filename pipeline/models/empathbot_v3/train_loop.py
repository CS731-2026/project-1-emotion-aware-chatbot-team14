"""Training loop — ported from notebook 5 cells 20, 22, 24, 25.

Three optimizer param groups (backbone / attention / classifier),
LR_HEAD=3e-3, no backbone freeze, MixUp from epoch 11 (MIXUP_START=10).
"""

from __future__ import annotations

import logging
import math
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
    lr_head           = 3.0e-3,
    lr_backbone       = 1.0e-4,
    weight_decay      = 1.0e-4,
    epochs            = 40,
    batch_size        = 32,
    num_workers       = 2,
    mixup_alpha       = 0.2,
    mixup_start_epoch = 10,
    label_smoothing   = 0.05,
    grad_clip         = 1.0,
    neg_boost         = 1.2,
    warmup_epochs     = 3,
)


def _config_overrides(ctx_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(CFG)
    for k in ("epochs", "batch_size", "num_workers", "lr_head", "lr_backbone",
              "weight_decay", "mixup_alpha", "mixup_start_epoch", "label_smoothing",
              "grad_clip", "neg_boost", "warmup_epochs"):
        if k in ctx_cfg:
            out[k] = ctx_cfg[k]
    return out


def _compute_class_weights(train_ds, num_classes, neg_boost):
    counts = train_ds.class_counts(num_classes)
    total = sum(counts)
    w = [total / (num_classes * max(c, 1)) for c in counts]
    for i in NEGATIVE_LABEL_IDS:
        if i < num_classes:
            w[i] *= neg_boost
    wt = torch.tensor(w, dtype=torch.float32)
    return wt / wt.sum() * num_classes


def _mixup(x, y, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _train_epoch(model, loader, optimizer, criterion, device,
                 use_mixup: bool, alpha: float, grad_clip: float):
    model.train()
    total_loss = correct = total = 0
    y_a = y_b = None
    lam = 1.0
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)
        if use_mixup:
            imgs, y_a, y_b, lam = _mixup(imgs, labels, alpha)
        optimizer.zero_grad()
        out = model(imgs)
        if use_mixup:
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
        else:
            loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        ref = y_a if use_mixup else labels
        correct += int(out.argmax(1).eq(ref).sum().item())
        total += int(labels.size(0))
    return total_loss / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def _eval(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)
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
                              num_workers=cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"])

    cls_w = _compute_class_weights(train_ds, num_classes, cfg["neg_boost"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=cfg["label_smoothing"])

    # Three param groups (notebook cell 22) — backbone / attention / classifier
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": cfg["lr_backbone"]},
        {"params": model.attention.parameters(),  "lr": cfg["lr_head"]},
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
        "empathbot_v3: device=%s epochs=%d head_lr=%.1e bb_lr=%.1e mixup_after=%d",
        device, epochs, cfg["lr_head"], cfg["lr_backbone"], cfg["mixup_start_epoch"],
    )

    history = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        use_mixup = epoch > cfg["mixup_start_epoch"]
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion,
                                        device, use_mixup, cfg["mixup_alpha"], cfg["grad_clip"])
        vl_loss, vl_acc = _eval(model, val_loader, criterion, device)
        scheduler.step()

        ctx.save_scalar("train/loss", tr_loss, step=epoch - 1)
        ctx.save_scalar("train/acc",  tr_acc,  step=epoch - 1)
        ctx.save_scalar("val/loss",   vl_loss, step=epoch - 1)
        ctx.save_scalar("val/acc",    vl_acc,  step=epoch - 1)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": vl_loss, "val_acc": vl_acc, "mixup": use_mixup})
        logger.info(
            "epoch %d/%d mix=%s tr_loss=%.4f tr_acc=%.3f vl_loss=%.4f vl_acc=%.3f",
            epoch, epochs, "on" if use_mixup else "off",
            tr_loss, tr_acc, vl_loss, vl_acc,
        )

        if vl_acc > best_val_acc:
            best_val_acc, best_epoch = vl_acc, epoch
            ctx.save_checkpoint("best", {
                "epoch": epoch, "val_acc": vl_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "empathbot_classes": {i: n for i, n in enumerate(dataset.class_names)},
                "architecture": "EmpathBotV1-resnet18-se",
            })

    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    test_loss, test_acc = _eval(model, test_loader, criterion, device)
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

    logger.info("empathbot_v3: complete. best_val=%.4f@%d test=%.4f",
                best_val_acc, best_epoch, test_acc)
    return TrainedModel(
        model_name=ctx.config.model, num_classes=num_classes,
        checkpoint_path=best_ckpt, history=history,
        final_val={"acc": best_val_acc},
        final_test={"acc": test_acc, "loss": test_loss},
    )

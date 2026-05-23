"""Training loop — ported from notebook 2 cells 4, 10, 14, 16.

Adam (not AdamW), StepLR every 10 epochs ×0.5, weighted CE, early
stopping with patience=10. Checkpoint envelope matches the notebook
(epoch, model_state_dict, optimizer_state_dict, val_acc, val_loss,
class_names, num_classes).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.loop import auto_device, collect_predictions
from pipeline.training.reporting import write_standard_artifacts

from .augment import TRAIN_TF, VAL_TF

logger = logging.getLogger(__name__)


CFG = dict(
    epochs          = 40,
    batch_size      = 32,
    img_size        = 224,
    lr              = 1.0e-4,
    weight_decay    = 1.0e-4,
    lr_decay_step   = 10,
    lr_decay_gamma  = 0.5,
    num_workers     = 0,
    early_stop      = 10,
)


class _CsvDataset(Dataset):
    """The notebook's EmpathBotDataset — skips missing files silently."""

    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        from pathlib import Path
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        return img, int(row["label"])


def _config_overrides(ctx_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(CFG)
    for k in ("epochs", "batch_size", "num_workers", "lr", "weight_decay",
              "lr_decay_step", "lr_decay_gamma", "early_stop"):
        if k in ctx_cfg:
            out[k] = ctx_cfg[k]
    return out


def _run_epoch(model, loader, criterion, optimizer, device,
               is_training: bool) -> tuple[float, float]:
    """Verbatim from notebook 2 cell 16."""
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += int((outputs.argmax(1) == labels).sum().item())
            total += int(images.size(0))
    return running_loss / max(1, total), correct / max(1, total)


def run(ctx: Context, dataset: DatasetSpec, model: nn.Module) -> TrainedModel:
    cfg = _config_overrides(ctx.config.train_cfg)
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes

    # Class weights (notebook cell 14)
    weight = None
    if dataset.class_weights is not None:
        weight = torch.tensor(dataset.class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["lr_decay_step"], gamma=cfg["lr_decay_gamma"],
    )

    train_loader = DataLoader(
        _CsvDataset(dataset.splits["train"], TRAIN_TF),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        _CsvDataset(dataset.splits["val"], VAL_TF),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        _CsvDataset(dataset.splits["test"], VAL_TF),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        "resnet18: device=%s epochs=%d batch=%d lr=%.1e early_stop=%d",
        device, cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["early_stop"],
    )

    history: list[dict] = []
    best_val_acc = 0.0
    best_epoch = 0
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer,
                                      device, is_training=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer,
                                        device, is_training=False)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        ctx.save_scalar("train/loss", tr_loss, step=epoch - 1)
        ctx.save_scalar("train/acc",  tr_acc,  step=epoch - 1)
        ctx.save_scalar("val/loss",   val_loss, step=epoch - 1)
        ctx.save_scalar("val/acc",    val_acc,  step=epoch - 1)
        ctx.save_scalar("lr",         current_lr, step=epoch - 1)
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
        })
        logger.info(
            "epoch %d/%d: train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f lr=%.1e",
            epoch, cfg["epochs"], tr_loss, tr_acc, val_loss, val_acc, current_lr,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience = 0
            ctx.save_checkpoint("best", {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": {i: n for i, n in enumerate(dataset.class_names)},
                "num_classes": num_classes,
            })
        else:
            patience += 1
            if patience >= cfg["early_stop"]:
                logger.info("early stop at epoch %d (no improvement for %d epochs)",
                            epoch, cfg["early_stop"])
                break

    # Final test eval on best
    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, optimizer,
                                      device, is_training=False)
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

    logger.info("resnet18: complete. best_val_acc=%.4f@epoch%d test_acc=%.4f",
                best_val_acc, best_epoch, test_acc)
    return TrainedModel(
        model_name=ctx.config.model,
        num_classes=num_classes,
        checkpoint_path=best_ckpt,
        history=history,
        final_val={"acc": best_val_acc},
        final_test={"acc": test_acc, "loss": test_loss},
    )

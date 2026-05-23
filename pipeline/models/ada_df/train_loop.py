"""Training loop — ported from notebook 4 cell 11.

AdamW + StepLR (×0.5 every 12 epochs) + plain CrossEntropy + early
stop on train_acc (the notebook patterns on train_acc rather than
val_acc; pipeline preserves that).

Checkpoint envelope: epoch, model_state_dict, train_acc.
"""

from __future__ import annotations

import logging
from pathlib import Path
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
    lr              = 3.0e-4,
    weight_decay    = 1.0e-4,
    lr_decay_step   = 12,
    lr_decay_gamma  = 0.5,
    early_stop      = 10,
    num_workers     = 0,
)


class _CsvDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        valid = df["path"].apply(lambda p: Path(p).exists())
        self.df = df[valid].reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])


def _config_overrides(ctx_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(CFG)
    for k in ("epochs", "batch_size", "num_workers", "lr", "weight_decay",
              "lr_decay_step", "lr_decay_gamma", "early_stop"):
        if k in ctx_cfg:
            out[k] = ctx_cfg[k]
    return out


@torch.no_grad()
def _eval(model, loader, device) -> tuple[float, float]:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)
        correct += int((model(imgs).argmax(1) == labels).sum().item())
        total += int(labels.size(0))
    return correct / max(1, total), 0.0


def run(ctx: Context, dataset: DatasetSpec, model: nn.Module) -> TrainedModel:
    cfg = _config_overrides(ctx.config.train_cfg)
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes

    train_loader = DataLoader(
        _CsvDataset(dataset.splits["train"], TRAIN_TF),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        _CsvDataset(dataset.splits["val"], VAL_TF),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
    )
    test_loader = DataLoader(
        _CsvDataset(dataset.splits["test"], VAL_TF),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["lr_decay_step"], gamma=cfg["lr_decay_gamma"]
    )

    logger.info(
        "ada_df: device=%s epochs=%d batch=%d lr=%.1e early_stop=%d",
        device, cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["early_stop"],
    )

    history: list[dict] = []
    best_train_acc = 0.0
    best_epoch = 0
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += int((outputs.argmax(1) == labels).sum().item())
            total += int(imgs.size(0))
        tr_loss = running_loss / max(1, total)
        tr_acc = correct / max(1, total)
        scheduler.step()

        # Notebook only logs train metrics in this cell, but we also
        # eval on val for the run dir's metrics.jsonl + leaderboard.
        val_acc, _ = _eval(model, val_loader, device)

        ctx.save_scalar("train/loss", tr_loss, step=epoch - 1)
        ctx.save_scalar("train/acc",  tr_acc,  step=epoch - 1)
        ctx.save_scalar("val/acc",    val_acc, step=epoch - 1)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_acc": val_acc})
        logger.info(
            "epoch %d/%d  tr_loss=%.4f tr_acc=%.3f  val_acc=%.3f",
            epoch, cfg["epochs"], tr_loss, tr_acc, val_acc,
        )

        # Notebook uses train_acc for best-checkpoint + patience — preserve.
        if tr_acc > best_train_acc:
            best_train_acc = tr_acc
            best_epoch = epoch
            patience = 0
            ctx.save_checkpoint("best", {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_acc": tr_acc,
            })
        else:
            patience += 1
            if patience >= cfg["early_stop"]:
                logger.info("early stop at epoch %d", epoch)
                break

    # Test eval on best
    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    test_acc, _ = _eval(model, test_loader, device)
    test_preds, test_labels = collect_predictions(model, test_loader, device)
    ctx.save_scalar("test/acc", test_acc)
    ctx.save_json("history", history)
    write_standard_artifacts(
        ctx, history=history,
        test_preds=test_preds, test_labels=test_labels,
        num_classes=num_classes, class_names=dataset.class_names,
        final_summary={
            "best_epoch": best_epoch, "best_train_acc": best_train_acc,
            "test_acc": test_acc,
        },
    )

    logger.info("ada_df: complete. best_train_acc=%.4f@epoch%d test_acc=%.4f",
                best_train_acc, best_epoch, test_acc)
    return TrainedModel(
        model_name=ctx.config.model,
        num_classes=num_classes,
        checkpoint_path=best_ckpt,
        history=history,
        final_val={"train_acc": best_train_acc},
        final_test={"acc": test_acc},
    )

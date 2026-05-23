"""Training loop — ported from notebook cells 38, 44, 48.

SGD(momentum=0.9, wd=1e-4) + OneCycleLR (scheduler.step() per batch, not
per epoch — notebook's train_step calls it inside the batch loop).
CrossEntropyLoss (no class weights). 50 epochs, early-stop patience=10,
min_delta=1e-4. Batch size 32.

The notebook discovers max_lr via fastai's lr_find; here we use a fixed
sensible default (3e-3) which can be overridden via cfg["max_lr"].
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
    epochs       = 50,
    batch_size   = 32,
    max_lr       = 3.0e-3,
    momentum     = 0.9,
    weight_decay = 1.0e-4,
    num_workers  = 0,
    early_stop   = 10,
    min_delta    = 1.0e-4,
)


class _CsvDataset(Dataset):
    def __init__(self, csv_path, transform):
        from pathlib import Path
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


def _config_overrides(ctx_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(CFG)
    for k in ("epochs", "batch_size", "num_workers", "max_lr", "momentum",
              "weight_decay", "early_stop", "min_delta"):
        if k in ctx_cfg:
            out[k] = ctx_cfg[k]
    return out


def _train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Verbatim from notebook cell 38 — scheduler steps **per batch**."""
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
        correct += int(out.argmax(1).eq(y).sum().item())
        total += int(y.size(0))
    return total_loss / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def _eval(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += float(loss.item())
        correct += int(out.argmax(1).eq(y).sum().item())
        total += int(y.size(0))
    return total_loss / max(1, len(loader)), correct / max(1, total)


def run(ctx: Context, dataset: DatasetSpec, model: nn.Module) -> TrainedModel:
    cfg = _config_overrides(ctx.config.train_cfg)
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes

    train_ds = _CsvDataset(dataset.splits["train"], TRAIN_TF)
    val_ds   = _CsvDataset(dataset.splits["val"],   VAL_TF)
    test_ds  = _CsvDataset(dataset.splits["test"],  VAL_TF)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg["max_lr"],
                          momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["max_lr"],
        steps_per_epoch=max(1, len(train_loader)),
        epochs=cfg["epochs"],
    )

    logger.info(
        "resnet18_fer_onecycle: device=%s epochs=%d max_lr=%.1e",
        device, cfg["epochs"], cfg["max_lr"],
    )

    history = []
    best_val_acc = 0.0
    best_epoch = 0
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion,
                                            optimizer, scheduler, device)
        vl_loss, vl_acc = _eval(model, val_loader, criterion, device)

        ctx.save_scalar("train/loss", tr_loss, step=epoch - 1)
        ctx.save_scalar("train/acc",  tr_acc,  step=epoch - 1)
        ctx.save_scalar("val/loss",   vl_loss, step=epoch - 1)
        ctx.save_scalar("val/acc",    vl_acc,  step=epoch - 1)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": vl_loss, "val_acc": vl_acc})
        logger.info("epoch %d/%d tr_loss=%.4f tr_acc=%.3f vl_loss=%.4f vl_acc=%.3f",
                    epoch, cfg["epochs"], tr_loss, tr_acc, vl_loss, vl_acc)

        if vl_acc - best_val_acc > cfg["min_delta"]:
            best_val_acc, best_epoch, patience = vl_acc, epoch, 0
            ctx.save_checkpoint("best", {
                "epoch": epoch, "val_acc": vl_acc, "val_loss": vl_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            })
        else:
            patience += 1
            if patience >= cfg["early_stop"]:
                logger.info("early stop at epoch %d (best val_acc=%.4f @ %d)",
                            epoch, best_val_acc, best_epoch)
                break

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

    logger.info("resnet18_fer_onecycle: complete. best_val=%.4f@%d test=%.4f",
                best_val_acc, best_epoch, test_acc)
    return TrainedModel(
        model_name=ctx.config.model, num_classes=num_classes,
        checkpoint_path=best_ckpt, history=history,
        final_val={"acc": best_val_acc},
        final_test={"acc": test_acc, "loss": test_loss},
    )

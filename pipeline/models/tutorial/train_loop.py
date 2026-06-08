"""train_loop.py, orchestrates training. The meat of a model module.

═══════════════════════════════════════════════════════════════════════════
FRAMEWORK CHEATSHEET, every affordance you have
═══════════════════════════════════════════════════════════════════════════

ctx, Context the driver hands you. Surface:
  ctx.config               read-only Config (dataset_name, model_name, config_name, seed, train_cfg)
  ctx.config.train_cfg     the merged hparam dict from configs/<name>.py + per-run yaml overrides
  ctx.run_dir              Path, you rarely touch this; ctx.save_* handles paths
  ctx.dataset_module       the dataset module currently running
  ctx.model_module         this module
  ctx.save_image(name, fig)            → artifacts/<name>.png   (matplotlib Figure)
  ctx.save_json(name, obj)             → artifacts/<name>.json
  ctx.save_text(name, txt)             → artifacts/<name>.txt
  ctx.save_scalar(name, val, step=)    → appends to metrics.jsonl
  ctx.save_checkpoint(name, dict)      → checkpoints/<name>.pth

dataset, DatasetSpec from the dataset module's prepare(). Surface:
  dataset.name, num_classes, class_names, class_weights
  dataset.splits["train"|"val"|"test"]   → CSV Path with `path,label` rows

pipeline.training.loop helpers:
  auto_device()                        CUDA → MPS → CPU
  merge_cfg(CFG, overrides)            resolved dict + auto-logs defaults/overrides/resolved
  train_one_epoch / evaluate / collect_predictions, generic loop primitives

pipeline.training.reporting.write_standard_artifacts(ctx, ...)
                                       4 files in one call: training_curves.png,
                                       confusion_matrix.png, classification_report.txt,
                                       per_class_metrics.json + final.json

pipeline.training.standard.train_classifier(ctx, dataset, model, preprocess)
                                       If your training is "CE + AdamW", skip this whole
                                       file and use train_classifier instead
                                       (see pipeline/models/mlp/__init__.py).

pipeline.training.{losses, optimizers, augmentations}  string-keyed registries
pipeline.kaggle  download_dataset / fetch_models / publish_models / creds_present
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.loop import auto_device, collect_predictions, merge_cfg
from pipeline.training.reporting import write_standard_artifacts

from .augment import TRAIN_TF, VAL_TF
from .data import CsvImageDataset

logger = logging.getLogger(__name__)


# Defaults, any key here becomes overridable from configs/*.py CONFIG or
# runs.yaml train_cfg (merge_cfg auto-whitelists by intersection with CFG).
CFG = dict(
    epochs       = 30,
    batch_size   = 32,
    num_workers  = 0,        # bump to 2-4 with real datasets if you have RAM
    lr           = 1.0e-4,   # safe for fine-tuning a pretrained backbone
    weight_decay = 1.0e-4,
    early_stop   = 10,       # epochs without val_acc improvement before bail
)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * imgs.size(0)
        correct += int(out.argmax(1).eq(labels).sum().item())
        total += int(labels.size(0))
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def _eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += float(loss.item()) * imgs.size(0)
        correct += int(out.argmax(1).eq(labels).sum().item())
        total += int(labels.size(0))
    return total_loss / max(1, total), correct / max(1, total)


def run(ctx: Context, dataset: DatasetSpec, model: nn.Module) -> TrainedModel:
    cfg = merge_cfg(CFG, ctx.config.train_cfg)        # auto-logs defaults/overrides/resolved
    device = auto_device()
    model = model.to(device)
    num_classes = dataset.num_classes
    logger.info("tutorial: device=%s epochs=%d batch=%d lr=%.1e",
                device, cfg["epochs"], cfg["batch_size"], cfg["lr"])

    # ── dataloaders ──────────────────────────────────────────────────────
    pin = device.type == "cuda"
    def _loader(split: str, tf, shuffle: bool) -> DataLoader:
        return DataLoader(CsvImageDataset(dataset.splits[split], tf),
                           batch_size=cfg["batch_size"], shuffle=shuffle,
                           num_workers=cfg["num_workers"], pin_memory=pin)
    train_loader = _loader("train", TRAIN_TF, shuffle=True)
    val_loader   = _loader("val",   VAL_TF,   shuffle=False)
    test_loader  = _loader("test",  VAL_TF,   shuffle=False)

    # ── loss / optimiser / scheduler ─────────────────────────────────────
    # Weighted CE handles class imbalance, weights come from the dataset module.
    weight = (torch.tensor(dataset.class_weights, dtype=torch.float32).to(device)
              if dataset.class_weights is not None else None)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # halve every 10ep

    # ── train loop ───────────────────────────────────────────────────────
    history: list[dict] = []
    best_val_acc = 0.0
    best_epoch = 0
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = _eval_one_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # save_scalar appends to metrics.jsonl, cheap; call for any metric you want plotted later
        for name, val in [("train/loss", tr_loss), ("train/acc", tr_acc),
                           ("val/loss", vl_loss), ("val/acc", vl_acc), ("lr", current_lr)]:
            ctx.save_scalar(name, val, step=epoch - 1)

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": vl_loss, "val_acc": vl_acc, "lr": current_lr})
        logger.info("epoch %d/%d  tr_loss=%.4f tr_acc=%.3f  vl_loss=%.4f vl_acc=%.3f  lr=%.1e",
                    epoch, cfg["epochs"], tr_loss, tr_acc, vl_loss, vl_acc, current_lr)

        if vl_acc > best_val_acc:
            best_val_acc, best_epoch, patience = vl_acc, epoch, 0
            # Envelope is your call, include what deploy + debug need.
            ctx.save_checkpoint("best", {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": vl_acc, "val_loss": vl_loss,
                "class_names": dataset.class_names,
            })
        else:
            patience += 1
            if patience >= cfg["early_stop"]:
                logger.info("early stop at epoch %d (best val_acc=%.4f @ %d)",
                            epoch, best_val_acc, best_epoch)
                break

    # ── final test on the best checkpoint ────────────────────────────────
    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"          # one of the rare run_dir touches
    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    test_loss, test_acc = _eval_one_epoch(model, test_loader, criterion, device)
    test_preds, test_labels = collect_predictions(model, test_loader, device)

    ctx.save_scalar("test/loss", test_loss)
    ctx.save_scalar("test/acc",  test_acc)
    ctx.save_json("history", history)

    # 4 files in one call: training_curves.png, confusion_matrix.png,
    # classification_report.txt, per_class_metrics.json + final.json
    write_standard_artifacts(
        ctx, history=history,
        test_preds=test_preds, test_labels=test_labels,
        num_classes=num_classes, class_names=dataset.class_names,
        final_summary={"best_epoch": best_epoch, "best_val_acc": best_val_acc,
                       "test_acc": test_acc, "test_loss": test_loss},
    )

    # Custom artifact example, any matplotlib Figure → ctx.save_image.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(num_classes), [test_labels.count(i) for i in range(num_classes)])
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(dataset.class_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("test set count")
        ax.set_title(f"{ctx.config.model_name}, test class distribution")
        ctx.save_image("test_class_distribution", fig)
        plt.close(fig)
    except Exception as e:
        logger.warning("tutorial: custom artifact skipped (%s)", e)

    ctx.save_text("notes",
                  f"best val acc {best_val_acc:.4f} @ epoch {best_epoch}\n"
                  f"test acc      {test_acc:.4f}\n"
                  f"device        {device}\n")

    return TrainedModel(
        model_name=ctx.config.model_name,
        num_classes=num_classes,
        checkpoint_path=best_ckpt,
        history=history,
        final_val={"acc": best_val_acc},
        final_test={"acc": test_acc, "loss": test_loss},
    )

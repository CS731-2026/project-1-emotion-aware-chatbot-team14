"""Shared classifier-training routine.

Utility category in the utility/state/composition taxonomy. The
default training loop a model module delegates to when its setup is
"build it, feed it batches of images, cross-entropy loss, optimizer
of choice, save the best checkpoint" — i.e. most baselines.

Custom architectures (multi-stage training, contrastive / GAN / RL
losses, alternating optimizers, etc.) write their own train() in
their model module and don't import this helper. The
pipeline.training.loop primitives (train_one_epoch, evaluate,
auto_device) are still available for them to compose.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torchvision.transforms as T

from ..framework.specs import DatasetSpec, TrainedModel
from .augmentations import get_augment
from .data import make_loader
from .loop import auto_device, evaluate as run_eval, train_one_epoch
from .losses import get_loss
from .optimizers import get_optimizer

if TYPE_CHECKING:
    from torch import nn
    from ..framework.context import Context

logger = logging.getLogger(__name__)


def train_classifier(
    ctx: "Context",
    dataset: DatasetSpec,
    *,
    model: "nn.Module",
    preprocess: T.Compose,
) -> TrainedModel:
    """Train `model` on `dataset` using ctx.config.train_cfg.

    The model arrives already built (CPU). This helper places it on
    the auto-selected device, composes the augmentation in front of
    `preprocess` for the train split, runs the loop, tracks the best
    checkpoint by val_acc, and returns a TrainedModel pointing at the
    best weights. Final test eval also happens here so the
    TrainedModel.final_test is filled in.
    """
    tcfg = ctx.config.train_cfg
    epochs      = int(tcfg.get("epochs", 5))
    batch_size  = int(tcfg.get("batch_size", 32))
    num_workers = int(tcfg.get("num_workers", 0))

    device = auto_device()
    logger.info("train_classifier: device=%s epochs=%d batch=%d",
                device, epochs, batch_size)

    aug_cfg = tcfg.get("augment", {"name": "none"})
    train_tf = T.Compose(
        list(get_augment(aug_cfg.get("name", "none"), aug_cfg.get("args")).transforms)
        + list(preprocess.transforms)
    )

    train_loader = make_loader(dataset.splits["train"], train_tf,
                               batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = make_loader(dataset.splits["val"],   preprocess,
                               batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = make_loader(dataset.splits["test"],  preprocess,
                               batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)

    loss_cfg = tcfg.get("loss", {"name": "ce"})
    loss_fn  = get_loss(loss_cfg.get("name", "ce"),
                        class_weights=dataset.class_weights,
                        args=loss_cfg.get("args")).to(device)

    opt_cfg = tcfg.get("optimizer", {"name": "adamw", "args": {"lr": 1e-3}})
    optimizer = get_optimizer(opt_cfg.get("name", "adamw"),
                              model.parameters(),
                              args=opt_cfg.get("args"))

    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                        device, epoch, ctx)
        val_metrics = run_eval(model, val_loader, loss_fn, device)

        ctx.save_scalar("train/epoch_loss", train_metrics["loss"], step=epoch)
        ctx.save_scalar("val/loss",         val_metrics["loss"],   step=epoch)
        ctx.save_scalar("val/acc",          val_metrics["acc"],    step=epoch)
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        })
        logger.info("epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch + 1, epochs,
                    train_metrics["loss"], val_metrics["loss"], val_metrics["acc"])

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            ctx.save_checkpoint("best", model.state_dict())

    last_ckpt = ctx.save_checkpoint("last", model.state_dict())
    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if not best_ckpt.exists():
        best_ckpt = last_ckpt

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = run_eval(model, test_loader, loss_fn, device)
    ctx.save_scalar("test/loss", test_metrics["loss"])
    ctx.save_scalar("test/acc",  test_metrics["acc"])
    ctx.save_json("history", history)
    ctx.save_json("final", {
        "best_epoch":  best_epoch,
        "best_val":    {"acc": best_val_acc},
        "final_val":   history[-1] if history else {},
        "test":        test_metrics,
    })

    logger.info("train_classifier: complete. best_val_acc=%.4f@epoch%d test_acc=%.4f",
                best_val_acc, best_epoch, test_metrics["acc"])
    return TrainedModel(
        model_name=ctx.config.model,
        num_classes=dataset.num_classes,
        checkpoint_path=best_ckpt,
        history=history,
        final_val=history[-1] if history else {},
        final_test=test_metrics,
    )

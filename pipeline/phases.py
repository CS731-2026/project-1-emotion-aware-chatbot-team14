"""Phase functions.

Composition file in the utility/state/composition taxonomy. Each
phase has signature `(ctx: Context) -> None` and is registered in
pipeline/driver.py's PHASES dict. Order of execution is set by the
phase list on Context.config; the driver iterates it.

What each phase does:

  setup            seed RNGs, drop a breadcrumb
  prepare_dataset  delegate to ctx.dataset_module.prepare(ctx) → DatasetSpec
  train            build the model, train + eval, save checkpoint + TrainedModel
"""

from __future__ import annotations

import logging
import random

from .framework import keys as K
from .framework.context import Context
from .framework.specs import DatasetSpec, TrainedModel

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------


def setup(ctx: Context) -> None:
    """Seed RNGs and drop a breadcrumb in the run dir. Bookkeeping only."""
    seed = ctx.config.seed
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.info("setup: run_dir=%s seed=%d", ctx.run_dir, seed)
    ctx.save_text("setup", f"slug: {ctx.config.slug()}\nseed: {seed}\n")


# ----------------------------------------------------------------------------


def prepare_dataset(ctx: Context) -> None:
    """Delegate to the registered dataset module's prepare() function.

    Each dataset module knows how to fetch its own source (Kaggle,
    synthetic, local, …) and uses the helpers in pipeline.ingest to
    walk, remap, split, and persist a DatasetSpec. The phase is just
    glue — store the spec + drop a self-describing artifact.
    """
    spec = ctx.dataset_module.prepare(ctx)
    if not isinstance(spec, DatasetSpec):
        raise TypeError(
            f"{ctx.dataset_module.__name__}.prepare(ctx) returned "
            f"{type(spec).__name__}, expected DatasetSpec"
        )
    ctx.store.put(K.DATASET, spec)
    ctx.save_json("dataset_used", spec.to_manifest())


# ----------------------------------------------------------------------------


def train(ctx: Context) -> None:
    """Build the model on the prepared dataset, train + evaluate, save
    checkpoint, hand a TrainedModel back through the store.

    Per-epoch val + final test eval are inline here for now — a future
    dedicated evaluate phase can split out cleanly using the saved
    checkpoint and the history this phase writes.
    """
    import torch
    import torchvision.transforms as T

    from .training.augmentations import get_augment
    from .training.data import make_loader
    from .training.losses import get_loss
    from .training.loop import auto_device, evaluate as run_eval, train_one_epoch
    from .training.optimizers import get_optimizer

    ds = ctx.store.get(K.DATASET, DatasetSpec)
    model_module = ctx.model_module

    tcfg = ctx.config.train_cfg
    epochs      = int(tcfg.get("epochs", 5))
    batch_size  = int(tcfg.get("batch_size", 32))
    num_workers = int(tcfg.get("num_workers", 0))

    device = auto_device()
    logger.info("train: device=%s epochs=%d batch=%d", device, epochs, batch_size)

    # Transforms: augmentation only in front of train; bare PREPROCESS for val/test
    aug_cfg = tcfg.get("augment", {"name": "none"})
    train_tf = T.Compose(
        list(get_augment(aug_cfg.get("name", "none"), aug_cfg.get("args")).transforms)
        + list(model_module.PREPROCESS.transforms)
    )
    eval_tf = model_module.PREPROCESS

    train_loader = make_loader(ds.splits["train"], train_tf,
                               batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = make_loader(ds.splits["val"],   eval_tf,
                               batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = make_loader(ds.splits["test"],  eval_tf,
                               batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model_module.build(ds.num_classes).to(device)

    loss_cfg = tcfg.get("loss", {"name": "ce"})
    loss_fn  = get_loss(loss_cfg.get("name", "ce"),
                        class_weights=ds.class_weights,
                        args=loss_cfg.get("args"))
    # Weighted CE keeps the class-weight tensor as a buffer that must
    # live on the same device as the logits.
    loss_fn = loss_fn.to(device)

    opt_cfg = tcfg.get("optimizer", {"name": "adamw", "args": {"lr": 1e-3}})
    optimizer = get_optimizer(opt_cfg.get("name", "adamw"),
                              model.parameters(),
                              args=opt_cfg.get("args"))

    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, ctx,
        )
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

    # Final test eval on the best checkpoint.
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

    trained = TrainedModel(
        model_name=ctx.config.model,
        num_classes=ds.num_classes,
        checkpoint_path=best_ckpt,
        history=history,
        final_val=history[-1] if history else {},
        final_test=test_metrics,
    )
    ctx.store.put(K.TRAINED_MODEL, trained)
    logger.info("train: complete. best_val_acc=%.4f@epoch%d test_acc=%.4f",
                best_val_acc, best_epoch, test_metrics["acc"])

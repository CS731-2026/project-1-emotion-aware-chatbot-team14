"""Phase functions.

Each phase has the same signature — `(ctx: Context) -> None` — and
participates in the pipeline by being registered in `pipeline.driver.PHASES`.
A phase reads what it needs from `ctx.store` (typed `get`s), does its
work, drops artifacts via `ctx.save_*`, and `put`s any produced
objects back into the store for later phases.

Order of execution is controlled by `cfg.phases` (a list of phase
names from the experiment yaml), so an experiment can skip phases for
fast iteration or insert new ones without code changes to the driver.

Real phase bodies (prepare_dataset, train, evaluate) land in later
commits — this file ships with `setup` so the driver has something to
exercise end-to-end.
"""

from __future__ import annotations

import importlib
import logging
import random
from pathlib import Path

from . import dataset_ingest as ingest
from .dataset_spec import DatasetSpec
from .framework import keys as K
from .framework.context import Context
from .trained_model import TrainedModel

logger = logging.getLogger(__name__)


# Where prepped datasets live. Same output/ tree the run dirs are under,
# so the existing .gitignore rule covers it and `rm -rf output/` is a
# clean slate.
_DATA_ROOT = Path("output/data")


def setup(ctx: Context) -> None:
    """First phase. Seed RNGs, log run identity, leave a breadcrumb in
    the run dir. No store puts — setup is bookkeeping only.

    Importing torch / numpy lives inside the function so the framework
    module itself stays import-time cheap; phases that don't need them
    (e.g. a notebook-driven dry run) won't pay the cost.
    """
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


def prepare_dataset(ctx: Context) -> None:
    """Resolve cfg.dataset_cfg into a DatasetSpec under output/data/<name>/.

    Idempotent — if a manifest already sits in the cache and its
    source_md5 matches the (re-hashed) extracted source, the prep is a
    no-op load. Otherwise: download via the kaggle CLI, walk the
    imagefolder, apply the label_remap from the yaml, carve val from
    train, write split CSVs + manifest, store the DatasetSpec.

    Cache lives outside the run dir (it's reused across runs); the run
    dir gets a small `dataset_used.json` artifact pointing at it so a
    run is still self-describing.
    """
    dcfg = ctx.config.dataset_cfg
    name = dcfg["name"]
    cache_dir = _DATA_ROOT / name
    manifest_path = cache_dir / "manifest.json"
    source_dir = cache_dir / "source"

    # Re-use prior prep when nothing has changed.
    if manifest_path.exists() and source_dir.exists():
        prior = DatasetSpec.from_manifest(manifest_path)
        current_md5 = ingest.md5_of_dir(source_dir)
        if prior.source_md5 == current_md5:
            logger.info("prepare_dataset: cache hit at %s", cache_dir)
            ctx.store.put(K.DATASET, prior)
            ctx.save_json("dataset_used", prior.to_manifest())
            return
        logger.info("prepare_dataset: source changed (md5 mismatch), re-prepping")

    # Fresh prep — download, walk, remap, split, write.
    source = dcfg["source"]
    if source["type"] != "kaggle":
        raise NotImplementedError(
            f"dataset source type {source['type']!r} not supported yet — "
            "currently only 'kaggle' has an ingest path"
        )
    ingest.download_kaggle(source["dataset_id"], source_dir)

    layout = source["archive_layout"]
    train_raw = ingest.scan_imagefolder(source_dir / layout["train_dir"])
    test_raw  = ingest.scan_imagefolder(source_dir / layout["test_dir"])

    class_names = list(dcfg["class_names"])
    train_remapped = ingest.apply_remap(train_raw, dcfg["label_remap"], class_names)
    test_remapped  = ingest.apply_remap(test_raw,  dcfg["label_remap"], class_names)

    splits_cfg = dcfg.get("splits", {})
    train_df, val_df = ingest.carve_val(
        train_remapped,
        val_fraction=float(splits_cfg.get("val_fraction", 0.10)),
        seed=int(splits_cfg.get("seed", 42)),
    )

    weights: list[float] | None
    if dcfg.get("class_weights") == "auto":
        weights = ingest.compute_class_weights(train_df["label"], num_classes=len(class_names))
    else:
        weights = dcfg.get("class_weights")  # explicit list or None

    splits = ingest.write_split_csvs(
        {"train": train_df, "val": val_df, "test": test_remapped},
        cache_dir,
    )

    spec = DatasetSpec(
        name=name,
        cache_dir=cache_dir,
        splits=splits,
        num_classes=len(class_names),
        class_names=class_names,
        class_weights=weights,
        source_md5=ingest.md5_of_dir(source_dir),
    )
    manifest_path.write_text(__import__("json").dumps(spec.to_manifest(), indent=2))

    ctx.store.put(K.DATASET, spec)
    ctx.save_json("dataset_used", spec.to_manifest())
    logger.info(
        "prepare_dataset: %s ready — train=%d val=%d test=%d",
        name, len(train_df), len(val_df), len(test_remapped),
    )


def train(ctx: Context) -> None:
    """Build the configured model on the prepared dataset, train for the
    configured epochs, evaluate on val each epoch + test at the end,
    and hand the TrainedModel back through the store.

    Per-epoch evaluation is intentionally bundled here for now — a
    later commit can split it into its own evaluate phase without
    changing the train phase's shape (the checkpoint + the per-epoch
    history are enough to re-run rich eval post-hoc).
    """
    import torchvision.transforms as T

    from training.data import make_loader
    from training.losses import get_loss
    from training.loop import auto_device, evaluate as run_eval, train_one_epoch
    from training.optimizers import get_optimizer
    from training.augmentations import get_augment

    ds = ctx.store.get(K.DATASET, DatasetSpec)
    model_module = importlib.import_module(f"models.{ctx.config.model}")
    ctx.store.put(K.MODEL_MODULE, model_module)

    tcfg = ctx.config.train_cfg
    epochs       = int(tcfg.get("epochs", 5))
    batch_size   = int(tcfg.get("batch_size", 32))
    num_workers  = int(tcfg.get("num_workers", 0))

    device = auto_device()
    logger.info("train: device=%s epochs=%d batch=%d", device, epochs, batch_size)

    # transforms: augmentation only in front of train; bare PREPROCESS for val/test
    aug_cfg = tcfg.get("augment", {"name": "none"})
    train_tf = T.Compose(list(get_augment(aug_cfg.get("name", "none"), aug_cfg.get("args"))
                              .transforms)
                         + list(model_module.PREPROCESS.transforms))
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
    # Move the loss to device too — weighted CE keeps the class-weight
    # tensor as a buffer that must live on the same device as the logits.
    loss_fn = loss_fn.to(device)

    opt_cfg = tcfg.get("optimizer", {"name": "adamw", "args": {"lr": 1e-3}})
    optimizer = get_optimizer(opt_cfg.get("name", "adamw"),
                              model.parameters(),
                              args=opt_cfg.get("args"))

    # ---- training loop ----
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

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                                **{f"val_{k}":   v for k, v in val_metrics.items()}}
        history.append(row)
        logger.info("epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch + 1, epochs, train_metrics["loss"], val_metrics["loss"], val_metrics["acc"])

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            ctx.save_checkpoint("best", model.state_dict())

    # always save the last epoch's weights too
    last_ckpt = ctx.save_checkpoint("last", model.state_dict())
    best_ckpt = ctx.run_dir / "checkpoints" / "best.pth"
    if not best_ckpt.exists():  # all epochs were equally bad — fall back to last
        best_ckpt = last_ckpt

    # ---- final test eval on the best checkpoint ----
    import torch
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

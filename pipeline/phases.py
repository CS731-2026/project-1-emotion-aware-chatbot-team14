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

import logging
import random
from pathlib import Path

from . import dataset_ingest as ingest
from . import keys as K
from .context import Context
from .dataset_spec import DatasetSpec

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

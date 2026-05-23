"""kash dataset — team-collected images on local disk.

Source of truth: Notebooks/7_kash_dataset_prep.ipynb.

This dataset isn't on Kaggle — it's a local folder the team collects
into. Point KASH_DATASET_DIR at the root before running, or place the
images at the default path (output/data/kash/raw/<class>/*.jpg).

Folder layout expected (matches the notebook's RAW_DIR):
    <root>/
        neutral/     img001.jpg  img002.jpg  ...
        trust_relief/  ...
        sad/         (alias for sadness — see KASH_FOLDER_REMAP)
        confused/    (alias for confusion)
        disgust/     (alias for distrust)
        ...

The notebook also performs face-detection + blur filtering during
prep (cells 9-12). Both are opt-in via env vars:
    KASH_FACE_CROP=1       run the shared YOLO detector + crop
    KASH_BLUR_FILTER=1     reject by Laplacian variance < KASH_BLUR_THR
    KASH_BLUR_THR=80.0     blur threshold (default = notebook 7)
When either is set, prepare() routes through pipeline/datasets/kash/
quality_filter.py and writes the filtered crops under
output/data/kash/crops/ before the split.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pipeline import ingest

from . import quality_filter

logger = logging.getLogger(__name__)


NAME = "kash"

CLASS_NAMES = ["neutral", "trust_relief", "sadness", "fear_anxiety",
               "confusion", "distrust"]

# Folder-name → target-class-name (extends the FOLDER_TO_LABEL dict
# from notebook cell 4 with the EmpathBot string names instead of
# integer indices). Aliases keep prep robust across slightly-different
# folder naming conventions from different collection sessions.
KASH_FOLDER_REMAP = {
    "neutral":       "neutral",
    "trust_relief":  "trust_relief",
    "happy":         "trust_relief",   # alias
    "happiness":     "trust_relief",   # alias
    "sadness":       "sadness",
    "sad":           "sadness",        # alias
    "fear_anxiety":  "fear_anxiety",
    "fear":          "fear_anxiety",   # alias
    "confusion":     "confusion",
    "confused":      "confusion",      # alias
    "distrust":      "distrust",
    "disgust":       "distrust",       # alias
    "anger":         "distrust",       # alias — notebook collapses anger into distrust
    "angry":         "distrust",       # alias
}


def _default_raw_dir() -> Path:
    return Path("output/data") / NAME / "raw"


def prepare(ctx) -> "ingest.DatasetSpec":  # noqa: F821
    """Walk a local imagefolder of team-collected face images, remap
    folder names to EmpathBot classes, split, persist a DatasetSpec.

    Source dir resolution:
      1. KASH_DATASET_DIR env var (absolute path)
      2. output/data/kash/raw/ (the gitignored default)

    Raises FileNotFoundError with both candidate paths in the message
    if neither resolves — surfaces what to set without grep.
    """
    cache_dir  = Path("output/data") / NAME
    source_dir = cache_dir / "source"
    raw_dir = Path(os.environ.get("KASH_DATASET_DIR") or _default_raw_dir())

    cached = ingest.try_load_cached(cache_dir, source_dir)
    if cached is not None:
        return cached

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"kash dataset not found. Looked for:\n"
            f"  1. KASH_DATASET_DIR env var (unset)\n"
            f"  2. {raw_dir} (default)\n"
            "Either set KASH_DATASET_DIR=<path/to/kash_dataset> or "
            "place the team's collected images there."
        )

    # Materialise into source/{train,test}/<class>/*.jpg under our cache
    # tree by symlinking the user's raw_dir as source/train (test split
    # is carved from train below at val_fraction time — kash doesn't
    # ship a separate test split).
    source_dir.mkdir(parents=True, exist_ok=True)
    train_link = source_dir / "train"
    if not train_link.exists():
        train_link.symlink_to(raw_dir.resolve(), target_is_directory=True)

    # The shared finalize_dataset helper expects both train/ and test/
    # subdirs. Until kash has its own test set, mirror train as test —
    # finalize_dataset will scan + remap + carve val from train and
    # treat the mirrored test the same way.
    test_link = source_dir / "test"
    if not test_link.exists():
        test_link.symlink_to(raw_dir.resolve(), target_is_directory=True)

    do_filter = (
        os.environ.get("KASH_FACE_CROP", "0").lower() in {"1", "true", "yes"}
        or os.environ.get("KASH_BLUR_FILTER", "0").lower() in {"1", "true", "yes"}
    )
    if do_filter:
        return _prepare_filtered(cache_dir, source_dir)

    return ingest.finalize_dataset(
        name=NAME,
        cache_dir=cache_dir,
        source_dir=source_dir,
        class_names=CLASS_NAMES,
        label_remap=KASH_FOLDER_REMAP,
        val_fraction=0.20,
        val_seed=42,
    )


def _prepare_filtered(cache_dir: Path, source_dir: Path):
    """Path taken when KASH_FACE_CROP or KASH_BLUR_FILTER is set.

    Scans source/train (the symlink to raw_dir), remaps folder names to
    EmpathBot labels, pushes the resulting (path, label) frame through
    quality_filter.filter_dataset (which writes crops under
    cache_dir/crops/), then carves train/val splits the same way the
    default path does. test = train (notebook 7 doesn't carve a test
    set — all images are train with a val_sample flag).
    """
    from pipeline.framework.specs import DatasetSpec

    train_raw = ingest.scan_imagefolder(source_dir / "train")
    train_remapped = ingest.apply_remap(train_raw, KASH_FOLDER_REMAP, CLASS_NAMES)
    logger.info("kash: %d raw images after remap", len(train_remapped))

    filtered = quality_filter.filter_dataset(
        train_remapped, cache_dir / "crops", NAME,
    )
    if len(filtered) < 10:
        raise RuntimeError(
            f"kash: only {len(filtered)} images passed the quality filter — "
            "loosen KASH_BLUR_THR or disable KASH_FACE_CROP."
        )

    filtered_pl = filtered[["path", "label"]].reset_index(drop=True)
    train_df, val_df = ingest.carve_val(filtered_pl, 0.20, 42)
    test_df = filtered_pl

    splits = ingest.write_split_csvs(
        {"train": train_df, "val": val_df, "test": test_df}, cache_dir,
    )
    weights = ingest.compute_class_weights(
        train_df["label"], num_classes=len(CLASS_NAMES),
    )
    spec = DatasetSpec(
        name=NAME, cache_dir=cache_dir, splits=splits,
        num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES,
        class_weights=weights,
        source_md5=ingest.md5_of_dir(source_dir),
    )
    (cache_dir / "manifest.json").write_text(json.dumps(spec.to_manifest(), indent=2))
    logger.info("kash (filtered) ready — train=%d val=%d test=%d",
                len(train_df), len(val_df), len(test_df))
    return spec

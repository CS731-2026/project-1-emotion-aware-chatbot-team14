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
prep. We skip those in the initial port — run face_cropper.py
separately if you want cropped faces, or extend this prepare() to
call ingest.face_crop_imagefolder later.
"""

from __future__ import annotations

import os
from pathlib import Path

from pipeline import ingest


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

    return ingest.finalize_dataset(
        name=NAME,
        cache_dir=cache_dir,
        source_dir=source_dir,
        class_names=CLASS_NAMES,
        label_remap=KASH_FOLDER_REMAP,
        val_fraction=0.20,
        val_seed=42,
    )

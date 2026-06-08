"""AffectNet-HQ loader — folder layout: <root>/<int_label>/*.jpg

Source label scheme (8-class, from notebook 1 cell 7):
  0=Neutral 1=Happy 2=Sad 3=Surprise 4=Fear 5=Disgust 6=Anger 7=Contempt

Remapped to EmpathBot 6-class via AFFECTNET_MAP (Disgust + Anger +
Contempt all → distrust, since EmpathBot has no separate "anger" class).

If EMPATH_AFFECTNET_DIR is set, the loader scans that directory. If
unset, it falls back to a cached download at output/data/empath/raw/affectnet/
populated from HuggingFace (Piro17/affectnethq).

The download path **always face-crops** each HF sample in-stream via the
YOLO face detector — there is no raw-image stage. Saved JPEGs are
224×224 face crops at quality 88. Images that fail the face filter
(no face / face too small / multiple faces) are skipped. Net on-disk
footprint after materialization: ~330 MB (vs ~4.4 GB without cropping).

Stratified split: AffectNet ships its own train/val/test partition in
the original release; we follow the notebook's assign_splits_affectnet
which is just a stratified random carve at 80/10/10.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# notebook cell 7 — AFFECTNET_MAP verbatim
AFFECTNET_MAP = {
    0: 0,   # Neutral   → neutral
    1: 1,   # Happy     → trust_relief
    2: 2,   # Sad       → sadness
    3: 4,   # Surprise  → confusion
    4: 3,   # Fear      → fear_anxiety
    5: 5,   # Disgust   → distrust
    6: 5,   # Anger     → distrust
    7: 5,   # Contempt  → distrust
}

_IMG_EXTS = {".png", ".jpg", ".jpeg"}

_DEFAULT_CACHE_DIR = Path("output/data/empath/raw/affectnet")
_HF_SLUG = "Piro17/affectnethq"


def _has_cached_images(root: Path) -> bool:
    """True iff `root` has at least one <int_label>/*.{jpg,png} file."""
    if not root.is_dir():
        return False
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        try:
            int(class_dir.name)
        except ValueError:
            continue
        for img in class_dir.iterdir():
            if img.suffix.lower() in _IMG_EXTS:
                return True
    return False


def _download_to(dest: Path) -> None:
    """Pull AffectNet-HQ from HuggingFace and face-crop in-stream to
    <dest>/<int_label>/<idx:06d>.jpg at 224×224.

    Idempotent on rerun: skips files that already exist on disk. The
    HF parquet cache is purged at the end (set EMPATH_KEEP_HF_CACHE=1
    to keep it for iteration).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "downloading AffectNet-HQ requires the `datasets` package. "
            "Install with `pip install datasets` (already in "
            "pipeline/requirements.txt — run `make install-training`)."
        ) from e
    from collections import Counter
    from PIL import Image

    from . import _hf_cache, face_crop

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("affectnet: downloading %s from HuggingFace → %s "
                "(face-cropping in-stream, output ~330 MB)", _HF_SLUG, dest)
    ds = load_dataset(_HF_SLUG, split="train")

    yolo, device = face_crop.load_yolo()

    saved = skipped_exist = 0
    skip_reasons: Counter[str] = Counter()
    for idx, sample in enumerate(ds):
        label = int(sample["label"])
        out = dest / str(label) / f"{idx:06d}.jpg"
        if out.exists():
            skipped_exist += 1
            continue
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        crop, reason = face_crop.crop_pil(img, yolo, device)
        if crop is None:
            skip_reasons[reason] += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out, "JPEG", quality=88)
        saved += 1
    logger.info("affectnet: download complete — %d saved (face-cropped), "
                "%d already on disk, %d filtered out (%s)",
                saved, skipped_exist, sum(skip_reasons.values()),
                dict(skip_reasons) or "none")
    _hf_cache.purge_hf_dataset_cache(_HF_SLUG)


def load() -> pd.DataFrame:
    """Return df[path, label, split] for AffectNet-HQ.

    Source resolution order:
      1. $EMPATH_AFFECTNET_DIR if set (must be a directory)
      2. output/data/empath/raw/affectnet/ — auto-downloaded from
         HuggingFace if missing
    """
    root = os.environ.get("EMPATH_AFFECTNET_DIR")
    if root:
        root_p = Path(root)
        if not root_p.is_dir():
            raise FileNotFoundError(f"EMPATH_AFFECTNET_DIR={root} is not a directory")
    else:
        root_p = _DEFAULT_CACHE_DIR
        if not _has_cached_images(root_p):
            _download_to(root_p)

    rows = []
    for class_dir in sorted(p for p in root_p.iterdir() if p.is_dir()):
        try:
            src_label = int(class_dir.name)
        except ValueError:
            continue
        if src_label not in AFFECTNET_MAP:
            continue
        target = AFFECTNET_MAP[src_label]
        for img in class_dir.iterdir():
            if img.suffix.lower() in _IMG_EXTS:
                rows.append({"path": str(img.resolve()), "label": target})

    if not rows:
        raise FileNotFoundError(
            f"no images found under {root} (expected <root>/<int>/*.{{png,jpg}})"
        )

    df = pd.DataFrame(rows)
    # 80 / 10 / 10 stratified random carve (notebook's
    # assign_splits_affectnet, simplified). Stratify by label so each
    # split sees every class proportionally.
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    splits = []
    for _, group in df.groupby("label"):
        n = len(group)
        n_test = int(n * 0.10)
        n_val = int(n * 0.10)
        sub = group.copy()
        sub["split"] = "train"
        sub.iloc[:n_test, sub.columns.get_loc("split")] = "test"
        sub.iloc[n_test:n_test + n_val, sub.columns.get_loc("split")] = "val"
        splits.append(sub)
    return pd.concat(splits, ignore_index=True)

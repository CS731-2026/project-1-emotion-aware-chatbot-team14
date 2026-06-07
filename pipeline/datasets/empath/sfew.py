"""SFEW loader — evaluation-only sub-dataset.

The notebook treats SFEW as eval-only (cell 19-21): every SFEW image
gets split='test'. Its labels follow the same 1-indexed convention as
RAF-DB, so we reuse RAFDB_MAP.

Expected layout (one subfolder per int label):
    $EMPATH_SFEW_DIR/
        1/*.png    (Surprise)
        2/*.png    (Fear)
        ...

If EMPATH_SFEW_DIR is unset, the loader falls back to a cached download
at output/data/empath/raw/sfew/ populated from HuggingFace. SFEW has no
canonical HF mirror, so we try a couple of community uploads in order;
if none work, the empty cache surfaces a FileNotFoundError and the
empath merge skips SFEW gracefully (it's eval-only — training is
unaffected). Notebook 2 cell `cell-dl-sfew` for the same fallback pattern.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from .rafdb import RAFDB_MAP

logger = logging.getLogger(__name__)

# Same label scheme as RAF-DB — kept as a separate alias for grep-ability.
SFEW_MAP = RAFDB_MAP

_IMG_EXTS = {".png", ".jpg", ".jpeg"}

_DEFAULT_CACHE_DIR = Path("output/data/empath/raw/sfew")

# HF SFEW mirrors expose labels as a ClassLabel with class-name strings
# (e.g. "angry", "happy"). We map name → RAF-DB 1-indexed label so the
# materialized folder layout matches the scan code below and SFEW_MAP /
# RAFDB_MAP work unchanged.
_NAME_TO_RAFDB = {
    "angry":     6, "anger":     6,
    "disgust":   3,
    "fear":      2,
    "happy":     4, "happiness": 4,
    "neutral":   7,
    "sad":       5, "sadness":   5,
    "surprise":  1,
}

# Tried in order; first one that loads wins. Most SFEW mirrors get
# unpublished from HF over time (the notebook's original candidates are
# both gone), so SFEW often falls through to "skip gracefully". That's
# fine — SFEW is eval-only and training continues without it.
_HF_CANDIDATES = [
    "Jayanth2002/SFEW-Dataset",
    "soerendip/sfew",
    "chrysnthmm/sfewa3",
    "niuzi1/SFEWQ34",
]


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


def _download_to(dest: Path) -> bool:
    """Try each HF candidate; on first success, materialize images to
    <dest>/<int_label>/<idx:06d>.jpg and return True. Returns False if
    no candidate loads (the empath merge will then skip SFEW)."""
    try:
        from datasets import DatasetDict, IterableDatasetDict, load_dataset
    except ImportError:
        logger.warning("sfew: `datasets` not installed — skipping HF fallback")
        return False
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    ds = None
    chosen_slug = None
    for slug in _HF_CANDIDATES:
        try:
            logger.info("sfew: trying HuggingFace slug %s", slug)
            ds = load_dataset(slug)
            chosen_slug = slug
            break
        except Exception as e:  # noqa: BLE001 — HF surfaces a wide error variety
            logger.info("sfew: %s failed (%s)", slug, e)

    if ds is None:
        logger.warning(
            "sfew: no HF candidate loaded — SFEW will be skipped. "
            "Set EMPATH_SFEW_DIR to a local copy if you need it.")
        return False

    # Pick a usable split (val/test preferred; otherwise the first).
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        split = ds.get("val") or ds.get("test") or next(iter(ds.values()))
    else:
        split = ds

    label_feat = split.features.get("label") if split.features else None
    names = getattr(label_feat, "names", None)
    if not names:
        logger.warning("sfew: %s has no label names — cannot map to RAF-DB "
                       "scheme; skipping", chosen_slug)
        return False

    saved = skipped = unmapped = 0
    for idx, sample in enumerate(split):
        name = names[int(sample["label"])].lower()
        rafdb_label = _NAME_TO_RAFDB.get(name)
        if rafdb_label is None:
            unmapped += 1
            continue
        out = dest / str(rafdb_label) / f"{idx:06d}.jpg"
        if out.exists():
            skipped += 1
            continue
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(out, "JPEG", quality=95)
        saved += 1

    logger.info("sfew: download complete from %s — %d saved, %d already "
                "present, %d unmapped labels skipped",
                chosen_slug, saved, skipped, unmapped)
    return True


def load() -> pd.DataFrame:
    """Return df[path, label, split] for SFEW (all rows split='test').

    Source resolution order:
      1. $EMPATH_SFEW_DIR if set (must be a directory)
      2. output/data/empath/raw/sfew/ — auto-downloaded from HuggingFace
         if missing; failure to download surfaces as FileNotFoundError
         and the empath merge skips SFEW gracefully (eval-only).
    """
    root = os.environ.get("EMPATH_SFEW_DIR")
    if root:
        root_p = Path(root)
        if not root_p.is_dir():
            raise FileNotFoundError(f"EMPATH_SFEW_DIR={root} is not a directory")
    else:
        root_p = _DEFAULT_CACHE_DIR
        if not _has_cached_images(root_p):
            _download_to(root_p)
        if not _has_cached_images(root_p):
            raise FileNotFoundError(
                f"SFEW unavailable — no HF mirror loaded and {root_p} is empty"
            )

    rows = []
    for class_dir in sorted(p for p in root_p.iterdir() if p.is_dir()):
        try:
            src_label = int(class_dir.name)
        except ValueError:
            continue
        if src_label not in SFEW_MAP:
            continue
        target = SFEW_MAP[src_label]
        for img in class_dir.iterdir():
            if img.suffix.lower() in _IMG_EXTS:
                rows.append({
                    "path": str(img.resolve()),
                    "label": target,
                    "split": "test",   # SFEW is eval-only
                })

    if not rows:
        raise FileNotFoundError(
            f"no SFEW images found under {root} (expected <root>/<int>/*.{{png,jpg}})"
        )
    return pd.DataFrame(rows)

"""RAF-DB loader.

RAF-DB ships a list_patition_label.txt at the dataset root with:
    train_00001.jpg 4
    test_00001.jpg  1
    ...
Image filenames carry the train/test prefix; the integer is the
1-indexed RAF-DB label (notebook cell 7).

Source label scheme:
  1=Surprise 2=Fear 3=Disgust 4=Happy 5=Sad 6=Angry 7=Neutral

Remapped to EmpathBot 6-class via RAFDB_MAP.

If EMPATH_RAFDB_DIR is set, the loader scans that directory. If unset,
it falls back to a cached download at output/data/empath/raw/rafdb/
populated from HuggingFace (~20k images).

The download path **always face-crops** each HF sample in-stream via
the YOLO face detector — no raw-image stage. Output is 224×224 face
crops at JPEG quality 88, materialized under Image/aligned/ so the
existing list_patition_label.txt loader works unchanged. Images that
fail the face filter are skipped. The HF version is train-only — there's
no separate official test split in that mirror — so all surviving rows
start as `train` and the loader carves a 10% val off train as usual.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# notebook cell 7 — RAFDB_MAP verbatim
RAFDB_MAP = {
    1: 4,   # Surprise → confusion
    2: 3,   # Fear     → fear_anxiety
    3: 5,   # Disgust  → distrust
    4: 1,   # Happy    → trust_relief
    5: 2,   # Sad      → sadness
    6: 5,   # Angry    → distrust
    7: 0,   # Neutral  → neutral
}

_DEFAULT_CACHE_DIR = Path("output/data/empath/raw/rafdb")

# Tried in order; first one that loads wins. Community mirrors get
# unpublished from HF periodically (the notebook's original
# soerendip/raf-db-7emotions is gone), so keep a couple of fallbacks.
_HF_CANDIDATES = [
    "deanngkl/raf-db-7emotions",
    "rhavill/raf-db-7emotions",
]

# HuggingFace 0-indexed labels (from notebook 1 cell 13's HF_RAF_LABEL_NAME)
# mapped to the official RAF-DB 1-indexed scheme so the materialized
# list_patition_label.txt matches what the scan logic below expects.
_HF_TO_RAFDB = {
    0: 6,   # anger     → 6 Angry
    1: 3,   # disgust   → 3 Disgust
    2: 2,   # fear      → 2 Fear
    3: 4,   # happiness → 4 Happy
    4: 7,   # neutral   → 7 Neutral
    5: 5,   # sadness   → 5 Sad
    6: 1,   # surprise  → 1 Surprise
}


def _download_to(dest: Path) -> None:
    """Pull RAF-DB from HuggingFace and materialize into the official
    layout (Image/aligned/train_NNNNNN.jpg + list_patition_label.txt
    with 1-indexed labels) so the scan code below works unchanged.

    Idempotent: skips images that already exist, rewrites the label
    file each time (cheap; ~20k lines)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "downloading RAF-DB requires the `datasets` package. "
            "Install with `pip install datasets` (already in "
            "pipeline/requirements.txt — run `make install-training`)."
        ) from e
    from PIL import Image

    aligned_dir = dest / "Image" / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    ds = None
    chosen_slug = None
    for slug in _HF_CANDIDATES:
        try:
            logger.info("rafdb: trying HuggingFace slug %s", slug)
            ds = load_dataset(slug, split="train")
            chosen_slug = slug
            break
        except Exception as e:  # noqa: BLE001 — HF surfaces a wide error variety
            logger.info("rafdb: %s failed (%s)", slug, e)

    if ds is None:
        raise RuntimeError(
            f"rafdb: no HF candidate loaded (tried {_HF_CANDIDATES}). "
            f"Set EMPATH_RAFDB_DIR to a local copy with list_patition_label.txt."
        )
    logger.info("rafdb: materializing %s → %s (face-cropping in-stream)",
                chosen_slug, dest)

    from collections import Counter
    from . import _hf_cache, face_crop
    yolo, device = face_crop.load_yolo()

    label_lines: list[str] = []
    saved = skipped = 0
    skip_reasons: Counter[str] = Counter()
    for idx, sample in enumerate(ds):
        hf_label = int(sample["label"])
        rafdb_label = _HF_TO_RAFDB[hf_label]
        filename = f"train_{idx:06d}.jpg"
        out = aligned_dir / filename
        if out.exists():
            label_lines.append(f"{filename} {rafdb_label}")
            skipped += 1
            continue
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        crop, reason = face_crop.crop_pil(img, yolo, device)
        if crop is None:
            skip_reasons[reason] += 1
            continue
        crop.save(out, "JPEG", quality=88)
        label_lines.append(f"{filename} {rafdb_label}")
        saved += 1

    (dest / "list_patition_label.txt").write_text("\n".join(label_lines) + "\n")
    logger.info("rafdb: download complete — %d saved (face-cropped), "
                "%d already on disk, %d filtered out (%s)",
                saved, skipped, sum(skip_reasons.values()),
                dict(skip_reasons) or "none")
    _hf_cache.purge_hf_dataset_cache(chosen_slug)


def load() -> pd.DataFrame:
    """Return df[path, label, split] for RAF-DB.

    Source resolution order:
      1. $EMPATH_RAFDB_DIR if set (must contain list_patition_label.txt)
      2. output/data/empath/raw/rafdb/ — auto-downloaded from
         HuggingFace if missing
    """
    root = os.environ.get("EMPATH_RAFDB_DIR")
    if root:
        root_p = Path(root)
    else:
        root_p = _DEFAULT_CACHE_DIR
        if not (root_p / "list_patition_label.txt").exists():
            _download_to(root_p)

    label_file = root_p / "list_patition_label.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"{label_file} not found")

    # The notebook's resolver tries both raw and aligned image folders;
    # use whichever exists. Aligned (face-cropped) is preferred when
    # both are present.
    image_root_candidates = [
        root_p / "Image" / "aligned",
        root_p / "Image" / "original",
        root_p / "aligned",
        root_p,
    ]
    image_root = next((c for c in image_root_candidates if c.is_dir()), None)
    if image_root is None:
        raise FileNotFoundError(
            f"no image dir under {root_p} (tried Image/aligned, Image/original, ...)"
        )

    rows = []
    for line in label_file.read_text().strip().split("\n"):
        parts = line.split()
        if len(parts) < 2:
            continue
        filename, src_label_str = parts[0], parts[1]
        src_label = int(src_label_str)
        if src_label not in RAFDB_MAP:
            continue

        # When aligned, RAF-DB conventionally renames *.jpg → *_aligned.jpg.
        # Try both filenames.
        candidates = [image_root / filename]
        if "_aligned" not in filename:
            stem = Path(filename).stem
            candidates.append(image_root / f"{stem}_aligned.jpg")
        img_path = next((c for c in candidates if c.exists()), None)
        if img_path is None:
            continue

        # RAF-DB encodes train/test in the filename prefix.
        split = "train" if filename.startswith("train") else "test"
        rows.append({
            "path": str(img_path.resolve()),
            "label": RAFDB_MAP[src_label],
            "split": split,
        })

    if not rows:
        raise FileNotFoundError(f"no usable RAF-DB rows found in {label_file}")

    df = pd.DataFrame(rows)
    # Carve a 10% stratified val off the train rows (notebook
    # assign_splits_rafdb).
    train_df = df[df["split"] == "train"]
    val_parts = []
    for _, group in train_df.groupby("label"):
        shuffled = group.sample(frac=1.0, random_state=42)
        n_val = max(1, int(len(shuffled) * 0.10))
        val_parts.append(shuffled.iloc[:n_val])
    val_indices = pd.concat(val_parts).index
    df.loc[val_indices, "split"] = "val"
    return df.reset_index(drop=True)

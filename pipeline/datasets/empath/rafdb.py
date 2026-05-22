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

The notebook carves a val split off RAF-DB train at 10% stratified;
test is the official RAF-DB test set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


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


def load() -> pd.DataFrame:
    """Return df[path, label, split] for RAF-DB.

    Expects:
        $EMPATH_RAFDB_DIR/
            list_patition_label.txt
            Image/aligned/*.jpg     (or wherever cropped images live)
    """
    root = os.environ.get("EMPATH_RAFDB_DIR")
    if not root:
        raise FileNotFoundError("EMPATH_RAFDB_DIR not set")
    root_p = Path(root)
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

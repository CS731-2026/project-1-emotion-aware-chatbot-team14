"""AffectNet-HQ loader — folder layout: <root>/<int_label>/*.jpg

Source label scheme (8-class, from notebook 1 cell 7):
  0=Neutral 1=Happy 2=Sad 3=Surprise 4=Fear 5=Disgust 6=Anger 7=Contempt

Remapped to EmpathBot 6-class via AFFECTNET_MAP (Disgust + Anger +
Contempt all → distrust, since EmpathBot has no separate "anger" class).

The notebook expects a face-cropped copy of AffectNet-HQ; we trust
EMPATH_AFFECTNET_DIR points at one. Run face_cropper.py against the
raw AffectNet-HQ Kaggle download first if you don't have crops.

Stratified split: AffectNet ships its own train/val/test partition in
the original release; we follow the notebook's assign_splits_affectnet
which is just a stratified random carve at 80/10/10.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


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


def load() -> pd.DataFrame:
    """Return df[path, label, split] for AffectNet-HQ.

    Raises FileNotFoundError if EMPATH_AFFECTNET_DIR is unset or empty.
    """
    root = os.environ.get("EMPATH_AFFECTNET_DIR")
    if not root:
        raise FileNotFoundError("EMPATH_AFFECTNET_DIR not set")
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"EMPATH_AFFECTNET_DIR={root} is not a directory")

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

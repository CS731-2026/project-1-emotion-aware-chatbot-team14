"""SFEW loader — evaluation-only sub-dataset.

The notebook treats SFEW as eval-only (cell 19-21): every SFEW image
gets split='test'. Its labels follow the same 1-indexed convention as
RAF-DB, so we reuse RAFDB_MAP.

Expected layout (one subfolder per int label):
    $EMPATH_SFEW_DIR/
        1/*.png    (Surprise)
        2/*.png    (Fear)
        ...
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .rafdb import RAFDB_MAP

# Same label scheme as RAF-DB — kept as a separate alias for grep-ability.
SFEW_MAP = RAFDB_MAP

_IMG_EXTS = {".png", ".jpg", ".jpeg"}


def load() -> pd.DataFrame:
    """Return df[path, label, split] for SFEW (all rows split='test')."""
    root = os.environ.get("EMPATH_SFEW_DIR")
    if not root:
        raise FileNotFoundError("EMPATH_SFEW_DIR not set")
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"EMPATH_SFEW_DIR={root} is not a directory")

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

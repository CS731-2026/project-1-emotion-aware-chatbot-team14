"""FER2013 from Kaggle (msambare/fer2013).

~35k 48×48 grayscale faces, 7 source classes. We remap to the
EmpathBot 6-class schema and drop disgust (~547 examples; overlaps
semantically with distrust).
"""

from __future__ import annotations

from pathlib import Path

from pipeline import ingest


NAME = "fer2013"

CLASS_NAMES = ["neutral", "trust_relief", "sadness", "fear_anxiety",
               "confusion", "distrust"]

# Source 7-class label → target class (or __drop__ to discard the row).
_REMAP = {
    "angry":    "distrust",
    "disgust":  "__drop__",
    "fear":     "fear_anxiety",
    "happy":    "trust_relief",
    "sad":      "sadness",
    "surprise": "confusion",
    "neutral":  "neutral",
}


def prepare(ctx) -> "ingest.DatasetSpec":  # noqa: F821 (forward-ref via ingest)
    cache_dir  = Path("output/data") / NAME
    source_dir = cache_dir / "source"

    cached = ingest.try_load_cached(cache_dir, source_dir)
    if cached is not None:
        return cached

    ingest.download_kaggle("msambare/fer2013", source_dir)
    return ingest.finalize_dataset(
        name=NAME,
        cache_dir=cache_dir,
        source_dir=source_dir,
        class_names=CLASS_NAMES,
        label_remap=_REMAP,
    )

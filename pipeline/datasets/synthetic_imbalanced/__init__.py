"""Synthetic imbalanced dataset — same shape as synthetic_smoke but
with heavily skewed class counts so we can validate class_weights:
auto actually moves the loss in the right direction. Useful when
tuning the imbalanced-data path before pointing it at real datasets
where the imbalance is structural (e.g. FER2013's disgust class).
"""

from __future__ import annotations

from pathlib import Path

from pipeline import ingest


NAME = "synthetic_imbalanced"

CLASS_NAMES = ["neutral", "trust_relief", "sadness", "fear_anxiety",
               "confusion", "distrust"]

# Per-class train counts — neutral dominates, distrust is rare.
_TRAIN_SAMPLES = {
    "neutral":      200,
    "trust_relief": 80,
    "sadness":      80,
    "fear_anxiety": 40,
    "confusion":    40,
    "distrust":     20,
}


def prepare(ctx) -> "ingest.DatasetSpec":  # noqa: F821
    cache_dir  = Path("output/data") / NAME
    source_dir = cache_dir / "source"

    cached = ingest.try_load_cached(cache_dir, source_dir)
    if cached is not None:
        return cached

    ingest.generate_synthetic(source_dir / "train",
                              class_names=CLASS_NAMES,
                              samples_per_class=_TRAIN_SAMPLES,
                              seed=7, image_size=32)
    # Balanced test set so per-class metrics aren't hidden by the imbalance.
    ingest.generate_synthetic(source_dir / "test",
                              class_names=CLASS_NAMES,
                              samples_per_class=20, seed=8, image_size=32)

    return ingest.finalize_dataset(
        name=NAME,
        cache_dir=cache_dir,
        source_dir=source_dir,
        class_names=CLASS_NAMES,
    )

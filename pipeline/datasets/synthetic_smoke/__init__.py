"""Synthetic smoke dataset — balanced, network-free. Prep + train
finish in seconds; the "did I break the pipeline?" sanity check.
"""

from __future__ import annotations

from pathlib import Path

from pipeline import ingest


NAME = "synthetic_smoke"

CLASS_NAMES = ["neutral", "trust_relief", "sadness", "fear_anxiety",
               "confusion", "distrust"]


def prepare(ctx) -> "ingest.DatasetSpec":  # noqa: F821
    cache_dir  = Path("output/data") / NAME
    source_dir = cache_dir / "source"

    cached = ingest.try_load_cached(cache_dir, source_dir)
    if cached is not None:
        return cached

    # Generate train + test imagefolders; deterministic seeds.
    ingest.generate_synthetic(source_dir / "train",
                              class_names=CLASS_NAMES,
                              samples_per_class=60, seed=42, image_size=32)
    ingest.generate_synthetic(source_dir / "test",
                              class_names=CLASS_NAMES,
                              samples_per_class=20, seed=43, image_size=32)

    return ingest.finalize_dataset(
        name=NAME,
        cache_dir=cache_dir,
        source_dir=source_dir,
        class_names=CLASS_NAMES,
        # No remap — synthetic generator already writes one subdir per class_name.
    )

"""tutorial — heavily commented reference dataset module.

A dataset module lands 3 CSVs (`path,label`) on disk + returns a
DatasetSpec pointing at them. Where the data comes from + how labels
get remapped + how splits are carved is your call.

═══════════════════════════════════════════════════════════════════════
DATASET CHEATSHEET — every helper you have
═══════════════════════════════════════════════════════════════════════

ctx — same surface as the model side (see pipeline/models/tutorial/
      train_loop.py for the full list). Mostly unused in prepare().

Return type — DatasetSpec (pipeline/framework/specs.py):
  name, cache_dir, splits["train"|"val"|"test"] → Path,
  num_classes, class_names, class_weights, source_md5

pipeline.ingest helpers:
  generate_synthetic(dest, class_names, samples_per_class, seed, image_size)
                                       fake imagefolder, no network
  download_kaggle(slug, dest)          kaggle CLI extract; cache-aware
  scan_imagefolder(folder)             DataFrame[path, src_label] from <folder>/<class>/*
  apply_remap(df, remap, class_names)  DataFrame[path, int_label]; drops "__drop__"
  carve_val(train_df, val_fraction, seed)  (train, val) split
  compute_class_weights(labels, num_classes)  inverse-frequency, mean 1.0
  write_split_csvs({"train":df, "val":df, "test":df}, cache_dir)
  finalize_dataset(...)                ONE call that does scan + remap + split + weights +
                                       CSVs + manifest. Use for the common case.
  try_load_cached(cache_dir, source_dir)  cached spec if source unchanged
  md5_of_dir(root)                     stable hash of file listing
  DROP_SENTINEL                        "__drop__" — discard rows of a class

pipeline.kaggle  download_dataset, creds_present, dataset_exists (finer control)
═══════════════════════════════════════════════════════════════════════

THIS TUTORIAL — step by step:
  1. cache check (try_load_cached — always your first line)
  2. acquire data (generate_synthetic / switch to download_kaggle for real)
  3. label remap (source → target int labels, with __drop__)
  4. split + class weights + CSVs (finalize_dataset one-shot)
  5. return DatasetSpec — train phase reads .splits["train|val|test"]
"""

from __future__ import annotations

from pathlib import Path

from pipeline import ingest


NAME = "tutorial"

# Target schema. Order = int label downstream + class_weights index.
CLASS_NAMES = ["happy", "sad", "neutral"]

# Source folder name → target class name (or DROP_SENTINEL to discard).
_REMAP = {
    "happy":     "happy",                  # 1:1
    "happiness": "happy",                  # alias
    "sad":       "sad",
    "sadness":   "sad",
    "neutral":   "neutral",
    "angry":     ingest.DROP_SENTINEL,     # discard
}


def prepare(ctx) -> "ingest.DatasetSpec":  # noqa: F821 — forward-ref via ingest
    cache_dir  = Path("output/data") / NAME
    source_dir = cache_dir / "source"

    # Cache hit when manifest + source md5 match — skip everything below.
    cached = ingest.try_load_cached(cache_dir, source_dir)
    if cached is not None:
        return cached

    # Generate fake imagefolder. Replace these two lines with
    # `ingest.download_kaggle("user/slug", source_dir)` for a real dataset
    # (auth: KAGGLE_USERNAME / KAGGLE_KEY in .env; see fer2013 module).
    ingest.generate_synthetic(source_dir / "train",
                               class_names=list(_REMAP),
                               samples_per_class=40, seed=42, image_size=32)
    ingest.generate_synthetic(source_dir / "test",
                               class_names=list(_REMAP),
                               samples_per_class=10, seed=43, image_size=32)

    # finalize_dataset = scan + remap + carve_val + class_weights + CSVs + manifest.
    # See pipeline/datasets/empath/__init__.py for the case where you skip
    # finalize_dataset and combine the lower-level helpers yourself.
    return ingest.finalize_dataset(
        name=NAME,
        cache_dir=cache_dir,
        source_dir=source_dir,
        class_names=CLASS_NAMES,
        label_remap=_REMAP,
        train_dir="train",          # subdir under source_dir
        test_dir="test",
        val_fraction=0.10,          # 10% of train → val
        val_seed=42,
        class_weights="auto",       # "auto" = inverse frequency; "uniform" = None
    )

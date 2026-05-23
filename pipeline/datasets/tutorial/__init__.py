"""tutorial — heavily commented reference dataset module.

A dataset module's only job is to land three CSVs (`path,label` rows)
on disk + return a DatasetSpec pointing at them. Everything else is
your call: where the data comes from (Kaggle / HF / synthetic /
hand-collected), how labels get remapped, how splits are carved.

═══════════════════════════════════════════════════════════════════════
DATASET CHEATSHEET — every helper you have available
═══════════════════════════════════════════════════════════════════════

ctx — Context handed to you by the driver (mostly unused in prepare()).
      Same surface as the model side (see pipeline/models/tutorial/
      train_loop.py for the full list). prepare() is called early in
      the pipeline so ctx.save_* mostly isn't relevant — the run's
      artifacts come from the train phase.

Return type — DatasetSpec (pipeline/framework/specs.py). Fields:
  name           str
  cache_dir      Path     (output/data/<name>/)
  splits         dict["train"|"val"|"test", Path]   the three CSVs
  num_classes    int
  class_names    list[str]
  class_weights  list[float] | None
  source_md5     str      (hash of the source tree, for cache-invalidation)

pipeline.ingest  helpers — your toolkit:
  generate_synthetic(dest, class_names, samples_per_class, seed, image_size)
                                       → fake imagefolder under dest, no network
  download_kaggle(slug, dest)          → kaggle CLI extract into dest
                                         (raises if creds missing; cache-aware)
  scan_imagefolder(folder)             → DataFrame[path, src_label] from <folder>/<class>/*
  apply_remap(df, remap, class_names)  → DataFrame[path, int_label]; drops "__drop__"
  carve_val(train_df, val_fraction, seed) → (train, val) split
  compute_class_weights(labels, num_classes) → list[float] inverse-frequency, mean 1.0
  write_split_csvs({"train":df, "val":df, "test":df}, cache_dir) → {split: Path}
  finalize_dataset(...)                → ONE call that does scan + remap + split + weights +
                                         CSVs + manifest. Use this for the common case.
  try_load_cached(cache_dir, source_dir) → returns cached spec if source unchanged
  md5_of_dir(root)                     → stable hash of file listing for cache invalidation

pipeline.kaggle  utilities (when you need finer control than download_kaggle):
  download_dataset, creds_present, dataset_exists

═══════════════════════════════════════════════════════════════════════

THIS TUTORIAL — what it does step by step:
  1. cache check (skip work if source hasn't changed)
  2. acquire data (generate_synthetic; switch to download_kaggle for real)
  3. label remap (source → target int labels, optionally dropping rows)
  4. split (train/val/test) via finalize_dataset
  5. return DatasetSpec — the train phase reads .splits["train|val|test"]
"""

from __future__ import annotations

from pathlib import Path

from pipeline import ingest


# ── module exports the driver looks for ─────────────────────────────────
NAME = "tutorial"

# The target schema your dataset will expose. Order matters — index in this
# list becomes the int label downstream. Class weights index the same way.
CLASS_NAMES = ["happy", "sad", "neutral"]

# Source-label → target-class mapping. Use "__drop__" to discard rows of a
# class you don't want in the final dataset. Source labels are strings the
# scanner reads from subfolder names; targets must be in CLASS_NAMES.
_REMAP = {
    "happy":     "happy",       # straight 1:1 mapping
    "happiness": "happy",       # alias the scanner might find
    "sad":       "sad",
    "sadness":   "sad",
    "neutral":   "neutral",
    "angry":     ingest.DROP_SENTINEL,  # drop entirely from the dataset
}


def prepare(ctx) -> "ingest.DatasetSpec":  # noqa: F821 — forward ref via ingest
    cache_dir  = Path("output/data") / NAME    # where CSVs + manifest land
    source_dir = cache_dir / "source"          # where raw imagefolder lives

    # ── 1. cache check ──
    # try_load_cached returns a DatasetSpec if both manifest.json AND a
    # source_dir of the same md5 exist. Skips ALL the work below on warm
    # re-runs. Always your first line in prepare().
    cached = ingest.try_load_cached(cache_dir, source_dir)
    if cached is not None:
        return cached

    # ── 2. acquire data ──
    # The synthetic generator writes <source_dir>/<class_name>/<idx>.png
    # with deterministic per-class colours so a model can actually learn
    # something. Different seeds for train/test so the imagefolders differ.
    ingest.generate_synthetic(source_dir / "train",
                               class_names=list(_REMAP.keys()),
                               samples_per_class=40, seed=42, image_size=32)
    ingest.generate_synthetic(source_dir / "test",
                               class_names=list(_REMAP.keys()),
                               samples_per_class=10, seed=43, image_size=32)

    # ── To use a real Kaggle dataset instead, replace the two
    # generate_synthetic calls above with one line:
    #
    #     ingest.download_kaggle("user/slug", source_dir)
    #
    # Auth via KAGGLE_USERNAME / KAGGLE_KEY in .env (see fer2013 for the
    # working example).
    # ────────────────────────────────────────────────────────────────────

    # ── 3-5. finalize ──
    # finalize_dataset does scan_imagefolder + apply_remap + carve_val +
    # compute_class_weights + write_split_csvs + manifest in one call.
    # For the common "imagefolder → CSVs" case this is all you need.
    # Reach for the lower-level helpers individually if you need to do
    # something unusual (see pipeline/datasets/empath/__init__.py for an
    # example that bypasses finalize_dataset to merge multiple sources).
    return ingest.finalize_dataset(
        name=NAME,
        cache_dir=cache_dir,
        source_dir=source_dir,
        class_names=CLASS_NAMES,
        label_remap=_REMAP,        # source-label-string → target-class-string
        train_dir="train",         # subdir under source_dir for train split
        test_dir="test",           # subdir under source_dir for test split
        val_fraction=0.10,         # carve 10% of train → val
        val_seed=42,
        class_weights="auto",      # "auto" = inverse frequency; "uniform" = None
    )

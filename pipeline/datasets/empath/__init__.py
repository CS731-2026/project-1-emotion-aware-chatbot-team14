"""EmpathBot canonical dataset — merge of AffectNet-HQ + RAF-DB + SFEW.

Source of truth: Notebooks/1_dataset_pipeline.ipynb. Three source
datasets, each with its own label scheme, all remapped onto the
EmpathBot 6-class schema and stratified together into train/val/test.

This package owns one prepare() function. The per-source loaders +
label remaps live in sibling files so each can be edited
independently when a source dataset's format drifts:

  empath/
    __init__.py    pipeline surface — NAME, CLASS_NAMES, prepare(ctx)
    affectnet.py   AffectNet-HQ loader + label remap (notebook cell 7-12)
    rafdb.py       RAF-DB loader + label remap (notebook cell 13-18)
    sfew.py        SFEW loader + label remap (notebook cell 19-21, eval-only)

Source data isn't bundled. Each loader resolves a directory in this
order:

  1. EMPATH_{AFFECTNET,RAFDB,SFEW}_DIR if set — use the team's
     pre-cropped local copy at that path
  2. Otherwise output/data/empath/raw/<source>/ (gitignored) —
     auto-downloaded from HuggingFace + **face-cropped in-stream**
     via the YOLO face detector. AffectNet and RAF-DB always succeed;
     SFEW depends on community mirrors and is skipped gracefully if
     none load (it's eval-only — training is unaffected).

Face cropping is always-on for the download path (matches the team's
notebook pipeline; gives training-distribution parity with the
hand-trained baselines under models/empathbot/).

The notebook also runs YOLO face detection + cropping (cells 24-26)
before merging. By default we skip that step — set EMPATH_FACE_CROP=1
to enable the in-line face-detection pre-pass (cached under
output/data/empath/crops/<source>/<label>/). Otherwise, run
face_cropper.py against the raw sources first and point the env vars
at the cropped outputs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from pipeline import ingest
from pipeline.framework.specs import DatasetSpec

from . import affectnet, face_crop, rafdb, sfew

logger = logging.getLogger(__name__)


NAME = "empath"

# EmpathBot 6-class schema (notebook cell 7).
# index 6 (fake_politeness) is synthetic / inference-only — never appears
# in training data, so it's not in CLASS_NAMES here.
CLASS_NAMES = ["neutral", "trust_relief", "sadness", "fear_anxiety",
               "confusion", "distrust"]


def prepare(ctx) -> DatasetSpec:
    """Resolve + remap each source dataset, concatenate, persist a
    DatasetSpec under output/data/empath/.

    Each source is optional — if EMPATH_AFFECTNET_DIR is unset, that
    source is skipped (with a log line) and the merge continues with
    what's available. Raises only if NO sources resolve.
    """
    cache_dir = Path("output/data") / NAME

    cached = ingest.try_load_cached(cache_dir, cache_dir / "merged")
    if cached is not None:
        return cached

    # Collect available sources. Each loader returns a DataFrame with
    # columns ['path', 'label', 'split'] where label is the EmpathBot
    # class index and split is one of {'train', 'val', 'test'}.
    #
    # Face-cropping happens **inside** each per-source loader's
    # _download_to() now — the YOLO face filter is mandatory and runs
    # in-stream during the HF download. The previous opt-in
    # EMPATH_FACE_CROP env var is gone (face crops are always-on,
    # because the team's hand-trained baselines were trained on
    # cropped data and we want training-distribution parity).
    parts: list[pd.DataFrame] = []
    for name, loader in [
        ("affectnet", affectnet.load),
        ("rafdb",     rafdb.load),
        ("sfew",      sfew.load),
    ]:
        try:
            df = loader()
            df["dataset"] = name
            parts.append(df)
            logger.info("empath: loaded %s — %d rows", name, len(df))
        except FileNotFoundError as e:
            logger.warning("empath: %s skipped (%s)", name, e)

    if not parts:
        raise FileNotFoundError(
            "no empath source datasets found. Set at least one of "
            "EMPATH_AFFECTNET_DIR / EMPATH_RAFDB_DIR / EMPATH_SFEW_DIR."
        )

    master = pd.concat(parts, ignore_index=True)
    logger.info("empath: merged %d rows across %d source(s)",
                len(master), len(parts))

    cache_dir.mkdir(parents=True, exist_ok=True)
    splits_out: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        sub = master[master["split"] == split_name][["path", "label"]]
        dest = cache_dir / f"{split_name}.csv"
        sub.to_csv(dest, index=False)
        splits_out[split_name] = dest

    # Stable md5 of the merged manifest (we don't have a single source_dir
    # since data lives across multiple env-var paths). Hash the
    # filename + label tuples for change detection on re-prep.
    merged_dir = cache_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    (merged_dir / "_inventory.txt").write_text(
        "\n".join(f"{r['path']}\t{r['label']}\t{r['split']}"
                  for r in master.to_dict("records"))
    )
    source_md5 = ingest.md5_of_dir(merged_dir)

    weights = ingest.compute_class_weights(
        master[master["split"] == "train"]["label"],
        num_classes=len(CLASS_NAMES),
    )

    spec = DatasetSpec(
        name=NAME,
        cache_dir=cache_dir,
        splits=splits_out,
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        class_weights=weights,
        source_md5=source_md5,
    )
    (cache_dir / "manifest.json").write_text(
        __import__("json").dumps(spec.to_manifest(), indent=2)
    )
    logger.info(
        "empath: ready — train=%d val=%d test=%d",
        len(master[master["split"] == "train"]),
        len(master[master["split"] == "val"]),
        len(master[master["split"] == "test"]),
    )
    return spec

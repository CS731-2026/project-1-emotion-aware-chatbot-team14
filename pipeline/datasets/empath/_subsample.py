"""Stratified subsampling of a prepared DatasetSpec.

Used by the empath_tiny / empath_small / etc. variant modules to expose
a smaller training split while leaving val/test untouched — so eval
numbers from a smoke-sized run are still comparable in shape to a full
run (just noisier).

Design notes:
  * Train-only subsample. Val/test pass through by reference (we just
    keep the parent's CSV paths). Subsampling val/test would defeat
    the purpose of comparable eval metrics across variants.
  * Stratified per class. Random fractional sampling can drop minority
    classes entirely from a tiny variant (empath has rare fear_anxiety);
    per-class N guarantees representation.
  * Subsample params hashed into source_md5 so cache invalidates if
    you change N. Otherwise the previous tiny variant's CSV would be
    silently reused.
  * Class weights recomputed from the subsampled train distribution.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from pipeline import ingest
from pipeline.framework.specs import DatasetSpec

logger = logging.getLogger(__name__)


def subsample_spec(
    parent: DatasetSpec,
    *,
    n_per_class: int,
    name: str,
    cache_dir: Path,
    seed: int = 42,
) -> DatasetSpec:
    """Return a new DatasetSpec whose train split is N rows per class
    drawn from `parent`'s train CSV. Val/test reference parent's CSVs
    unchanged.

    Idempotent: if cache_dir/manifest.json already exists with a
    matching subsample-params hash, reuse it. Re-running with a new N
    rebuilds.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Hash the (parent_md5, n_per_class, seed) tuple so the variant's
    # source_md5 changes when ANY of those change — cache reuse logic
    # downstream still works as expected.
    params_blob = json.dumps({
        "parent_md5":   parent.source_md5,
        "n_per_class":  n_per_class,
        "seed":         seed,
    }, sort_keys=True).encode()
    new_md5 = hashlib.md5(params_blob).hexdigest()

    train_csv_out = cache_dir / "train.csv"
    manifest_path = cache_dir / "manifest.json"

    # Cache hit: same params + manifest already on disk → reuse.
    if manifest_path.exists():
        try:
            cached = DatasetSpec.from_manifest(manifest_path)
            if cached.source_md5 == new_md5 and train_csv_out.exists():
                logger.info("%s: cache hit — %d rows / class from parent %s",
                            name, n_per_class, parent.name)
                return cached
        except Exception:  # noqa: BLE001
            pass  # rebuild on any deserialization issue

    # Read parent's train CSV and take N stratified rows per class.
    parent_train = pd.read_csv(parent.splits["train"])
    if "label" not in parent_train.columns:
        raise ValueError(
            f"{name}: parent train CSV {parent.splits['train']} lacks 'label' column"
        )

    sampled_parts: list[pd.DataFrame] = []
    actual_per_class: dict[int, int] = {}
    for label, group in parent_train.groupby("label"):
        take = min(n_per_class, len(group))
        sampled = group.sample(n=take, random_state=seed)
        sampled_parts.append(sampled)
        actual_per_class[int(str(label))] = int(take)

    subsampled = pd.concat(sampled_parts, ignore_index=True)
    subsampled = subsampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    subsampled.to_csv(train_csv_out, index=False)
    train_labels: pd.Series = subsampled["label"]  # type: ignore[assignment]

    # Val/test pass through — just reference parent's CSV paths so we
    # don't duplicate large files for every variant. Compare metrics
    # are still apples-to-apples across variants.
    splits_out = {
        "train": train_csv_out,
        "val":   parent.splits["val"],
        "test":  parent.splits["test"],
    }

    new_weights = ingest.compute_class_weights(
        train_labels, num_classes=parent.num_classes,
    )

    spec = DatasetSpec(
        name=name,
        cache_dir=cache_dir,
        splits=splits_out,
        num_classes=parent.num_classes,
        class_names=parent.class_names,
        class_weights=new_weights,
        source_md5=new_md5,
    )
    manifest_path.write_text(json.dumps(spec.to_manifest(), indent=2))
    logger.info(
        "%s: subsampled %d → %d train rows (%d/class target, actual: %s); "
        "val=%d test=%d (passthrough)",
        name, len(parent_train), len(subsampled), n_per_class,
        actual_per_class,
        _csv_len(parent.splits["val"]),
        _csv_len(parent.splits["test"]),
    )
    return spec


def _csv_len(path: Path) -> int:
    try:
        return len(pd.read_csv(path))
    except Exception:
        return -1

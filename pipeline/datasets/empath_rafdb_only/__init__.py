"""empath_rafdb_only — single-source variant of empath.

Uses only the RAF-DB source loader (20k images), skipping AffectNet.
Useful when:
  * AffectNet hasn't downloaded yet (its 5.8 GB pull is the slow step)
  * isolating "is the bug in AffectNet's loader or in RAF-DB's?"
  * source-ablation experiments — does training on RAF-DB alone
    generalize differently than training on the merged set?

Class scheme + remaps reuse parent module's RAFDB_MAP. Train/val/test
splits computed in-place (90/10 train/val carve on RAF-DB train,
official RAF-DB test as test) — same logic as the multi-source merge
applies internally, just with one source.

Cache: separate dir under output/data/empath_rafdb_only/ so it
coexists with the full empath cache without conflict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from pipeline import ingest
from pipeline.framework.specs import DatasetSpec

from ..empath import CLASS_NAMES, rafdb


logger = logging.getLogger(__name__)


NAME = "empath_rafdb_only"


def prepare(ctx) -> DatasetSpec:
    cache_dir = Path("output/data") / NAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse parent module's RAF-DB loader — that handles the HF
    # download + materialisation + label remap to EmpathBot 6-class.
    # Result: a df with [path, label, split].
    df = rafdb.load()

    splits_out: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        sub: pd.DataFrame = df[df["split"] == split_name][["path", "label"]]  # type: ignore[assignment]
        dest = cache_dir / f"{split_name}.csv"
        sub.to_csv(dest, index=False)
        splits_out[split_name] = dest

    # Cheap stable hash — same shape as empath/__init__.py uses.
    inventory = cache_dir / "_inventory.txt"
    inventory.write_text(
        "\n".join(f"{r['path']}\t{r['label']}\t{r['split']}"
                  for r in df.to_dict("records"))
    )
    source_md5 = ingest.md5_of_dir(cache_dir)

    train_labels: pd.Series = df[df["split"] == "train"]["label"]  # type: ignore[assignment]
    weights = ingest.compute_class_weights(train_labels, num_classes=len(CLASS_NAMES))

    spec = DatasetSpec(
        name=NAME,
        cache_dir=cache_dir,
        splits=splits_out,
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        class_weights=weights,
        source_md5=source_md5,
    )
    (cache_dir / "manifest.json").write_text(json.dumps(spec.to_manifest(), indent=2))
    logger.info(
        "%s: ready — train=%d val=%d test=%d",
        NAME,
        int((df["split"] == "train").sum()),
        int((df["split"] == "val").sum()),
        int((df["split"] == "test").sum()),
    )
    return spec

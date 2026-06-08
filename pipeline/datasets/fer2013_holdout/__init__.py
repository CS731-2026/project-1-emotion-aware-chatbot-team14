"""FER2013 as held-out out-of-distribution eval set.

Wraps pipeline.datasets.fer2013 and exposes only its test split (no
train, no val) so it's structurally impossible to accidentally train
on FER2013 via this dataset name. Use only as an `eval_datasets` entry
in the evaluate phase.

Rationale: AffectNet + RAF-DB form the in-distribution train/val/test
set for empath models. FER2013 is sourced differently (Google image
search ~2013, 48×48 grayscale upscaled, crowdworker majority-vote
labels) — different enough to function as a generalization test. The
delta between empath_test_acc and fer2013_holdout_acc is the
generalization gap.

Caching: shares fer2013's source download + remapped CSVs. No extra
disk used beyond what `fer2013` already produces.
"""

from __future__ import annotations

from pipeline.datasets import fer2013
from pipeline.framework.specs import DatasetSpec


NAME = "fer2013_holdout"
CLASS_NAMES = fer2013.CLASS_NAMES


def prepare(ctx) -> DatasetSpec:
    """Run the full fer2013 prepare (downloads + remaps + writes CSVs
    if not cached), then return a DatasetSpec exposing only the test
    split. The empty train/val mapping is the safety guarantee — any
    train_loop that grabs splits["train"] on this spec KeyErrors
    immediately."""
    parent = fer2013.prepare(ctx)
    return DatasetSpec(
        name=NAME,
        cache_dir=parent.cache_dir,
        splits={"test": parent.splits["test"]},
        num_classes=parent.num_classes,
        class_names=parent.class_names,
        class_weights=parent.class_weights,
        source_md5=parent.source_md5,
    )

"""empath_small — dev-rung variant of empath.

200 stratified images per class on the train split (~1200 train rows
across 6 classes); val/test pass through unchanged. Sized for the
10-20 min iteration loop when sweeping hyperparameters on the real
empathbot architecture — fast enough for ~10 sweep cells per hour on
MPS, large enough that loss curves are informative.

See pipeline/datasets/empath/_subsample.py for the slicing logic.
"""

from __future__ import annotations

from pathlib import Path

from .. import empath
from ..empath._subsample import subsample_spec
from pipeline.framework.specs import DatasetSpec


NAME = "empath_small"
CLASS_NAMES = empath.CLASS_NAMES

N_PER_CLASS = 200


def prepare(ctx) -> DatasetSpec:
    parent = empath.prepare(ctx)
    return subsample_spec(
        parent,
        n_per_class=N_PER_CLASS,
        name=NAME,
        cache_dir=Path("output/data") / NAME,
    )

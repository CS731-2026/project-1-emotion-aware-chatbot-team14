"""empath_tiny — smoke-rung variant of empath.

50 stratified images per class on the train split; val/test pass through
unchanged so eval metrics from a tiny-scale run are still meaningful
in shape (just noisier).

Used by `runs.yaml` smoke rows to validate end-to-end wiring in ~30s
before committing to a real training run. The full empath cache must
exist (parent.prepare() will trigger the AffectNet + RAF-DB download
on first call — that's the slow step; subsamples are instant after).

See pipeline/datasets/empath/_subsample.py for the slicing logic.
"""

from __future__ import annotations

from pathlib import Path

from .. import empath
from ..empath._subsample import subsample_spec
from pipeline.framework.specs import DatasetSpec


NAME = "empath_tiny"
CLASS_NAMES = empath.CLASS_NAMES

N_PER_CLASS = 50


def prepare(ctx) -> DatasetSpec:
    parent = empath.prepare(ctx)
    return subsample_spec(
        parent,
        n_per_class=N_PER_CLASS,
        name=NAME,
        cache_dir=Path("output/data") / NAME,
    )

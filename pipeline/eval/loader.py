"""Resolve an eval-dataset name to a DatasetSpec.

Mirrors how pipeline.runs_loader resolves training dataset names ,
importlib finds pipeline.datasets.<name> and calls its prepare(ctx).
Kept separate so the eval phase doesn't pull in the rest of runs_loader's
schema validation just to load one dataset.

Eval datasets are expected to be test-only (no train/val splits). The
fer2013_holdout module is the canonical example, it wraps fer2013 and
drops the train+val splits before returning.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from pipeline.framework.specs import DatasetSpec

if TYPE_CHECKING:
    from pipeline.framework.context import Context


def load_eval_dataset(name: str, ctx: "Context") -> DatasetSpec:
    """Import pipeline.datasets.<name> and run its prepare(ctx).

    Raises ImportError with a clear message if the module doesn't
    exist, most likely cause is a typo in `eval_datasets:` in
    runs.yaml or in a default phase list.
    """
    try:
        module = importlib.import_module(f"pipeline.datasets.{name}")
    except ImportError as e:
        raise ImportError(
            f"eval dataset {name!r} not found at pipeline/datasets/{name}/. "
            f"Available eval datasets must be Python modules under "
            f"pipeline/datasets/. Original error: {e}"
        ) from e
    spec = module.prepare(ctx)
    if not isinstance(spec, DatasetSpec):
        raise TypeError(
            f"pipeline.datasets.{name}.prepare(ctx) returned "
            f"{type(spec).__name__}, expected DatasetSpec"
        )
    return spec

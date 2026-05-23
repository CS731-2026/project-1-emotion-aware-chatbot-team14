"""Ada-DF equivalent — faithful port of Notebooks/4_benchmark_ada_df.ipynb.

The notebook frames this as a trainable fallback for the official
Ada-DF checkpoint: EfficientNet-B2 with timm's built-in SE attention,
trained on RAF-DB train split with AdamW + StepLR.

  __init__.py    pipeline surface (PREPROCESS + train)
  model.py       AdaDFEquivalent architecture (cell 11)
  augment.py     TRAIN_TF / VAL_TF (cell 11)
  train_loop.py  AdamW + StepLR + plain CE + early-stop-on-train-acc
                 (cell 11)
"""

from __future__ import annotations

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

from .augment import VAL_TF as PREPROCESS
from .model import build
from .train_loop import run as _run


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return _run(ctx, dataset, model=build(dataset.num_classes))

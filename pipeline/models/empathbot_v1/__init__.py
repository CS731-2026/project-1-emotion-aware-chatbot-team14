"""EmpathBot V1, faithful port of Notebooks/6b_empathbot_v1_improvements.ipynb.

The whole training procedure is lifted: FocalLoss with class weights +
label smoothing, AdamW with split LR (backbone vs head), linear-
warmup + cosine LR schedule, head-only freeze for the first
CFG['freeze_epochs'], gradient clipping, WeightedRandomSampler
with 1.3x bias toward HARD_LABEL_IDS, early stopping on val_acc,
checkpoint envelope matching what model_service expects.

Package layout, each file mirrors one section of the notebook:

  __init__.py     pipeline surface (PREPROCESS + train)
  model.py        architecture (notebook cell 9)
  augment.py      BASE_AUG / STRONG_AUG / VAL_TF (notebook cell 7)
  data.py         EmpathBotDataset (notebook cell 7)
  loss.py         FocalLoss (notebook cell 11)
  train_loop.py   optimizer + scheduler + per-epoch loop (cells 13 + 15)

The notebook stays the source of truth; any tweak should land there
first and be re-lifted across these files.
"""

from __future__ import annotations

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

from .augment import VAL_TF as PREPROCESS  # val-time transform, no augmentation
from .model import build
from .train_loop import run as _run


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return _run(ctx, dataset, model=build(dataset.num_classes))

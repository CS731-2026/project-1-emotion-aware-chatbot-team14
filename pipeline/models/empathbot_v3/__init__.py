"""EmpathBot V3, faithful port of Notebooks/5_final_empathbot_training.ipynb.

Predecessor to empathbot_final (notebook 5_v4). Hard-coded ResNet-18
backbone (no selectable EfficientNet) + single SqueezeExcitation
block on the 512-channel features (not per-layer SE like
empathbot_resnet18).

Notable differences from empathbot_final:
  - No backbone freeze (final uses 5-epoch freeze)
  - MixUp starts at epoch 10 (final starts at epoch 1)
  - LR_HEAD = 3e-3 (final = 1e-3)
  - 3 optimizer param groups (backbone / attention / classifier)
    instead of 2 (backbone / classifier)
"""

from __future__ import annotations

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

from .augment import VAL_TF as PREPROCESS
from .model import build
from .train_loop import run as _run


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return _run(ctx, dataset, model=build(dataset.num_classes))

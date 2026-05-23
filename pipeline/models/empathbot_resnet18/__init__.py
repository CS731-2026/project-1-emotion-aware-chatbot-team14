"""EmpathBot V1 (ResNet-18 backbone) — faithful port of
Notebooks/6_empathbot_v1_resnet18.ipynb.

Predecessor to the 6b (timm EfficientNet-B2) and 5_v4 (torchvision
EfficientNet-B2 with selectable backbone) variants. This one uses
ResNet-18 with explicit SE blocks inserted after each residual stage,
plus a 3-layer BN classifier head.

Package layout — each file mirrors one notebook section:

  __init__.py     pipeline surface (PREPROCESS + train)
  model.py        SEBlock + EmpathBotV1 architecture (cell 11)
  augment.py      BASE_AUG / STRONG_AUG / VAL_TF (cell 8)
  data.py         EmpathBotDataset — per-class augment (cell 9 ish)
  train_loop.py   AdamW split-LR + warmup+cosine LambdaLR + freeze
                  schedule + MixUp + label-smoothed weighted-CE
                  + early stop (cells 13 + 15)
"""

from __future__ import annotations

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

from .augment import VAL_TF as PREPROCESS
from .model import build
from .train_loop import run as _run


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return _run(ctx, dataset, model=build(dataset.num_classes))

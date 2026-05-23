"""ResNet18 baseline — faithful port of Notebooks/2_benchmark_resnet18.ipynb.

The notebook is the source of truth; any architecture or training
procedure tweak should land there first and be re-lifted across this
package's files.

  __init__.py     pipeline surface (PREPROCESS + train)
  augment.py      TRAIN_TF / VAL_TF (cell 10)
  train_loop.py   Adam + StepLR + weighted-CE + early stopping (cells 4, 14, 16)

Architecture (cell 12 of notebook):
  - torchvision resnet18, ImageNet-pretrained
  - Dropout(0.3) → Linear(512, num_classes) replacement head
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

from .augment import VAL_TF as PREPROCESS
from .train_loop import run as _run


def build(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """Verbatim from Notebooks/2_benchmark_resnet18.ipynb cell 12."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return _run(ctx, dataset, model=build(dataset.num_classes))

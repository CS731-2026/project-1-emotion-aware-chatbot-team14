"""ResNet18 FER + OneCycleLR — faithful port of
Notebooks/2_emotion-recognition-resnet18.ipynb.

Distinct from pipeline/models/resnet18/ in three concrete ways:

  * 7-class FER2013 (not the EmpathBot 6-class mapping).
  * Deeper head: Linear(512, 512) → ReLU → Dropout(0.5) → Linear(512, 256)
    → ReLU → Dropout(0.5) → Linear(256, num_classes).
  * SGD(momentum=0.9, wd=1e-4) + OneCycleLR (max_lr discovered by fastai
    lr_find in the notebook; the port uses a fixed sensible default
    that can be overridden via cfg).

Train transforms (cell 31): Grayscale→3ch, Resize(256), RandomCrop(224),
RandomHorizontalFlip, RandomRotation(10), ColorJitter(0.2, 0.2).
Val transforms: Grayscale→3ch, Resize(256), CenterCrop(224).
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel

from .augment import VAL_TF as PREPROCESS
from .train_loop import run as _run


def build(num_classes: int) -> nn.Module:
    """Verbatim from notebook cell 25 (with num_classes parameterised)."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return _run(ctx, dataset, model=build(dataset.num_classes))

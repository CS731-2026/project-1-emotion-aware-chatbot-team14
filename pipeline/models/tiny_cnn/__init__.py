"""Tiny 3-conv CNN, the simplest baseline. Trains in minutes on a
laptop CPU. Useful as a "does the pipeline actually run?" reference
before reaching for anything heavier.

This module owns both the architecture and its training procedure.
For a vanilla classifier the training procedure is just "delegate to
pipeline.training.standard.train_classifier". Custom architectures
(multi-stage, GAN, contrastive) replace train() with their own loop.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.transforms as T

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier


# Structural part of the input pipeline, resize + normalize. The
# training loop composes augmentation on top of this for the train
# split and uses it bare for val/test, so what the model "sees" stays
# consistent across splits and across training/inference.
PREPROCESS = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),                                              # PIL RGB → [0,1] CHW
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),    # → [-1,1]
])


def build(num_classes: int) -> nn.Module:
    """The architecture. ~24k params."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),                              # 32 → 16
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),                              # 16 → 8
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),                      # → (B, 64, 1, 1)
        nn.Flatten(),
        nn.Linear(64, num_classes),
    )


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    """Vanilla classifier, delegate to the shared helper."""
    return train_classifier(
        ctx, dataset,
        model=build(dataset.num_classes),
        preprocess=PREPROCESS,
    )

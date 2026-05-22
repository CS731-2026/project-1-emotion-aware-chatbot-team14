"""MLP baseline — the simplest classifier the pipeline can train.

Flatten → 256 → 128 → num_classes. No convolutions, no inductive bias.
Useful as a "is the dataset learnable at all?" floor — if MLP beats
chance, the pipeline is working; if a CNN can't beat MLP, something
about the CNN setup is wrong.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.transforms as T

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier


PREPROCESS = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def build(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),                 # (B, 3, 32, 32) → (B, 3072)
        nn.Linear(3 * 32 * 32, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return train_classifier(ctx, dataset,
                            model=build(dataset.num_classes),
                            preprocess=PREPROCESS)

"""ResNet18 — torchvision's standard small ResNet. The real-model
reference: anything that beats the team's empath_final.pth needs to
clear ResNet18 first.

We load it without ImageNet weights (weights=None) so a fresh run
trains from scratch; flip to weights="DEFAULT" in build() if you want
pretrained init. Input is 224×224 to match what ResNet18 expects out
of the box; smaller inputs hurt accuracy more than they save compute.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier


PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    # ImageNet stats — the right normalization for ResNet18 whether or
    # not we use pretrained weights; the architecture is tuned for
    # inputs in this distribution.
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build(num_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    # Replace the 1000-class ImageNet head with our num_classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return train_classifier(ctx, dataset,
                            model=build(dataset.num_classes),
                            preprocess=PREPROCESS)

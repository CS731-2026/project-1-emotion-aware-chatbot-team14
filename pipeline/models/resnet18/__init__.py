"""ResNet18 baseline — pretrained on ImageNet, fine-tuned for our
classes with a dropout-regularised head.

Source of truth: Notebooks/2_benchmark_resnet18.ipynb (cell 12,
build_resnet18). The notebook positions ResNet18 as the general-
purpose baseline specialised FER models compare against — anything
below this isn't beating "off-the-shelf".

Architecture (verbatim from notebook):
  - torchvision resnet18, pretrained on ImageNet
  - Replacement head: Dropout(0.3) → Linear(512, num_classes)
  - ~11M params, 224×224 input, ImageNet normalisation
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights, resnet18

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier


PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return train_classifier(ctx, dataset,
                            model=build(dataset.num_classes),
                            preprocess=PREPROCESS)

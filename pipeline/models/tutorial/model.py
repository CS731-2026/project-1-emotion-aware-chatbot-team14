"""Architecture — define the nn.Module + a build() factory. Nothing else."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class Tutorial(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        # ImageNet-pretrained: face features transfer from natural images well
        # enough that random init takes ~10× longer. Pin the weights for reproducibility.
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Swap ResNet's Linear(512, 1000) head for our own.
        in_features = self.backbone.fc.in_features                  # 512 for ResNet-18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),                                  # 0.1 if underfit, 0.5 if overfit
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        # See pipeline/models/empathbot_v1/model.py for a richer example
        # (SE attention + 3-layer classifier).
        return self.backbone(x)


def build(num_classes: int) -> nn.Module:
    # Fresh instance every call — never cache. Tests + deploy + sweep expect a clean module.
    return Tutorial(num_classes=num_classes)

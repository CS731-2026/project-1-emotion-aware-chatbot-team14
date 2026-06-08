"""Architecture, verbatim from notebook 5 cell 16.

ResNet-18 backbone + single SqueezeExcitation block on the 512-channel
features + 3-layer classifier (no batchnorm-flatten head). Notebook
loads pretrained ResNet-18 weights from an external checkpoint; we use
torchvision's ImageNet pretrained as a default (matches v4's behaviour
when no specific checkpoint is provided).
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm


class SqueezeExcitation(nn.Module):
    """Lightweight channel attention, verbatim from notebook 5 cell 16."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class EmpathBotV1(nn.Module):
    """ResNet-18 + single SE on 512-channel features + 3-layer classifier."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.4,
                 pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = tvm.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(bb.children())[:-2])  # → (B, 512, 7, 7)
        self.attention = SqueezeExcitation(512, reduction=16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.attention(self.backbone(x))))


def build(num_classes: int) -> nn.Module:
    return EmpathBotV1(num_classes=num_classes, pretrained=True)

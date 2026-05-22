"""EmpathBot V4 architecture — ported verbatim from notebook 5 cell 16.

Same architecture as the notebook; selectable backbone (efficientnet_b2
or resnet18). For resnet18 the notebook builds a manual SE attention
block on top of the backbone — efficientnet_b2 already has SE built in
so it gets nn.Identity() in that slot.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm


class EmpathBotV1(nn.Module):
    """EmpathBotV1 with selectable backbone (efficientnet_b2 / resnet18).

    The notebook's class name is preserved; we live under
    pipeline/models/empathbot_final/ to avoid colliding with the
    notebook-6b port that also calls itself EmpathBotV1.
    """

    _FEAT_DIM = {"efficientnet_b2": 1408, "resnet18": 512}

    def __init__(self, backbone: str = "efficientnet_b2",
                 num_classes: int = 6, dropout: float = 0.4,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone
        feat_dim = self._FEAT_DIM[backbone]

        if backbone == "efficientnet_b2":
            weights = tvm.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            base = tvm.efficientnet_b2(weights=weights)
            self.backbone  = base.features
            self.gap       = nn.AdaptiveAvgPool2d(1)
            self.attention = nn.Identity()       # EfficientNet has SE inside
        else:  # resnet18
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = tvm.resnet18(weights=weights)
            self.backbone  = nn.Sequential(*list(base.children())[:-2])
            self.gap       = nn.AdaptiveAvgPool2d(1)
            self.attention = nn.Sequential(      # manual SE for ResNet-18
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(feat_dim, feat_dim // 16, bias=False), nn.ReLU(),
                nn.Linear(feat_dim // 16, feat_dim, bias=False), nn.Sigmoid(),
            )

        mid = max(feat_dim // 2, 256)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, mid), nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(mid, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        if self.backbone_name == "efficientnet_b2":
            x = self.gap(x)
        else:
            # SE channel attention then pool
            scale = self.attention(x)
            x = x * scale.view(x.size(0), x.size(1), 1, 1)
            x = self.gap(x)
        return self.classifier(x)


def build(num_classes: int) -> nn.Module:
    """Default to the EfficientNet-B2 variant — the notebook's primary recommendation."""
    return EmpathBotV1(
        backbone="efficientnet_b2",
        num_classes=num_classes,
        pretrained=True,
    )

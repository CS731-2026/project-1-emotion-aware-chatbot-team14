"""Architecture — ONE concern: define the nn.Module + a build() factory.

Separated from train_loop.py so tests + deploy + experiments can build
the model without dragging the training loop in. Swap the backbone
(timm, custom, etc) here without touching anything else.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class Tutorial(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        # ImageNet-pretrained: face features transfer from natural images well
        # enough that random init takes ~10× longer to converge. Pin the
        # specific weights file so all teammates get the same checkpoint.
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # ResNet's last layer is Linear(512, 1000) for ImageNet. Swap for our
        # 6-class head. Earlier conv blocks stay pretrained.
        in_features = self.backbone.fc.in_features                 # 512 for ResNet-18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),                                 # 0.1 if underfitting, 0.5 if overfit
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        # See pipeline/models/empathbot_v1/model.py for a richer architecture
        # (SE attention + 3-layer classifier) if you need more than this.
        return self.backbone(x)


def build(num_classes: int) -> nn.Module:
    # Fresh instance every call — never cache. Tests + deploy + sweep all
    # expect a clean module they can move to a device and load weights into.
    return Tutorial(num_classes=num_classes)

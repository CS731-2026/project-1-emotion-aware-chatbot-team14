"""Architecture, verbatim from notebook 6 cell 11.

SEBlock (Squeeze-and-Excitation channel attention) + EmpathBotV1 with
ResNet-18 backbone, SE block after each residual stage, 3-layer BN head.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        s = self.avg(x).view(b, c)
        return x * self.fc(s).view(b, c, 1, 1)


def _make_head(in_features: int, num_classes: int) -> nn.Sequential:
    """3-layer BN head from notebook cell 11."""
    mid = max(in_features // 2, 256)
    return nn.Sequential(
        nn.Linear(in_features, mid),
        nn.BatchNorm1d(mid),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(mid, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )


class EmpathBotV1(nn.Module):
    """ResNet-18 + SE attention after each residual stage + BN head.
    Selectable backbone, efficientnet_b0 supported as in the notebook."""

    _FEAT_DIMS = {"resnet18": 512, "efficientnet_b0": 1280}

    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 se_reduction: int = 16, pretrained: bool = True) -> None:
        super().__init__()
        assert backbone in self._FEAT_DIMS
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            b = tvm.resnet18(weights=weights)
            self.stem   = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool)
            self.layer1 = b.layer1;  self.se1 = SEBlock(64,  se_reduction)
            self.layer2 = b.layer2;  self.se2 = SEBlock(128, se_reduction)
            self.layer3 = b.layer3;  self.se3 = SEBlock(256, se_reduction)
            self.layer4 = b.layer4;  self.se4 = SEBlock(512, se_reduction)
        else:  # efficientnet_b0
            weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            b = tvm.efficientnet_b0(weights=weights)
            self.features = b.features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = _make_head(self._FEAT_DIMS[backbone], num_classes)

    def forward(self, x):
        if self.backbone_name == "resnet18":
            x = self.stem(x)
            x = self.se1(self.layer1(x))
            x = self.se2(self.layer2(x))
            x = self.se3(self.layer3(x))
            x = self.se4(self.layer4(x))
        else:
            x = self.features(x)
        return self.head(self.pool(x).flatten(1))

    def backbone_params(self):
        if self.backbone_name == "resnet18":
            return [
                *self.stem.parameters(),
                *self.layer1.parameters(), *self.se1.parameters(),
                *self.layer2.parameters(), *self.se2.parameters(),
                *self.layer3.parameters(), *self.se3.parameters(),
                *self.layer4.parameters(), *self.se4.parameters(),
            ]
        return list(self.features.parameters())

    def head_params(self):
        return list(self.head.parameters())


def build(num_classes: int) -> nn.Module:
    return EmpathBotV1(
        num_classes=num_classes,
        backbone="resnet18",
        se_reduction=16,
        pretrained=True,
    )

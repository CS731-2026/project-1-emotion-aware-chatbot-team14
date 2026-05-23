"""Ada-DF equivalent architecture — ported verbatim from notebook 4.

EfficientNet-B2 backbone (with SE attention built into timm) +
2-stage classifier head. The notebook treats this as a fallback for
the official Ada-DF weights; we use it as the trainable form here.
"""

from __future__ import annotations

import torch.nn as nn


class AdaDFEquivalent(nn.Module):
    """EfficientNet-B2 fine-tuned for FER.

    Uses timm's built-in SE blocks for adaptive channel weighting,
    matching Ada-DF's core idea of adaptive feature selection.
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.3,
                 pretrained: bool = True) -> None:
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,         # strip classifier head
            drop_rate=dropout,
        )
        feat_dim = self.backbone.num_features  # 1408 for EfficientNet-B2
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


def build(num_classes: int) -> nn.Module:
    return AdaDFEquivalent(num_classes=num_classes, pretrained=True)

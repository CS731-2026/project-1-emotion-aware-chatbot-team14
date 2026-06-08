"""EmpathBotV1 architecture, ported verbatim from notebook 6b cell 9.

The notebook is the source of truth: any tweak here should also be
made in `Notebooks/6b_empathbot_v1_improvements.ipynb` to keep the
two in sync. Notebook does not import this module, porting goes
notebook → here, never the other way.
"""

from __future__ import annotations

import torch.nn as nn


def _make_head(in_features: int, num_classes: int) -> nn.Sequential:
    """3-layer BN head with progressive dropout. Verbatim from notebook."""
    mid = max(in_features // 2, 256)
    return nn.Sequential(
        nn.Linear(in_features, mid),
        nn.BatchNorm1d(mid),
        nn.ReLU(inplace=True),
        nn.Dropout(0.35),
        nn.Linear(mid, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )


class EmpathBotV1(nn.Module):
    """EfficientNet-B2 backbone (timm, pretrained) + 3-layer BN head.

    Falls back to torchvision EfficientNet-B2 if timm is unavailable
    (the notebook supports both via its CFG['use_timm']).
    """

    def __init__(self, num_classes: int, backbone: str = "efficientnet_b2",
                 use_timm: bool = True, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.use_timm = use_timm

        if use_timm:
            import timm
            self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            feat_dim = self.encoder.num_features
        else:
            from torchvision import models as tvm
            base = tvm.efficientnet_b2(weights=tvm.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = 1408

        self.head = _make_head(feat_dim, num_classes)

    def forward(self, x):
        if self.use_timm:
            x = self.encoder(x)            # timm handles pooling internally with num_classes=0
        else:
            x = self.pool(self.encoder(x)).flatten(1)
        return self.head(x)

    # Notebook-side helpers preserved verbatim, split-LR optimizer setup
    # uses these when the train phase eventually wires it up.
    def backbone_params(self):
        return list(self.encoder.parameters())

    def head_params(self):
        return list(self.head.parameters())


def build(num_classes: int) -> nn.Module:
    """The pipeline-side constructor, fixed backbone choice (timm +
    pretrained efficientnet_b2) to match notebook defaults. Override
    via a custom train() in a sibling model file if you need to vary."""
    return EmpathBotV1(
        num_classes=num_classes,
        backbone="efficientnet_b2",
        use_timm=True,
        pretrained=True,
    )

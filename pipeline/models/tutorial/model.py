"""model.py — your architecture. ONE concern: define the nn.Module.

Why split this from train_loop.py?
  - Tests can import build() without dragging the training loop in
  - The deploy step needs build() to reconstruct the architecture
    before loading the saved state dict
  - When you want to swap architectures, you edit one file, not a 200-line training script

The pattern:
  - One nn.Module subclass with the layers
  - One module-level build(num_classes) factory the rest of the
    package uses
That's it. No training logic in here.

This tutorial uses an ImageNet-pretrained ResNet-18 with a dropout +
linear head. It's the simplest "real" model — small enough to train
in seconds, large enough to actually learn something. Swap the
backbone (e.g. timm.create_model("efficientnet_b2", pretrained=True))
without touching the rest of the package.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class Tutorial(nn.Module):
    """ResNet-18 backbone + Dropout + Linear head.

    Why ImageNet-pretrained?
      Face images differ from ImageNet but the low-level features
      (edges, textures, colour gradients) transfer surprisingly well.
      Starting from random would take 10× longer to converge on a
      dataset our size.

    Why a separate Dropout layer instead of just nn.Linear?
      Reduces overfitting on small datasets. 0.3 is a good default;
      lower it (0.1) if you see underfitting, raise it (0.5) if val
      loss diverges from train loss.

    Why expose `dropout` as a __init__ kwarg?
      So you can experiment from a notebook or a custom test without
      monkey-patching. Real models usually go further and read this
      from CFG; we keep it simple here.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        # torchvision's pretrained ResNet-18. The .IMAGENET1K_V1 alias
        # pins the specific weights file so different machines all get
        # the same checkpoint (defaults can shift between torchvision
        # releases).
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # ResNet's final layer is `fc: Linear(512, 1000)` (1000 = ImageNet
        # classes). We swap that one layer for our own head. The
        # backbone's earlier conv blocks stay pretrained and untouched.
        in_features = self.backbone.fc.in_features  # 512 for ResNet-18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        # Single-line forward. Most architectures need more structure
        # (multi-branch outputs, attention, etc) — see
        # pipeline/models/empathbot_v1/model.py for a richer example
        # with SE attention + a 3-layer classifier.
        return self.backbone(x)


def build(num_classes: int) -> nn.Module:
    """Factory the rest of the package calls. Always returns a fresh
    instance — never cache. Tests + deploy expect a clean nn.Module
    they can move to a device and load weights into.
    """
    return Tutorial(num_classes=num_classes)

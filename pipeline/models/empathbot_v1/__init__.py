"""EmpathBot V1 — the team's best model so far (~73% val acc).

Source of truth: `Notebooks/6b_empathbot_v1_improvements.ipynb`. This
package ports the architecture + the val-time preprocessing (VAL_TF)
verbatim, then trains via the shared pipeline.training.standard
helper. Train-time augmentation strength is config-controlled
(configs/<name>.yaml's `augment:` section) rather than hard-coded
here, so the same model can be swept across multiple augment regimes.

Architecture:
  - EfficientNet-B2 backbone via timm (pretrained=True)
  - 3-layer BN classifier head (in → max(in//2,256) → 128 → num_classes)
  - ~7M params

See `model.py` for the architecture, this `__init__.py` for the
pipeline-facing surface (PREPROCESS + train).
"""

from __future__ import annotations

import torchvision.transforms as T

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier

from .model import build


# Val-time preprocessing — verbatim from notebook 6b's VAL_TF.
# 224×224 input (the notebook's CFG['img_size']), ImageNet stats
# (EfficientNet-B2 is tuned for this distribution regardless of
# whether we load pretrained or from-scratch).
_IMG_SIZE = 224
PREPROCESS = T.Compose([
    T.Resize((_IMG_SIZE, _IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    """Delegate to the shared classifier loop. The notebook's bespoke
    bits (split LR for backbone/head, focal loss, two-phase freeze→
    unfreeze schedule) live behind the augment / loss / optimizer
    config keys — extend pipeline.training.registries to enable them
    if a config asks."""
    return train_classifier(
        ctx, dataset,
        model=build(dataset.num_classes),
        preprocess=PREPROCESS,
    )

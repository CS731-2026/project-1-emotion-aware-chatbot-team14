"""Ada-DF equivalent — EfficientNet-B2 with timm's built-in SE attention.

Source of truth: Notebooks/4_benchmark_ada_df.ipynb. The notebook
calls this an "Ada-DF equivalent": it doesn't load the official Ada-DF
weights (those require the paper's external code), but it captures the
core idea — adaptive channel weighting via SE blocks — on top of an
EfficientNet-B2 backbone we can train end-to-end with our pipeline.

If you later get the official Ada-DF checkpoint and want to load it,
that needs the paper repo's model class; this module is for the
trainable approximation only.

Architecture (from notebook):
  - EfficientNet-B2 (timm, pretrained, SE blocks built in)
  - Linear(1408 → 256) → BN → ReLU → Dropout → Linear(256 → num_classes)
"""

from __future__ import annotations

import torchvision.transforms as T

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier

from .model import build


# Val-time preprocessing — 224×224 matches EfficientNet-B2's tuned input
# distribution; ImageNet stats since the backbone is pretrained.
_IMG_SIZE = 224
PREPROCESS = T.Compose([
    T.Resize((_IMG_SIZE, _IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return train_classifier(
        ctx, dataset,
        model=build(dataset.num_classes),
        preprocess=PREPROCESS,
    )

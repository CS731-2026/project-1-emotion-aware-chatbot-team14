"""EmpathBot final (v4) — torchvision-based with selectable backbone.

Source of truth: Notebooks/5_final_empathbot_training_v4.ipynb. Sibling
to empathbot_v1 (which is ported from notebook 6b). They share the
class name `EmpathBotV1` in their respective notebooks but differ
materially:

  6b (empathbot_v1)        timm EfficientNet-B2, no custom SE
  5_v4 (empathbot_final)   torchvision; selectable backbone with
                           manual SE attention for resnet18

Default backbone here is EfficientNet-B2 (matches notebook 5's
default). To train the ResNet-18 variant, build a sibling model
module that overrides build() — or expose a config knob if you find
yourself doing it often.
"""

from __future__ import annotations

import torchvision.transforms as T

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier

from .model import build


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

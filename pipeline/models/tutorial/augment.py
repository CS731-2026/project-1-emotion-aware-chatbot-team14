"""TRAIN_TF (with augmentation) / VAL_TF (deterministic).

Splitting them matters — augmenting val/test would change reported metrics
every run, making "did my last change help?" unanswerable.

PREPROCESS = VAL_TF — re-exported from __init__.py for the live service.
Ensures train-time and inference-time preprocessing match exactly.
"""

from __future__ import annotations

import torchvision.transforms as T


IMG_SIZE = 224

# ImageNet mean/std — REQUIRED for pretrained backbones (conv filters
# expect inputs in this range). Custom backbones can use different stats.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


TRAIN_TF = T.Compose([
    T.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),    # larger so RandomCrop has somewhere to move
    T.RandomCrop(IMG_SIZE),                      # spatial variation
    T.RandomHorizontalFlip(p=0.5),               # faces are roughly symmetric → free aug
    T.ColorJitter(brightness=0.2, contrast=0.2), # webcam exposure variation; 0.2 is safe
    T.RandomRotation(degrees=10),                # heads tilt a bit, not 90°; keep < 15
    T.ToTensor(),                                # PIL → tensor + scales to [0,1]
    T.Normalize(MEAN, STD),                      # always last
])

VAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),              # no randomness
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

"""TRAIN_TF (with augmentation) vs VAL_TF (deterministic).

PREPROCESS = VAL_TF, re-exported from __init__.py so the live model
service uses the same eval transforms at inference (no train/serve drift).
"""

from __future__ import annotations

import torchvision.transforms as T


IMG_SIZE = 224

# REQUIRED for pretrained backbones, ImageNet stats.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


TRAIN_TF = T.Compose([
    T.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),    # larger so RandomCrop has room to move
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(p=0.5),               # faces ~ symmetric → free aug
    T.ColorJitter(brightness=0.2, contrast=0.2), # webcam exposure variation; 0.2 is safe
    T.RandomRotation(degrees=10),                # heads tilt, not 90°; keep < 15
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

VAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

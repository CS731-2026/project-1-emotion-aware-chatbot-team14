"""Train + val transforms — verbatim from notebook 5_v4 cell 14.

Two train-time augmentation strengths:
  STD_TF   for non-NEGATIVE classes (mild)
  NEG_TF   for "hard" classes (NEGATIVE_CLASSES) — stronger augment
            with grayscale + affine + tighter jitter

VAL_TF is bare resize + normalise.
"""

from __future__ import annotations

import torchvision.transforms as T


IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Notebook's NEGATIVE_CLASSES = {sadness, fear_anxiety, distrust} per
# EmpathBot 6-class indexing.
NEGATIVE_LABEL_IDS = {2, 3, 5}


STD_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

NEG_TF = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.1),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

VAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

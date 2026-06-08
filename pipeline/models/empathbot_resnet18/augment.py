"""Augmentations from notebook 6 cell 8.

Mirrors notebook 6b's augment.py but with the older notebook's
slightly more aggressive jitter on the BASE variant. HARD_LABEL_IDS
is the same set as 6b, {sadness=2, fear_anxiety=3, distrust=5}.
"""

from __future__ import annotations

import torchvision.transforms as T


IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
HARD_LABEL_IDS = {2, 3, 5}


BASE_AUG = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

STRONG_AUG = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05),
    T.RandomRotation(15),
    T.RandomGrayscale(p=0.10),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

VAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

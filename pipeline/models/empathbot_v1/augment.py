"""Train + val transforms, verbatim from notebook 6b cell 7.

BASE_AUG    geometric + mild colour jitter, default for most classes
STRONG_AUG  more aggressive colour + grayscale, used for "hard" classes
            (sadness, fear_anxiety, distrust per notebook's HARD_LABEL_IDS)
VAL_TF      no augmentation, used at val / test time
"""

from __future__ import annotations

import torchvision.transforms as T


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# Notebook's HARD_LABEL_IDS, sadness=2, fear_anxiety=3, distrust=5.
# These classes get STRONG_AUG; everything else gets BASE_AUG.
HARD_LABEL_IDS = {2, 3, 5}


BASE_AUG = T.Compose([
    T.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    T.RandomRotation(8),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

STRONG_AUG = T.Compose([
    T.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04),
    T.RandomRotation(12),
    T.RandomGrayscale(p=0.08),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

VAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

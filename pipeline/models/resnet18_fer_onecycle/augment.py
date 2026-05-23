"""Transforms — verbatim from notebook cell 31.

FER2013 is single-channel grayscale; Grayscale(3) repeats it across the
3 ImageNet-normalised channels expected by ResNet-18. Resize(256) +
RandomCrop(224) gives a small augmentation crop; val uses CenterCrop.
"""

from __future__ import annotations

import torchvision.transforms as T


IMG_SIZE = 224
RESIZE   = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


TRAIN_TF = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize(RESIZE),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

VAL_TF = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize(RESIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

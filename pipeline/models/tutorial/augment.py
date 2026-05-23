"""augment.py — transforms (what images look like going in).

Two transforms minimum:
  - TRAIN_TF: applied to training-set images. Includes augmentation
    (random crops, flips, jitter) to simulate variability the model
    will see at inference.
  - VAL_TF:   applied to val + test images. Deterministic — no
    augmentation, because you want repeatable eval numbers.

The split matters: augmenting val/test would change your reported
metrics every run, making "did my last change help?" unanswerable.

We also export PREPROCESS = VAL_TF — same transforms the live
model_service uses at inference. Defining it once ensures the
augment used at train-time matches the deterministic preprocess at
serve-time, no drift.

Common transforms to consider (commented out below — uncomment as you
need them):
  - RandomCrop / RandomResizedCrop  — spatial variation
  - ColorJitter                     — lighting / camera variation
  - RandomRotation                  — small rotations (faces shouldn't go upside down)
  - RandomGrayscale                 — robustness to monochrome
  - RandomErasing                   — adversarial "missing pixels"
  - Normalize                       — REQUIRED for pretrained backbones
"""

from __future__ import annotations

import torchvision.transforms as T


IMG_SIZE = 224

# ImageNet mean/std — the values our pretrained ResNet-18 was trained
# with. Always normalise to these when using ImageNet-pretrained
# backbones; the conv filters expect inputs in this range. Custom
# backbones can use different stats (e.g. [0.5, 0.5, 0.5]).
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


TRAIN_TF = T.Compose([
    # Why Resize before RandomCrop? To get a slightly-larger image so
    # the random crop has somewhere to move. If you Resize to 224 and
    # RandomCrop(224), the crop is always the same — pointless.
    T.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    T.RandomCrop(IMG_SIZE),

    # Horizontal flip — faces are roughly symmetric so this is free
    # data augmentation. Don't do vertical flip for faces.
    T.RandomHorizontalFlip(p=0.5),

    # Mild colour jitter — simulates webcam exposure variation. Going
    # higher (0.4–0.5) hurts on small / subtle classes (confusion vs
    # neutral). 0.2 is usually safe.
    T.ColorJitter(brightness=0.2, contrast=0.2),

    # Small rotation — heads tilt, but not 90 degrees. Keep this < 15.
    T.RandomRotation(degrees=10),

    # Final: tensor + normalise. Always last.
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

VAL_TF = T.Compose([
    # Same target size, no randomness. CenterCrop instead of
    # RandomCrop if you want the exact ImageNet eval preprocess.
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

"""Augmentation registry. Returns the *random-only* transform stack that
gets composed in front of the model's PREPROCESS for the train split.

Crucially, augmentation never includes normalization or resize-to-input
— those are PREPROCESS's job. That keeps "what does the model see"
unambiguous and lets the same augment work across models with different
input sizes.
"""

from __future__ import annotations

from typing import Any

import torchvision.transforms as T


SUPPORTED = ("none", "mild", "strong")


def get_augment(name: str, args: dict[str, Any] | None = None) -> T.Compose:
    args = args or {}
    if name == "none":
        return T.Compose([])
    if name == "mild":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        ])
    if name == "strong":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(float(args.get("rotation", 12))),
            T.ColorJitter(
                brightness=float(args.get("brightness", 0.25)),
                contrast=float(args.get("contrast", 0.25)),
                saturation=float(args.get("saturation", 0.15)),
            ),
            T.RandomGrayscale(p=float(args.get("grayscale_p", 0.08))),
        ])
    raise ValueError(f"unknown augmentation {name!r}. supported: {SUPPORTED}")

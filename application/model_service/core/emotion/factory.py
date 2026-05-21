"""Emotion model factory."""

from __future__ import annotations

import config

from .base import EmotionModel


def create_emotion_model(variant: str = "placeholder") -> EmotionModel:
    """Create and return an EmotionModel for the requested variant.

    Args:
        variant: Model variant to instantiate. Supported: "placeholder",
            "resnet18", "empathbot".

    Returns:
        A ready-to-use EmotionModel instance.

    Raises:
        ValueError: If the variant is not recognised.
    """
    if variant == "placeholder":
        from .placeholder import PlaceholderEmotionModel

        return PlaceholderEmotionModel()

    if variant == "resnet18":
        from .resnet18 import ResNet18EmotionModel

        return ResNet18EmotionModel(
            checkpoint_path=config.EMOTION_CHECKPOINT_PATH,
            device=config.EMOTION_DEVICE,
        )

    if variant == "empathbot":
        from .empathbot import EmpathBotEmotionModel

        return EmpathBotEmotionModel(
            checkpoint_path=config.EMOTION_CHECKPOINT_PATH,
            device=config.EMOTION_DEVICE,
        )

    raise ValueError(
        f"Unknown emotion model variant: '{variant}'. "
        "Valid options: 'placeholder', 'resnet18', 'empathbot'."
    )

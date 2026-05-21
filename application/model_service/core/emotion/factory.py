"""Emotion model factory."""

from __future__ import annotations

from .base import EmotionModel


def create_emotion_model(variant: str = "placeholder") -> EmotionModel:
    """Create and return an EmotionModel for the requested variant.

    Args:
        variant: Model variant to instantiate. Currently supported: "placeholder".

    Returns:
        A ready-to-use EmotionModel instance.

    Raises:
        ValueError: If the variant is not recognised.
    """
    if variant == "placeholder":
        from .placeholder import PlaceholderEmotionModel

        return PlaceholderEmotionModel()

    else:
        raise ValueError(
            f"Unknown emotion model variant: '{variant}'. "
            "Valid options: 'placeholder'."
        )

"""Emotion model factory."""

from __future__ import annotations

import config

from ..model_resolver import resolve_checkpoint
from .base import EmotionModel


def create_emotion_model(variant: str = "placeholder") -> EmotionModel:
    """Create and return an EmotionModel.

    Resolution order:
      1. If ``config.EMOTION_MODEL_ID`` is set, look it up in models.yaml and
         use the entry's variant + path. Preferred path.
      2. Otherwise, fall back to the ``variant`` arg + ``EMOTION_CHECKPOINT_PATH``.

    Args:
        variant: Fallback variant when no model id is configured. Supported:
            "placeholder", "resnet18", "empathbot".

    Returns:
        A ready-to-use EmotionModel instance.

    Raises:
        ValueError: If the resolved variant is unknown or the model id is not
            in the registry.
    """
    checkpoint_path: str = config.EMOTION_CHECKPOINT_PATH

    if config.EMOTION_MODEL_ID:
        registry = config.load_model_registry()
        entry = registry.get(config.EMOTION_MODEL_ID)
        if entry is None:
            raise ValueError(
                f"EMOTION_MODEL_ID='{config.EMOTION_MODEL_ID}' not found in "
                f"models.yaml. Known ids: {sorted(registry.keys())}"
            )
        variant = entry["variant"]
        # Auto-fetches from Kaggle if missing locally — see model_resolver.py
        # for the resolution flow ("creds-only" team distribution).
        checkpoint_path = str(resolve_checkpoint(
            entry["path"], repo_root=config.REPO_ROOT,
            model_id=config.EMOTION_MODEL_ID,
        ))

    if variant == "placeholder":
        from .placeholder import PlaceholderEmotionModel

        return PlaceholderEmotionModel()

    if variant == "resnet18":
        from .resnet18 import ResNet18EmotionModel

        return ResNet18EmotionModel(
            checkpoint_path=checkpoint_path,
            device=config.EMOTION_DEVICE,
        )

    if variant == "empathbot":
        from .empathbot import EmpathBotEmotionModel

        return EmpathBotEmotionModel(
            checkpoint_path=checkpoint_path,
            device=config.EMOTION_DEVICE,
        )

    raise ValueError(
        f"Unknown emotion model variant: '{variant}'. "
        "Valid options: 'placeholder', 'resnet18', 'empathbot'."
    )

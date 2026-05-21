"""Base interface for emotion recognition models."""

from abc import ABC, abstractmethod

import numpy as np


# EmpathBot 6-class schema. Source of truth: dataset/master_split.csv eb_label.
# Index order matches the model's output logit order — do not reorder.
EMOTIONS = [
    "neutral",        # 0
    "trust_relief",   # 1
    "sadness",        # 2
    "fear_anxiety",   # 3
    "confusion",      # 4
    "distrust",       # 5
]

# Human-readable prose for LLM prompts and HUD display.
EMOTION_PROSE: dict[str, str] = {
    "neutral":      "neutral",
    "trust_relief": "calm and at ease",
    "sadness":      "sad or low",
    "fear_anxiety": "anxious or fearful",
    "confusion":    "confused",
    "distrust":     "uneasy or guarded",
}


class EmotionModel(ABC):
    """Abstract base class for all emotion recognition backends."""

    EMOTIONS = EMOTIONS

    @abstractmethod
    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        """Predict the dominant emotion for a face crop.

        Args:
            face_bgr: A BGR image crop of a detected face (H, W, 3), uint8.

        Returns:
            (emotion, confidence) where emotion is one of EMOTIONS and
            confidence is a float in [0.0, 1.0].
        """
        ...

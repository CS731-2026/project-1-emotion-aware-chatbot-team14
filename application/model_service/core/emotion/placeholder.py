"""Placeholder emotion model, returns a random emotion with random confidence.

No model loading. Used during development and as the default fallback until a
real emotion recognition model is integrated.
"""

from __future__ import annotations

import random

import numpy as np

from .base import EmotionModel


class PlaceholderEmotionModel(EmotionModel):
    """Returns a random emotion label with a random confidence score.

    Confidence range: 0.5 – 0.95 (uniform).
    """

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        """Return a random emotion and confidence.

        Args:
            face_bgr: Ignored, no inference is performed.

        Returns:
            (emotion, confidence)
        """
        emotion = random.choice(self.EMOTIONS)
        confidence = random.uniform(0.5, 0.95)
        return emotion, confidence

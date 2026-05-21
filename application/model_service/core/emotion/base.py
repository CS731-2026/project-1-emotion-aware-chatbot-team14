"""Base interface for emotion recognition models."""

from abc import ABC, abstractmethod

import numpy as np


class EmotionModel(ABC):
    """Abstract base class for all emotion recognition backends."""

    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

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

"""Base interface for transcription services."""

from abc import ABC, abstractmethod

import numpy as np


class TranscriptionService(ABC):

    REQUIRES_DEPS: list[str] = []
    REQUIREMENTS_FILE: str | None = None

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> tuple[str, str, float | None]:
        """Transcribe audio data.

        Args:
            audio_data: numpy array of audio samples (float32).

        Returns:
            (transcript, language, confidence) where confidence is 0.0–1.0
            or None if the backend does not expose a confidence score.
        """
        ...

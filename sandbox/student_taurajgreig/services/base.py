"""Base interface for transcription services."""

from abc import ABC, abstractmethod
import numpy as np


class TranscriptionService(ABC):
    """Abstract base class for transcription services.

    Any transcription service (Whisper, OpenAI API, etc.) should inherit
    from this and implement the transcribe method.
    """

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> tuple[str, str]:
        """Transcribe audio data.

        Args:
            audio_data: numpy array of audio samples (float32)

        Returns:
            (transcript, language) tuple
        """
        pass

"""Base interface for transcription services."""

from abc import ABC, abstractmethod
import numpy as np


class TranscriptionService(ABC):
    """Abstract base class for transcription services.

    Any transcription service (Whisper, OpenAI API, etc.) should inherit
    from this and implement the transcribe method.

    Each service subclass should define:
        - REQUIRES_DEPS: list of required package names
        - REQUIREMENTS_FILE: path to requirements file (relative to services/)
    """

    REQUIRES_DEPS = []  # List of required packages
    REQUIREMENTS_FILE = None  # Path to requirements file

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray):
        """Transcribe audio data.

        Args:
            audio_data: numpy array of audio samples (float32)

        Returns:
            (transcript, language) tuple, or
            (transcript, language, confidence) tuple if confidence is available
            where confidence is 0.0-1.0 representing model confidence
        """
        pass

"""Mock transcription service for testing without a real model.

Uses simple volume-based heuristics to simulate transcription.
Useful for testing VAD logic without resource constraints.
"""

import numpy as np
from .base import TranscriptionService


class MockTranscriptionService(TranscriptionService):
    """Mock transcription service - returns dummy output.

    Simulates transcription without loading any models.
    Useful for testing the recording and UI logic.
    """

    def __init__(self):
        """Initialize mock service."""
        self.call_count = 0
        self.responses = [
            "Hello, this is a test.",
            "The speech recognition system is working.",
            "This is a mock transcription.",
            "Testing voice activation detection.",
            "Speech recognition demo in progress.",
        ]

    def transcribe(self, audio_data: np.ndarray) -> tuple[str, str]:
        """Return mock transcription based on audio duration.

        Args:
            audio_data: numpy array (not actually used for mock)

        Returns:
            (transcript, language) tuple
        """
        # Cycle through responses
        transcript = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return transcript, "en"

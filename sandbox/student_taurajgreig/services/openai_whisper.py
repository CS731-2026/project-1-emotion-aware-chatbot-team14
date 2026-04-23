"""OpenAI Whisper API transcription service (cloud-based reference).

Use this to test against a well-known, reliable API for debugging purposes.

Requires:
  - openai library: pip install openai
  - OPENAI_API_KEY environment variable set

Get API key from: https://platform.openai.com/api/keys
"""

import numpy as np
import io
from .base import TranscriptionService


class OpenAIWhisperTranscriptionService(TranscriptionService):
    """Transcription service using OpenAI's Whisper API.

    This is a cloud-based reference implementation for debugging.
    Use this to verify if audio quality is the issue vs. the local model.

    Required dependencies: openai
    Install with: pip install openai

    Set OPENAI_API_KEY environment variable with your API key.
    Get key from: https://platform.openai.com/api/keys
    """

    REQUIRES_DEPS = ["openai"]
    REQUIREMENTS_FILE = None

    def __init__(self):
        """Initialize OpenAI Whisper API client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai library not installed. "
                "Install with: pip install openai"
            )

        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk_your"):
            raise ValueError(
                "OPENAI_API_KEY not set in .env file. "
                "Get your key from: https://platform.openai.com/api/keys"
            )

        self.client = OpenAI(api_key=api_key)

    def transcribe(self, audio_data: np.ndarray) -> tuple[str, str]:
        """Transcribe audio using OpenAI's Whisper API.

        Args:
            audio_data: numpy array of audio samples (float32)

        Returns:
            (transcript, language) tuple
        """
        # Convert numpy array to WAV bytes in memory
        import scipy.io.wavfile as wavfile

        sample_rate = 16000  # Standard sample rate
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Write to bytes buffer
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)

        # Send to OpenAI API
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", buffer, "audio/wav"),
        )

        return transcript.text, "EN"

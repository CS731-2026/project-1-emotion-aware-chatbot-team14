"""STT service factory."""

from __future__ import annotations

from .base import TranscriptionService


def create_stt(
    engine: str = "whisper-cpp",
    model: str = "base.en",
) -> TranscriptionService:
    """Create and return a TranscriptionService for the requested engine.

    Args:
        engine: One of "whisper-cpp" or "faster-whisper".
        model:  Model name passed to the chosen engine.

    Returns:
        A ready-to-use TranscriptionService instance.

    Raises:
        ValueError: If the engine name is not recognised.
    """
    if engine == "whisper-cpp":
        from config import WHISPER_CPP_DIR
        from .whisper_cpp import WhisperCppTranscriptionService

        return WhisperCppTranscriptionService(
            model_name=model,
            repo_dir=WHISPER_CPP_DIR,
        )

    elif engine == "faster-whisper":
        from .whisper_faster import FasterWhisperTranscriptionService

        return FasterWhisperTranscriptionService(model_name=model)

    else:
        raise ValueError(
            f"Unknown STT engine: '{engine}'. "
            "Valid options: 'whisper-cpp', 'faster-whisper'."
        )

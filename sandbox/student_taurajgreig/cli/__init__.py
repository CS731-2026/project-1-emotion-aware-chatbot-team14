"""CLI and configuration utilities."""

from .config import (
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_SIZE,
    SPEECH_THRESHOLD,
    SILENCE_DURATION,
    MIN_SPEECH_DURATION,
    MAX_SPEECH_DURATION,
    DEBUG_MODE,
    MODEL_SERVICE,
    MODEL_NAME,
    LOCAL_MODELS_TO_TEST,
)
from .args import parse_arguments, create_service

__all__ = [
    "SAMPLE_RATE",
    "CHANNELS",
    "CHUNK_SIZE",
    "SPEECH_THRESHOLD",
    "SILENCE_DURATION",
    "MIN_SPEECH_DURATION",
    "MAX_SPEECH_DURATION",
    "DEBUG_MODE",
    "MODEL_SERVICE",
    "MODEL_NAME",
    "LOCAL_MODELS_TO_TEST",
    "parse_arguments",
    "create_service",
]

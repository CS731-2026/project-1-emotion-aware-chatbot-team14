"""Transcription services - pluggable implementations."""

from .base import TranscriptionService
from .mock import MockTranscriptionService
from .whisper import WhisperTranscriptionService

__all__ = [
    "TranscriptionService",
    "MockTranscriptionService",
    "WhisperTranscriptionService",
]

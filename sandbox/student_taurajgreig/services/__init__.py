"""Transcription services - pluggable implementations."""

from .base import TranscriptionService
from .mock import MockTranscriptionService
from .whisper import WhisperTranscriptionService
from .whisper_distilled import WhisperDistilledTranscriptionService
from .openai_whisper import OpenAIWhisperTranscriptionService

__all__ = [
    "TranscriptionService",
    "MockTranscriptionService",
    "WhisperTranscriptionService",
    "WhisperDistilledTranscriptionService",
    "OpenAIWhisperTranscriptionService",
]

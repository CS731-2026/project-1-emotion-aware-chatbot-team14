"""Session state shared between the WebSocket handler and the HTTP chat route."""

import logging
from dataclasses import dataclass
import time

from core.emotion.buffer import EmotionBuffer

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    text: str
    timestamp: float  # unix seconds — from audio_chunk message
    # STT confidence in [0, 1] when the engine exposes it (faster-whisper),
    # None otherwise (whisper-cpp). Used by the reasoner to flag transcripts
    # that may be mis-heard.
    confidence: float | None = None


@dataclass
class HarnessSession:
    profile_id: str
    emotion_buffer: EmotionBuffer
    transcript_buffer: list[TranscriptSegment]
    frame_count: int = 0
    audio_chunk_count: int = 0
    emotion_cycle_started_at: float = 0.0


# Keyed by profile_id. Written by the WS handler; read by the HTTP /chat route.
_sessions: dict[str, HarnessSession] = {}


def emit_debug(message: str) -> None:
    """Write a debug line to both logger and stdout."""
    logger.info(message)
    print(f"[harness] {message}", flush=True)


def get_session(profile_id: str) -> HarnessSession | None:
    """Look up an active session by profile ID; returns None if not found.

    Used by the HTTP /chat route to read the emotion buffer from the
    corresponding WebSocket session without sharing any global state directly.
    """
    return _sessions.get(profile_id)


def create_session(profile_id: str) -> HarnessSession:
    """Create and register a new HarnessSession, replacing any existing one."""
    session = HarnessSession(
        profile_id=profile_id,
        emotion_buffer=EmotionBuffer(),
        transcript_buffer=[],
        emotion_cycle_started_at=time.time(),
    )
    _sessions[profile_id] = session
    return session


def remove_session(profile_id: str) -> None:
    """Remove a session from the store (called on session_end or WS disconnect)."""
    _sessions.pop(profile_id, None)

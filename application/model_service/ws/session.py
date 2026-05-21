"""Session state shared between the WebSocket handler and the HTTP chat route."""

import logging
from dataclasses import dataclass, field
import time

from core.conductor import Conductor
from core.conductor.states import SESSION_FLOW
from core.emotion.buffer import EmotionBuffer
from core.events import SystemEvent

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
    # Per-session state machine. Each session gets its own Conductor instance
    # so different profiles can be at different points in the flow concurrently.
    conductor: Conductor = field(default_factory=lambda: Conductor(SESSION_FLOW))
    # Turns spent in the current conductor state. Reset on transition.
    turn_in_state: int = 0
    # Append-only log of typed system events (form answers, emotion windows,
    # segment summaries, silences). Merged with transcript_buffer at
    # LLM-prompt-assembly time via core.transcript_render.compose_stream.
    system_events: list[SystemEvent] = field(default_factory=list)
    # When the current state began, in session-clock seconds. Used to slice
    # the transcript / events down to "this state's conversation" for the
    # end-of-state fact extraction.
    state_started_at: float = 0.0
    # Sequential id incremented on each transition; used as the public
    # identifier for a segment in segment_summary events (so the LLM never
    # sees the internal state name).
    segment_id_counter: int = 0
    # Facts extracted at each state's end, keyed by internal state.name.
    state_facts: dict[str, dict] = field(default_factory=dict)
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
    now = time.time()
    session = HarnessSession(
        profile_id=profile_id,
        emotion_buffer=EmotionBuffer(),
        transcript_buffer=[],
        state_started_at=now,
        emotion_cycle_started_at=now,
    )
    _sessions[profile_id] = session
    return session


def remove_session(profile_id: str) -> None:
    """Remove a session from the store (called on session_end or WS disconnect)."""
    _sessions.pop(profile_id, None)

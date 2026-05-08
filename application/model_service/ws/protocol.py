from dataclasses import dataclass
from typing import Literal


# --- Inbound (frontend → harness) ---

@dataclass
class SessionStart:
    type: Literal["session_start"]
    profile_id: str


@dataclass
class VideoFrame:
    type: Literal["video_frame"]
    data: str       # base64 JPEG
    timestamp: float  # unix seconds from frontend


@dataclass
class AudioChunk:
    type: Literal["audio_chunk"]
    data: str       # base64 WAV
    timestamp: float  # unix seconds from frontend


@dataclass
class SessionEnd:
    type: Literal["session_end"]


# --- Outbound (harness → frontend) ---

@dataclass
class EmotionUpdate:
    type: Literal["emotion_update"]
    emotion: str
    confidence: float
    timestamp: float


@dataclass
class TranscriptChunk:
    type: Literal["transcript_chunk"]
    text: str
    timestamp: float


@dataclass
class ErrorMessage:
    type: Literal["error"]
    message: str

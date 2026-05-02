import json
import logging
import random
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect

from core.emotion.buffer import EmotionBuffer, EmotionObservation

logger = logging.getLogger(__name__)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


@dataclass
class TranscriptSegment:
    text: str
    timestamp: float  # unix seconds — from audio_chunk message


@dataclass
class HarnessSession:
    profile_id: str
    emotion_buffer: EmotionBuffer
    transcript_buffer: list[TranscriptSegment]


# Module-level session store — HTTP chat endpoint reads this by profile_id
_sessions: dict[str, HarnessSession] = {}


def get_session(profile_id: str) -> HarnessSession | None:
    return _sessions.get(profile_id)


async def handle_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    session: HarnessSession | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "session_start":
                profile_id = msg["profile_id"]
                session = HarnessSession(
                    profile_id=profile_id,
                    emotion_buffer=EmotionBuffer(),
                    transcript_buffer=[],
                )
                _sessions[profile_id] = session
                logger.info("Session started: %s", profile_id)

            elif msg_type == "session_end":
                if session:
                    _sessions.pop(session.profile_id, None)
                    logger.info("Session ended: %s", session.profile_id)
                    session = None
                break

            elif msg_type == "video_frame":
                if session is None:
                    continue
                timestamp = float(msg.get("timestamp", 0))

                # Stage 5: random emotion stub (Stage 6 will swap in real face detector)
                emotion = random.choice(EMOTIONS)
                confidence = round(random.uniform(0.5, 0.95), 2)
                session.emotion_buffer.update(emotion, confidence, timestamp)

                await websocket.send_text(json.dumps({
                    "type": "emotion_update",
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": timestamp,
                }))

            elif msg_type == "audio_chunk":
                if session is None:
                    continue
                timestamp = float(msg.get("timestamp", 0))

                # Stage 5: stub transcript (Stage 7 will swap in whisper.cpp)
                text = ""  # silent until STT wired
                if text:
                    segment = TranscriptSegment(text=text, timestamp=timestamp)
                    session.transcript_buffer.append(segment)
                    if len(session.transcript_buffer) > 20:
                        session.transcript_buffer = session.transcript_buffer[-20:]
                    await websocket.send_text(json.dumps({
                        "type": "transcript_chunk",
                        "text": text,
                        "timestamp": timestamp,
                    }))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                }))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        if session:
            _sessions.pop(session.profile_id, None)

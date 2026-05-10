"""WebSocket dispatcher — accepts connections and routes messages to handlers."""

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

import config
from ws.session import (
    HarnessSession,
    create_session,
    remove_session,
    emit_debug,
)
from ws.audio import process_audio_chunk
from ws.video import process_video_frame

logger = logging.getLogger(__name__)


def _harness_status(app_state) -> dict:
    """Build the harness_status payload from app.state component flags."""
    fd = getattr(app_state, "face_detector", None)
    return {
        "type": "harness_status",
        "face_detector_loaded": fd is not None,
        "face_detector_device": getattr(fd, "device", None),
        "face_detector_device_reason": getattr(fd, "device_reason", None),
        "torch_version": getattr(fd, "torch_version", None),
        "mps_built": getattr(fd, "mps_built", None),
        "mps_available": getattr(fd, "mps_available", None),
        "stt_loaded": getattr(app_state, "stt", None) is not None,
        "emotion_model_loaded": getattr(app_state, "emotion_model", None) is not None,
        "llm_loaded": getattr(app_state, "llm", None) is not None,
        "test_emotions": config.TEST_EMOTIONS,
        "stt_engine": config.STT_ENGINE,
        "stt_model": config.STT_MODEL,
    }


async def handle_websocket(websocket: WebSocket) -> None:
    """Accept a WebSocket connection and dispatch incoming messages.

    Message protocol (all JSON):
      Inbound:  session_start | session_end | video_frame | audio_chunk
      Outbound: connection_ack | message_ack | harness_status | face_detection
                | frame_debug | emotion_update | transcript_chunk | audio_debug | error

    A HarnessSession is created on session_start and stored in _sessions so the
    HTTP /chat route can read its emotion buffer when generating LLM responses.
    """
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "connection_ack", "message": "Harness websocket accepted"}))

    session: HarnessSession | None = None
    app_state = websocket.app.state

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            await websocket.send_text(json.dumps({"type": "message_ack", "message_type": msg_type}))

            if msg_type == "session_start":
                profile_id = msg["profile_id"]
                session = create_session(profile_id)
                emit_debug(
                    f"Session started: {profile_id} "
                    f"(test_emotions={config.TEST_EMOTIONS}, "
                    f"face_detector_loaded={getattr(app_state, 'face_detector', None) is not None})"
                )
                await websocket.send_text(json.dumps({**_harness_status(app_state), "profile_id": profile_id}))

            elif msg_type == "session_end":
                if session:
                    remove_session(session.profile_id)
                    emit_debug(f"Session ended: {session.profile_id}")
                    session = None
                break

            elif msg_type == "video_frame":
                if session is None:
                    await websocket.send_text(json.dumps({"type": "error", "message": "video_frame received before session_start"}))
                    continue
                await process_video_frame(
                    websocket, session,
                    getattr(app_state, "face_detector", None),
                    getattr(app_state, "emotion_model", None),
                    msg,
                )

            elif msg_type == "audio_chunk":
                if session is None:
                    await websocket.send_text(json.dumps({"type": "error", "message": "audio_chunk received before session_start"}))
                    continue
                timestamp = float(msg.get("timestamp", 0))
                session.audio_chunk_count += 1
                await websocket.send_text(json.dumps({
                    "type": "audio_received",
                    "audio_chunk_count": session.audio_chunk_count,
                    "byte_length": len(msg.get("data", "")),
                    "timestamp": timestamp,
                }))
                asyncio.create_task(process_audio_chunk(
                    websocket, session,
                    getattr(app_state, "stt", None),
                    session.audio_chunk_count,
                    msg.get("data", ""),
                    timestamp,
                ))

            else:
                await websocket.send_text(json.dumps({"type": "error", "message": f"Unknown message type: {msg_type}"}))

    except WebSocketDisconnect:
        emit_debug("WebSocket disconnected")
    except Exception as exc:
        logger.exception("WebSocket handler crashed")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": f"WebSocket handler crashed: {exc}"}))
        except Exception:
            pass
    finally:
        if session:
            remove_session(session.profile_id)

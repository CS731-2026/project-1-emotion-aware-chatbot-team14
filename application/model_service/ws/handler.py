"""WebSocket dispatcher — accepts connections and routes messages to typed handlers."""

import asyncio
import json
import logging
from typing import Callable, Awaitable

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

# Type alias: every message handler is an async function that returns True to
# continue the loop, or False to break (i.e. close the connection).
MessageHandler = Callable[[dict], Awaitable[bool]]


def _send(websocket: WebSocket, payload: dict) -> Awaitable[None]:
    return websocket.send_text(json.dumps(payload))


def _error(websocket: WebSocket, message: str) -> Awaitable[None]:
    return _send(websocket, {"type": "error", "message": message})


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


def _make_handlers(
    websocket: WebSocket,
    app_state,
    get_session: Callable[[], HarnessSession | None],
    set_session: Callable[[HarnessSession | None], None],
) -> dict[str, MessageHandler]:
    """Return a dispatch table mapping message type → handler coroutine.

    Each handler receives the raw parsed message dict and returns True to keep
    the loop running, or False to signal a clean close.

    Using a dispatch table instead of if/elif means adding a new message type
    is a single dict entry — no touching the main loop.
    """

    async def on_session_start(msg: dict) -> bool:
        profile_id = msg["profile_id"]
        session = create_session(profile_id)
        set_session(session)
        emit_debug(
            f"Session started: {profile_id} "
            f"(test_emotions={config.TEST_EMOTIONS}, "
            f"face_detector_loaded={getattr(app_state, 'face_detector', None) is not None})"
        )
        await _send(websocket, {**_harness_status(app_state), "profile_id": profile_id})
        return True

    async def on_session_end(_msg: dict) -> bool:
        session = get_session()
        if session:
            remove_session(session.profile_id)
            emit_debug(f"Session ended: {session.profile_id}")
            set_session(None)
        return False  # signal the loop to close

    async def on_video_frame(msg: dict) -> bool:
        session = get_session()
        if session is None:
            await _error(websocket, "video_frame received before session_start")
            return True
        await process_video_frame(
            websocket, session,
            getattr(app_state, "face_detector", None),
            getattr(app_state, "emotion_model", None),
            msg,
        )
        return True

    async def on_audio_chunk(msg: dict) -> bool:
        session = get_session()
        if session is None:
            await _error(websocket, "audio_chunk received before session_start")
            return True
        timestamp = float(msg.get("timestamp", 0))
        session.audio_chunk_count += 1
        await _send(websocket, {
            "type": "audio_received",
            "audio_chunk_count": session.audio_chunk_count,
            "byte_length": len(msg.get("data", "")),
            "timestamp": timestamp,
        })
        asyncio.create_task(process_audio_chunk(
            websocket, session,
            getattr(app_state, "stt", None),
            session.audio_chunk_count,
            msg.get("data", ""),
            timestamp,
        ))
        return True

    return {
        "session_start": on_session_start,
        "session_end":   on_session_end,
        "video_frame":   on_video_frame,
        "audio_chunk":   on_audio_chunk,
    }


async def handle_websocket(websocket: WebSocket) -> None:
    """Accept a WebSocket connection and dispatch messages via a handler table.

    Message protocol (all JSON):
      Inbound:  session_start | session_end | video_frame | audio_chunk
      Outbound: connection_ack | message_ack | harness_status | face_detection
                | frame_debug | emotion_update | transcript_chunk | audio_debug | error
    """
    await websocket.accept()
    await _send(websocket, {"type": "connection_ack", "message": "Harness websocket accepted"})

    # One-element list acts as a mutable cell so handler closures can rebind
    # the session reference without needing nonlocal or a class.
    cell: list[HarnessSession | None] = [None]

    handlers = _make_handlers(
        websocket,
        app_state=websocket.app.state,
        get_session=lambda: cell[0],
        set_session=lambda s: cell.__setitem__(0, s),
    )

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            await _send(websocket, {"type": "message_ack", "message_type": msg_type})

            handler = handlers.get(msg_type)
            if handler is None:
                await _error(websocket, f"Unknown message type: {msg_type}")
                continue

            keep_going = await handler(msg)
            if not keep_going:
                break

    except WebSocketDisconnect:
        emit_debug("WebSocket disconnected")
    except Exception as exc:
        logger.exception("WebSocket handler crashed")
        try:
            await _error(websocket, f"WebSocket handler crashed: {exc}")
        except Exception:
            pass
    finally:
        if cell[0]:
            remove_session(cell[0].profile_id)

"""WebSocket dispatcher — accepts connections and routes messages to typed handlers."""

import asyncio
import json
import logging
import random
from typing import Any, Callable, Awaitable, cast

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

import config
from core.app_state import HRIAppState
from core.emotion.base import EMOTIONS, EmotionModel
from ws.session import (
    HarnessSession,
    create_session,
    remove_session,
    emit_debug,
)
from ws.audio import process_audio_chunk
from ws.video import FrameDetectionResult, detect_from_message

logger = logging.getLogger(__name__)

# Type alias: every message handler is an async function that returns True to
# continue the loop, or False to break (i.e. close the connection).
MessageHandler = Callable[[dict[str, Any]], Awaitable[bool]]


def _send(websocket: WebSocket, payload: dict[str, Any]) -> Awaitable[None]:
    return websocket.send_text(json.dumps(payload))


def _error(websocket: WebSocket, message: str) -> Awaitable[None]:
    return _send(websocket, {"type": "error", "message": message})


def _harness_status(hri: HRIAppState) -> dict[str, Any]:
    """Build the harness_status payload sent to the frontend on session_start.

    Observability only — reports which components loaded. The emotion_model
    is never invoked here; invocation is in pick_emotion() below.
    """
    fd = hri.face_detector
    return {
        "type": "harness_status",
        "face_detector_loaded": fd is not None,
        "face_detector_device": fd.device if fd is not None else None,
        "face_detector_device_reason": fd.device_reason if fd is not None else None,
        "torch_version": fd.torch_version if fd is not None else None,
        "mps_built": fd.mps_built if fd is not None else None,
        "mps_available": fd.mps_available if fd is not None else None,
        "stt_loaded": hri.stt is not None,
        "emotion_model_loaded": hri.emotion_model is not None,
        "llm_loaded": hri.llm is not None,
        "test_emotions": config.TEST_EMOTIONS,
        "stt_engine": config.STT_ENGINE,
        "stt_model": config.STT_MODEL,
    }


def pick_emotion(
    face_crop: np.ndarray | None,
    emotion_model: EmotionModel | None,
    detected: bool,
) -> tuple[str, float]:
    """Invoke the emotion model or fall back to debug/neutral values.

    Priority:
      1. Real model  — face detected + model loaded + TEST_EMOTIONS=false
      2. Random      — DEBUG: TEST_EMOTIONS=true; bypasses model entirely
      3. Neutral     — no face and TEST_EMOTIONS=false

    emotion_model.predict() is the only model invocation point in the codebase.
    Lives here (not in ws/video.py) because it owns the model call — video.py
    only prepares the face crop.
    """
    if detected and face_crop is not None and emotion_model is not None:
        return emotion_model.predict(face_crop)

    if config.TEST_EMOTIONS:
        return random.choice(EMOTIONS), round(random.uniform(0.5, 0.8), 2)

    return "neutral", 0.5


async def _send_frame_messages(
    websocket: WebSocket,
    session: HarnessSession,
    result: FrameDetectionResult,
    emotion: str,
    confidence: float,
    timestamp: float,
) -> None:
    """Send the three per-frame WS messages: face_detection, frame_debug, emotion_update."""
    await websocket.send_text(json.dumps({
        "type": "face_detection",
        "detected": result.detected,
        "detector_loaded": result.detector_loaded,
        "timestamp": timestamp,
    }))
    await websocket.send_text(json.dumps({
        "type": "frame_debug",
        "frame_count": session.frame_count,
        "detected": result.detected,
        "detector_loaded": result.detector_loaded,
        "box": result.box,
        "timings_ms": result.timings_ms,
        "image_data": result.annotated_image_data,
        "face_crop_data": result.face_crop_data,
        "timestamp": timestamp,
    }))
    await websocket.send_text(json.dumps({
        "type": "emotion_update",
        "emotion": emotion,
        "confidence": confidence,
        "timestamp": timestamp,
    }))


def _make_handlers(
    websocket: WebSocket,
    hri: HRIAppState,
    get_session: Callable[[], HarnessSession | None],
    set_session: Callable[[HarnessSession | None], None],
) -> dict[str, MessageHandler]:
    """Return a dispatch table mapping message type → handler coroutine.

    Each handler receives the raw parsed message dict and returns True to keep
    the loop running, or False to signal a clean close.

    Using a dispatch table instead of if/elif means adding a new message type
    is a single dict entry — no touching the main loop.
    """

    async def on_session_start(msg: dict[str, Any]) -> bool:
        profile_id = msg["profile_id"]
        session = create_session(profile_id)
        set_session(session)
        emit_debug(
            f"Session started: {profile_id} "
            f"(test_emotions={config.TEST_EMOTIONS}, "
            f"face_detector_loaded={hri.face_detector is not None})"
        )
        await _send(websocket, {**_harness_status(hri), "profile_id": profile_id})
        return True

    async def on_session_end(_msg: dict[str, Any]) -> bool:
        session = get_session()
        if session:
            remove_session(session.profile_id)
            emit_debug(f"Session ended: {session.profile_id}")
            set_session(None)
        return False  # signal the loop to close

    async def on_video_frame(msg: dict[str, Any]) -> bool:
        session = get_session()
        if session is None:
            await _error(websocket, "video_frame received before session_start")
            return True

        timestamp = float(msg.get("timestamp", 0))
        session.frame_count += 1

        result = detect_from_message(hri.face_detector, msg, session.frame_count)
        emotion, confidence = pick_emotion(result.face_crop, hri.emotion_model, result.detected)
        session.emotion_buffer.update(emotion, confidence, timestamp)

        if session.frame_count == 1 or session.frame_count % 10 == 0:
            emit_debug(
                f"Frame {session.frame_count} [{session.profile_id}]: "
                f"detector_loaded={result.detector_loaded} detected={result.detected} emotion={emotion}"
            )

        await _send_frame_messages(websocket, session, result, emotion, confidence, timestamp)
        return True

    async def on_audio_chunk(msg: dict[str, Any]) -> bool:
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
            hri.stt,
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

    hri = cast(HRIAppState, websocket.app.state.hri)

    # One-element list acts as a mutable cell so handler closures can rebind
    # the session reference without needing nonlocal or a class.
    cell: list[HarnessSession | None] = [None]

    handlers = _make_handlers(
        websocket,
        hri,
        get_session=lambda: cell[0],
        set_session=lambda s: cell.__setitem__(0, s),
    )

    try:
        while True:
            raw = await websocket.receive_text()
            msg: dict[str, Any] = json.loads(raw)
            msg_type: str | None = msg.get("type")

            await _send(websocket, {"type": "message_ack", "message_type": msg_type})

            handler = handlers.get(msg_type) if msg_type else None
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

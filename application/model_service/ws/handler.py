import asyncio
import json
import logging
import random
import base64
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

import config
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
    frame_count: int = 0
    audio_chunk_count: int = 0


# Module-level session store — HTTP chat endpoint reads this by profile_id
_sessions: dict[str, HarnessSession] = {}


def emit_debug(message: str) -> None:
    logger.info(message)
    print(f"[harness] {message}", flush=True)


def get_session(profile_id: str) -> HarnessSession | None:
    return _sessions.get(profile_id)


def decode_browser_audio_to_numpy(data: str):
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="hri_audio_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "chunk.webm"
        output_path = tmp_dir / "chunk.s16le"
        input_path.write_bytes(base64.b64decode(data))

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            str(output_path),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or f"exit status {exc.returncode}"
            raise RuntimeError(f"ffmpeg browser audio decode failed: {detail}") from exc

        pcm = np.frombuffer(output_path.read_bytes(), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0


def encode_jpeg_b64(frame_bgr) -> str | None:
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


async def process_audio_chunk(
    websocket: WebSocket,
    session: HarnessSession,
    stt,
    chunk_count: int,
    data: str,
    timestamp: float,
) -> None:
    text = ""
    stt_error = None
    audio_timings_ms: dict[str, float] = {}

    if stt is not None:
        try:
            audio_started = time.perf_counter()
            emit_debug(
                "Audio chunk "
                f"{chunk_count}: decoding browser audio before whisper.cpp"
            )
            audio_np = await asyncio.to_thread(decode_browser_audio_to_numpy, data)
            audio_timings_ms["ffmpeg_decode"] = round((time.perf_counter() - audio_started) * 1000, 1)
            emit_debug(
                "Audio chunk "
                f"{chunk_count}: decoded {audio_np.shape[0]} samples; "
                "running whisper.cpp in worker thread"
            )
            whisper_started = time.perf_counter()
            text, _language, _confidence = await asyncio.to_thread(stt.transcribe, audio_np)
            audio_timings_ms["whisper_cpp"] = round((time.perf_counter() - whisper_started) * 1000, 1)
            audio_timings_ms["total"] = round((time.perf_counter() - audio_started) * 1000, 1)
            emit_debug(
                "Audio chunk "
                f"{chunk_count}: whisper.cpp returned {text!r}; "
                f"timings_ms={audio_timings_ms}"
            )
        except Exception as exc:
            stt_error = str(exc)
            logger.warning("STT failed for audio chunk: %s", exc)
            print(f"[harness] STT failed for audio chunk: {exc}", flush=True)

    if not text:
        text = "[whisper.cpp returned no text]"

    emit_debug(
        "Audio chunk "
        f"{chunk_count} for {session.profile_id} "
        f"received at {timestamp:.3f}; transcript={text!r}"
    )

    segment = TranscriptSegment(text=text, timestamp=timestamp)
    session.transcript_buffer.append(segment)
    if len(session.transcript_buffer) > 20:
        session.transcript_buffer = session.transcript_buffer[-20:]

    try:
        await websocket.send_text(json.dumps({
            "type": "transcript_chunk",
            "text": text,
            "audio_chunk_count": chunk_count,
            "timestamp": timestamp,
        }))
        await websocket.send_text(json.dumps({
            "type": "audio_debug",
            "audio_chunk_count": chunk_count,
            "byte_length": len(data),
            "text": text,
            "stt_engine": config.STT_ENGINE,
            "stt_loaded": stt is not None,
            "stt_error": stt_error,
            "timings_ms": audio_timings_ms,
            "timestamp": timestamp,
        }))
    except Exception as exc:
        emit_debug(f"Audio chunk {chunk_count}: could not send STT result to frontend: {exc}")


async def handle_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_text(json.dumps({
        "type": "connection_ack",
        "message": "Harness websocket accepted",
    }))
    session: HarnessSession | None = None
    app = websocket.app

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")
            await websocket.send_text(json.dumps({
                "type": "message_ack",
                "message_type": msg_type,
            }))

            if msg_type == "session_start":
                profile_id = msg["profile_id"]
                session = HarnessSession(
                    profile_id=profile_id,
                    emotion_buffer=EmotionBuffer(),
                    transcript_buffer=[],
                )
                _sessions[profile_id] = session
                emit_debug(
                    "Session started: "
                    f"{profile_id} "
                    f"(test_emotions={config.TEST_EMOTIONS}, "
                    f"face_detector_loaded={getattr(app.state, 'face_detector', None) is not None}, "
                    f"face_detector_device={getattr(getattr(app.state, 'face_detector', None), 'device', 'none')}, "
                    f"device_reason={getattr(getattr(app.state, 'face_detector', None), 'device_reason', 'unknown')})"
                )
                await websocket.send_text(json.dumps({
                    "type": "harness_status",
                    "profile_id": profile_id,
                    "face_detector_loaded": getattr(app.state, "face_detector", None) is not None,
                    "face_detector_device": getattr(getattr(app.state, "face_detector", None), "device", None),
                    "face_detector_device_reason": getattr(getattr(app.state, "face_detector", None), "device_reason", None),
                    "torch_version": getattr(getattr(app.state, "face_detector", None), "torch_version", None),
                    "mps_built": getattr(getattr(app.state, "face_detector", None), "mps_built", None),
                    "mps_available": getattr(getattr(app.state, "face_detector", None), "mps_available", None),
                    "stt_loaded": getattr(app.state, "stt", None) is not None,
                    "emotion_model_loaded": getattr(app.state, "emotion_model", None) is not None,
                    "llm_loaded": getattr(app.state, "llm", None) is not None,
                    "test_emotions": config.TEST_EMOTIONS,
                    "stt_engine": config.STT_ENGINE,
                    "stt_model": config.STT_MODEL,
                }))

            elif msg_type == "session_end":
                if session:
                    _sessions.pop(session.profile_id, None)
                    emit_debug(f"Session ended: {session.profile_id}")
                    session = None
                break

            elif msg_type == "video_frame":
                if session is None:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "video_frame received before session_start",
                    }))
                    continue
                timestamp = float(msg.get("timestamp", 0))
                session.frame_count += 1

                detected = False
                box = None
                annotated_image_data = msg["data"]
                face_crop_data = None
                detector_loaded = getattr(app.state, "face_detector", None) is not None
                face_detector = getattr(app.state, "face_detector", None)
                timings_ms: dict[str, float] = {}

                if face_detector is not None:
                    try:
                        import cv2
                        import numpy as np

                        frame_started = time.perf_counter()
                        emit_debug(
                            "Frame "
                            f"{session.frame_count}: received {len(msg.get('data', ''))} base64 chars; "
                            "decoding before YOLO"
                        )
                        frame_bytes = base64.b64decode(msg["data"])
                        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        timings_ms["decode"] = round((time.perf_counter() - frame_started) * 1000, 1)

                        if frame_bgr is not None:
                            emit_debug(
                                "Frame "
                                f"{session.frame_count}: decoded image shape={frame_bgr.shape}; "
                                f"running YOLOv8 face detector on {getattr(face_detector, 'device', 'unknown')}"
                            )
                            yolo_started = time.perf_counter()
                            face_crop, detected_box = face_detector.detect_best(frame_bgr)
                            timings_ms["yolo"] = round((time.perf_counter() - yolo_started) * 1000, 1)
                            detected = face_crop is not None
                            box = detected_box.tolist() if detected_box is not None else None
                            if detected_box is not None:
                                x1, y1, x2, y2 = detected_box.astype(int)
                                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(
                                    frame_bgr,
                                    "YOLOv8 face",
                                    (x1, max(24, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                )
                            encode_started = time.perf_counter()
                            annotated_image_data = encode_jpeg_b64(frame_bgr) or msg["data"]
                            if face_crop is not None:
                                face_crop_data = encode_jpeg_b64(face_crop)
                            timings_ms["jpeg_encode"] = round((time.perf_counter() - encode_started) * 1000, 1)
                            timings_ms["total_before_send"] = round((time.perf_counter() - frame_started) * 1000, 1)
                            emit_debug(
                                "Frame "
                                f"{session.frame_count}: YOLO complete; "
                                f"detected={detected}; box={box}; timings_ms={timings_ms}"
                            )
                        else:
                            emit_debug(f"Frame {session.frame_count}: cv2.imdecode returned None")
                    except Exception as exc:
                        logger.warning("Face detection failed for frame: %s", exc)
                        print(f"[harness] Face detection failed for frame: {exc}", flush=True)

                if config.TEST_EMOTIONS:
                    emotion = random.choice(EMOTIONS)
                else:
                    emotion = "happy" if detected else random.choice(EMOTIONS)
                confidence = 0.95 if detected else round(random.uniform(0.5, 0.8), 2)
                session.emotion_buffer.update(emotion, confidence, timestamp)

                if session.frame_count == 1 or session.frame_count % 10 == 0:
                    emit_debug(
                        "Video frame "
                        f"{session.frame_count} for {session.profile_id}: "
                        f"detector_loaded={detector_loaded} "
                        f"detected={detected} "
                        f"emotion={emotion}"
                    )

                await websocket.send_text(json.dumps({
                    "type": "face_detection",
                    "detected": detected,
                    "detector_loaded": detector_loaded,
                    "timestamp": timestamp,
                }))

                await websocket.send_text(json.dumps({
                    "type": "frame_debug",
                    "frame_count": session.frame_count,
                    "detected": detected,
                    "detector_loaded": detector_loaded,
                    "detector_device": getattr(face_detector, "device", None),
                    "box": box,
                    "timings_ms": timings_ms,
                    "image_data": annotated_image_data,
                    "face_crop_data": face_crop_data,
                    "timestamp": timestamp,
                }))

                await websocket.send_text(json.dumps({
                    "type": "emotion_update",
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": timestamp,
                }))

            elif msg_type == "audio_chunk":
                if session is None:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "audio_chunk received before session_start",
                    }))
                    continue
                timestamp = float(msg.get("timestamp", 0))
                session.audio_chunk_count += 1
                await websocket.send_text(json.dumps({
                    "type": "audio_received",
                    "audio_chunk_count": session.audio_chunk_count,
                    "byte_length": len(msg.get("data", "")),
                    "timestamp": timestamp,
                }))

                stt = getattr(app.state, "stt", None)
                asyncio.create_task(process_audio_chunk(
                    websocket,
                    session,
                    stt,
                    session.audio_chunk_count,
                    msg.get("data", ""),
                    timestamp,
                ))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                }))

    except WebSocketDisconnect:
        emit_debug("WebSocket disconnected")
    except Exception as exc:
        logger.exception("WebSocket handler crashed")
        print(f"[harness] WebSocket handler crashed: {exc}", flush=True)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"WebSocket handler crashed: {exc}",
            }))
        except Exception:
            pass
    finally:
        if session:
            _sessions.pop(session.profile_id, None)

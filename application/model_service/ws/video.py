"""Video frame processing: JPEG decode → face detection → emotion → WS responses."""

import base64
import json
import logging
import random
import time

import numpy as np
from fastapi import WebSocket

import config
from ws.session import HarnessSession, EMOTIONS, emit_debug

logger = logging.getLogger(__name__)


def encode_jpeg_b64(frame_bgr: np.ndarray) -> str | None:
    """JPEG-encode a BGR frame and return it as a base64 ASCII string.

    Quality is set to 70 — enough for debug display; reduces WS payload size.
    Returns None if encoding fails.
    """
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _run_face_detection(
    face_detector,
    frame_bgr: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray | None, list | None, np.ndarray, dict]:
    """Run YOLO face detection on a decoded frame.

    Returns:
        face_crop:      BGR crop of the best face, or None
        box:            [x1,y1,x2,y2] as a plain list, or None
        annotated_bgr:  frame with bounding box drawn on it
        timings_ms:     dict of per-step latencies
    """
    import cv2

    timings_ms: dict[str, float] = {}
    face_crop = None
    box = None

    t0 = time.perf_counter()
    emit_debug(
        f"Frame {frame_count}: decoded image shape={frame_bgr.shape}; "
        f"running YOLOv8 on {getattr(face_detector, 'device', 'unknown')}"
    )

    face_crop, detected_box = face_detector.detect_best(frame_bgr)
    timings_ms["yolo"] = round((time.perf_counter() - t0) * 1000, 1)

    if detected_box is not None:
        box = detected_box.tolist()
        x1, y1, x2, y2 = detected_box.astype(int)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame_bgr, "YOLOv8 face",
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

    return face_crop, box, frame_bgr, timings_ms


def _pick_emotion(
    face_crop: np.ndarray | None,
    emotion_model,
    detected: bool,
) -> tuple[str, float]:
    """Select an emotion label and confidence from the available sources.

    Priority:
      1. Real emotion model (when loaded and a face crop is available)
      2. Random placeholder (TEST_EMOTIONS=true, default)
      3. Neutral fallback
    """
    if detected and face_crop is not None and emotion_model is not None:
        return emotion_model.predict(face_crop)

    if config.TEST_EMOTIONS:
        return random.choice(EMOTIONS), round(random.uniform(0.5, 0.8), 2)

    return "neutral", 0.5


async def process_video_frame(
    websocket: WebSocket,
    session: HarnessSession,
    face_detector,
    emotion_model,
    msg: dict,
) -> None:
    """Handle one video_frame WebSocket message end-to-end.

    Steps:
      1. Decode base64 JPEG → BGR numpy array
      2. Run YOLO face detector → face crop + bounding box
      3. Pick emotion (real model if available, otherwise placeholder)
      4. Update emotion buffer
      5. Send face_detection, frame_debug, and emotion_update WS messages
    """
    import cv2

    timestamp = float(msg.get("timestamp", 0))
    session.frame_count += 1

    detected = False
    box = None
    annotated_image_data = msg["data"]
    face_crop_data = None
    face_crop = None
    timings_ms: dict[str, float] = {}
    detector_loaded = face_detector is not None

    if face_detector is not None:
        try:
            t0 = time.perf_counter()
            emit_debug(
                f"Frame {session.frame_count}: received {len(msg.get('data', ''))} base64 chars; "
                "decoding before YOLO"
            )

            frame_bytes = base64.b64decode(msg["data"])
            frame_bgr = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            timings_ms["decode"] = round((time.perf_counter() - t0) * 1000, 1)

            if frame_bgr is not None:
                face_crop, box, frame_bgr, yolo_timings = _run_face_detection(
                    face_detector, frame_bgr, session.frame_count
                )
                timings_ms.update(yolo_timings)
                detected = face_crop is not None

                t_enc = time.perf_counter()
                annotated_image_data = encode_jpeg_b64(frame_bgr) or msg["data"]
                if face_crop is not None:
                    face_crop_data = encode_jpeg_b64(face_crop)
                timings_ms["jpeg_encode"] = round((time.perf_counter() - t_enc) * 1000, 1)
                timings_ms["total_before_send"] = round((time.perf_counter() - t0) * 1000, 1)

                emit_debug(
                    f"Frame {session.frame_count}: YOLO complete; "
                    f"detected={detected}; box={box}; timings_ms={timings_ms}"
                )
            else:
                emit_debug(f"Frame {session.frame_count}: cv2.imdecode returned None")

        except Exception as exc:
            logger.warning("Face detection failed for frame: %s", exc)

    emotion, confidence = _pick_emotion(face_crop, emotion_model, detected)
    session.emotion_buffer.update(emotion, confidence, timestamp)

    if session.frame_count == 1 or session.frame_count % 10 == 0:
        emit_debug(
            f"Video frame {session.frame_count} for {session.profile_id}: "
            f"detector_loaded={detector_loaded} detected={detected} emotion={emotion}"
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

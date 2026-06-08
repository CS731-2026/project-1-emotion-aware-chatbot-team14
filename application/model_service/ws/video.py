"""Video frame utilities: JPEG decode, face detection, encoding.

Pure frame-processing functions, no emotion logic, no WebSocket state.
The caller (ws/handler.py:on_video_frame) composes these into the pipeline.
"""

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.face_detector import FaceDetector
from ws.session import emit_debug

logger = logging.getLogger(__name__)


@dataclass
class FrameDetectionResult:
    """Output of detect_from_message: everything the handler needs after frame processing."""
    detected: bool = False
    detector_loaded: bool = False
    face_crop: np.ndarray | None = None
    face_crop_data: str | None = None       # base64 JPEG of face crop, or None
    box: list[float] | None = None          # [x1, y1, x2, y2] or None
    annotated_image_data: str = ""          # base64 JPEG with bounding box drawn, or original
    timings_ms: dict[str, float] = field(default_factory=dict)


def encode_jpeg_b64(frame_bgr: np.ndarray) -> str | None:
    """JPEG-encode a BGR frame and return it as a base64 ASCII string.

    Quality 70, enough for debug display; reduces WS payload size.
    Returns None if encoding fails.
    """
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_frame(data: str) -> tuple[np.ndarray | None, float]:
    """Decode a base64 JPEG string to a BGR numpy array.

    Returns (frame_bgr, decode_ms). frame_bgr is None on decode failure.
    """
    import cv2

    t0 = time.perf_counter()
    try:
        frame_bytes = base64.b64decode(data)
        frame_bgr = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame_bgr, round((time.perf_counter() - t0) * 1000, 1)
    except Exception:
        return None, 0.0


def run_face_detection(
    face_detector: FaceDetector,
    frame_bgr: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray | None, list[float] | None, np.ndarray, dict[str, float]]:
    """Run YOLO face detection on a decoded frame.

    Returns:
        face_crop:      BGR crop of the best face, or None
        box:            [x1, y1, x2, y2] as a plain list, or None
        annotated_bgr:  frame with bounding box drawn on it
        timings_ms:     dict of per-step latencies
    """
    import cv2

    emit_debug(
        f"Frame {frame_count}: shape={frame_bgr.shape}; "
        f"running YOLOv8 on {face_detector.device}"
    )

    t0 = time.perf_counter()
    face_crop, detected_box = face_detector.detect_best(frame_bgr)
    timings_ms: dict[str, float] = {"yolo": round((time.perf_counter() - t0) * 1000, 1)}

    box: list[float] | None = None
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


def detect_from_message(
    face_detector: FaceDetector | None,
    msg: dict[str, Any],
    frame_count: int,
) -> FrameDetectionResult:
    """Decode a video_frame WS message and run face detection.

    Combines decode_frame → run_face_detection → encode annotated output into a
    single call so the handler doesn't manage intermediate variables.
    Returns a FrameDetectionResult regardless of whether detection succeeded.
    """
    result = FrameDetectionResult(
        detector_loaded=face_detector is not None,
        annotated_image_data=msg["data"],
    )

    emit_debug(
        f"Frame {frame_count}: received {len(msg.get('data', ''))} base64 chars; decoding"
    )

    frame_bgr, decode_ms = decode_frame(msg["data"])
    if decode_ms:
        result.timings_ms["decode"] = decode_ms

    if face_detector is None or frame_bgr is None:
        return result

    try:
        face_crop, box, annotated_bgr, yolo_timings = run_face_detection(
            face_detector, frame_bgr, frame_count
        )
        result.timings_ms.update(yolo_timings)

        t_enc = time.perf_counter()
        result.annotated_image_data = encode_jpeg_b64(annotated_bgr) or msg["data"]
        result.timings_ms["jpeg_encode"] = round((time.perf_counter() - t_enc) * 1000, 1)

        result.face_crop = face_crop
        result.detected = face_crop is not None
        result.box = box
        if face_crop is not None:
            result.face_crop_data = encode_jpeg_b64(face_crop)

        emit_debug(
            f"Frame {frame_count}: YOLO done; "
            f"detected={result.detected}; box={box}; timings_ms={result.timings_ms}"
        )
    except Exception as exc:
        logger.warning("Face detection failed for frame: %s", exc)

    return result

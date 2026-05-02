"""Face detector using YOLOv8 from HuggingFace.

Model: arnabdhar/YOLOv8-Face-Detection
Uses supervision.Detections.from_ultralytics() to parse results.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np


# Resolve models/ directory relative to this file's location inside the service.
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _SERVICE_ROOT / "models"


class FaceDetector:
    """Detects faces in a BGR frame using YOLOv8.

    On init the model is downloaded from HuggingFace (cached after first run)
    and loaded via ultralytics.YOLO.
    """

    REPO_ID = "arnabdhar/YOLOv8-Face-Detection"
    FILENAME = "model.pt"
    CONF_THRESHOLD = 0.35

    def __init__(self) -> None:
        from huggingface_hub import hf_hub_download
        import torch
        from ultralytics import YOLO

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.FILENAME,
            cache_dir=str(_MODELS_DIR),
            force_download=False,
            resume_download=True,
        )

        self._model = YOLO(model_path)
        self.torch_version = getattr(torch, "__version__", "unknown")
        self.mps_built = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()
        )
        self.mps_available = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        self.device = self._select_device(torch)
        self.device_reason = self._describe_device(torch)
        self.last_inference_ms: float | None = None

    def _select_device(self, torch) -> str:
        if self.mps_available:
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _describe_device(self, torch) -> str:
        if self.device == "mps":
            return "PyTorch MPS is available"
        if self.device == "cuda":
            return "CUDA is available"
        if self.mps_built and not self.mps_available:
            return "PyTorch was built with MPS but reports MPS unavailable"
        return "No GPU backend reported by PyTorch"

    def detect_best(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Detect faces and return the crop + bounding box of the largest face.

        Args:
            frame_bgr: A BGR image as a numpy array (H, W, 3).

        Returns:
            (face_crop_bgr, box_xyxy) for the highest-confidence face, or
            (None, None) if no face is detected.

            face_crop_bgr: uint8 BGR crop of the detected face region.
            box_xyxy:      float32 array [x1, y1, x2, y2] in pixel coordinates.
        """
        from supervision import Detections

        start = time.perf_counter()
        results = self._model(
            frame_bgr,
            conf=self.CONF_THRESHOLD,
            verbose=False,
            device=self.device,
        )
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        r = results[0]

        detections = Detections.from_ultralytics(r)

        if len(detections) == 0:
            return None, None

        # Pick the detection with the highest confidence score.
        confidences = detections.confidence  # shape (N,)
        best_idx = int(np.argmax(confidences))
        box = detections.xyxy[best_idx]  # [x1, y1, x2, y2]

        x1, y1, x2, y2 = box.astype(int)

        # Clamp to frame boundaries.
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, None

        face_crop = frame_bgr[y1:y2, x1:x2]
        return face_crop, box.astype(np.float32)

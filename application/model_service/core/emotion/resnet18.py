"""ResNet18 emotion classifier (EmpathBot 6-class).

Loads a torch state_dict produced by the team training pipeline
(experiments/emotion_recognition/training.ipynb or Notebooks/2_benchmark_resnet18.ipynb)
and emits an EmpathBot label per face crop.

Expected checkpoint shape: ResNet18 with the final fc replaced by
Linear(in_features, len(EMOTIONS)). Pure state_dict — not a pickled module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import EMOTIONS, EmotionModel

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 224


class ResNet18EmotionModel(EmotionModel):
    """Fine-tuned ResNet18 head; outputs softmax over EmpathBot 6 classes."""

    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        # Lazy import: torch is heavy; only paid when this variant is actually used.
        import torch
        from torchvision.models import resnet18

        self._torch = torch

        resolved_device = device or self._auto_device(torch)
        self._device = torch.device(resolved_device)

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"ResNet18 checkpoint not found at {ckpt.resolve()}. "
                "Set EMOTION_CHECKPOINT_PATH or place the file at this path."
            )

        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(EMOTIONS))

        state_dict = torch.load(ckpt, map_location=self._device)
        # Tolerate a few common checkpoint envelopes.
        if isinstance(state_dict, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break

        model.load_state_dict(state_dict, strict=True)
        model.to(self._device).eval()
        self._model = model

        logger.info(
            "ResNet18 emotion model loaded: checkpoint=%s device=%s classes=%d",
            ckpt, self._device, len(EMOTIONS),
        )

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        torch = self._torch

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_AREA)

        # uint8 [0, 255] -> float32 [0, 1] -> ImageNet-normalised CHW.
        x = face_rgb.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            confidence = float(probs[idx].item())

        return EMOTIONS[idx], confidence

    @staticmethod
    def _auto_device(torch_module) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        # MPS is fine for ResNet18 inference; fall back to CPU when unavailable.
        if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

"""EmpathBotV1 emotion classifier.

EfficientNet-B2 backbone (via timm) + 3-layer BN classifier head. Ported
from Notebooks/6b_empathbot_v1_improvements.ipynb — the team's trained
EmpathBot model. Checkpoint envelope keys: model_state, class_names, cfg.

Output classes match the EmpathBot 6-label schema in base.EMOTIONS:
  0 neutral · 1 trust_relief · 2 sadness · 3 fear_anxiety · 4 confusion · 5 distrust
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


def _make_head(in_features: int, num_classes: int):
    """Reproduce the head used in 6b_empathbot_v1_improvements.ipynb."""
    import torch.nn as nn

    mid = max(in_features // 2, 256)
    return nn.Sequential(
        nn.Linear(in_features, mid),
        nn.BatchNorm1d(mid),
        nn.ReLU(inplace=True),
        nn.Dropout(0.35),
        nn.Linear(mid, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )


class EmpathBotEmotionModel(EmotionModel):
    """EfficientNet-B2 + 3-layer BN head. Softmax over EmpathBot 6 classes."""

    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        # Lazy imports — torch + timm are heavy; only loaded when this
        # variant is selected.
        import timm
        import torch
        import torch.nn as nn

        self._torch = torch

        resolved_device = device or self._auto_device(torch)
        self._device = torch.device(resolved_device)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"EmpathBot checkpoint not found at {ckpt_path.resolve()}. "
                "Set EMOTION_CHECKPOINT_PATH or place the file at this path."
            )

        # Build the same architecture the notebook trained — encoder + head.
        encoder = timm.create_model("efficientnet_b2", pretrained=False, num_classes=0)
        feat_dim = encoder.num_features
        head = _make_head(feat_dim, len(EMOTIONS))

        class _EmpathBot(nn.Module):
            def __init__(self, encoder, head):
                super().__init__()
                self.encoder = encoder
                self.head = head

            def forward(self, x):
                return self.head(self.encoder(x))

        model = _EmpathBot(encoder, head)

        # The training checkpoint wraps the state under "model_state" and
        # also stores class_names + cfg metadata.
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            ckpt_classes = ckpt.get("class_names")
            if ckpt_classes is not None:
                # Sanity-check: checkpoint class order must match EMOTIONS.
                ckpt_order = [ckpt_classes[i] for i in range(len(ckpt_classes))]
                if ckpt_order != EMOTIONS:
                    raise ValueError(
                        f"Checkpoint class order {ckpt_order} does not match "
                        f"base.EMOTIONS {EMOTIONS}. Reorder or update one."
                    )
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=True)
        model.to(self._device).eval()
        self._model = model

        logger.info(
            "EmpathBot emotion model loaded: checkpoint=%s device=%s classes=%d",
            ckpt_path, self._device, len(EMOTIONS),
        )

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        torch = self._torch

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_AREA)

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
        if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

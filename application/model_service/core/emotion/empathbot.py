"""EmpathBotV1 emotion classifier (EmpathBot 6-class).

Loads the checkpoint produced by Notebooks/6b_empathbot_v1_improvements.ipynb.
The training notebook is the source of truth: this module ports the
EmpathBotV1 module + the val-time preprocessing (VAL_TF) verbatim.

Expected checkpoint envelope:
    {
        'epoch':          int,
        'model_state':    state_dict,
        'val_acc':        float,
        'per_cls_recall': list[float],
        'class_names':    dict[int, str],   # asserted against EMOTIONS
        'cfg':            dict,              # provides backbone / use_timm / img_size
    }
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import EMOTIONS, EmotionModel

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _make_head(in_features: int, num_classes: int):
    """Ported from Notebooks/6b_empathbot_v1_improvements.ipynb cell 9."""
    from torch import nn

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


def _build_empathbot(num_classes: int, backbone: str, use_timm: bool):
    """Build the EmpathBotV1 architecture without downloading pretrained weights.

    pretrained=False matters: we're about to overwrite the encoder with our
    own checkpoint, so downloading ImageNet weights wastes time and bandwidth.
    """
    import timm
    from torch import nn
    from torchvision import models as tvm

    class EmpathBotV1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone_name = backbone
            self.use_timm = use_timm

            if use_timm:
                self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
                feat_dim = self.encoder.num_features
            else:
                base = tvm.efficientnet_b2(weights=None)
                self.encoder = base.features
                self.pool = nn.AdaptiveAvgPool2d(1)
                feat_dim = 1408

            self.head = _make_head(feat_dim, num_classes)

        def forward(self, x):
            if self.use_timm:
                x = self.encoder(x)
            else:
                x = self.pool(self.encoder(x)).flatten(1)
            return self.head(x)

    return EmpathBotV1()


class EmpathBotEmotionModel(EmotionModel):
    """EmpathBotV1 inference wrapper, loads a notebook-6b-style checkpoint."""

    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        import torch
        from torchvision import transforms as T

        self._torch = torch

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"EmpathBot checkpoint not found at {ckpt_path.resolve()}. "
                "Set the path in models.yaml or place the file there."
            )

        resolved_device = device or self._auto_device(torch)
        self._device = torch.device(resolved_device)

        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)

        # Schema assertion, model is the source of truth.
        ckpt_classes = ckpt.get("class_names")
        if ckpt_classes is None:
            raise ValueError(
                f"Checkpoint {ckpt_path} missing 'class_names'. Re-export from "
                "Notebooks/6b_empathbot_v1_improvements.ipynb."
            )
        ordered = [ckpt_classes[i] for i in range(len(EMOTIONS))]
        if ordered != EMOTIONS:
            raise ValueError(
                f"Checkpoint class labels {ordered} do not match the service "
                f"EMOTIONS schema {EMOTIONS}. Update one to match the other."
            )

        cfg = ckpt.get("cfg") or {}
        backbone = cfg.get("backbone", "efficientnet_b2")
        use_timm = bool(cfg.get("use_timm", True))
        img_size = int(cfg.get("img_size", 224))

        model = _build_empathbot(num_classes=len(EMOTIONS), backbone=backbone, use_timm=use_timm)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(self._device).eval()
        self._model = model

        self._transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        logger.info(
            "EmpathBot emotion model loaded: backbone=%s use_timm=%s img_size=%d "
            "device=%s val_acc=%.4f epoch=%s",
            backbone, use_timm, img_size, self._device,
            float(ckpt.get("val_acc", 0.0)), ckpt.get("epoch", "?"),
        )

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        torch = self._torch
        from PIL import Image

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self._transform(pil).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(x)
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

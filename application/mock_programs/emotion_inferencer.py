"""
CS731 — Real-Time Emotion Inferencer
======================================
Wraps a trained checkpoint for single-image inference.
Includes the EmotionBuffer (rolling window mode) used in both exemplar pipelines.

Usage
-----
  from emotion_inferencer import EmotionInferencer, EmotionBuffer

  inferencer = EmotionInferencer('models/checkpoints/swin_tiny_ekman6_best.pt')
  buffer     = EmotionBuffer(window=10)

  face_crop = ...  # numpy BGR from OpenCV
  emotion, confidence = inferencer.predict(face_crop)
  buffer.update(emotion)
  smoothed_emotion = buffer.get_emotion()
"""

from collections import deque
from pathlib import Path
from statistics import mode as stat_mode

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import build_model

# ── Default emotion lists ──────────────────────────────────────────────────────
EKMAN_6 = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
EKMAN_7 = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
ALL_8   = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# ── Emotion Inferencer ────────────────────────────────────────────────────────

class EmotionInferencer:
    """
    Wraps a trained checkpoint for single-image emotion prediction.

    The checkpoint must have been saved by models/train.py:
      {'model_name', 'num_classes', 'mode', 'state_dict', ...}
    """

    # Standard ImageNet normalisation (same as training)
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, checkpoint_path: str | Path,
                 emotions: list[str] | None = None,
                 img_size: int = 224,
                 device:   str | None = None):
        """
        Args:
            checkpoint_path: path to .pt checkpoint saved by train.py
            emotions:        ordered list of class names (auto-detected from checkpoint)
            img_size:        input image size expected by the model (default 224)
            device:          'cuda' | 'cpu' | 'mps' | None (auto-detect)
        """
        self.ckpt_path = Path(checkpoint_path)
        self.img_size  = img_size

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        model_name  = ckpt['model_name']
        num_classes = ckpt['num_classes']
        mode        = ckpt.get('mode', 'ekman6')

        # Resolve emotion names
        if emotions is not None:
            self.emotions = emotions
        else:
            # Auto-detect from mode stored in checkpoint
            if num_classes == 6:
                self.emotions = EKMAN_6
            elif num_classes == 7:
                self.emotions = EKMAN_7
            else:
                self.emotions = ALL_8[:num_classes]

        assert len(self.emotions) == num_classes, (
            f'Emotion list length ({len(self.emotions)}) != '
            f'num_classes ({num_classes}). Pass emotions= explicitly.'
        )

        # Build and load model
        self.model = build_model(model_name, num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Inference transform (no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])

        print(f'[INFO] EmotionInferencer loaded: {model_name} '
              f'({num_classes} classes, {mode}, {device})')
        print(f'       Classes: {self.emotions}')

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        """
        Predict emotion from a face crop.

        Args:
            face_bgr: numpy array (H, W, 3) in BGR format (OpenCV convention)

        Returns:
            (emotion_string, confidence_float)
            e.g. ('happy', 0.93)
        """
        if face_bgr is None or face_bgr.size == 0:
            return 'unknown', 0.0

        # BGR → RGB → PIL → tensor
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(face_rgb)
        tensor   = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)                    # (1, num_classes)
            probs  = F.softmax(logits, dim=1)              # (1, num_classes)
            conf, idx = probs.max(dim=1)

        emotion    = self.emotions[idx.item()]
        confidence = conf.item()
        return emotion, confidence

    def predict_topk(self, face_bgr: np.ndarray, k: int = 3) -> list[tuple]:
        """
        Return the top-k emotions with probabilities.

        Returns: [(emotion, prob), ...] sorted by probability descending
        """
        if face_bgr is None or face_bgr.size == 0:
            return [('unknown', 0.0)]

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(face_rgb)
        tensor   = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits      = self.model(tensor)
            probs       = F.softmax(logits, dim=1)[0]     # (num_classes,)
            top_probs, top_idx = probs.topk(min(k, len(self.emotions)))

        return [(self.emotions[i.item()], p.item())
                for i, p in zip(top_idx, top_probs)]

    def __repr__(self) -> str:
        return (f'EmotionInferencer(model={self.ckpt_path.name}, '
                f'emotions={self.emotions}, device={self.device})')


# ── Emotion Buffer ────────────────────────────────────────────────────────────

class EmotionBuffer:
    """
    Rolling-window majority-vote smoother for real-time emotion predictions.

    Both exemplar pipelines use this pattern to avoid flickering:
      - Group 15: window=10 (mode of last 10 frames)
      - Team 7:   window=5  (moving average of last 5 frames)

    Usage
    -----
      buffer = EmotionBuffer(window=10)
      buffer.update('happy')
      smooth = buffer.get_emotion()   # → 'happy'
    """

    def __init__(self, window: int = 10):
        self.window = window
        self._buffer: deque[str] = deque(maxlen=window)

    def update(self, emotion: str) -> None:
        """Add the latest prediction to the buffer."""
        self._buffer.append(emotion)

    def get_emotion(self) -> str | None:
        """Return the most frequent emotion in the window (mode)."""
        if not self._buffer:
            return None
        try:
            return stat_mode(self._buffer)
        except Exception:
            return self._buffer[-1]   # fallback: most recent

    def get_emotion_moving_avg(self) -> str | None:
        """
        Alternative: return the most common emotion weighted by recency.
        Counts occurrences; more robust than pure mode when window is small.
        """
        if not self._buffer:
            return None
        from collections import Counter
        counts = Counter(self._buffer)
        return counts.most_common(1)[0][0]

    def clear(self) -> None:
        """Reset the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f'EmotionBuffer(window={self.window}, current={list(self._buffer)})'


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    # Test the buffer independently (no model needed)
    print('── EmotionBuffer smoke test ──')
    buf = EmotionBuffer(window=10)
    sequence = ['happy', 'happy', 'sad', 'happy', 'neutral', 'happy', 'happy']
    for e in sequence:
        buf.update(e)
        print(f'  Added: {e:8s} → smoothed: {buf.get_emotion()}')

    print(f'\nBuffer state: {buf}')

    # Test the inferencer if a checkpoint is given
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
        img_path  = sys.argv[2] if len(sys.argv) > 2 else None

        inferencer = EmotionInferencer(ckpt_path)

        if img_path:
            img = cv2.imread(img_path)
            if img is not None:
                emotion, conf = inferencer.predict(img)
                print(f'\nPrediction: {emotion} ({conf:.2%})')
                topk = inferencer.predict_topk(img, k=3)
                print('Top-3:', topk)
            else:
                print(f'Cannot load image: {img_path}')
        else:
            # Create a dummy image
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            emotion, conf = inferencer.predict(dummy)
            print(f'\nDummy image prediction: {emotion} ({conf:.2%})')

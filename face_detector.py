"""
CS731 — Face Detector Module
==============================
Implements all four detectors compared in the notebook:
  1. YOLOv8-face  (recommended — best speed/accuracy for real-time)
  2. RetinaFace   (skipped gracefully if TF dependency conflict exists)
  3. MediaPipe    (skipped gracefully if not installed)
  4. Haar Cascade (CPU baseline — always available)

All detectors share the same interface:
    detector.detect(img_bgr) → list of {'bbox': [x1,y1,x2,y2], 'confidence': float}

For the live pipeline only YOLOv8-face is used (chosen after comparison).
"""

import time
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.5
YOLO_WEIGHTS_URL = (
    'https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt'
)
YOLO_WEIGHTS_PATH = Path('weights/yolov8n-face.pt')


# ── Base class ────────────────────────────────────────────────────────────────

class FaceDetector(ABC):
    """Abstract base for all face detectors."""
    name: str = 'BaseDetector'

    @abstractmethod
    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR image (OpenCV format).

        Returns
        -------
        list of dict, each with keys:
            'bbox':       [x1, y1, x2, y2]  (pixel coords, ints)
            'confidence': float              (0–1)
        """
        ...

    def detect_best(self, img_bgr: np.ndarray) -> dict | None:
        """Return only the highest-confidence detection, or None."""
        dets = self.detect(img_bgr)
        if not dets:
            return None
        return max(dets, key=lambda d: d['confidence'])

    def crop_face(self, img_bgr: np.ndarray,
                  padding: int = 10) -> tuple[np.ndarray | None, list | None]:
        """
        Detect the best face, add padding, and return the cropped image.

        Returns (face_crop, [x1,y1,x2,y2]) or (None, None) if no face.
        """
        best = self.detect_best(img_bgr)
        if best is None:
            return None, None
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = best['bbox']
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return img_bgr[y1:y2, x1:x2], [x1, y1, x2, y2]

    def __repr__(self) -> str:
        return f'<{self.name}>'


# ── 1. YOLOv8-face ───────────────────────────────────────────────────────────

class YOLOFaceDetector(FaceDetector):
    """
    YOLOv8n-face — fine-tuned on WIDERFace.
    Best speed-accuracy trade-off for real-time webcam use.

    Weights are auto-downloaded from:
      https://github.com/akanametov/yolov8-face
    """
    name = 'YOLOv8-face'

    def __init__(self, conf: float = CONF_THRESHOLD,
                 weights_path: Path = YOLO_WEIGHTS_PATH):
        from ultralytics import YOLO  # imported here to keep module importable

        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        if not weights_path.exists():
            print(f'[INFO] Downloading YOLOv8-face weights → {weights_path}')
            try:
                urllib.request.urlretrieve(YOLO_WEIGHTS_URL, weights_path)
                print('[INFO] Download complete.')
            except Exception as e:
                print(f'[WARN] Download failed ({e}). Falling back to yolov8n.')
                weights_path = None

        try:
            self.model = YOLO(str(weights_path) if weights_path else 'yolov8n.pt')
        except Exception:
            print('[WARN] Could not load yolov8n-face, falling back to yolov8n.')
            self.model = YOLO('yolov8n.pt')

        self.conf = conf

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        results = self.model(img_bgr, conf=self.conf, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu())
                detections.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})
        return detections


# ── 2. RetinaFace ────────────────────────────────────────────────────────────

class RetinaFaceDetector(FaceDetector):
    """
    RetinaFace — state-of-the-art anchor-based detector.
    Requires tensorflow, which conflicts with newer Python versions.
    Skipped gracefully if unavailable.
    """
    name = 'RetinaFace'

    def __init__(self, conf: float = CONF_THRESHOLD):
        self.conf = conf
        try:
            from retinaface import RetinaFace as RF
            self._rf = RF
            self._available = True
        except ImportError:
            print('[WARN] RetinaFace unavailable (TensorFlow dependency conflict). '
                  'Install tensorflow to enable it.')
            self._available = False

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        if not self._available:
            return []
        try:
            faces = self._rf.detect_faces(img_bgr)
        except Exception as e:
            print(f'[ERROR] RetinaFace: {e}')
            return []
        detections = []
        if isinstance(faces, dict):
            for face_info in faces.values():
                score = float(face_info.get('score', 1.0))
                if score < self.conf:
                    continue
                fa = face_info['facial_area']   # [x1, y1, x2, y2]
                detections.append({'bbox': fa, 'confidence': score})
        return detections


# ── 3. MediaPipe BlazeFace ────────────────────────────────────────────────────

class MediaPipeDetector(FaceDetector):
    """
    MediaPipe BlazeFace — ultra-fast mobile-grade detector.
    Works well for selfie-style inputs; lower IoU on academic crop datasets.
    Skipped gracefully if mediapipe is not installed.
    """
    name = 'MediaPipe'

    def __init__(self, conf: float = CONF_THRESHOLD):
        self.conf = conf
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection
            self._detector = self._mp_face.FaceDetection(
                model_selection=1,              # 1 = full-range model
                min_detection_confidence=conf
            )
            self._available = True
        except ImportError:
            print('[WARN] MediaPipe unavailable. Install mediapipe to enable it.')
            self._available = False

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        if not self._available:
            return []
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result  = self._detector.process(img_rgb)
        detections = []
        if result.detections:
            for det in result.detections:
                bb  = det.location_data.relative_bounding_box
                x1  = max(0, int(bb.xmin * w))
                y1  = max(0, int(bb.ymin * h))
                x2  = min(w, int((bb.xmin + bb.width)  * w))
                y2  = min(h, int((bb.ymin + bb.height) * h))
                score = float(det.score[0]) if det.score else 0.0
                detections.append({'bbox': [x1, y1, x2, y2], 'confidence': score})
        return detections


# ── 4. Haar Cascade (baseline) ────────────────────────────────────────────────

class HaarCascadeDetector(FaceDetector):
    """
    OpenCV Haar Cascade — classical CPU baseline.
    Always available (bundled with opencv-python).
    Good for frontal faces, weaker on angled or low-contrast images.
    """
    name = 'Haar Cascade'

    def __init__(self, scale_factor: float = 1.1, min_neighbours: int = 5):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector      = cv2.CascadeClassifier(cascade_path)
        self.scale_factor  = scale_factor
        self.min_neighbours = min_neighbours

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)   # contrast normalisation
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor  = self.scale_factor,
            minNeighbors = self.min_neighbours,
            minSize      = (30, 30),
            flags        = cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return []
        return [{'bbox': [x, y, x + w, y + h], 'confidence': 1.0}
                for (x, y, w, h) in faces]


# ── Convenience factory ───────────────────────────────────────────────────────

def get_detector(name: str = 'yolo') -> FaceDetector:
    """
    Factory function. Returns a detector by short name.

    Args:
        name: 'yolo' | 'retina' | 'mediapipe' | 'haar'
    """
    name = name.lower()
    if name in ('yolo', 'yolov8'):
        return YOLOFaceDetector()
    elif name in ('retina', 'retinaface'):
        return RetinaFaceDetector()
    elif name in ('mediapipe', 'mp'):
        return MediaPipeDetector()
    elif name in ('haar', 'cascade'):
        return HaarCascadeDetector()
    else:
        raise ValueError(f'Unknown detector: {name}. '
                         f'Choose from yolo, retina, mediapipe, haar.')


# ── Helper: draw detections on frame ─────────────────────────────────────────

DETECTOR_COLORS = {
    'YOLOv8-face': (0,   255, 0),    # green
    'RetinaFace':  (0,   0,   255),  # red
    'MediaPipe':   (255, 165, 0),    # orange
    'Haar Cascade':(255, 0,   255),  # magenta
}

def draw_detections(img_bgr: np.ndarray,
                    detections: list[dict],
                    detector_name: str,
                    emotion: str = '',
                    conf: float = 0.0) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of the frame.

    Args:
        img_bgr:       OpenCV BGR image
        detections:    list from detector.detect()
        detector_name: for colour lookup
        emotion:       emotion string to overlay (from emotion inferencer)
        conf:          emotion confidence to overlay
    """
    out   = img_bgr.copy()
    color = DETECTOR_COLORS.get(detector_name, (0, 255, 0))

    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        det_conf = d.get('confidence', 0.0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f'{detector_name} {det_conf:.2f}'
        if emotion:
            label = f'{emotion} ({conf:.0%})'

        cv2.putText(out, label, (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return out


# ── Metric helpers (used by the comparison notebook) ─────────────────────────

def compute_iou(box_a: list, box_b: list) -> float:
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def measure_speed(detector: FaceDetector,
                  img: np.ndarray,
                  n_runs: int = 10,
                  warmup: int = 3) -> float:
    """Return mean inference time in ms over n_runs."""
    for _ in range(warmup):
        detector.detect(img)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        detector.detect(img)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def image_to_pseudo_gt(img: np.ndarray) -> list:
    """
    AffectNetHQ images are face-cropped, so the entire image
    is a reasonable pseudo ground-truth bounding box.
    """
    h, w = img.shape[:2]
    return [0, 0, w, h]


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else None

    for detector_name in ('yolo', 'haar'):
        det = get_detector(detector_name)
        print(f'\nTesting {det}')

        if img_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f'  Cannot read {img_path}')
                continue
        else:
            # Create a blank test image if no path given
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        dets = det.detect(img)
        ms   = (time.perf_counter() - t0) * 1000
        print(f'  Detections: {len(dets)}  |  Time: {ms:.1f}ms')
        for d in dets:
            print(f'    bbox={d["bbox"]}, conf={d["confidence"]:.3f}')

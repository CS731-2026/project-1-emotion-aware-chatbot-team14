"""Face-crop pre-step — ported from Notebooks/1_dataset_pipeline.ipynb cells 24-26.

The notebook runs YOLOv8 face detection over every raw image and
discards images that fail any of:

  - no_face         → detector returned nothing
  - face_too_small  → largest detected face < min_face_ratio (0.15)
  - multiple_faces  → second-largest face also > 0.10 (ambiguous)
  - empty_crop      → degenerate crop after padding
  - unreadable      → cv2.imread returned None

Kept images are written to `out_root/<eb_label>/<original_filename>`
and the manifest path is rewritten to point there. Resumable: skips
output files that already exist.

We reuse the production `FaceDetector` to share the cached YOLO weights
file, but call its underlying `_model` directly so we can read *all*
boxes (FaceDetector.detect_best() returns only the highest-confidence
one — insufficient for the notebook's multi-face filter).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_MIN_FACE_RATIO = 0.15
DEFAULT_PAD_RATIO      = 0.20
DEFAULT_TARGET_SIZE    = 224
DEFAULT_MULTI_FACE_THR = 0.10
DEFAULT_CONF           = 0.4   # notebook cell 25 default


def load_yolo():
    """Reuse FaceDetector's cached weights file by instantiating it
    (the download is hf_hub_download with resume — fast after first
    call) and returning its underlying ultralytics.YOLO model.

    Public name so the in-stream cropping path (affectnet/rafdb
    _download_to) can call it once outside the per-sample loop."""
    from pipeline.face_cropper import FaceDetector  # type: ignore[attr-defined]
    fd = FaceDetector()
    return fd._model, fd.device


# Backwards-compat alias for the existing crop_dataset path.
_yolo = load_yolo


def crop_pil(pil_img, yolo, device, *,
             min_face_ratio: float = DEFAULT_MIN_FACE_RATIO,
             pad_ratio:      float = DEFAULT_PAD_RATIO,
             target_size:    int   = DEFAULT_TARGET_SIZE,
             multi_face_thr: float = DEFAULT_MULTI_FACE_THR,
             conf:           float = DEFAULT_CONF):
    """In-memory variant of _filter_and_crop — takes a PIL image,
    returns (cropped PIL image or None, reason).

    Returns (PIL.Image, "ok") on success; (None, reason_str) otherwise
    where reason is one of "no_face" / "face_too_small" / "multiple_faces"
    / "empty_crop". Used by in-stream materialization so we never write
    the raw image to disk just to feed it to YOLO.
    """
    import cv2
    import numpy as np
    from PIL import Image

    # PIL → BGR numpy for cv2/YOLO consistency with the file-path path.
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    img_bgr = arr[:, :, ::-1].copy()                 # RGB → BGR

    h, w = img_bgr.shape[:2]
    img_area = float(h * w)
    if img_area <= 0:
        return None, "empty_crop"

    results = yolo(img_bgr, verbose=False, conf=conf, device=device)
    boxes_attr = results[0].boxes
    if boxes_attr is None or len(boxes_attr) == 0:
        return None, "no_face"

    xyxy = boxes_attr.xyxy.cpu().numpy()
    valid = []
    for box in xyxy:
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        ratio = ((x2 - x1) * (y2 - y1)) / img_area
        if ratio >= min_face_ratio:
            valid.append((x1, y1, x2, y2, ratio))

    if not valid:
        return None, "face_too_small"

    valid.sort(key=lambda b: b[4], reverse=True)
    if len(valid) > 1 and valid[1][4] > multi_face_thr:
        return None, "multiple_faces"

    x1, y1, x2, y2, _ = valid[0]
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "empty_crop"

    resized = cv2.resize(crop, (target_size, target_size),
                         interpolation=cv2.INTER_LANCZOS4)
    # BGR → RGB → PIL for the caller.
    rgb = resized[:, :, ::-1]
    return Image.fromarray(rgb), "ok"


def _filter_and_crop(img_path: str, out_path: str, yolo, device,
                     min_face_ratio: float, pad_ratio: float,
                     target_size: int, multi_face_thr: float,
                     conf: float) -> dict:
    """Faithful port of notebook 1 cell 25's `filter_and_crop_face()`."""
    import cv2

    img = cv2.imread(img_path)
    if img is None:
        return {"status": "skip", "reason": "unreadable", "face_ratio": 0.0}

    h, w = img.shape[:2]
    img_area = float(h * w)

    results = yolo(img, verbose=False, conf=conf, device=device)
    boxes_attr = results[0].boxes
    if boxes_attr is None or len(boxes_attr) == 0:
        return {"status": "skip", "reason": "no_face", "face_ratio": 0.0}

    xyxy = boxes_attr.xyxy.cpu().numpy()
    valid = []
    for box in xyxy:
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        ratio = ((x2 - x1) * (y2 - y1)) / img_area
        if ratio >= min_face_ratio:
            valid.append((x1, y1, x2, y2, ratio))

    if not valid:
        return {"status": "skip", "reason": "face_too_small", "face_ratio": 0.0}

    valid.sort(key=lambda b: b[4], reverse=True)
    if len(valid) > 1 and valid[1][4] > multi_face_thr:
        return {"status": "skip", "reason": "multiple_faces",
                "face_ratio": valid[0][4]}

    x1, y1, x2, y2, ratio = valid[0]
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return {"status": "skip", "reason": "empty_crop", "face_ratio": ratio}

    resized = cv2.resize(crop, (target_size, target_size),
                         interpolation=cv2.INTER_LANCZOS4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, resized)
    return {"status": "ok", "reason": "", "face_ratio": float(ratio)}


def crop_dataset(df_raw: pd.DataFrame, out_root: Path, dataset_name: str,
                 *, min_face_ratio: float = DEFAULT_MIN_FACE_RATIO,
                 pad_ratio: float = DEFAULT_PAD_RATIO,
                 target_size: int = DEFAULT_TARGET_SIZE,
                 multi_face_thr: float = DEFAULT_MULTI_FACE_THR,
                 conf: float = DEFAULT_CONF) -> pd.DataFrame:
    """Faithful port of notebook 1 cell 26's `process_dataset()`.

    Iterates over `df_raw` (must have columns `path` + `label`), writes
    crops to `out_root/<label>/<filename>`, and returns a new DataFrame
    pointing at the crops. Skip statistics are logged.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    yolo, device = _yolo()
    processed: list[dict] = []
    skip_reasons: Counter[str] = Counter()

    for _, row in df_raw.iterrows():
        src = str(row["path"])
        lbl = int(row["label"])
        fname = Path(src).name
        dst = out_root / str(lbl) / fname

        if dst.exists():
            new = dict(row); new["path"] = str(dst.resolve())
            processed.append(new)
            continue

        result = _filter_and_crop(src, str(dst), yolo, device,
                                   min_face_ratio, pad_ratio,
                                   target_size, multi_face_thr, conf)
        if result["status"] == "ok":
            new = dict(row)
            new["path"] = str(dst.resolve())
            new["face_ratio"] = result["face_ratio"]
            processed.append(new)
        else:
            skip_reasons[result["reason"]] += 1

    kept = len(processed); total = len(df_raw)
    logger.info("empath.face_crop[%s]: kept %d / %d (%.1f%%)",
                dataset_name, kept, total, 100 * kept / max(1, total))
    for reason, n in skip_reasons.most_common():
        logger.info("  skipped %s: %d", reason, n)

    cols = list(df_raw.columns)
    if "face_ratio" not in cols:
        cols.append("face_ratio")
    return pd.DataFrame(processed, columns=cols) if processed else pd.DataFrame(columns=cols)

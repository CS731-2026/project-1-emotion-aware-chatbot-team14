"""Quality filter — ported from Notebooks/7_kash_dataset_prep.ipynb cells 9-12.

The notebook applies two independent filters to each raw image:

  * Face detection + crop (optional — notebook shows three variants:
    MediaPipe, Haar cascade, no-op). When enabled here we reuse the
    same YOLO-based detector as empath/face_crop.py so kash and empath
    share one face-crop implementation.

  * Laplacian-variance blur score (cell 9): reject crops below
    BLUR_THRESHOLD (notebook default 80.0). This catches motion blur
    and out-of-focus selfies that dominate hand-collected datasets.

Both are opt-in via env vars:
    KASH_FACE_CROP=1       run the face detector before saving
    KASH_BLUR_FILTER=1     reject by Laplacian variance < KASH_BLUR_THR
    KASH_BLUR_THR=80.0     blur threshold (default matches notebook 7)

Outputs are written to `out_root/<eb_label>/<filename>` and the
manifest path is rewritten to point there. Resumable.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_OUT_SIZE       = 224
DEFAULT_MIN_FACE_PX    = 48
DEFAULT_BLUR_THRESHOLD = 80.0
DEFAULT_FACE_PADDING   = 0.20


def _blur_score(rgb) -> float:
    """Verbatim from notebook cell 9 — Laplacian variance on the
    grayscale version of the image."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _yolo():
    """Reuse the shared FaceDetector to avoid duplicate YOLO downloads."""
    from pipeline.face_cropper import FaceDetector  # type: ignore[attr-defined]
    fd = FaceDetector()
    return fd._model, fd.device


def _detect_and_crop(img_bgr, yolo, device, min_face_px: int,
                     padding: float):
    """Closest-match to notebook cell 11's Haar-cascade variant but
    using the team's production YOLOv8 detector. Returns
    (cropped_bgr, reason) — reason is None on success."""
    h, w = img_bgr.shape[:2]
    results = yolo(img_bgr, verbose=False, conf=0.4, device=device)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, "no_face"
    xyxy = boxes.xyxy.cpu().numpy()
    largest = max(
        ((int(x1), int(y1), int(x2), int(y2))
         for x1, y1, x2, y2 in (b[:4] for b in xyxy)),
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
    )
    x1, y1, x2, y2 = largest
    fw, fh = x2 - x1, y2 - y1
    if fw < min_face_px or fh < min_face_px:
        return None, "face_too_small"
    pad_x = int(fw * padding); pad_y = int(fh * padding)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "empty_crop"
    return crop, None


def _enabled(var: str, default: str = "0") -> bool:
    return os.environ.get(var, default).lower() in {"1", "true", "yes"}


def filter_dataset(df_raw: pd.DataFrame, out_root: Path, dataset_name: str,
                   *, out_size: int = DEFAULT_OUT_SIZE,
                   min_face_px: int = DEFAULT_MIN_FACE_PX,
                   face_padding: float = DEFAULT_FACE_PADDING,
                   blur_threshold: float | None = None) -> pd.DataFrame:
    """Faithful port of notebook 7 cell 12's accept/reject loop.

    `df_raw` must have columns `path` and `label`. Returns a new
    DataFrame of accepted rows with the path column rewritten to the
    saved crop, plus a `blur_score` column when blur filtering ran.

    Face detection runs when KASH_FACE_CROP=1; blur filtering when
    KASH_BLUR_FILTER=1 (default off, matching the notebook's active
    "Option B" cell 10 which skips both).
    """
    import cv2

    do_face = _enabled("KASH_FACE_CROP")
    do_blur = _enabled("KASH_BLUR_FILTER")
    blur_thr = blur_threshold if blur_threshold is not None else float(
        os.environ.get("KASH_BLUR_THR", DEFAULT_BLUR_THRESHOLD)
    )

    if not (do_face or do_blur):
        logger.info("kash.quality_filter[%s]: both filters disabled — pass-through",
                    dataset_name)
        return df_raw

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    yolo = device = None
    if do_face:
        yolo, device = _yolo()

    accepted: list[dict] = []
    rejected: Counter[str] = Counter()

    for i, row in df_raw.iterrows():
        src = str(row["path"])
        lbl = int(row["label"])
        fname = f"{lbl}_{Path(src).stem}_{int(i):05d}.jpg"
        dst = out_root / str(lbl) / fname

        if dst.exists():
            new = dict(row); new["path"] = str(dst.resolve())
            accepted.append(new)
            continue

        img_bgr = cv2.imread(src)
        if img_bgr is None:
            rejected["unreadable"] += 1
            continue

        crop_bgr = img_bgr
        if do_face:
            crop_bgr, reason = _detect_and_crop(img_bgr, yolo, device,
                                                 min_face_px, face_padding)
            if crop_bgr is None:
                rejected[reason or "no_face"] += 1
                continue

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        blur = None
        if do_blur:
            blur = _blur_score(crop_rgb)
            if blur < blur_thr:
                rejected["blurry"] += 1
                continue

        resized = cv2.resize(crop_rgb, (out_size, out_size),
                              interpolation=cv2.INTER_LANCZOS4)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        new = dict(row)
        new["path"] = str(dst.resolve())
        if blur is not None:
            new["blur_score"] = round(blur, 1)
        accepted.append(new)

    kept = len(accepted); total = len(df_raw)
    logger.info("kash.quality_filter[%s]: kept %d / %d (%.1f%%) | face=%s blur=%s",
                dataset_name, kept, total, 100 * kept / max(1, total),
                do_face, do_blur)
    for reason, n in rejected.most_common():
        logger.info("  rejected %s: %d", reason, n)

    cols = list(df_raw.columns)
    if do_blur and "blur_score" not in cols:
        cols.append("blur_score")
    return pd.DataFrame(accepted, columns=cols) if accepted else pd.DataFrame(columns=cols)

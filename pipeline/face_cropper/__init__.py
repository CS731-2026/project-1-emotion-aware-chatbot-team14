"""Face-cropping CLI + library wrapper for the production face detector.

Single source of truth for the detector is
``application/model_service/core/face_detector.py``. This file re-exports
``FaceDetector`` from there (so notebooks use the exact same YOLOv8 weights,
confidence threshold, and "largest face" heuristic that the model service uses
at inference time) and adds:

    - Input coercion: path | PIL.Image | numpy(BGR) → numpy(BGR)
    - ``crop_face(image)`` shortcut for one-line notebook use
    - ``crop_directory(in, out)`` for batch dataset preprocessing
    - A CLI for the same operations (``python -m pipeline.face_cropper --help``)

Behavioural changes belong in ``face_detector.py``, not here.

Usage in notebooks
------------------

    from pipeline.face_cropper import crop_face
    face = crop_face("dataset/raw/img_001.jpg")
    if face is not None:
        face.save("dataset/crops/img_001.jpg")

Usage from the shell
--------------------

    # One image
    python -m pipeline.face_cropper crop input.jpg output.jpg

    # Whole directory
    python -m pipeline.face_cropper crop-dir ./raw ./crops --recursive --resize 224

    # Idempotent rerun with a summary report
    python -m pipeline.face_cropper crop-dir ./raw ./crops --recursive \\
        --skip-existing --report report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Union

# Reach into the model service so we use the production FaceDetector class.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "application" / "model_service"))
from core.face_detector import FaceDetector  # noqa: E402

import numpy as np  # noqa: E402

# Re-export FaceDetector so `from pipeline.face_cropper import FaceDetector` works.
__all__ = ["FaceDetector", "crop_face", "crop_directory"]

ImageLike = Union[str, Path, "np.ndarray", "object"]  # str/Path | numpy | PIL.Image

_VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logger = logging.getLogger("face_cropper")

# Cache the detector so repeated crop_face() calls don't reload YOLO every time.
_default_detector: FaceDetector | None = None


def _detector() -> FaceDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = FaceDetector()
    return _default_detector


def _to_bgr(image: ImageLike) -> np.ndarray:
    """Coerce path | PIL.Image | numpy → uint8 BGR numpy (H, W, 3)."""
    import cv2

    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image at {image}")
        return bgr

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) numpy image, got shape {image.shape}")
        return image

    # PIL.Image, duck-typed to avoid hard PIL import for callers who don't use it.
    if hasattr(image, "convert") and hasattr(image, "size"):
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def crop_face(
    image: ImageLike,
    *,
    padding: float = 0.0,
    resize: int | None = None,
    return_pil: bool = True,
):
    """Crop the highest-confidence face from a single image.

    Args:
        image:      Path string, ``pathlib.Path``, ``PIL.Image``, or uint8 BGR numpy.
        padding:    Fractional margin around the detected box (0.1 = 10%).
        resize:     If set, resize the crop to ``(resize, resize)``.
        return_pil: If True, return a ``PIL.Image``. If False, return uint8 BGR numpy.

    Returns:
        The face crop, or ``None`` if no face was detected.
    """
    import cv2

    bgr = _to_bgr(image)
    face_bgr, box = _detector().detect_best(bgr)
    if face_bgr is None:
        return None

    if padding > 0 and box is not None:
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * padding, bh * padding
        x1 = int(max(0, x1 - pad_x))
        y1 = int(max(0, y1 - pad_y))
        x2 = int(min(w, x2 + pad_x))
        y2 = int(min(h, y2 + pad_y))
        face_bgr = bgr[y1:y2, x1:x2]

    if resize is not None:
        face_bgr = cv2.resize(face_bgr, (resize, resize), interpolation=cv2.INTER_AREA)

    if return_pil:
        from PIL import Image

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    return face_bgr


def _iter_image_paths(root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in _VALID_SUFFIXES:
            yield p


def crop_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    recursive: bool = False,
    padding: float = 0.0,
    resize: int | None = None,
    skip_existing: bool = False,
) -> dict:
    """Crop every image in a directory and write the results to ``output_dir``.

    Output paths mirror the input layout relative to ``input_dir`` (so subdirs
    are preserved when ``recursive=True``).

    Args:
        input_dir:     Directory containing source images.
        output_dir:    Destination directory (created if missing).
        recursive:     Walk subdirectories.
        padding:       Fractional margin around each detected box.
        resize:        Resize each crop to ``(resize, resize)`` if set.
        skip_existing: Skip writing if the output file already exists.

    Returns:
        A summary dict: ``{total, cropped, no_face, skipped_existing, errors}``.
    """
    import cv2

    in_root = Path(input_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    detector = _detector()
    summary = {"total": 0, "cropped": 0, "no_face": 0, "skipped_existing": 0, "errors": 0}
    started = time.time()

    for src in _iter_image_paths(in_root, recursive):
        summary["total"] += 1
        dst = out_root / src.relative_to(in_root)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and dst.exists():
            summary["skipped_existing"] += 1
            continue

        try:
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                summary["errors"] += 1
                logger.warning("could not read %s", src)
                continue

            face_bgr, box = detector.detect_best(bgr)
            if face_bgr is None:
                summary["no_face"] += 1
                continue

            if padding > 0 and box is not None:
                h, w = bgr.shape[:2]
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                pad_x, pad_y = bw * padding, bh * padding
                x1 = int(max(0, x1 - pad_x))
                y1 = int(max(0, y1 - pad_y))
                x2 = int(min(w, x2 + pad_x))
                y2 = int(min(h, y2 + pad_y))
                face_bgr = bgr[y1:y2, x1:x2]

            if resize is not None:
                face_bgr = cv2.resize(face_bgr, (resize, resize), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(dst), face_bgr)
            summary["cropped"] += 1

        except Exception as exc:
            summary["errors"] += 1
            logger.warning("failed on %s: %s", src, exc)

    summary["elapsed_seconds"] = round(time.time() - started, 2)
    return summary


# ─── CLI ────────────────────────────────────────────────────────────────────

def _cli_crop(args: argparse.Namespace) -> int:
    import cv2

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"error: could not read {args.input}", file=sys.stderr)
        return 2

    face_bgr, _box = _detector().detect_best(bgr)
    if face_bgr is None:
        print(f"no face detected in {args.input}", file=sys.stderr)
        return 1

    if args.resize:
        face_bgr = cv2.resize(face_bgr, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, face_bgr)
    print(f"wrote {args.output}")
    return 0


def _cli_crop_dir(args: argparse.Namespace) -> int:
    summary = crop_directory(
        args.input,
        args.output,
        recursive=args.recursive,
        padding=args.padding,
        resize=args.resize,
        skip_existing=args.skip_existing,
    )

    print(json.dumps(summary, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2))
        print(f"report → {args.report}")
    return 0 if summary["errors"] == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face_cropper",
        description=(
            "Crop faces from images using the production HRI face detector "
            "(application/model_service/core/face_detector.py)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_one = sub.add_parser("crop", help="crop the largest face from one image")
    p_one.add_argument("input", help="input image path")
    p_one.add_argument("output", help="output image path")
    p_one.add_argument("--resize", type=int, default=None, help="resize crop to NxN pixels")
    p_one.set_defaults(func=_cli_crop)

    p_dir = sub.add_parser("crop-dir", help="crop every image in a directory")
    p_dir.add_argument("input", help="input directory")
    p_dir.add_argument("output", help="output directory")
    p_dir.add_argument("--recursive", action="store_true", help="walk subdirectories")
    p_dir.add_argument("--padding", type=float, default=0.0,
                       help="fractional margin around detected box (e.g. 0.1)")
    p_dir.add_argument("--resize", type=int, default=None, help="resize each crop to NxN pixels")
    p_dir.add_argument("--skip-existing", action="store_true",
                       help="skip if the output file already exists")
    p_dir.add_argument("--report", type=str, default=None,
                       help="write the summary JSON to this path")
    p_dir.set_defaults(func=_cli_crop_dir)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[face_cropper] %(message)s")
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

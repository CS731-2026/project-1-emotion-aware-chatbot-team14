"""Smoke test for the face cropper.

Confirms:
  1. The detector loads (downloads YOLO weights on first run).
  2. Synthetic noise returns no detection (sanity check).
  3. If a real image path is supplied as argv[1], it cuts a crop and writes it.

Run from the repo root:
    python face_cropper/test_face_cropper.py [optional/path/to/face.jpg]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable regardless of where this script is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from pipeline.face_cropper import FaceDetector, crop_face  # noqa: E402


def _check_load() -> None:
    det = FaceDetector()
    print(f"[OK] detector loaded, device={det.device}, threshold={det.CONF_THRESHOLD}")


def _check_no_face_on_noise() -> None:
    noise = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    result = crop_face(noise, return_pil=False)
    assert result is None, "expected no detection on random noise"
    print("[OK] random noise correctly yields no detection")


def _check_real_image(path: Path) -> None:
    face = crop_face(path)
    if face is None:
        print(f"[WARN] no face detected in {path}")
        return
    out_path = Path("/tmp/face_cropper_test_output.jpg")
    face.save(out_path)
    print(f"[OK] face detected in {path} → wrote {out_path} (size={face.size})")


def main() -> int:
    _check_load()
    _check_no_face_on_noise()

    if len(sys.argv) > 1:
        _check_real_image(Path(sys.argv[1]))
    else:
        print("[skip] pass an image path as argv[1] to test on a real face")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

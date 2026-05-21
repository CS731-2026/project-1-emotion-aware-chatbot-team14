# face_cropper

Standalone face-cropping CLI + Python library that wraps the **same** face detector the model service uses at inference time. Use it to preprocess training datasets so your emotion classifier sees the same crops the production pipeline will hand it.

> **Source of truth:** `application/model_service/core/face_detector.py`
> `face_cropper.py` re-exports `FaceDetector` from there — it does not reimplement detection. Behaviour changes (confidence threshold, "largest face" heuristic, etc.) belong in the model service file, not here.

## Three ways to use it

### 1. Bulk preprocess a dataset (the common case)

From a shell:

```bash
# One image
python face_cropper.py crop input.jpg output.jpg

# A whole dataset
python face_cropper.py crop-dir ./raw_dataset ./crops \
    --recursive \
    --resize 224 \
    --padding 0.1 \
    --skip-existing \
    --report crops/_summary.json
```

From the repo root with `make`:

```bash
make crop-faces INPUT=./raw_dataset OUTPUT=./crops RESIZE=224 PADDING=0.1
```

The `--report` JSON looks like:

```json
{
  "total": 24380,
  "cropped": 23104,
  "no_face": 1109,
  "skipped_existing": 167,
  "errors": 0,
  "elapsed_seconds": 412.3
}
```

`no_face` is the number that worth checking — those images either have no face or a face below the detector's confidence threshold. Inspecting a sample of them is the fastest way to spot dataset quality issues before you start training.

### 2. From inside a notebook

```python
from face_cropper import crop_face

# Inputs: path, PIL.Image, or numpy (BGR)
face = crop_face("raw_dataset/img_001.jpg")     # returns PIL.Image or None
if face is not None:
    face.save("crops/img_001.jpg")

# As a numpy array instead of PIL
face_bgr = crop_face(img, return_pil=False)     # uint8 BGR (H, W, 3)
```

Repeated calls reuse a cached detector — YOLO loads once.

### 3. Verify your setup

```bash
python face_cropper/test_face_cropper.py /path/to/a/face.jpg
```

Prints `[OK] ...` lines if the detector loads, refuses noise, and crops a real image. Omit the path to skip the real-image step.

## What it does, exactly

For each input image:

1. Run YOLOv8-Face-Detection from HuggingFace (auto-downloaded; cached locally after first run).
2. Pick the **highest-confidence** face (not the largest one — the threshold is `0.35`).
3. Clamp the bounding box to the image edges.
4. Optionally expand the box by `--padding` (fraction of box size).
5. Optionally resize to `--resize N` (square, `INTER_AREA`).
6. Write to the output directory, preserving the relative input path.

## Common gotchas

- **First run is slow.** YOLO weights download from HuggingFace (~6 MB). After that the cache is reused.
- **GPU is auto-detected** — MPS on Apple Silicon, CUDA on NVIDIA, otherwise CPU. CPU is fine for ~100 images, painful for ~10k.
- **Outputs mirror inputs.** `./raw/0/foo.jpg` → `./crops/0/foo.jpg`. If your training loader expects class-as-directory, you can run on subset dirs in a loop.
- **`--skip-existing` makes reruns cheap.** Useful when adding new images to an existing dataset.
- **Padding can hurt or help.** With `padding=0.0` you get a tight face crop, which is what the model service does. For training, a small padding (`0.1`) often improves emotion-classifier accuracy because it preserves more facial context. Worth experimenting per-dataset.

## Why not a microservice?

We considered hosting this behind an HTTP endpoint. It loses in this use case:

- Thousands of images per dataset = thousands of round trips. Local inference is ~30x faster than HTTP.
- The detector is small and stateless — there's nothing to centralise.
- Service downtime blocks training.

Shipping the same code (not a service wrapping the code) keeps the model service and training preprocessing exactly aligned.

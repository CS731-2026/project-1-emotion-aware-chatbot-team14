"""
Arnabdhar. (n.d.). YOLOv8-Face-Detection [Model]. Hugging Face. Retrieved April 23, 2026, from https://huggingface.co/arnabdhar/YOLOv8-Face-Detection

Deng, J., Guo, J., Zhou, Y., Yu, J., Kotsia, I., & Zafeiriou, S. (2020). RetinaFace: Single-shot multi-level face localisation in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5203–5212).

Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0) [Computer software]. GitHub. https://github.com/ultralytics/ultralytics

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 779–788).

Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). WIDER FACE: A face detection benchmark. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5525–5533).
"""

"""
Face detection test script using OpenCV and other libraries.
"""

import cv2
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections


def test_opencv_cascade(duration_seconds: int = 10):
    """Test face detection using OpenCV Haar Cascades."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting OpenCV Cascade test (press 'q' to quit)")
    start_time = time.time()
    frames = 0

    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        elapsed = time.time() - frame_start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        frames += 1

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('OpenCV Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Test complete: {frames} frames processed")


def test_yolo_facedetection(duration_seconds: int = 10):
    """Test face detection using YOLOv8 face detection model from HuggingFace."""
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Download model from HuggingFace with progress
    print("Downloading YOLOv8 Face Detection model from HuggingFace...")
    try:
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt",
            cache_dir=str(models_dir),
            force_download=False,
            resume_download=True
        )
        print(f"✓ Model loaded from: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("✓ Model loaded successfully")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting YOLO Face Detection test (press 'q' to quit)")
    start_time = time.time()
    frames = 0

    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        # Run YOLO inference
        results = model(frame, conf=0.35)
        r = results[0]

        # Convert to supervision Detections format
        detections = Detections.from_ultralytics(r)

        # Draw bounding boxes
        if len(detections) > 0:
            boxes = detections.xyxy
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        elapsed = time.time() - frame_start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        frames += 1

        num_detections = len(detections)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {num_detections}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('YOLO Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Test complete: {frames} frames processed")

if __name__ == "__main__":
    print("=== Face Detection Test ===\n")
    # test_opencv_cascade(duration_seconds=10)
    test_yolo_facedetection(duration_seconds=10)

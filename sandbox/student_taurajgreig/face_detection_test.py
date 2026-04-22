"""
Face detection test script using OpenCV and other libraries.
"""

import cv2
import time


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


if __name__ == "__main__":
    test_opencv_cascade(duration_seconds=10)

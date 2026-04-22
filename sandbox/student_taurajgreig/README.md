# Face Detection & Speech Recognition Testing

Boilerplate for testing face detection and speech recognition libraries.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Face Detection Testing

Edit `face_detection_test.py` and implement:
1. Library initialization (MediaPipe or RetinaFace)
2. Face detection logic in `detect_faces()`

Run tests:
```bash
python face_detection_test.py
```

Test conditions:
- Frontal face in good lighting
- Turned head
- Dimmer lighting
- Natural movement

## Speech Recognition Testing

Edit `speech_recognition_test.py` and implement:
1. Audio recording from microphone
2. Transcription with selected method (FasterWhisper or SpeechRecognition)

Run tests:
```bash
python speech_recognition_test.py
```

Test phrases in `test_phrases.txt`:
1. Normal sentence
2. Technical sentence
3. Short command
4. Fast sentence
5. Mumbled sentence

## Results

Test results are saved as JSON:
- `face_detection_results.json`
- `speech_recognition_results.json`

"""
CS731, Emotion-Aware Chatbot: Terminal Application
=====================================================
Full end-to-end pipeline:
  1. Webcam captures frames in a background thread
  2. YOLOv8-face detects and crops the face
  3. Trained emotion model classifies the emotion
  4. EmotionBuffer smooths predictions over a rolling window
  5. User types or speaks a message
  6. Emotion + message sent to OpenAI chatbot
  7. Response printed as separate text bubbles

Usage
-----
  # With a trained checkpoint
  python main.py --checkpoint models/checkpoints/swin_tiny_ekman6_best.pt

  # With voice input enabled
  python main.py --checkpoint models/checkpoints/swin_tiny_ekman6_best.pt --voice

  # Text-only (no webcam) for quick testing
  python main.py --no_webcam --no_voice --mock_emotion happy

Environment
-----------
  OPENAI_API_KEY must be set (in .env or shell)
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
from face_detector    import YOLOFaceDetector, draw_detections
from emotion_inferencer import EmotionInferencer, EmotionBuffer
from chatbot          import EmotionChatbot
from speech           import record_and_transcribe, WHISPER_AVAILABLE, SOUNDDEVICE_AVAILABLE


# ── Config ────────────────────────────────────────────────────────────────────
BUFFER_WINDOW  = 10        # frames for emotion smoothing (Group 15: 10)
WEBCAM_INDEX   = 0         # default webcam
RECORD_SECS    = 5         # voice recording duration
CHATBOT_MODEL  = 'o4-mini' # chosen after 3-way comparison (see report)
CHATBOT_TEMP   = 1.0       # chosen after temp comparison (0.5 / 1.0 / 1.5)


# ── Webcam Thread ─────────────────────────────────────────────────────────────

class WebcamEmotionThread(threading.Thread):
    """
    Background thread that continuously:
      1. Reads frames from webcam
      2. Runs face detection (YOLOv8-face)
      3. Runs emotion inference
      4. Updates a shared EmotionBuffer
      5. Displays annotated frame in a cv2 window
    """

    def __init__(self, inferencer: EmotionInferencer,
                 buffer: EmotionBuffer,
                 cam_index: int = WEBCAM_INDEX,
                 show_window: bool = True):
        super().__init__(daemon=True)
        self.inferencer   = inferencer
        self.buffer       = buffer
        self.cam_index    = cam_index
        self.show_window  = show_window
        self.stop_event   = threading.Event()
        self.face_detector= YOLOFaceDetector()
        self._last_emotion = 'neutral'
        self._last_conf    = 0.0

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            print('[WARN] Could not open webcam. Continuing without video feed.')
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Detect face
            face_crop, bbox = self.face_detector.crop_face(frame, padding=10)

            if face_crop is not None and face_crop.size > 0:
                emotion, conf = self.inferencer.predict(face_crop)
                self.buffer.update(emotion)
                self._last_emotion = emotion
                self._last_conf    = conf

                # Draw bounding box + emotion label on frame
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{emotion} ({conf:.0%})'
                    cv2.putText(frame, label, (x1, max(y1 - 8, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display
            if self.show_window:
                smoothed = self.buffer.get_emotion() or 'detecting...'
                cv2.putText(frame, f'Smoothed: {smoothed}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow('CS731, Emotion Detection (press Q to quit)', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.stop_event.set()

    @property
    def current_emotion(self) -> str:
        return self.buffer.get_emotion() or 'neutral'


# ── Input helpers ─────────────────────────────────────────────────────────────

def get_user_input(voice_enabled: bool) -> str:
    """
    Prompt the user for input. If voice_enabled and they press Enter
    without typing, record from microphone.
    """
    if voice_enabled:
        prompt = '\nYou (type message, or press ENTER to speak): '
    else:
        prompt = '\nYou: '

    user_input = input(prompt).strip()

    if user_input == '' and voice_enabled:
        if not WHISPER_AVAILABLE or not SOUNDDEVICE_AVAILABLE:
            print('[WARN] Voice input unavailable. Type your message instead.')
            return input('You: ').strip()
        print(f'🎤 Speak now ({RECORD_SECS}s)...')
        user_input = record_and_transcribe(duration=RECORD_SECS)
        print(f'📝 Transcribed: "{user_input}"')

    return user_input


def print_response(response: str, chatbot_name: str = 'Frienderly') -> None:
    """Print the chatbot response as separate bubbles (simulates texting)."""
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [response]
    for para in paragraphs:
        print(f'\n  {chatbot_name}: {para}')
        time.sleep(0.3)  # small delay to simulate typing


# ── Main application ──────────────────────────────────────────────────────────

def main(args):
    print('\n' + '='*60)
    print('  CS731, Emotion-Aware Chatbot')
    print('  Type "quit" to exit | "reset" to clear history')
    print('='*60)

    # ── Emotion buffer ────────────────────────────────────────────────────────
    emotion_buffer = EmotionBuffer(window=BUFFER_WINDOW)

    # ── Webcam + inferencer ───────────────────────────────────────────────────
    webcam_thread = None
    if not args.no_webcam and Path(args.checkpoint).exists():
        print(f'\n[INFO] Loading emotion model: {args.checkpoint}')
        inferencer    = EmotionInferencer(args.checkpoint)
        webcam_thread = WebcamEmotionThread(
            inferencer  = inferencer,
            buffer      = emotion_buffer,
            show_window = not args.no_display,
        )
        webcam_thread.start()
        print('[INFO] Webcam started. Look at the camera.')
        time.sleep(2)   # give the model a moment to warm up
    else:
        if args.no_webcam:
            print('[INFO] Webcam disabled (--no_webcam).')
        else:
            print(f'[WARN] Checkpoint not found: {args.checkpoint}. '
                  f'Running without emotion recognition.')
        if args.mock_emotion:
            # Fill buffer with mock emotion for testing
            for _ in range(BUFFER_WINDOW):
                emotion_buffer.update(args.mock_emotion)
            print(f'[INFO] Mock emotion: {args.mock_emotion}')

    # ── Chatbot ────────────────────────────────────────────────────────────────
    print(f'\n[INFO] Connecting to {CHATBOT_MODEL}...')
    try:
        bot = EmotionChatbot(
            model       = CHATBOT_MODEL,
            temperature = CHATBOT_TEMP,
            max_tokens  = 500,
        )
        print('[INFO] Chatbot ready.\n')
    except Exception as e:
        print(f'[ERROR] Could not initialise chatbot: {e}')
        if webcam_thread:
            webcam_thread.stop()
        sys.exit(1)

    # ── Chat loop ─────────────────────────────────────────────────────────────
    print('─'*60)
    print('  Chat started. The chatbot will adapt to your emotions.')
    print('─'*60)

    voice_enabled = args.voice and not args.no_voice

    while True:
        try:
            user_text = get_user_input(voice_enabled)
        except (KeyboardInterrupt, EOFError):
            print('\n\nGoodbye!')
            break

        # Control commands
        if not user_text:
            continue
        if user_text.lower() in ('quit', 'exit', 'q', 'bye'):
            print('\nGoodbye! Take care.')
            break
        if user_text.lower() == 'reset':
            bot.reset_history()
            emotion_buffer.clear()
            print('  [History cleared]')
            continue

        # Get current emotion
        current_emotion = emotion_buffer.get_emotion() or 'neutral'
        print(f'  [Detected emotion: {current_emotion}]')

        # Get chatbot response
        try:
            response = bot.chat(user_text, emotion=current_emotion)
            print_response(response)
        except Exception as e:
            print(f'  [ERROR] Chatbot failed: {e}')

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if webcam_thread:
        webcam_thread.stop()
        webcam_thread.join(timeout=3)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='CS731 Emotion-Aware Chatbot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--checkpoint',   type=str,
                   default='models/checkpoints/swin_tiny_ekman6_best.pt',
                   help='Path to trained emotion model checkpoint')
    p.add_argument('--no_webcam',    action='store_true',
                   help='Disable webcam (text-only mode)')
    p.add_argument('--no_display',   action='store_true',
                   help='Run webcam in background without OpenCV window')
    p.add_argument('--voice',        action='store_true',
                   help='Enable voice input via FasterWhisper')
    p.add_argument('--no_voice',     action='store_true',
                   help='Force disable voice input')
    p.add_argument('--mock_emotion', type=str, default=None,
                   help='Inject a fixed mock emotion (for testing without webcam)')
    p.add_argument('--model',        type=str, default=CHATBOT_MODEL,
                   help='OpenAI model to use for the chatbot')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())

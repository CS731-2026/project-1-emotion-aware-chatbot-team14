"""
Real-time speech recognition with voice activity detection (VAD).

Records audio, transcribes speech locally with Whisper-based models, and shows
results in a GUI as each model finishes.

Press Ctrl+C to quit.

Examples:
  python speech_recognition_test.py
  python speech_recognition_test.py --service mock --duration 5
  python speech_recognition_test.py --service whisper --model tiny --duration 5
  python speech_recognition_test.py --service whisper-distilled --model distil-small.en
  python speech_recognition_test.py --save-audio ./audio_debug

Config:
  Edit cli/config.py to change tested models, VAD settings, and debug mode.
"""

import os
import time
import queue
import threading
from datetime import datetime

import tkinter as tk
from tkinter import scrolledtext
from dotenv import load_dotenv

from audio import AudioRecorder, save_audio
from cli import (
    parse_arguments,
    create_service,
    MODEL_SERVICE,
    MODEL_NAME,
    DEBUG_MODE,
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_SIZE,
    SPEECH_THRESHOLD,
    SILENCE_DURATION,
    MIN_SPEECH_DURATION,
    MAX_SPEECH_DURATION,
    LOCAL_MODELS_TO_TEST,
)


def choose_recording_mode():
    """Return True for manual mode, False for automatic VAD mode."""
    print("\n" + "=" * 60)
    print("🎤 RECORDING MODE")
    print("=" * 60)
    print("\n[1] Manual  - press ENTER to start/stop")
    print("[2] Auto    - record when speech is detected\n")

    while True:
        choice = input("Choose 1 or 2: ").strip()
        if choice in {"1", "2"}:
            manual = choice == "1"
            mode = "Manual" if manual else "Auto"
            print(f"\n→ {mode} mode selected\n")
            return manual
        print("Invalid choice. Enter 1 or 2.")


def load_env_token():
    """Load Hugging Face token from ../../.env into HF_TOKEN."""
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")
    if not os.path.exists(env_path):
        return False

    load_dotenv(env_path)
    token = os.getenv("HUGGING_FACE")
    if not token:
        return False

    os.environ["HF_TOKEN"] = token
    print("[INIT] ✓ HF token loaded")
    return True


def print_startup_info(service, record_duration=None, save_audio_dir=None):
    """Print startup configuration."""
    if record_duration:
        mode = f"FIXED DURATION ({record_duration}s)"
    elif DEBUG_MODE:
        mode = "DEBUG (manual recording)"
    else:
        mode = "VAD"

    print("=" * 60)
    print("🎤 SPEECH RECOGNITION")
    print("=" * 60)
    print(f"Service: {service.__class__.__name__}")
    print(f"Mode: {mode}")
    print("Config:")
    print(f"  • Speech Threshold: {SPEECH_THRESHOLD:.4f}")
    print(f"  • Min Duration: {MIN_SPEECH_DURATION}s")
    print(f"  • Max Duration: {MAX_SPEECH_DURATION}s")
    print(f"  • Silence Timeout: {SILENCE_DURATION}s")
    if save_audio_dir:
        print(f"  • Save Audio: {save_audio_dir}")
    print("\nPress Ctrl+C to quit\n")


def format_result(model, transcript, elapsed, confidence=None):
    """Build a result payload for the GUI queue."""
    return {
        "model": model,
        "transcript": transcript,
        "elapsed_time": elapsed,
        "confidence": confidence,
        "timestamp": time.time(),
    }


def enqueue_recording(recording, clip_queue, save_audio_dir):
    """Add a recording to the transcription queue."""
    if recording is None:
        return
    clip_queue.put(
        (recording.audio_data, recording.start_time, recording.duration, save_audio_dir)
    )


def iter_recordings(recorder, record_duration=None, manual_mode=None):
    """Yield recordings for the selected mode."""
    if record_duration:
        yield recorder.record_for_duration(record_duration)
        return

    if manual_mode is True or (manual_mode is None and DEBUG_MODE):
        yield recorder.record_with_key_press()
        return

    if manual_mode is False:
        yield from recorder.record_with_vad_continuous()
        return

    yield recorder.record_with_vad()


def transcribe_with_model(service_name, model_name, audio_data, results_queue):
    """Run one transcription model and push its result to the queue."""
    label = f"{service_name} - {model_name}"
    service = create_service(service_name, model_name)

    try:
        start = time.time()
        result = service.transcribe(audio_data)
        elapsed = time.time() - start

        if len(result) == 3:
            transcript, _language, confidence = result
        else:
            transcript, _language = result
            confidence = None

        results_queue.put(format_result(label, transcript, elapsed, confidence))
    except Exception as exc:
        results_queue.put(
            format_result(label, f"❌ Error: {str(exc)[:60]}", 0, None)
        )


def transcription_worker(clip_queue, results_queue, stop_event):
    """Process audio clips and run all configured models in parallel."""
    while not stop_event.is_set():
        try:
            audio_data, _start_time, duration, save_dir = clip_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            if save_dir:
                save_audio(audio_data, save_dir, SAMPLE_RATE)

            print(f"\n⏳ Processing audio clip ({duration:.2f}s)...")

            threads = []
            for service_name, model_name in LOCAL_MODELS_TO_TEST:
                thread = threading.Thread(
                    target=transcribe_with_model,
                    args=(service_name, model_name, audio_data, results_queue),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            results_queue.put({"type": "clip_complete"})
        finally:
            clip_queue.task_done()


def process_recording(
    recorder,
    record_duration=None,
    save_audio_dir=None,
    manual_mode=None,
    results_queue=None,
):
    """Record audio and send clips to the background transcription worker."""
    results_queue = results_queue or queue.Queue()
    clip_queue = queue.Queue()
    stop_event = threading.Event()

    worker = threading.Thread(
        target=transcription_worker,
        args=(clip_queue, results_queue, stop_event),
        daemon=True,
    )
    worker.start()

    try:
        for recording in iter_recordings(recorder, record_duration, manual_mode):
            enqueue_recording(recording, clip_queue, save_audio_dir)

            # Wait only for single-shot modes.
            if record_duration or manual_mode is True or (manual_mode is None and DEBUG_MODE):
                break

        clip_queue.join()
    finally:
        stop_event.set()
        worker.join(timeout=2.0)

    return True


class TranscriptionGUI:
    """Simple GUI that shows transcription results as they arrive."""

    def __init__(self, root, results_queue):
        self.root = root
        self.results_queue = results_queue

        self.root.title("Speech Recognition - Live Results")
        self.root.geometry("900x600")
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        self.text = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            font=("Courier", 11),
            bg="#1e1e1e",
            fg="#00ff00",
        )
        self.text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.text.tag_config("local", foreground="#ffaa00")
        self.text.tag_config("time", foreground="#888888")

        self.update_display()

    def append_result(self, result):
        """Render one result entry."""
        model = result["model"]
        transcript = result["transcript"]
        elapsed = result["elapsed_time"]
        confidence = result.get("confidence")
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.text.insert(tk.END, f"[{timestamp}] ", "time")

        if confidence is None:
            header = f"{model:45} ({elapsed:5.2f}s)\n"
        else:
            header = f"{model:40} ({elapsed:5.2f}s, {confidence * 100:.1f}%)\n"

        self.text.insert(tk.END, header, "local")
        self.text.insert(tk.END, f"  → {transcript}\n\n")
        self.text.see(tk.END)

    def update_display(self):
        """Drain the queue and refresh the text view."""
        try:
            while True:
                result = self.results_queue.get_nowait()
                if result.get("type") == "clip_complete":
                    self.text.insert(tk.END, "\n" + "─" * 80 + "\n", "time")
                else:
                    self.append_result(result)
        except queue.Empty:
            pass

        self.root.after(100, self.update_display)


def create_recorder():
    """Create an AudioRecorder from config."""
    return AudioRecorder(
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        chunk_size=CHUNK_SIZE,
        speech_threshold=SPEECH_THRESHOLD,
        silence_duration=SILENCE_DURATION,
        min_speech_duration=MIN_SPEECH_DURATION,
        max_speech_duration=MAX_SPEECH_DURATION,
    )


def main(service, record_duration=None, save_audio_dir=None):
    """Start the GUI and background recording/transcription pipeline."""
    print()
    print_startup_info(service, record_duration, save_audio_dir)

    manual_mode = None if record_duration else choose_recording_mode()
    results_queue = queue.Queue()
    recorder = create_recorder()

    root = tk.Tk()
    TranscriptionGUI(root, results_queue)

    worker = threading.Thread(
        target=process_recording,
        args=(recorder, record_duration, save_audio_dir, manual_mode, results_queue),
        daemon=True,
    )
    worker.start()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] Shutting down...")
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Goodbye!\n")


if __name__ == "__main__":
    load_env_token()
    args = parse_arguments()
    service = create_service(MODEL_SERVICE, MODEL_NAME)
    main(service, args.duration, args.save_audio)
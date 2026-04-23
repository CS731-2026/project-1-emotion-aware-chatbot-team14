"""
Real-time speech recognition with voice activation detection.

Records audio and transcribes it using local Whisper models in real-time.
Results appear in GUI window as they complete, with confidence scores.

Press Ctrl+C to quit.

Usage:
  python speech_recognition_test.py              # Uses config (MODEL_SERVICE)
  python speech_recognition_test.py --service mock --duration 5   # Mock service with fixed recording
  python speech_recognition_test.py --service whisper --model tiny --duration 5  # Whisper tiny
  python speech_recognition_test.py --service whisper-distilled --model distil-small.en  # Distilled model
  python speech_recognition_test.py --save-audio ./audio_debug   # Save audio for inspection

Configuration:
  Edit cli/config.py to:
    - Change LOCAL_MODELS_TO_TEST to test different models
    - Adjust VAD thresholds (SPEECH_THRESHOLD, SILENCE_DURATION)
    - Set DEBUG_MODE to use manual recording or VAD

Real-time Flow:
  1. Audio is recorded using VAD (voice activation detection)
  2. Each detected speech clip is queued for transcription
  3. Models from LOCAL_MODELS_TO_TEST transcribe in parallel
  4. Results appear in GUI as soon as they complete with confidence scores
  5. Recording continues unblocked while transcription happens
"""

import os
import time
import queue
import threading
from datetime import datetime
from dotenv import load_dotenv
import tkinter as tk
from tkinter import scrolledtext

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


# ── Mode Selection ────────────────────────────────────────────────────────


def choose_recording_mode():
    """Ask user to choose between manual or automatic recording mode.

    Returns:
        bool: True for manual mode (DEBUG), False for automatic mode (VAD)
    """
    print("\n" + "="*60)
    print("🎤 RECORDING MODE")
    print("="*60)
    print("\nChoose recording mode:\n")
    print("  [1] MANUAL MODE   - Press ENTER to record/stop (DEBUG)")
    print("  [2] AUTO MODE     - Records when threshold is exceeded (VAD)")
    print()

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            print("\n→ Manual mode selected (press ENTER to start/stop recording)\n")
            return True
        elif choice == "2":
            print("\n→ Auto mode selected (will record when sound level exceeds threshold)\n")
            return False
        else:
            print("Invalid choice. Please enter 1 or 2.")


# ── Initialization ─────────────────────────────────────────────────────────


def load_env_token():
    """Load Hugging Face token from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")
    if not os.path.exists(env_path):
        return False

    load_dotenv(env_path)
    token = os.getenv("HUGGING_FACE")
    if token:
        os.environ["HF_TOKEN"] = token
        print("[INIT] ✓ HF token loaded")
        return True
    return False


# ── UI / Display ────────────────────────────────────────────────────────────


def print_startup_info(service, record_duration=None, save_audio_dir=None):
    """Print application startup information."""
    print("=" * 60)
    print("🎤 SPEECH RECOGNITION")
    print("=" * 60)
    print(f"Service: {service.__class__.__name__}")
    if record_duration:
        print(f"Mode: FIXED DURATION ({record_duration}s auto-record)")
    elif DEBUG_MODE:
        print(f"Mode: DEBUG (key press to record)")
    else:
        print(f"Mode: VAD (voice activation detection)")
    print(f"Config:")
    print(f"  • Speech Threshold: {SPEECH_THRESHOLD:.4f}")
    print(f"  • Min Duration: {MIN_SPEECH_DURATION}s")
    print(f"  • Max Duration: {MAX_SPEECH_DURATION}s")
    print(f"  • Silence Timeout: {SILENCE_DURATION}s")
    if save_audio_dir:
        print(f"  • Saving audio to: {save_audio_dir}")
    print(f"\nPress Ctrl+C to quit\n")


def display_results_table(results, recording_start_time=None, recording_duration=None):
    """Display transcription results in a formatted table.

    Args:
        results: List of dicts with keys: 'model', 'transcript', 'elapsed_time'
        recording_start_time: Unix timestamp when recording started
        recording_duration: Duration of recording in seconds
    """
    if not results:
        return

    # Clear line and show results
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║ ✅ RESULTS" + " "*67 + "║")
    print("╠" + "═"*78 + "╣")

    # Show recording metadata
    if recording_start_time and recording_duration:
        from datetime import datetime
        start_dt = datetime.fromtimestamp(recording_start_time).strftime("%H:%M:%S")
        print(f"║ ⏱️  Recorded at {start_dt} | Duration: {recording_duration:.2f}s" + " "*37 + "║")
        print("╠" + "═"*78 + "╣")

    # Results table
    for i, result in enumerate(results, 1):
        model_name = result['model']
        transcript = result['transcript']
        elapsed = result['elapsed_time']

        if i == 1:
            icon = "☁️ "
        else:
            icon = "🔬"

        print(f"║ {icon} {model_name:38} {elapsed:6.2f}s" + " "*24 + "║")

        if transcript:
            # Wrap long transcripts
            text = f"    {transcript}"
            if len(text) > 76:
                print(f"║ {text[:76]}" + " "*(78-len(text[:76])) + "║")
                remaining = text[76:]
                while remaining:
                    chunk = remaining[:76]
                    print(f"║ {chunk}" + " "*(78-len(chunk)) + "║")
                    remaining = remaining[76:]
            else:
                print(f"║ {text}" + " "*(78-len(text)) + "║")
        else:
            print(f"║     ⚠️ NO SPEECH DETECTED" + " "*51 + "║")

        if i < len(results):
            print("╟" + "─"*78 + "╢")

    print("╚" + "═"*78 + "╝\n")


# ── Transcription Cycle ────────────────────────────────────────────────────


def transcription_worker(service, queue_in, results_queue, stop_event):
    """Background worker thread that processes audio clips from queue.

    Spawns parallel transcription threads so models don't block each other.

    Args:
        service: OpenAI TranscriptionService instance
        queue_in: Queue of (audio_data, start_time, duration, save_dir) tuples
        results_queue: Queue to put results for GUI
        stop_event: Threading event to signal when to stop
    """
    def transcribe_model(model_name, service_instance, audio_data, start_proc):
        """Transcribe with a single model in a thread."""
        try:
            start_time = time.time()
            result = service_instance.transcribe(audio_data)
            elapsed = time.time() - start_time

            # Handle both 2-tuple (legacy) and 3-tuple (with confidence) returns
            if len(result) == 3:
                transcript, language, confidence = result
            else:
                transcript, language = result
                confidence = None

            results_queue.put({
                "model": model_name,
                "transcript": transcript,
                "elapsed_time": elapsed,
                "confidence": confidence,
                "timestamp": time.time(),
            })
        except Exception as e:
            results_queue.put({
                "model": model_name,
                "transcript": f"❌ Error: {str(e)[:60]}",
                "elapsed_time": 0,
                "confidence": None,
                "timestamp": time.time(),
            })

    while not stop_event.is_set():
        try:
            # Wait for next clip
            audio_data, start_time, duration, save_dir = queue_in.get(timeout=0.5)

            # Save audio if requested
            if save_dir:
                save_audio(audio_data, save_dir, SAMPLE_RATE)

            print(f"\n⏳ Processing audio clip ({duration:.2f}s)...")

            # Spawn parallel threads for each model
            threads = []

            # Start local models in parallel
            for service_name, model_name in LOCAL_MODELS_TO_TEST:
                local_service = create_service(service_name, model_name)
                t = threading.Thread(
                    target=transcribe_model,
                    args=(
                        f"{service_name} - {model_name}",
                        local_service,
                        audio_data,
                        time.time(),
                    ),
                )
                t.daemon = True
                t.start()
                threads.append(t)

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Signal completion
            results_queue.put({"type": "clip_complete"})
            queue_in.task_done()

        except queue.Empty:
            continue


def process_recording(service, recorder, record_duration=None, save_audio_dir=None, manual_mode=None, results_queue=None):
    """Record audio and queue it for transcription (background processing).

    Args:
        service: OpenAI TranscriptionService instance (reference)
        recorder: AudioRecorder instance
        record_duration: Fixed duration in seconds (overrides manual/VAD mode)
        save_audio_dir: Directory to save audio files for debugging
        manual_mode: If True use manual mode, if False use continuous VAD
        results_queue: Queue to put results for GUI display

    Returns:
        bool: True if completed successfully
    """
    if results_queue is None:
        results_queue = queue.Queue()

    # Create queue and start background worker
    clip_queue = queue.Queue()
    stop_event = threading.Event()

    worker_thread = threading.Thread(
        target=transcription_worker,
        args=(service, clip_queue, results_queue, stop_event),
        daemon=True,
    )
    worker_thread.start()

    try:
        # Determine which recording mode to use
        if record_duration:
            recording = recorder.record_for_duration(record_duration)
            if recording is not None:
                clip_queue.put(
                    (
                        recording.audio_data,
                        recording.start_time,
                        recording.duration,
                        save_audio_dir,
                    )
                )
            clip_queue.join()  # Wait for processing to complete

        elif manual_mode is not None:
            if manual_mode:
                # Manual mode - single recording
                recording = recorder.record_with_key_press()
                if recording is not None:
                    clip_queue.put(
                        (
                            recording.audio_data,
                            recording.start_time,
                            recording.duration,
                            save_audio_dir,
                        )
                    )
                clip_queue.join()
            else:
                # Continuous VAD mode - queue clips as they come in
                for recording in recorder.record_with_vad_continuous():
                    clip_queue.put(
                        (
                            recording.audio_data,
                            recording.start_time,
                            recording.duration,
                            save_audio_dir,
                        )
                    )

        elif DEBUG_MODE:
            # Debug mode - single recording
            recording = recorder.record_with_key_press()
            if recording is not None:
                clip_queue.put(
                    (
                        recording.audio_data,
                        recording.start_time,
                        recording.duration,
                        save_audio_dir,
                    )
                )
            clip_queue.join()

        else:
            # Default VAD mode - single recording
            recording = recorder.record_with_vad()
            if recording is not None:
                clip_queue.put(
                    (
                        recording.audio_data,
                        recording.start_time,
                        recording.duration,
                        save_audio_dir,
                    )
                )
            clip_queue.join()

    finally:
        stop_event.set()
        worker_thread.join(timeout=2.0)

    return True


# ── Main Loop ──────────────────────────────────────────────────────────────


class TranscriptionGUI:
    """Simple GUI to display transcription results as they arrive."""

    def __init__(self, root, results_queue):
        """Initialize GUI.

        Args:
            root: tkinter root window
            results_queue: Queue of results from transcription worker
        """
        self.root = root
        self.results_queue = results_queue
        self.root.title("Speech Recognition - Live Results")
        self.root.geometry("900x600")

        # Create text display
        self.text = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=("Courier", 11), bg="#1e1e1e", fg="#00ff00"
        )
        self.text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.text.tag_config("openai", foreground="#00ccff")
        self.text.tag_config("local", foreground="#ffaa00")
        self.text.tag_config("time", foreground="#888888")
        self.text.tag_config("header", foreground="#00ff00", font=("Courier", 12, "bold"))

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_display()

    def update_display(self):
        """Update display with new results from queue."""
        try:
            while True:
                result = self.results_queue.get_nowait()

                if result.get("type") == "clip_complete":
                    self.text.insert(tk.END, "\n" + "─" * 80 + "\n", "time")
                    self.text.see(tk.END)
                else:
                    # Display individual result
                    model = result["model"]
                    transcript = result["transcript"]
                    elapsed = result["elapsed_time"]
                    confidence = result.get("confidence")

                    # Format output
                    tag = "local"

                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.text.insert(
                        tk.END, f"[{timestamp}] ", "time"
                    )

                    # Show confidence if available
                    if confidence is not None:
                        conf_pct = f"{confidence*100:.1f}%"
                        self.text.insert(
                            tk.END, f"{model:40} ({elapsed:5.2f}s, {conf_pct})\n", tag
                        )
                    else:
                        self.text.insert(
                            tk.END, f"{model:45} ({elapsed:5.2f}s)\n", tag
                        )
                    self.text.insert(tk.END, f"  → {transcript}\n\n")
                    self.text.see(tk.END)

        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(100, self.update_display)

    def on_closing(self):
        """Handle window close."""
        self.root.destroy()


def main(service, record_duration=None, save_audio_dir=None):
    """Main application loop with GUI.

    Args:
        service: TranscriptionService instance
        record_duration: Fixed duration in seconds (overrides manual/VAD)
        save_audio_dir: Directory to save audio files for debugging
    """
    print()  # Blank line after init
    print_startup_info(service, record_duration, save_audio_dir)

    # Choose recording mode if not using fixed duration
    manual_mode = None
    if not record_duration:
        manual_mode = choose_recording_mode()

    # Create results queue for GUI
    results_queue = queue.Queue()

    # Start GUI in separate thread
    root = tk.Tk()
    gui = TranscriptionGUI(root, results_queue)

    # Create recorder with configured settings
    recorder = AudioRecorder(
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        chunk_size=CHUNK_SIZE,
        speech_threshold=SPEECH_THRESHOLD,
        silence_duration=SILENCE_DURATION,
        min_speech_duration=MIN_SPEECH_DURATION,
        max_speech_duration=MAX_SPEECH_DURATION,
    )

    try:
        # Start processing in a background thread
        process_thread = threading.Thread(
            target=process_recording,
            args=(service, recorder, record_duration, save_audio_dir, manual_mode, results_queue),
            daemon=True,
        )
        process_thread.start()

        # Run GUI main loop
        root.mainloop()

    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] Shutting down...")
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Goodbye!\n")


# ── Entry Point ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Load environment variables before creating service
    load_env_token()

    # Parse command-line arguments
    args = parse_arguments()

    # Create service (uses config defaults or CLI args)
    # The LOCAL_MODELS_TO_TEST are spawned in parallel threads
    service = create_service(MODEL_SERVICE, MODEL_NAME)

    main(service, args.duration, args.save_audio)

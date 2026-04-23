"""
Speech recognition with voice activation detection.
Terminal-only mode with live mic volume monitoring.
Uses pluggable transcription services.
Press Ctrl+C to quit.

Usage:
  python speech_recognition_test.py              # Uses mock service (no model)
  python speech_recognition_test.py --service mock   # Mock service
  python speech_recognition_test.py --service whisper --model tiny  # Whisper (if available)
"""

import time
import argparse
import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np

from services import MockTranscriptionService, WhisperTranscriptionService


# ── Tunable Constants ─────────────────────────────────────────────────────
# Adjust these to fine-tune speech detection behavior

SAMPLE_RATE = 16000  # Audio sample rate (Hz)
CHANNELS = 1  # Mono audio
CHUNK_SIZE = 512  # Samples per recording chunk

# Voice Activation Detection (VAD) thresholds
SPEECH_THRESHOLD = 0.0005  # RMS level to trigger recording (0.0 - 1.0)
SILENCE_DURATION = 0.6  # Seconds of silence to stop recording

# Recording duration limits
MIN_SPEECH_DURATION = 0.4  # Minimum recording duration to process (seconds)
MAX_SPEECH_DURATION = 30  # Maximum recording duration (seconds)


# ── Audio Utilities ───────────────────────────────────────────────────────

def get_rms_level(audio_chunk):
    """Calculate RMS (volume) level of audio"""
    return np.sqrt(np.mean(audio_chunk**2))


def record_with_vad():
    """Record audio using voice activation detection.

    States:
      1. LISTENING   - Waiting for speech above threshold
      2. RECORDING   - Speech detected, capturing audio
      3. PROCESSING  - Silence detected, analyzing

    Returns:
        numpy array of audio samples or None
    """
    print(f"[{time.strftime('%H:%M:%S')}] 🎤 LISTENING (threshold: {SPEECH_THRESHOLD:.4f})")

    # ── LISTENING PHASE ───────────────────────────────────────────────────
    # Wait for volume to exceed threshold
    while True:
        chunk = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32)
        sd.wait()

        level = get_rms_level(chunk)
        status = "🎙" if level > SPEECH_THRESHOLD else " "
        print(f"\r[{status}] {level:.5f} {'(above threshold!)' if level > SPEECH_THRESHOLD else ''}", end="", flush=True)

        if level > SPEECH_THRESHOLD:
            print(f"\n[{time.strftime('%H:%M:%S')}] 🎙 RECORDING ▶ (will stop after {SILENCE_DURATION}s silence)")
            break

    # ── RECORDING PHASE ───────────────────────────────────────────────────
    # Capture speech until silence or max duration
    audio_buffer = [chunk]
    silence_counter = 0
    speech_duration = 0

    while True:
        chunk = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32)
        sd.wait()

        level = get_rms_level(chunk)
        audio_buffer.append(chunk)
        speech_duration += len(chunk) / SAMPLE_RATE

        status = "🎙" if level > SPEECH_THRESHOLD else " "
        print(f"\r[{status}] {level:.5f} │ {speech_duration:.1f}s", end="", flush=True)

        # Reset silence counter if still speaking
        if level > SPEECH_THRESHOLD:
            silence_counter = 0
        else:
            silence_counter += len(chunk) / SAMPLE_RATE

        # Stop conditions
        if silence_counter > SILENCE_DURATION:
            print(f"\n[{time.strftime('%H:%M:%S')}] ⏹ STOPPED (silence detected)")
            break

        if speech_duration > MAX_SPEECH_DURATION:
            print(f"\n[{time.strftime('%H:%M:%S')}] ⏹ STOPPED (max duration reached)")
            break

    # ── PROCESSING PHASE ──────────────────────────────────────────────────
    audio_data = np.concatenate(audio_buffer)
    total_duration = len(audio_data) / SAMPLE_RATE
    audio_kb = len(audio_data) * 4 / 1024

    # Check minimum duration
    if total_duration < MIN_SPEECH_DURATION:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ DISCARDED ({total_duration:.2f}s < {MIN_SPEECH_DURATION}s minimum)\n")
        return None

    print(f"[{time.strftime('%H:%M:%S')}] ✓ READY ({audio_kb:.1f} KB, {total_duration:.2f}s)\n")
    return audio_data


# ── Main Loop ───────────────────────────────────────────────────

def main(service):
    """Main application

    Args:
        service: TranscriptionService instance
    """
    # Load environment
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        token = os.getenv("HUGGING_FACE")
        if token:
            os.environ["HF_TOKEN"] = token
            print("[INIT] ✓ HF token loaded\n")

    print("="*60)
    print("🎤 SPEECH RECOGNITION")
    print("="*60)
    print(f"Service: {service.__class__.__name__}")
    print(f"Config:")
    print(f"  • Speech Threshold: {SPEECH_THRESHOLD:.4f}")
    print(f"  • Min Duration: {MIN_SPEECH_DURATION}s")
    print(f"  • Max Duration: {MAX_SPEECH_DURATION}s")
    print(f"  • Silence Timeout: {SILENCE_DURATION}s")
    print(f"\nPress Ctrl+C to quit\n")

    # Main loop
    try:
        while True:
            audio_data = record_with_vad()

            if audio_data is None:
                continue

            print(f"[{time.strftime('%H:%M:%S')}] Transcribing...")
            start_time = time.time()
            transcript, language = service.transcribe(audio_data)
            elapsed = time.time() - start_time

            if transcript:
                print(f"[{time.strftime('%H:%M:%S')}] ✅ TRANSCRIBED in {elapsed:.2f}s")
                print(f"   Language: {language.upper()}")
                print(f"   Text: {transcript}\n")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ⚠️ NO SPEECH DETECTED\n")

    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] Shutting down...")
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Goodbye!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech recognition with voice activation detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python speech_recognition_test.py              # mock (default)
  python speech_recognition_test.py --service mock  # explicit mock
  python speech_recognition_test.py --service whisper --model tiny  # Whisper
        """,
    )
    parser.add_argument(
        "--service",
        type=str,
        default="mock",
        choices=["mock", "whisper"],
        help="Transcription service (default: mock - no model)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Model size for Whisper (default: tiny)",
    )

    args = parser.parse_args()

    # Create service based on argument
    if args.service == "mock":
        print("[INIT] Using MockTranscriptionService (no model loaded)\n")
        service = MockTranscriptionService()
    elif args.service == "whisper":
        print(f"[INIT] Using WhisperTranscriptionService ({args.model} model)\n")
        service = WhisperTranscriptionService(args.model)
    else:
        raise ValueError(f"Unknown service: {args.service}")

    main(service)

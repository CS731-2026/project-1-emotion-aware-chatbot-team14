"""
Speech recognition with voice activation detection.
Terminal-only mode with live mic volume monitoring.
Press Ctrl+C to quit.
"""

import time
from faster_whisper import WhisperModel
import torch
import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np


# ── Tunable Constants ─────────────────────────────────────────────────────
# Adjust these to fine-tune speech detection behavior

SAMPLE_RATE = 16000  # Audio sample rate (Hz)
CHANNELS = 1  # Mono audio
CHUNK_SIZE = 512  # Samples per recording chunk

# Voice Activation Detection (VAD) thresholds
SPEECH_THRESHOLD = 0.015  # RMS level to trigger recording (0.0 - 1.0)
SILENCE_DURATION = 0.6  # Seconds of silence to stop recording

# Recording duration limits
MIN_SPEECH_DURATION = 0.4  # Minimum recording duration to process (seconds)
MAX_SPEECH_DURATION = 30  # Maximum recording duration (seconds)


# ── Device Selection ───────────────────────────────────────────────────────

def get_device_priority():
    """Get ordered list of devices to try: CUDA -> CPU"""
    devices = []
    if torch.cuda.is_available():
        devices.append(("cuda", "int8_float16"))
    devices.append(("cpu", "int8"))
    return devices


def select_device_with_fallback(model_name="base"):
    """Try devices in order until one works: CUDA -> CPU"""
    devices = get_device_priority()

    for device, compute_type in devices:
        try:
            print(f"🔄 Trying {device.upper()}...", end=" ", flush=True)
            model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            print(f"✓ Success!")
            return device, compute_type, model
        except (ValueError, RuntimeError) as e:
            print(f"✗ Failed")
            continue

    raise RuntimeError("No compatible device found")


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


# ── Model Management ───────────────────────────────────────────────────────

def check_model_cached(model_name="base"):
    """Check if Whisper model is cached"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, f"models--openai--whisper-{model_name}")
    return os.path.exists(model_cache)


def download_model(model_name="base"):
    """Download Whisper model"""
    print(f"\n{'='*60}")
    print(f"📥 Downloading Whisper '{model_name}' model")
    print(f"{'='*60}")

    repo_id = f"openai/whisper-{model_name}"
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    start_time = time.time()
    snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        resume_download=True,
        local_files_only=False,
    )
    elapsed = time.time() - start_time
    print(f"✓ Download complete in {elapsed:.1f}s\n")


def load_whisper_model(model_name="base"):
    """Load Whisper model with automatic device fallback"""
    print(f"[{time.strftime('%H:%M:%S')}] Loading Whisper '{model_name}'...")

    if not check_model_cached(model_name):
        download_model(model_name)

    print(f"[{time.strftime('%H:%M:%S')}] Detecting compatible device...")
    start_time = time.time()
    device, compute_type, model = select_device_with_fallback(model_name)
    elapsed = time.time() - start_time

    print(f"[{time.strftime('%H:%M:%S')}] ✓ Using {device.upper()} in {elapsed:.2f}s\n")

    return model


# ── Transcription ───────────────────────────────────────────────────────

def transcribe_audio(audio_data, model):
    """Transcribe audio"""
    print(f"[{time.strftime('%H:%M:%S')}] Transcribing...")

    start_time = time.time()
    segments, info = model.transcribe(audio_data, beam_size=5)
    transcript = " ".join([seg.text for seg in segments]).strip()
    elapsed = time.time() - start_time

    return transcript, info.language, elapsed


# ── Main Loop ───────────────────────────────────────────────────────

def main():
    """Main application"""
    # Load environment
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        token = os.getenv("HUGGING_FACE")
        if token:
            os.environ["HF_TOKEN"] = token
            print("[INIT] ✓ HF token loaded\n")

    # Setup model
    model = load_whisper_model("base")

    print("="*60)
    print("🎤 SPEECH RECOGNITION")
    print("="*60)
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

            transcript, language, elapsed = transcribe_audio(audio_data, model)

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
    main()

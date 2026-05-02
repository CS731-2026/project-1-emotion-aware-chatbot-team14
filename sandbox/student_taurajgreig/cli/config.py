"""Configuration constants for speech recognition system."""

# ── Audio Configuration ────────────────────────────────────────────────────
# Basic audio settings for recording and playback

SAMPLE_RATE = 16000  # Audio sample rate (Hz)
CHANNELS = 1  # Mono audio
CHUNK_SIZE = 512  # Samples per recording chunk

# ── Voice Activation Detection (VAD) ───────────────────────────────────────
# Thresholds for automatic speech detection

SPEECH_THRESHOLD = 0.00075  # RMS level to trigger recording (0.0 - 1.0)
SILENCE_DURATION = 0.6  # Seconds of silence to stop recording

# ── Recording Duration Limits ──────────────────────────────────────────────
# Constraints on how long recordings can be

MIN_SPEECH_DURATION = 1.2  # Minimum recording duration to process (seconds)
MAX_SPEECH_DURATION = 30  # Maximum recording duration (seconds)

# ── Debug Mode ─────────────────────────────────────────────────────────────
# Enable/disable debug recording mode (press ENTER to start/stop)

DEBUG_MODE = True  # Set to False to use VAD threshold detection

# ── Model Configuration ────────────────────────────────────────────────────
# Default model and service to use when CLI args are not provided

MODEL_SERVICE = "whisper-cpp" 
MODEL_NAME = "base.en"

# ── Local Models to Test ──────────────────────────────────────────────────
# List of (service, model) tuples to test locally on each recording
# Edit this list to control which models get tested

LOCAL_MODELS_TO_TEST = [
    ("whisper-cpp", "base.en"),
    # ("whisper-distilled", "distil-small.en"),
    # ("whisper", "tiny"),
    # ("whisper", "base"),
    # ("whisper-distilled", "distil-medium.en"),
]

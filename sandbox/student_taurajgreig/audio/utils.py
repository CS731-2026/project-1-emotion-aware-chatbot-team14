"""Audio utilities for recording and processing."""

import os
import time
import numpy as np
import soundfile as sf
from pathlib import Path


def get_rms_level(audio_chunk):
    """Calculate RMS (volume) level of audio.

    Args:
        audio_chunk: numpy array of audio samples

    Returns:
        float: RMS level (0.0 to 1.0)
    """
    return np.sqrt(np.mean(audio_chunk**2))


def save_audio(audio_data, save_dir, sample_rate=16000):
    """Save audio to WAV file for debugging.

    Args:
        audio_data: numpy array of audio samples (float32)
        save_dir: Directory to save audio files
        sample_rate: Sample rate of audio (default: 16000 Hz)

    Returns:
        str: Path to saved file, or None if save_dir is not provided
    """
    if not save_dir:
        return None

    # Create directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"audio_{timestamp}.wav")

    # Save audio file
    sf.write(filepath, audio_data, sample_rate)
    print(f"[{time.strftime('%H:%M:%S')}] 💾 Saved to: {filepath}")

    return filepath

"""
Speech recognition test script using FasterWhisper.
Captures audio from microphone and transcribes it in real-time.
"""

import tempfile
import os
from faster_whisper import WhisperModel
import speech_recognition as sr
import torch


def select_device():
    """
    Select device in order of preference: mps -> cuda -> gpu -> cpu.
    Returns the device name and compute type.
    """
    # Try MPS (Metal Performance Shaders for Apple Silicon)
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Metal Performance Shaders)")
        return "mps", "float32"

    # Try CUDA (NVIDIA)
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return "cuda", "int8_float16"

    # Fallback to CPU
    print("Using CPU")
    return "cpu", "int8"


def record_audio(duration=10, output_path=None):
    """Record audio from microphone and save to file."""
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print(f"Recording for {duration} seconds... Speak now!")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=duration)

    # Save audio to file
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "audio_recording.wav")

    with open(output_path, "wb") as f:
        f.write(audio.get_wav_data())

    print(f"Audio saved to {output_path}")
    return output_path


def test_faster_whisper(audio_path=None, duration=10, device=None, model_size="base"):
    """
    Test speech recognition with faster-whisper.

    Args:
        audio_path: Path to audio file. If None, records from microphone.
        duration: Duration to record in seconds (only used if audio_path is None).
        device: Device to use ("mps", "cuda", "cpu"). If None, auto-detects.
        model_size: "tiny", "base", "small", "medium", or "large-v3".
    """
    # Record audio if no path provided
    if audio_path is None:
        audio_path = record_audio(duration=duration)

    # Auto-select device if not specified
    if device is None:
        device, compute_type = select_device()
    else:
        # Map device preference
        if device == "mps" and torch.backends.mps.is_available():
            compute_type = "float32"
        elif device == "cuda" and torch.cuda.is_available():
            compute_type = "int8_float16"
        else:
            device = "cpu"
            compute_type = "int8"

    # Load model
    print(f"Loading {model_size} model on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Transcribe
    print("Transcribing audio...")
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"\nDetected language: '{info.language}' (confidence: {info.language_probability:.2f})")
    print("\nTranscription:")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")


if __name__ == "__main__":
    print("Speech Recognition Test\n")

    # Record for 10 seconds and transcribe (auto-detects best device)
    test_faster_whisper(duration=10, model_size="base")

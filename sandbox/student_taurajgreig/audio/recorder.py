"""Audio recording with multiple modes (VAD, key press, fixed duration)."""

import time
import threading
from dataclasses import dataclass
import sounddevice as sd
import numpy as np
from .utils import get_rms_level


@dataclass
class Recording:
    """Audio recording with metadata."""
    audio_data: np.ndarray
    start_time: float  # Timestamp when recording started (seconds since epoch)
    duration: float    # Duration of recording in seconds


class AudioRecorder:
    """Record audio in different modes.

    Supports three recording modes:
    - Fixed duration: Records for a specified number of seconds
    - Key press (debug): Press ENTER to start/stop recording
    - VAD: Voice activation detection with silence threshold
    """

    def __init__(
        self,
        sample_rate=16000,
        channels=1,
        chunk_size=512,
        speech_threshold=0.0005,
        silence_duration=0.6,
        min_speech_duration=0.4,
        max_speech_duration=30,
    ):
        """Initialize audio recorder with configuration.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_size: Number of samples per chunk
            speech_threshold: RMS level to trigger recording (0.0-1.0)
            silence_duration: Seconds of silence to stop recording
            min_speech_duration: Minimum recording duration to process
            max_speech_duration: Maximum recording duration
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.speech_threshold = speech_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration

    def record_for_duration(self, duration):
        """Record audio for a fixed duration (continuous stream, no flickering).

        Args:
            duration: Duration to record in seconds

        Returns:
            Recording object with audio data and timestamps
        """
        print(f"[{time.strftime('%H:%M:%S')}] 🎤 Recording for {duration}s...")

        recording_start_time = time.time()

        # Record entire duration in ONE continuous stream (no on/off flickering)
        duration_samples = int(duration * self.sample_rate)
        audio_data = sd.rec(
            duration_samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
        )

        # Monitor progress while recording
        start_time = time.time()
        while sd.get_stream().active:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            level = get_rms_level(audio_data[: int(elapsed * self.sample_rate)])
            print(f"\r[ ] {level:.5f} │ {elapsed:.1f}s", end="", flush=True)
            time.sleep(0.1)  # Update display every 100ms

        sd.wait()  # Wait for recording to finish
        print(f"\n[{time.strftime('%H:%M:%S')}] ⏹ STOPPED")

        # Process audio
        total_duration = len(audio_data) / self.sample_rate
        audio_kb = len(audio_data) * 4 / 1024

        print(
            f"[{time.strftime('%H:%M:%S')}] ✓ READY ({audio_kb:.1f} KB, {total_duration:.2f}s)\n"
        )
        return Recording(
            audio_data=audio_data,
            start_time=recording_start_time,
            duration=total_duration,
        )

    def record_with_key_press(self):
        """Record audio triggered by key press (DEBUG MODE).

        Instructions:
          1. Press ENTER to START recording
          2. Press ENTER again to STOP recording

        Uses continuous audio stream (no flickering).

        Returns:
            Recording object with audio data and timestamps, or None
        """
        print(
            f"[{time.strftime('%H:%M:%S')}] 🎤 DEBUG MODE - Press ENTER to START recording..."
        )

        # Wait for key press to start
        input()  # Blocking wait for Enter key

        recording_start_time = time.time()

        print(
            f"[{time.strftime('%H:%M:%S')}] 🎙 RECORDING ▶ (Press ENTER to stop)"
        )

        # Use continuous input stream (keeps mic open without flickering)
        audio_buffer = []
        stop_recording = threading.Event()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_size,
        ) as stream:

            def monitor_recording():
                """Monitor recording progress in background"""
                while not stop_recording.is_set():
                    # Read continuously from stream
                    chunk, overflowed = stream.read(self.chunk_size)
                    if not overflowed:
                        audio_buffer.append(chunk)
                        level = get_rms_level(chunk)
                        print(f"\r[ ] {level:.5f}", end="", flush=True)
                    time.sleep(0.01)  # Small sleep to prevent busy waiting

            # Start monitoring thread
            thread = threading.Thread(target=monitor_recording, daemon=True)
            thread.start()

            # Wait for key press to stop
            input()
            stop_recording.set()
            thread.join(timeout=2.0)

        print(f"\n[{time.strftime('%H:%M:%S')}] ⏹ STOPPED")

        # Process audio
        if not audio_buffer:
            return None

        audio_data = np.concatenate(audio_buffer)
        total_duration = len(audio_data) / self.sample_rate
        audio_kb = len(audio_data) * 4 / 1024

        print(
            f"[{time.strftime('%H:%M:%S')}] ✓ READY ({audio_kb:.1f} KB, {total_duration:.2f}s)\n"
        )
        return Recording(
            audio_data=audio_data,
            start_time=recording_start_time,
            duration=total_duration,
        )

    def record_with_vad_continuous(self):
        """Record audio using voice activation detection with continuous stream.

        Uses a single continuous audio stream (no flickering).
        Yields Recording objects as speech clips are detected.

        Yields:
            Recording object when a speech clip is detected and silence follows
        """
        print(
            f"[{time.strftime('%H:%M:%S')}] 🎤 LISTENING (threshold: {self.speech_threshold:.4f})"
        )
        print("Press Ctrl+C to stop listening\n")

        # Open ONE continuous stream (no flickering!)
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_size,
        ) as stream:

            # ── State machine for VAD ──────────────────────────────────────────
            state = "LISTENING"  # LISTENING, RECORDING, SILENCE_DETECTED
            audio_buffer = []
            silence_counter = 0
            speech_duration = 0
            recording_start_time = None

            while True:
                # Read continuous chunk from stream
                chunk, overflowed = stream.read(self.chunk_size)
                if overflowed:
                    continue

                level = get_rms_level(chunk)

                # ── STATE: LISTENING ──────────────────────────────────────────
                if state == "LISTENING":
                    status = "🎙" if level > self.speech_threshold else " "
                    print(
                        f"\r[{status}] {level:.5f}",
                        end="",
                        flush=True,
                    )

                    if level > self.speech_threshold:
                        # Threshold crossed! Start recording
                        state = "RECORDING"
                        recording_start_time = time.time()
                        audio_buffer = [chunk]
                        silence_counter = 0
                        speech_duration = len(chunk) / self.sample_rate
                        print(
                            f"\n[{time.strftime('%H:%M:%S')}] 🎙 RECORDING ▶ (will stop after {self.silence_duration}s silence)"
                        )

                # ── STATE: RECORDING ──────────────────────────────────────────
                elif state == "RECORDING":
                    audio_buffer.append(chunk)
                    speech_duration += len(chunk) / self.sample_rate

                    status = "🎙" if level > self.speech_threshold else " "
                    print(
                        f"\r[{status}] {level:.5f} │ {speech_duration:.1f}s",
                        end="",
                        flush=True,
                    )

                    # Check if still speaking or silence started
                    if level > self.speech_threshold:
                        silence_counter = 0
                    else:
                        silence_counter += len(chunk) / self.sample_rate

                    # Stop conditions
                    if silence_counter > self.silence_duration:
                        state = "SILENCE_DETECTED"
                        print(
                            f"\n[{time.strftime('%H:%M:%S')}] ⏹ STOPPED (silence detected)"
                        )

                    elif speech_duration > self.max_speech_duration:
                        state = "SILENCE_DETECTED"
                        print(
                            f"\n[{time.strftime('%H:%M:%S')}] ⏹ STOPPED (max duration reached)"
                        )

                # ── STATE: SILENCE_DETECTED ───────────────────────────────────
                elif state == "SILENCE_DETECTED":
                    # Process the clip
                    audio_data = np.concatenate(audio_buffer)
                    total_duration = len(audio_data) / self.sample_rate
                    audio_kb = len(audio_data) * 4 / 1024

                    # Check minimum duration
                    if total_duration >= self.min_speech_duration:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] ✓ READY ({audio_kb:.1f} KB, {total_duration:.2f}s)\n"
                        )
                        yield Recording(
                            audio_data=audio_data,
                            start_time=recording_start_time,
                            duration=total_duration,
                        )
                    else:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] ⚠️ DISCARDED ({total_duration:.2f}s < {self.min_speech_duration}s minimum)\n"
                        )

                    # Reset to listening state
                    state = "LISTENING"
                    audio_buffer = []
                    silence_counter = 0
                    speech_duration = 0
                    recording_start_time = None
                    print(
                        f"[{time.strftime('%H:%M:%S')}] 🎤 LISTENING (threshold: {self.speech_threshold:.4f})"
                    )

    def record_with_vad(self):
        """Record audio using voice activation detection (single clip mode).

        Returns:
            Recording object with first detected speech clip, or None
        """
        for recording in self.record_with_vad_continuous():
            return recording
        return None

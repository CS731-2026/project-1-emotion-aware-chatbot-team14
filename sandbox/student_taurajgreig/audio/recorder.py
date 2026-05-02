"""Audio recording with fixed-duration, manual, and VAD modes."""

import time
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .utils import get_rms_level


@dataclass
class Recording:
    audio_data: np.ndarray
    start_time: float
    duration: float


class AudioRecorder:
    """Record audio using fixed duration, manual start/stop, or VAD."""

    def __init__(
        self,
        sample_rate=16000,
        channels=1,
        chunk_size=512,
        speech_threshold=0.0005,
        silence_duration=0.6,
        min_speech_duration=0.8,
        max_speech_duration=30,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.speech_threshold = speech_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration

    def _log(self, message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _chunk_seconds(self, chunk):
        return len(chunk) / self.sample_rate

    def _finalize_recording(self, audio_data, start_time, enforce_min=True):
        if audio_data is None or len(audio_data) == 0:
            return None

        duration = len(audio_data) / self.sample_rate
        size_kb = len(audio_data) * 4 / 1024

        if enforce_min and duration < self.min_speech_duration:
            self._log(
                f"⚠️ DISCARDED ({duration:.2f}s < {self.min_speech_duration}s minimum)\n"
            )
            return None

        self._log(f"✓ READY ({size_kb:.1f} KB, {duration:.2f}s)\n")
        return Recording(audio_data=audio_data, start_time=start_time, duration=duration)

    def _input_stream(self):
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_size,
        )

    def record_for_duration(self, duration):
        """Record audio for a fixed duration."""
        self._log(f"🎤 Recording for {duration}s...")
        start_time = time.time()

        frames = int(duration * self.sample_rate)
        audio_data = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
        )

        monitor_start = time.time()
        while sd.get_stream().active:
            elapsed = time.time() - monitor_start
            if elapsed >= duration:
                break
            level = get_rms_level(audio_data[: int(elapsed * self.sample_rate)])
            print(f"\r[ ] {level:.5f} │ {elapsed:.1f}s", end="", flush=True)
            time.sleep(0.1)

        sd.wait()
        self._log("⏹ STOPPED")
        return self._finalize_recording(audio_data, start_time, enforce_min=True)

    def record_with_key_press(self):
        """Record until ENTER is pressed again."""
        self._log("🎤 DEBUG MODE - Press ENTER to START recording...")
        input()

        start_time = time.time()
        self._log("🎙 RECORDING ▶ (Press ENTER to stop)")

        audio_buffer = []
        stop_event = threading.Event()

        with self._input_stream() as stream:

            def read_audio():
                while not stop_event.is_set():
                    chunk, overflowed = stream.read(self.chunk_size)
                    if overflowed:
                        continue
                    audio_buffer.append(chunk)
                    print(f"\r[ ] {get_rms_level(chunk):.5f}", end="", flush=True)
                    time.sleep(0.01)

            thread = threading.Thread(target=read_audio, daemon=True)
            thread.start()

            input()
            stop_event.set()
            thread.join(timeout=2.0)

        self._log("⏹ STOPPED")
        if not audio_buffer:
            return None

        audio_data = np.concatenate(audio_buffer)
        return self._finalize_recording(audio_data, start_time, enforce_min=True)

    def record_with_vad_continuous(self):
        """Yield recordings whenever VAD detects a complete speech clip."""
        self._log(f"🎤 LISTENING (threshold: {self.speech_threshold:.4f})")
        print("Press Ctrl+C to stop listening\n")

        state = "listening"
        audio_buffer = []
        silence_time = 0.0
        speech_time = 0.0
        recording_start_time = None

        with self._input_stream() as stream:
            while True:
                chunk, overflowed = stream.read(self.chunk_size)
                if overflowed:
                    continue

                level = get_rms_level(chunk)
                chunk_seconds = self._chunk_seconds(chunk)
                speaking = level > self.speech_threshold
                status = "🎙" if speaking else " "

                if state == "listening":
                    print(f"\r[{status}] {level:.5f}", end="", flush=True)

                    if speaking:
                        state = "recording"
                        recording_start_time = time.time()
                        audio_buffer = [chunk]
                        silence_time = 0.0
                        speech_time = chunk_seconds
                        self._log(
                            f"🎙 RECORDING ▶ (will stop after {self.silence_duration}s silence)"
                        )

                elif state == "recording":
                    audio_buffer.append(chunk)
                    speech_time += chunk_seconds
                    silence_time = 0.0 if speaking else silence_time + chunk_seconds

                    print(
                        f"\r[{status}] {level:.5f} │ {speech_time:.1f}s",
                        end="",
                        flush=True,
                    )

                    if silence_time > self.silence_duration:
                        self._log("⏹ STOPPED (silence detected)")
                        state = "finalizing"
                    elif speech_time > self.max_speech_duration:
                        self._log("⏹ STOPPED (max duration reached)")
                        state = "finalizing"

                if state == "finalizing":
                    recording = self._finalize_recording(
                        np.concatenate(audio_buffer),
                        recording_start_time,
                        enforce_min=True,
                    )
                    if recording:
                        yield recording

                    state = "listening"
                    audio_buffer = []
                    silence_time = 0.0
                    speech_time = 0.0
                    recording_start_time = None
                    self._log(f"🎤 LISTENING (threshold: {self.speech_threshold:.4f})")

    def record_with_vad(self):
        """Return the first VAD-detected speech clip."""
        return next(self.record_with_vad_continuous(), None)
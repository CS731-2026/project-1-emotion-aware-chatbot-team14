"""
whisper.cpp transcription service.

This service wraps the upstream whisper.cpp CLI instead of using Python bindings.
That keeps setup simple on a MacBook and avoids fighting with extra packaging.

===============================================================================
TUTORIAL: SIMPLE LOCAL SETUP INSIDE YOUR REPO
===============================================================================

Goal:
- Keep whisper.cpp inside a child folder of your repo
- Git-ignore it
- Build it with plain `make`
- Let this Python service call the compiled CLI

Recommended folder layout:

your_project/
├─ services/
│  └─ whisper_cpp.py        <-- this file
├─ vendor/
│  └─ whisper.cpp/          <-- cloned here, git-ignored
└─ .gitignore

1) Add the child folder to .gitignore

    vendor/whisper.cpp/
    vendor/whisper.cpp-build/
    tmp_audio/

2) Clone whisper.cpp into that ignored child folder

    git clone https://github.com/ggml-org/whisper.cpp.git vendor/whisper.cpp

3) Build it with the simplest path possible

    cd vendor/whisper.cpp
    make

That should produce the CLI binary without you needing to think about build flags.

4) Download a model

    ./models/download-ggml-model.sh base.en

This typically creates something like:

    vendor/whisper.cpp/models/ggml-base.en.bin

5) Test it manually once

    ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav

6) Then point this Python service at:
    - repo_dir = "vendor/whisper.cpp"
    - model_name = "base.en"

===============================================================================
WHY THIS APPROACH
===============================================================================

- It uses the official whisper.cpp CLI path instead of unofficial bindings
- It is easy to inspect and debug
- It works well with a Python app that already has a service abstraction
- It is easy to benchmark because every transcription is an explicit subprocess call

===============================================================================
NOTES
===============================================================================

- whisper.cpp's whisper-cli expects 16-bit WAV input, so this class writes a
  temporary PCM WAV file before running the binary.
- This wrapper returns:
      (transcript, language, confidence)
  but whisper.cpp CLI does not expose confidence in a simple stable way here,
  so confidence is returned as None.
- If you want lower latency later, the next step would be:
      keep the repo/build/model logic the same
      but switch from whisper-cli subprocess calls to a long-lived server mode
      or native binding
  For now, CLI is the least painful option.

===============================================================================
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from .base import TranscriptionService


class WhisperCppTranscriptionService(TranscriptionService):
    """Transcription service backed by the whisper.cpp CLI."""

    REQUIRES_DEPS = []
    REQUIREMENTS_FILE = None

    def __init__(
        self,
        model_name: str = "base.en",
        repo_dir: str | os.PathLike = "vendor/whisper.cpp",
        sample_rate: int = 16000,
        auto_setup: bool = True,
    ):
        """
        Args:
            model_name:
                whisper.cpp model name, e.g.:
                tiny.en, base.en, small.en, medium.en, large-v3, large-v3-turbo

            repo_dir:
                Local path where whisper.cpp is cloned.
                Recommended: vendor/whisper.cpp

            sample_rate:
                Expected input sample rate for audio_data.
                This wrapper writes 16 kHz mono WAV for whisper-cli.

            auto_setup:
                If True:
                  - verifies repo exists
                  - builds whisper.cpp with `make` if needed
                  - downloads model if needed

                If False:
                  assumes everything is already installed.
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.repo_dir = Path(repo_dir).resolve()

        self.models_dir = self.repo_dir / "models"
        self.build_bin_dir = self.repo_dir / "build" / "bin"

        # Expected binary and model paths.
        self.binary_path = self.build_bin_dir / "whisper-cli"
        self.model_path = self.models_dir / f"ggml-{model_name}.bin"

        if auto_setup:
            self._ensure_repo_exists()
            self._ensure_built()
            self._ensure_model_downloaded()

    # -------------------------------------------------------------------------
    # Setup helpers
    # -------------------------------------------------------------------------

    def _log(self, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _ensure_repo_exists(self) -> None:
        """Fail fast with clear instructions if whisper.cpp has not been cloned."""
        if self.repo_dir.exists():
            return

        raise FileNotFoundError(
            "\nwhisper.cpp repo not found.\n\n"
            f"Expected here:\n  {self.repo_dir}\n\n"
            "Fix:\n"
            f"  git clone https://github.com/ggml-org/whisper.cpp.git {self.repo_dir}\n"
        )

    def _ensure_built(self) -> None:
        """
        Build whisper.cpp using plain `make` if whisper-cli is missing.

        This intentionally avoids custom flags and avoids overengineering.
        """
        if self.binary_path.exists():
            return

        self._log("Building whisper.cpp with `make`...")

        # We use `make` because you explicitly asked for the simplest route.
        result = subprocess.run(
            ["make"],
            cwd=self.repo_dir,
            text=True,
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Failed to build whisper.cpp with `make`.\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        if not self.binary_path.exists():
            raise FileNotFoundError(
                "Build completed, but whisper-cli was not found at:\n"
                f"  {self.binary_path}\n\n"
                "Check the build output above and confirm the repo built correctly."
            )

        self._log("✓ whisper.cpp built successfully")

    def _ensure_model_downloaded(self) -> None:
        """
        Download the requested ggml model if missing.

        Uses whisper.cpp's own helper script so you do not have to guess URLs.
        """
        if self.model_path.exists():
            return

        script_path = self.models_dir / "download-ggml-model.sh"
        if not script_path.exists():
            raise FileNotFoundError(
                "Model download script not found:\n"
                f"  {script_path}\n"
            )

        self._log(f"Downloading whisper.cpp model '{self.model_name}'...")

        result = subprocess.run(
            ["sh", str(script_path), self.model_name],
            cwd=self.repo_dir,
            text=True,
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to download model '{self.model_name}'.\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Download command finished, but model file was not found at:\n"
                f"  {self.model_path}\n"
            )

        self._log(f"✓ Model ready: {self.model_path.name}")

    # -------------------------------------------------------------------------
    # Audio preparation
    # -------------------------------------------------------------------------

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert input audio into mono float32 in range [-1, 1].

        Accepts:
        - shape (samples,)
        - shape (samples, 1)
        - shape (samples, channels)
        """
        if audio_data is None:
            raise ValueError("audio_data cannot be None")

        audio = np.asarray(audio_data, dtype=np.float32)

        if audio.ndim == 2:
            # Collapse stereo or multi-channel input to mono.
            audio = audio.mean(axis=1)

        audio = np.squeeze(audio)
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D mono audio after squeeze; got shape {audio.shape}")

        # Clamp just in case upstream audio is slightly out of range.
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def _write_temp_wav(self, audio_data: np.ndarray) -> Path:
        """
        Write audio as 16-bit mono PCM WAV for whisper-cli.

        whisper.cpp CLI wants 16-bit WAV input, so we convert explicitly here.
        """
        audio = self._normalize_audio(audio_data)

        # Convert float32 [-1, 1] -> int16 PCM.
        pcm16 = (audio * 32767.0).astype(np.int16)

        tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_cpp_"))
        wav_path = tmp_dir / "input.wav"

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # int16 = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm16.tobytes())

        return wav_path

    # -------------------------------------------------------------------------
    # CLI invocation + parsing
    # -------------------------------------------------------------------------

    def _build_command(self, wav_path: Path) -> list[str]:
        """
        Build the whisper.cpp command.

        We keep it intentionally minimal:
        - -m model path
        - -f wav input
        - -l en because your existing examples were English-focused
        - -nt to suppress timestamps in the transcript output if supported
          (if you prefer, remove it; parsing still works either way)

        Note:
        If you want multilingual auto-detection later, remove `-l en`.
        """
        return [
            str(self.binary_path),
            "-m",
            str(self.model_path),
            "-f",
            str(wav_path),
            "-l",
            "en",
            "-nt",
        ]

    def _parse_transcript(self, stdout: str) -> str:
        """
        Extract transcript text from whisper-cli stdout.

        whisper.cpp prints some diagnostic lines plus transcript lines.
        We strip common noise and timestamps, then join the spoken text.
        """
        lines: list[str] = []

        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Skip common non-transcript diagnostic lines.
            if any(
                line.startswith(prefix)
                for prefix in (
                    "whisper_",
                    "main:",
                    "system_info:",
                    "samples_to_mel:",
                    "encode ",
                    "decode ",
                    "timings:",
                    "ggml_",
                    "load ",
                    "processing ",
                )
            ):
                continue

            # Remove timestamp ranges like:
            # [00:00:00.000 --> 00:00:01.200] Hello there
            line = re.sub(
                r"^\[[0-9:\.\-\-> ]+\]\s*",
                "",
                line,
            ).strip()

            if line:
                lines.append(line)

        return " ".join(lines).strip()

    def transcribe(self, audio_data: np.ndarray) -> tuple[str, str, float | None]:
        """
        Transcribe audio via whisper.cpp CLI.

        Returns:
            (transcript, language, confidence)

        Notes:
        - language is fixed to "en" here because the CLI command uses `-l en`
        - confidence is not exposed by this wrapper, so it returns None
        """
        if not self.binary_path.exists():
            raise FileNotFoundError(f"whisper-cli not found: {self.binary_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        wav_path = self._write_temp_wav(audio_data)

        try:
            start = time.time()
            result = subprocess.run(
                self._build_command(wav_path),
                cwd=self.repo_dir,
                text=True,
                capture_output=True,
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                raise RuntimeError(
                    "whisper.cpp transcription failed.\n\n"
                    f"STDOUT:\n{result.stdout}\n\n"
                    f"STDERR:\n{result.stderr}"
                )

            transcript = self._parse_transcript(result.stdout)

            if not transcript:
                transcript = ""

            self._log(f"✓ whisper.cpp transcription completed in {elapsed:.2f}s")
            return transcript, "en", None

        finally:
            # Clean up temp WAV + temp dir.
            try:
                shutil.rmtree(wav_path.parent)
            except Exception:
                pass
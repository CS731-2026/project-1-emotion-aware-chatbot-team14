"""FasterWhisper transcription service (fallback).

Requires: faster-whisper, torch, huggingface-hub
Install:  pip install faster-whisper torch huggingface-hub

This is a fallback engine. Prefer whisper-cpp for local use on macOS.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .base import TranscriptionService


class FasterWhisperTranscriptionService(TranscriptionService):
    """Transcription service using faster-whisper (fallback).

    Required dependencies: faster-whisper, torch, huggingface-hub
    """

    REQUIRES_DEPS = ["faster_whisper", "torch", "huggingface_hub"]
    REQUIREMENTS_FILE = None

    def __init__(self, model_name: str = "base") -> None:
        """
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large-v3).
        """
        self.model_name = model_name
        self._model = self._load_model(model_name)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _select_device() -> tuple[str, str]:
        """Select device: CUDA -> CPU."""
        import torch

        if torch.cuda.is_available():
            print("Using CUDA")
            return "cuda", "int8_float16"
        else:
            print("Using CPU")
            return "cpu", "int8"

    @staticmethod
    def _check_model_cached(model_name: str = "base") -> bool:
        """Check if the Whisper model is already cached locally."""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(
            cache_dir, f"models--openai--whisper-{model_name}"
        )
        return os.path.exists(model_cache)

    @staticmethod
    def _download_model(model_name: str = "base") -> None:
        """Download the Whisper model via huggingface_hub."""
        from huggingface_hub import snapshot_download

        print(f"Downloading Whisper '{model_name}' model...")

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
        print(f"Download complete in {elapsed:.1f}s")

    def _load_model(self, model_name: str = "base"):
        """Load WhisperModel with automatic device fallback."""
        from faster_whisper import WhisperModel

        print(
            f"[{time.strftime('%H:%M:%S')}] Loading Whisper '{model_name}'..."
        )

        if not self._check_model_cached(model_name):
            self._download_model(model_name)

        device, compute_type = self._select_device()

        start_time = time.time()
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        elapsed = time.time() - start_time
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"Loaded on {device.upper()} in {elapsed:.2f}s"
        )

        return model

    # -------------------------------------------------------------------------
    # TranscriptionService interface
    # -------------------------------------------------------------------------

    def transcribe(
        self, audio_data: np.ndarray
    ) -> tuple[str, str, float]:
        """Transcribe audio using faster-whisper.

        Args:
            audio_data: numpy float32 array, shape (samples,) or (samples, 1).

        Returns:
            (transcript, language, confidence)
            confidence: average (1 - no_speech_prob) across segments.
        """
        if self._model is None:
            return "[Model not loaded]", "en", 0.0

        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        segments, info = self._model.transcribe(audio_data, beam_size=5)
        segments_list = list(segments)

        if segments_list:
            confidences = [1.0 - seg.no_speech_prob for seg in segments_list]
            confidence = sum(confidences) / len(confidences)
        else:
            confidence = 0.0

        transcript = " ".join(seg.text for seg in segments_list).strip()
        return transcript, info.language, confidence

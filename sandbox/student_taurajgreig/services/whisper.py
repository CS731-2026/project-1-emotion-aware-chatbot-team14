"""Whisper transcription service.

NOTE: Requires faster-whisper library and sufficient system resources.
      Currently disabled due to memory constraints on this machine.

To use:
  1. Ensure faster-whisper is installed: pip install faster-whisper
  2. Update speech_recognition_test.py to import and use WhisperTranscriptionService
  3. Change: service = MockTranscriptionService()
     To:     service = WhisperTranscriptionService("tiny")
"""

import numpy as np
import torch
import time
import os
from huggingface_hub import snapshot_download
from .base import TranscriptionService

# NOTE: Commented out to avoid import errors without faster-whisper
# from faster_whisper import WhisperModel


class WhisperTranscriptionService(TranscriptionService):
    """Whisper transcription service using faster-whisper.

    NOTE: This is currently disabled (commented out) due to memory constraints.
    Uncomment the imports above to use.
    """

    def __init__(self, model_name="tiny"):
        """Initialize Whisper model.

        Args:
            model_name: Model size (tiny, base, small, medium, large-v3)
        """
        # NOTE: Uncomment these when you have a compatible system
        # self.model = self._load_model(model_name)
        raise RuntimeError(
            "WhisperTranscriptionService is disabled (memory constrained).\n"
            "Use MockTranscriptionService for testing, or find an alternative service."
        )

    @staticmethod
    def _select_device():
        """Select device: CUDA -> CPU"""
        if torch.cuda.is_available():
            print("✓ Using CUDA")
            return "cuda", "int8_float16"
        else:
            print("✓ Using CPU")
            return "cpu", "int8"

    @staticmethod
    def _check_model_cached(model_name="base"):
        """Check if Whisper model is cached"""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, f"models--openai--whisper-{model_name}")
        return os.path.exists(model_cache)

    @staticmethod
    def _download_model(model_name="base"):
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

    def _load_model(self, model_name="base"):
        """Load Whisper model with automatic device fallback"""
        # NOTE: Uncomment when enabled
        # print(f"[{time.strftime('%H:%M:%S')}] Loading Whisper '{model_name}'...")
        #
        # if not self._check_model_cached(model_name):
        #     self._download_model(model_name)
        #
        # device, compute_type = self._select_device()
        #
        # print(f"[{time.strftime('%H:%M:%S')}] Detecting compatible device...")
        # start_time = time.time()
        # model = WhisperModel(
        #     model_name,
        #     device=device,
        #     compute_type=compute_type,
        # )
        # elapsed = time.time() - start_time
        # print(f"[{time.strftime('%H:%M:%S')}] ✓ Using {device.upper()} in {elapsed:.2f}s\n")
        #
        # return model
        pass

    def transcribe(self, audio_data: np.ndarray) -> tuple[str, str]:
        """Transcribe audio using Whisper.

        Args:
            audio_data: numpy array of audio samples (float32)

        Returns:
            (transcript, language) tuple
        """
        # NOTE: Uncomment when enabled
        # segments, info = self.model.transcribe(audio_data, beam_size=5)
        # transcript = " ".join([seg.text for seg in segments]).strip()
        # return transcript, info.language
        pass

"""
CS731 — Speech-to-Text Module
================================
Uses FasterWhisper (Group 15's choice) for transcription.
Records audio from microphone via sounddevice and saves as a temp WAV.

Usage
-----
  from speech import record_and_transcribe, SpeechTranscriber

  # Quick one-liner
  text = record_and_transcribe(duration=5)

  # Or use the class for persistent model
  stt  = SpeechTranscriber()
  text = stt.record_and_transcribe(duration=5)
"""

import os
import tempfile
from pathlib import Path

import numpy as np

# Optional imports — handled gracefully
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print('[WARN] sounddevice not installed. Voice input disabled. '
          'Run: pip install sounddevice')

try:
    import scipy.io.wavfile as wav
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print('[WARN] faster-whisper not installed. Voice input disabled. '
          'Run: pip install faster-whisper')


# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000    # Hz — Whisper expects 16 kHz
DEFAULT_DURATION = 5      # seconds
WHISPER_MODEL  = 'base'   # 'tiny' | 'base' | 'small' | 'medium' | 'large-v3'


# ── Transcriber class ─────────────────────────────────────────────────────────

class SpeechTranscriber:
    """
    Wraps FasterWhisper for efficient CPU inference.
    Model is loaded once and reused across calls.

    FasterWhisper (Group 15's choice) is significantly faster than
    standard Whisper due to CTranslate2 quantization.
    """

    def __init__(self, model_size: str = WHISPER_MODEL,
                 device: str = 'cpu', compute_type: str = 'int8'):
        """
        Args:
            model_size:   'tiny' | 'base' | 'small' | 'medium' | 'large-v3'
                          'base' is the best trade-off for real-time use
            device:       'cpu' | 'cuda'
            compute_type: 'int8' (CPU) | 'float16' (GPU)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError('faster-whisper not installed. pip install faster-whisper')

        print(f'[INFO] Loading FasterWhisper ({model_size}, {device}, {compute_type})...')
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate = SAMPLE_RATE
        print('[INFO] FasterWhisper ready.')

    def record(self, duration: int = DEFAULT_DURATION,
               feedback: bool = True) -> str:
        """
        Record audio from the default microphone.

        Args:
            duration: recording length in seconds
            feedback: print countdown messages

        Returns:
            path to a temporary WAV file (caller should delete after use)
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError('sounddevice not installed. pip install sounddevice')
        if not SCIPY_AVAILABLE:
            raise ImportError('scipy not installed. pip install scipy')

        if feedback:
            print(f'🎤 Recording for {duration}s... (speak now)')

        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate = self.sample_rate,
            channels   = 1,
            dtype      = 'int16',
        )
        sd.wait()  # block until recording finishes

        if feedback:
            print('✓ Recording complete.')

        # Save to temp file
        tmp_path = tempfile.mktemp(suffix='.wav')
        wav.write(tmp_path, self.sample_rate, audio)
        return tmp_path

    def transcribe(self, audio_path: str, language: str = 'en',
                   delete_after: bool = True) -> str:
        """
        Transcribe a WAV file to text.

        Args:
            audio_path:    path to WAV file
            language:      ISO 639-1 code (None for auto-detect)
            delete_after:  remove temp file after transcription

        Returns:
            transcribed text string
        """
        lang_arg  = {'language': language} if language else {}
        segments, info = self.model.transcribe(
            audio_path,
            beam_size = 5,
            **lang_arg
        )
        text = ' '.join(seg.text for seg in segments).strip()

        if delete_after and os.path.exists(audio_path):
            os.remove(audio_path)

        return text

    def record_and_transcribe(self, duration: int = DEFAULT_DURATION,
                               language: str = 'en') -> str:
        """Record then immediately transcribe. Returns text."""
        audio_path = self.record(duration=duration)
        return self.transcribe(audio_path, language=language)


# ── Module-level convenience function ────────────────────────────────────────

_shared_transcriber: SpeechTranscriber | None = None

def record_and_transcribe(duration: int = DEFAULT_DURATION,
                           language: str = 'en') -> str:
    """
    One-liner convenience function.
    Creates a shared transcriber on first call (loads model once).

    Args:
        duration: seconds to record
        language: ISO 639-1 code

    Returns:
        transcribed text
    """
    global _shared_transcriber
    if _shared_transcriber is None:
        if not WHISPER_AVAILABLE or not SOUNDDEVICE_AVAILABLE:
            return ''
        _shared_transcriber = SpeechTranscriber()

    return _shared_transcriber.record_and_transcribe(duration, language)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('speech.py module')
    print(f'  sounddevice available: {SOUNDDEVICE_AVAILABLE}')
    print(f'  faster-whisper available: {WHISPER_AVAILABLE}')
    print(f'  scipy available: {SCIPY_AVAILABLE}')

    if WHISPER_AVAILABLE and SOUNDDEVICE_AVAILABLE:
        import sys
        duration = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DURATION
        print(f'\nRecording {duration}s test...')
        text = record_and_transcribe(duration=duration)
        print(f'Transcription: "{text}"')
    else:
        print('\n[INFO] Install missing dependencies to enable voice input.')

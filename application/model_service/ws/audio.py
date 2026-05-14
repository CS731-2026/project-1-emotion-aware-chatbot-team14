"""Audio processing: browser WebM/Opus → PCM → STT transcript."""

import asyncio
import base64
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fastapi import WebSocket

import config
from core.stt.base import TranscriptionService
from ws.session import HarnessSession, TranscriptSegment, emit_debug

logger = logging.getLogger(__name__)

_MAX_TRANSCRIPT_SEGMENTS = 20


def _clean_transcript(text: str) -> str:
    return " ".join(text.strip().split())


def _looks_like_speech(text: str) -> bool:
    cleaned = _clean_transcript(text)
    if len(cleaned) < config.STT_MIN_TEXT_CHARS:
        return False
    return any(character.isalpha() for character in cleaned)


def _looks_like_placeholder_token(text: str) -> bool:
    cleaned = _clean_transcript(text)
    return bool(re.fullmatch(r"\[[A-Z0-9_\- ]+\]", cleaned))


def _passes_transcript_filter(text: str, confidence: float | None) -> tuple[bool, str | None]:
    """Gate STT output before it becomes conversation state.

    When the backend exposes confidence, we use it directly.
    whisper.cpp in the current CLI wrapper does not, so for that path we fall
    back to a lightweight transcript-quality heuristic until richer confidence
    signals are available.
    """
    cleaned = _clean_transcript(text)
    if not cleaned:
      return False, "empty transcript"

    if confidence is not None and confidence < config.STT_MIN_CONFIDENCE:
      return False, f"low confidence ({confidence:.2f} < {config.STT_MIN_CONFIDENCE:.2f})"

    if confidence is None:
      if _looks_like_placeholder_token(cleaned):
        return False, "placeholder transcript token"

      if not _looks_like_speech(cleaned):
        return False, "transcript failed whisper.cpp quality gate"

    return True, None


def decode_browser_audio_to_numpy(data: str) -> np.ndarray:
    """Convert a base64-encoded WebM/Opus blob to a float32 PCM numpy array.

    The browser records audio as WebM; ffmpeg re-encodes it to raw 16-bit
    PCM at 16 kHz mono (the format whisper.cpp expects), which is then
    normalised to the [-1.0, 1.0] float32 range.
    """
    with tempfile.TemporaryDirectory(prefix="hri_audio_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "chunk.webm"
        output_path = tmp_dir / "chunk.s16le"
        input_path.write_bytes(base64.b64decode(data))

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(input_path),
            "-ac", "1", "-ar", "16000", "-f", "s16le",
            str(output_path),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or f"exit status {exc.returncode}"
            raise RuntimeError(f"ffmpeg browser audio decode failed: {detail}") from exc

        pcm = np.frombuffer(output_path.read_bytes(), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0


async def process_audio_chunk(
    websocket: WebSocket,
    session: HarnessSession,
    stt: TranscriptionService | None,
    chunk_count: int,
    data: str,
    timestamp: float,
) -> None:
    """Decode and transcribe one audio chunk, then push the result back over WS.

    Runs decode and STT in worker threads via asyncio.to_thread so the event
    loop stays unblocked. Sends transcript_chunk (text) and audio_debug
    (timing metadata for the debug dashboard).
    """
    import time

    text = ""
    confidence: float | None = None
    stt_error: str | None = None
    timings_ms: dict[str, float] = {}

    if stt is not None:
        try:
            t0 = time.perf_counter()
            emit_debug(f"Audio chunk {chunk_count}: decoding browser audio before whisper.cpp")

            audio_np = await asyncio.to_thread(decode_browser_audio_to_numpy, data)
            timings_ms["ffmpeg_decode"] = round((time.perf_counter() - t0) * 1000, 1)

            emit_debug(f"Audio chunk {chunk_count}: decoded {audio_np.shape[0]} samples; running STT")

            t1 = time.perf_counter()
            text, _lang, confidence = await asyncio.to_thread(stt.transcribe, audio_np)
            timings_ms["whisper_cpp"] = round((time.perf_counter() - t1) * 1000, 1)
            timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)

            emit_debug(f"Audio chunk {chunk_count}: STT returned {text!r}; timings_ms={timings_ms}")
        except Exception as exc:
            stt_error = str(exc)
            logger.warning("STT failed for audio chunk: %s", exc)

    accepted, filter_reason = _passes_transcript_filter(text, confidence)

    if text and not accepted:
        emit_debug(
            f"Audio chunk {chunk_count}: transcript rejected by filter "
            f"(reason={filter_reason}, confidence={confidence})"
        )
        text = "[whisper.cpp transcript filtered]"

    if not text:
        text = "[whisper.cpp returned no text]"

    emit_debug(f"Audio chunk {chunk_count} for {session.profile_id} at {timestamp:.3f}; transcript={text!r}")

    session.transcript_buffer.append(TranscriptSegment(text=text, timestamp=timestamp))
    if len(session.transcript_buffer) > _MAX_TRANSCRIPT_SEGMENTS:
        session.transcript_buffer = session.transcript_buffer[-_MAX_TRANSCRIPT_SEGMENTS:]

    try:
        await websocket.send_text(json.dumps({
            "type": "transcript_chunk",
            "text": text,
            "audio_chunk_count": chunk_count,
            "timestamp": timestamp,
        }))
        await websocket.send_text(json.dumps({
            "type": "audio_debug",
            "audio_chunk_count": chunk_count,
            "byte_length": len(data),
            "text": text,
            "stt_engine": config.STT_ENGINE,
            "stt_loaded": stt is not None,
            "stt_error": stt_error,
            "stt_confidence": confidence,
            "stt_filter_reason": filter_reason,
            "timings_ms": timings_ms,
            "timestamp": timestamp,
        }))
    except Exception as exc:
        emit_debug(f"Audio chunk {chunk_count}: could not send STT result to frontend: {exc}")

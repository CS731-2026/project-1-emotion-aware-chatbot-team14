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

# Transcript-filter tuning. Lives here (not in config / .env) because these
# are algorithm thresholds that change with our STT engine + acoustic setup,
# not per-environment values. Tune in code; commit the result.
#
# Short clips are more often misheard noise — we hold them to a higher
# confidence bar than sustained speech. Long clips use config.STT_MIN_CONFIDENCE
# (kept in config.py because the reasoning agent also uses it to compute the
# percent-above-threshold value it shows the LLM).
_SHORT_CLIP_MS = 1500
_MIN_CONFIDENCE_SHORT = 0.85


def _clean_transcript(text: str) -> str:
    return " ".join(text.strip().split())


def _looks_like_speech(text: str) -> bool:
    cleaned = _clean_transcript(text)
    if len(cleaned) < config.STT_MIN_TEXT_CHARS:
        return False
    return any(character.isalpha() for character in cleaned)


def _looks_like_placeholder_token(text: str) -> bool:
    """Reject transcripts that are *only* a non-speech sound tag.

    Whisper emits tags like [BLANK_AUDIO], (sighs), (upbeat music), [ Pause ]
    when it hears ambient noise instead of speech. We don't want those in the
    transcript buffer or the LLM context.
    """
    cleaned = _clean_transcript(text)
    # [ALL_CAPS], [Pause], [ Blank Audio ] etc.
    if re.fullmatch(r"\[[A-Za-z0-9_\- ]+\]", cleaned):
        return True
    # (sighs), (upbeat music), (air whooshing) etc.
    if re.fullmatch(r"\([A-Za-z0-9_\- ,]+\)", cleaned):
        return True
    return False


def _passes_transcript_filter(
    text: str,
    confidence: float | None,
    duration_ms: float | None,
) -> tuple[bool, str | None]:
    """Gate STT output before it becomes conversation state.

    Rules, in order:
    1. Reject empty transcripts.
    2. Reject placeholder/non-speech tags like [BLANK_AUDIO] or (sighs).
       This check runs regardless of confidence — whisper happily assigns
       high confidence to these even when there was no real speech.
    3. Confidence gate, scaled by clip duration:
         - long clips (>= 1.5s) get a lenient threshold (STT_MIN_CONFIDENCE)
         - short clips need a stricter threshold (STT_MIN_CONFIDENCE_SHORT)
       so brief noise hallucinations have to clear a higher bar than
       sustained speech.
    4. When confidence is unavailable, fall back to a text-quality heuristic.
    """
    cleaned = _clean_transcript(text)
    if not cleaned:
        return False, "empty transcript"

    # Always reject non-speech tags first — they slip through high-confidence
    # gates because whisper is "confident" the token *is* [BLANK_AUDIO].
    if _looks_like_placeholder_token(cleaned):
        return False, "placeholder transcript token"

    if confidence is not None:
        is_short = duration_ms is not None and duration_ms < _SHORT_CLIP_MS
        threshold = _MIN_CONFIDENCE_SHORT if is_short else config.STT_MIN_CONFIDENCE
        if confidence < threshold:
            kind = "short" if is_short else "long"
            return False, (
                f"low confidence on {kind} clip "
                f"({confidence:.2f} < {threshold:.2f})"
            )
        return True, None

    # No confidence available — use the legacy text-quality heuristic.
    if not _looks_like_speech(cleaned):
        return False, "transcript failed quality gate (no confidence available)"
    return True, None


def decode_browser_audio_to_numpy(data: str, fmt: str = "webm") -> np.ndarray:
    """Convert a base64-encoded audio blob to a float32 PCM numpy array.

    The browser sends either WebM/Opus (legacy MediaRecorder path) or WAV
    (new AudioWorklet PCM-capture path). ffmpeg auto-detects either, but
    the file extension is set from `fmt` so it gets the right hint.
    """
    extension = "wav" if fmt.lower() == "wav" else "webm"
    with tempfile.TemporaryDirectory(prefix="hri_audio_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / f"chunk.{extension}"
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
    fmt: str = "webm",
    duration_ms: float | None = None,
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

            audio_np = await asyncio.to_thread(decode_browser_audio_to_numpy, data, fmt)
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

    accepted, filter_reason = _passes_transcript_filter(text, confidence, duration_ms)
    # Preserve what whisper actually heard so the debug panel can surface it
    # even when we reject the clip.
    raw_text = text

    if text and not accepted:
        emit_debug(
            f"Audio chunk {chunk_count}: transcript rejected by filter "
            f"(reason={filter_reason}, confidence={confidence}, raw={raw_text!r})"
        )
        # Keep the placeholder text so the frontend can see what happened, but
        # do NOT append it to the conversation transcript buffer below.
        text = "[whisper.cpp transcript filtered]"

    if not text:
        text = "[whisper.cpp returned no text]"

    emit_debug(f"Audio chunk {chunk_count} for {session.profile_id} at {timestamp:.3f}; transcript={text!r}")

    # Only accepted, real transcripts feed the LLM context. Rejected clips
    # (non-speech tags, low confidence, etc.) are debug-only.
    if accepted:
        session.transcript_buffer.append(
            TranscriptSegment(text=text, timestamp=timestamp, confidence=confidence)
        )
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
            "raw_text": raw_text,
            "accepted": accepted,
            "stt_engine": config.STT_ENGINE,
            "stt_loaded": stt is not None,
            "stt_error": stt_error,
            "stt_confidence": confidence,
            "stt_filter_reason": filter_reason,
            "duration_ms": duration_ms,
            "timings_ms": timings_ms,
            "timestamp": timestamp,
        }))
    except Exception as exc:
        emit_debug(f"Audio chunk {chunk_count}: could not send STT result to frontend: {exc}")

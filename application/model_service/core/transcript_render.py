"""Merge user-speech segments and system events into one chronological view.

The reasoning agent calls `compose_stream(...)` per turn to produce the
list of transcript lines the LLM sees. Speech and events are sorted by
timestamp and rendered with distinct syntaxes:

  speech:  [t.t] (conf NN%) the user said this thing
  event:   [t.t] {{kind: {…compact json…}}}

Confidence tag rendering for speech mirrors what was in
reasoning_agent before — it stays here so the LLM-prompt-assembly code
has one source of truth for stream rendering.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from core.events import SystemEvent


def _confidence_tag(confidence: float | None, remap: Any) -> str:
    """Format the (conf NN%) tag using the same remap the agent uses.

    `remap` is the sigmoid function from reasoning_agent — passed in
    rather than imported to avoid a circular dependency.
    """
    pct = remap(confidence)
    return f" (conf {pct}%)" if pct is not None else ""


def compose_stream(
    transcript_segments: Sequence[Any],
    system_events: Iterable[SystemEvent],
    *,
    confidence_remap: Any,
) -> list[str]:
    """Return the merged chronological stream of transcript lines.

    Each transcript_segment is expected to have `text`, `timestamp`,
    `confidence` attributes (TranscriptSegment shape).
    """
    speech_entries: list[tuple[float, str]] = []
    for seg in transcript_segments:
        t = float(seg.timestamp)
        tag = _confidence_tag(getattr(seg, "confidence", None), confidence_remap)
        speech_entries.append((t, f"[{t:.1f}s]{tag} {seg.text}"))

    event_entries: list[tuple[float, str]] = [(ev.t, ev.render()) for ev in system_events]

    merged = sorted(speech_entries + event_entries, key=lambda pair: pair[0])
    return [line for _, line in merged]

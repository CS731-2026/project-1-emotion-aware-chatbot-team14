"""LLM-backed reasoning agent with emotional context and transcript injection."""

from __future__ import annotations

from .base import LLMProvider, Message

# TODO: prompt engineering flow not yet decided — this is a placeholder persona.
# The two inputs (emotional_context + transcript_context) are wired in correctly;
# the exact wording and structure of the system prompt should be refined once
# the team has agreed on the empathy bot's interaction design.
SYSTEM_PROMPT = (
    "You are an empathetic conversational companion. "
    "You will receive two pieces of context before each user message: "
    "the user's detected emotional state, and a timestamped transcript of what they recently said. "
    "Use both to calibrate your tone and response — but never mention, reference, or "
    "acknowledge either directly. "
    "Prioritise emotional validation before any other content. "
    "If the person seems distressed: lead with warmth, slow down, keep it simple. "
    "If calm: match their energy. "
    "Keep responses concise. One idea at a time."
)

_TRANSCRIPT_HEADER = "Recent speech transcript (with timestamps):"
_NO_TRANSCRIPT = "No speech transcript available."


def _format_transcript(transcript_segments: list) -> str:
    """Format timestamped transcript segments into a readable context string.

    TODO: the exact format here should be refined as part of prompt engineering.
    """
    if not transcript_segments:
        return _NO_TRANSCRIPT
    lines = [_TRANSCRIPT_HEADER]
    for seg in transcript_segments:
        lines.append(f"  [{seg.timestamp:.1f}s] {seg.text}")
    return "\n".join(lines)


class LLMReasoningAgent:
    """Wraps an LLMProvider and assembles context-aware prompts for each turn.

    Message assembly order per turn (matches system architecture diagram):
    1. Static system prompt (empathy bot persona).
    2. Windowed conversation history (user/assistant turns only).
    3. Emotional context — from EmotionalReasoningAgent, injected as system message.
    4. Transcript context — timestamped STT output, injected as system message.
    5. Current user message.

    Both emotional_context and transcript_context are separate inputs as per the
    architecture spec. Prompt engineering for how to combine them is TBD.
    """

    def __init__(
        self,
        llm: LLMProvider,
        history_window: int = 10,
    ) -> None:
        """
        Args:
            llm:            An instantiated LLMProvider.
            history_window: Max number of prior messages (user+assistant) to include.
        """
        self._llm = llm
        self._history_window = history_window

    def reason(
        self,
        message: str,
        emotional_context: str,
        history: list[Message],
        transcript_segments: list | None = None,
    ) -> str:
        """Build the full prompt and call the LLM.

        Args:
            message:             The user's current message text.
            emotional_context:   Produced by EmotionalReasoningAgent.analyse().
                                 Injected as a system message — never merged into
                                 the user message.
            history:             Previous conversation turns (user/assistant only).
                                 The most recent `history_window` messages are used.
            transcript_segments: Recent TranscriptSegment records from the STT
                                 pipeline, injected as a separate system message.
                                 None or empty list → fallback string is used.

        Returns:
            The assistant's response string.
        """
        messages: list[Message] = []

        # 1. Static persona system prompt.
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # 2. Windowed history — keep only user/assistant turns.
        prior = [m for m in history if m["role"] in ("user", "assistant")]
        messages.extend(prior[-self._history_window:])

        # 3. Emotional context (from Emotional Affect Model → EmotionalReasoningAgent).
        if emotional_context:
            messages.append({"role": "system", "content": emotional_context})

        # 4. Transcript context (from STT pipeline — separate input per architecture spec).
        transcript_context = _format_transcript(transcript_segments or [])
        messages.append({"role": "system", "content": transcript_context})

        # 5. Current user turn.
        messages.append({"role": "user", "content": message})

        return self._llm.chat(messages)

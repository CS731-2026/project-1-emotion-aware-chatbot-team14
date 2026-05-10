"""LLM reasoning agent — assembles the prompt and calls the LLM.

PROMPT ENGINEERING — OPEN QUESTIONS (decisions needed before production):
─────────────────────────────────────────────────────────────────────────
1. SYSTEM PROMPT
   What persona, constraints, and tone directives should the empathy bot have?
   → Edit SYSTEM_PROMPT below.

2. CONVERSATION HISTORY
   - How many prior turns should be included?
   - Should history be windowed (last N), summarised, or passed in full?
   - Should system messages from prior turns be stripped or kept?
   - How should the history be formatted (raw role/content, or compressed)?
   → See _build_history_messages().

3. EMOTIONAL CONTEXT REPRESENTATION
   - How should the emotional signal be injected into the prompt?
     Options: system message, prefix on user message, structured JSON block, etc.
   - Should it appear before or after the transcript?
   - Should low-confidence readings be suppressed or down-weighted?
   → See _build_emotional_context_message().

4. TRANSCRIPT REPRESENTATION
   - How should timestamped transcript segments be formatted?
   - Should timestamps be included, relative or absolute?
   - Should older segments be trimmed or summarised?
   → See _build_transcript_message().

5. RESPONSE / HISTORY APPENDING
   - After the LLM replies, how should that turn be stored?
   - Should the emotional context and transcript be appended alongside it
     so future turns have that context in history?
   - Is history owned here, by the caller (routers/chat.py), or by the backend?
   → Currently the backend (chat.router.ts) appends via profileStore.
     Decide whether emotional metadata should also be persisted there.
"""

from __future__ import annotations

from ws.session import TranscriptSegment
from .base import LLMProvider, Message

# TODO: decide empathy bot persona, tone directives, and constraints.
# This is a placeholder — do not treat it as final.
SYSTEM_PROMPT = (
    "You are an empathetic conversational companion. "
    "You will receive context about the user's emotional state and recent speech. "
    "Use both to calibrate your response. "
    "Prioritise emotional validation. Never reference the context signals directly."
)


def _build_history_messages(history: list[Message], window: int) -> list[Message]:
    # TODO: decide history strategy — windowed, summarised, or full.
    # TODO: decide whether to filter out system-role messages from prior turns.
    # TODO: decide window size (currently hardcoded to caller-supplied value).
    prior = [m for m in history if m["role"] in ("user", "assistant")]
    return prior[-window:]


def _build_emotional_context_message(emotional_context: str) -> Message | None:
    # TODO: decide how to represent emotional context in the prompt.
    # Options: system message, prefix on user message, structured block, omit entirely.
    # TODO: decide whether to suppress when confidence is low (requires passing confidence here).
    if not emotional_context:
        return None
    return {"role": "system", "content": emotional_context}


def _build_transcript_message(transcript_segments: list[TranscriptSegment]) -> Message | None:
    # TODO: decide how to format the transcript for the LLM.
    # TODO: decide whether timestamps should be included, and in what format.
    # TODO: decide whether to summarise long transcripts rather than truncate.
    if not transcript_segments:
        return None
    lines = ["Recent speech (with timestamps):"]
    for seg in transcript_segments:
        lines.append(f"  [{seg.timestamp:.1f}s] {seg.text}")
    return {"role": "system", "content": "\n".join(lines)}


class LLMReasoningAgent:
    """Calls the LLM with an assembled prompt.

    The three inputs — conversation history, emotional context, and transcript —
    are each built by a dedicated function above. All prompt engineering
    decisions are isolated to those functions and SYSTEM_PROMPT; this class
    only orchestrates the assembly order and the LLM call.

    TODO: once the prompt engineering questions above are resolved, the
    assembly order in reason() may also need to change.
    """

    def __init__(self, llm: LLMProvider, history_window: int = 10) -> None:
        # TODO: history_window default is arbitrary — revisit once history
        # strategy is decided (question 2 above).
        self._llm = llm
        self._history_window = history_window

    def reason(
        self,
        message: str,
        emotional_context: str,
        history: list[Message],
        transcript_segments: list[TranscriptSegment] | None = None,
    ) -> str:
        """Assemble the prompt from all inputs and return the LLM reply.

        Args:
            message:             The user's current message.
            emotional_context:   Output of EmotionalReasoningAgent.analyse().
            history:             Prior conversation turns (user/assistant).
            transcript_segments: Recent TranscriptSegment records from STT.

        Returns:
            The LLM's response string.

        TODO: the assembly order below is a placeholder — revisit once
        prompt engineering questions 2–4 above are decided.
        """
        messages: list[Message] = []

        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        messages.extend(_build_history_messages(history, self._history_window))

        emotional_msg = _build_emotional_context_message(emotional_context)
        if emotional_msg:
            messages.append(emotional_msg)

        transcript_msg = _build_transcript_message(transcript_segments or [])
        if transcript_msg:
            messages.append(transcript_msg)

        messages.append({"role": "user", "content": message})

        return self._llm.chat(messages)

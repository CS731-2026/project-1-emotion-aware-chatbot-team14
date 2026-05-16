"""LLM reasoning agent — lays out the current reasoning pipeline explicitly.

This module is meant to be a planning surface for future reasoning work.
The current implementation is intentionally simple, but the code is structured
so the eventual reasoning steps are easy to discuss and replace:

1. Collect the current inputs for the turn.
2. Derive the prompt context from those inputs.
3. Assemble the final prompt messages.
4. Call the LLM.

Future improvements should generally fit into one of those stages rather than
being threaded through the code ad hoc.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, get_args

from ws.session import TranscriptSegment
from .base import LLMProvider, Message

logger = logging.getLogger(__name__)

# Which page / overall conversational mode the user is currently in.
Mode = Literal["qa", "feedback", "consent", "done"]

# Within `qa` mode, which scripted stage of the anxiety-reduction arc.
Stage = Literal["open", "explore", "ground", "close"]

# Shared persona used across every mode. Mode / stage prompts layer on top.
BASE_PERSONA = (
    "You are the live empathy system the user is speaking with inside this application. "
    "You can hear the user's speech and the application can also observe the user's face through the webcam. "
    "You will receive hidden context derived from recent face-based emotional cues and recent transcript history. "
    "Respond as the system itself, not as a component inside it. "
    "Speak naturally, warmly, clearly, and briefly. "
    "Use the hidden emotional and transcript context to calibrate your tone, "
    "but do not explicitly mention the hidden signals, the webcam analysis, or internal system components unless the user directly asks about them."
)

# TODO: tune mode prompts after user testing.
MODE_PROMPTS: dict[Mode, str] = {
    "qa": (
        "You are guiding the user through a brief anxiety-reduction conversation. "
        "Stay focused on listening and helping the user feel heard. "
        "Do not give clinical advice."
    ),
    "feedback": (
        "The user is on the feedback check-in page and is periodically self-reporting how they feel. "
        "Stay mostly silent and brief. Only speak up when recent self-reports or emotional context suggest the user would benefit from acknowledgement. "
        "When you do speak, keep it to one or two short sentences."
    ),
    "consent": "",
    "done": "",
}

# TODO: tune stage prompts after user testing. These only apply when mode == "qa".
STAGE_PROMPTS: dict[Stage, str] = {
    "open": (
        "You are in the OPENING stage. "
        "Greet the user warmly in one or two sentences and invite them to share what is on their mind. "
        "Do not problem-solve. Do not ask multiple questions."
    ),
    "explore": (
        "You are in the EXPLORE stage. Actively listen. "
        "Reflect what the user has said in your own words, then ask one curious follow-up. "
        "Do not give advice."
    ),
    "ground": (
        "You are in the GROUND stage. "
        "Offer one short, concrete grounding or regulation exercise appropriate to the user's current emotional state "
        "(for example: a brief breathing pattern, a 5-4-3-2-1 sensory exercise, or a body-awareness check). "
        "Lead the exercise — do not just suggest it. Keep it to a few short turns."
    ),
    "close": (
        "You are in the CLOSE stage. "
        "Briefly summarise one or two things the user shared, affirm them, and end gently. "
        "Do not open new threads."
    ),
}


# Appended to every system prompt so the LLM emits the structured transition
# response the rest of the stack expects. The parser is forgiving — see
# parse_reasoning_output — but the prompt is written as if the schema is strict.
OUTPUT_INSTRUCTIONS = (
    "Respond with a single JSON object and nothing else. No markdown fences. "
    "The object must have exactly these keys:\n"
    '  "reply": a string — your spoken reply to the user, natural conversational text.\n'
    '  "next_mode": one of "qa", "feedback", or "done".\n'
    '  "next_stage": when next_mode is "qa", one of "open", "explore", "ground", "close"; otherwise null.\n'
    "\n"
    "You decide where the conversation goes next:\n"
    "- Stay in qa while supportive conversation is still useful.\n"
    "- Advance through qa stages: open → explore → ground → close.\n"
    '- Move to "feedback" when a brief self-report check-in would help (for example, a natural pause, or a disconnect between what the user says and how they sound).\n'
    '- Move to "done" when the conversation has reached a meaningful close.\n'
    "- Returning to qa from feedback is fine if the user wants to keep talking.\n"
    "\n"
    "Do not announce these transitions to the user. They are internal."
)


def _system_prompt(mode: Mode, stage: Stage | None) -> str:
    """Compose the system prompt from base persona + mode + (optional) stage + output instructions."""
    parts = [BASE_PERSONA]
    mode_part = MODE_PROMPTS.get(mode, "")
    if mode_part:
        parts.append(mode_part)
    if stage is not None and mode == "qa":
        stage_part = STAGE_PROMPTS.get(stage, "")
        if stage_part:
            parts.append(stage_part)
    parts.append(OUTPUT_INSTRUCTIONS)
    return "\n\n".join(parts)


@dataclass(frozen=True)
class ReasoningResult:
    """Structured output of one reasoning turn.

    The LLM is asked to emit JSON containing all three fields. When parsing
    fails, `reply` falls back to the raw text and the mode/stage carry over
    from the caller's current state — so a malformed response degrades to a
    plain text reply with no transition rather than a hard failure.
    """

    reply: str
    next_mode: Mode
    next_stage: Stage | None


_MODES: tuple[str, ...] = get_args(Mode)
_STAGES: tuple[str, ...] = get_args(Stage)
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_reasoning_output(
    raw_text: str,
    current_mode: Mode,
    current_stage: Stage | None,
) -> ReasoningResult:
    """Parse the LLM's JSON output. Degrade gracefully on any failure."""
    stripped = _JSON_FENCE_RE.sub("", raw_text).strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("reasoner output was not valid JSON; falling back to raw text")
        return ReasoningResult(reply=raw_text.strip(), next_mode=current_mode, next_stage=current_stage)

    if not isinstance(data, dict):
        logger.warning("reasoner output JSON was not an object; falling back to raw text")
        return ReasoningResult(reply=raw_text.strip(), next_mode=current_mode, next_stage=current_stage)

    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        logger.warning("reasoner output missing 'reply' field; falling back to raw text")
        reply = raw_text.strip()

    raw_next_mode = data.get("next_mode")
    next_mode: Mode = current_mode
    if isinstance(raw_next_mode, str) and raw_next_mode in _MODES:
        next_mode = raw_next_mode  # type: ignore[assignment]
    elif raw_next_mode is not None:
        logger.warning("reasoner emitted unknown next_mode=%r; keeping current=%s", raw_next_mode, current_mode)

    raw_next_stage = data.get("next_stage")
    next_stage: Stage | None = current_stage
    if raw_next_stage is None:
        next_stage = None
    elif isinstance(raw_next_stage, str) and raw_next_stage in _STAGES:
        next_stage = raw_next_stage  # type: ignore[assignment]
    else:
        logger.warning("reasoner emitted unknown next_stage=%r; keeping current=%s", raw_next_stage, current_stage)

    return ReasoningResult(reply=reply, next_mode=next_mode, next_stage=next_stage)


@dataclass(frozen=True)
class ReasoningInputs:
    """All raw inputs available to the reasoning layer for one turn."""

    current_message: str
    emotional_context: str
    history: list[Message]
    transcript_segments: list
    mode: Mode = "qa"
    stage: Stage | None = None


@dataclass(frozen=True)
class PromptContext:
    """Structured intermediate prompt context derived from ReasoningInputs.

    This is the main place to extend later when we want richer reasoning:
    summarised history, confidence-aware emotion handling, transcript pruning,
    tool traces, chain-of-thought scaffolding, and so on.
    """

    system_prompt: str
    history_messages: list[Message]
    emotional_message: Message | None
    transcript_message: Message | None
    # Populated in Phase 4 from session.feedback_buffer. Until then, always None.
    feedback_message: Message | None
    transcript_lines: list[str]


def _history_window(history: list[Message], window: int) -> list[Message]:
    # TODO: decide history strategy — windowed, summarised, or full.
    # TODO: decide whether to filter out system-role messages from prior turns.
    prior = [message for message in history if message["role"] in ("user", "assistant")]
    return prior[-window:]


def _emotional_message(emotional_context: str) -> Message | None:
    # TODO: decide how to represent emotional context in the prompt.
    # Options: system message, prefix on user message, structured block, omit entirely.
    if not emotional_context:
        return None
    return {"role": "system", "content": emotional_context}

def _transcript_lines(transcript_segments: list) -> list[str]:
    if not transcript_segments:
        return []
    return [f"[{segment.timestamp:.1f}s] {segment.text}" for segment in transcript_segments]


def _build_transcript_message(transcript_segments: list[TranscriptSegment]) -> Message | None:
    # TODO: decide how to format the transcript for the LLM.
    # TODO: decide whether timestamps should be included, and in what format.
    transcript_lines = _transcript_lines(transcript_segments)
    if not transcript_lines:
        return None

    lines = ["Recent speech (with timestamps):", *[f"  {line}" for line in transcript_lines]]
    return {"role": "system", "content": "\n".join(lines)}


class LLMReasoningAgent:
    """Owns the high-level reasoning pipeline for one conversational turn."""

    def __init__(self, llm: LLMProvider, history_window: int = 10) -> None:
        # TODO: history_window default is arbitrary — revisit once history
        # strategy is decided.
        self._llm = llm
        self._history_window = history_window

    def collect_inputs(
        self,
        message: str,
        emotional_context: str,
        history: list[Message],
        transcript_segments: list | None = None,
        mode: Mode = "qa",
        stage: Stage | None = None,
    ) -> ReasoningInputs:
        """Normalise raw inputs into one explicit turn object."""
        return ReasoningInputs(
            current_message=message,
            emotional_context=emotional_context,
            history=history,
            transcript_segments=transcript_segments or [],
            mode=mode,
            stage=stage,
        )

    def derive_prompt_context(self, inputs: ReasoningInputs) -> PromptContext:
        """Turn raw inputs into structured prompt context.

        This is the best place to change future reasoning behavior without
        touching the chat route or provider adapters.
        """
        return PromptContext(
            system_prompt=_system_prompt(inputs.mode, inputs.stage),
            history_messages=_history_window(inputs.history, self._history_window),
            emotional_message=_emotional_message(inputs.emotional_context),
            transcript_message=_build_transcript_message(inputs.transcript_segments),
            feedback_message=None,
            transcript_lines=_transcript_lines(inputs.transcript_segments),
        )

    def assemble_messages(
        self,
        inputs: ReasoningInputs,
        context: PromptContext,
    ) -> list[Message]:
        """Build the final provider-facing message list."""
        messages: list[Message] = [{"role": "system", "content": context.system_prompt}]
        messages.extend(context.history_messages)

        if context.emotional_message:
            messages.append(context.emotional_message)

        if context.transcript_message:
            messages.append(context.transcript_message)

        if context.feedback_message:
            messages.append(context.feedback_message)

        messages.append({"role": "user", "content": inputs.current_message})
        return messages

    def debug_snapshot(
        self,
        message: str,
        emotional_context: str,
        history: list[Message],
        transcript_segments: list | None = None,
        mode: Mode = "qa",
        stage: Stage | None = None,
    ) -> dict:
        """Return a structured snapshot of the current reasoning pipeline."""
        inputs = self.collect_inputs(message, emotional_context, history, transcript_segments, mode, stage)
        context = self.derive_prompt_context(inputs)
        prompt_messages = self.assemble_messages(inputs, context)

        return {
            "provider": self._llm.provider_name,
            "model": self._llm.model_name,
            "current_message": inputs.current_message,
            "mode": inputs.mode,
            "stage": inputs.stage,
            "system_prompt": context.system_prompt,
            "history_window": self._history_window,
            "history_messages": context.history_messages,
            "emotional_context": inputs.emotional_context,
            "transcript_lines": context.transcript_lines,
            "prompt_messages": prompt_messages,
        }

    def reason(
        self,
        message: str,
        emotional_context: str,
        history: list[Message],
        transcript_segments: list[TranscriptSegment] | None = None,
        mode: Mode = "qa",
        stage: Stage | None = None,
    ) -> ReasoningResult:
        """Run the current reasoning pipeline and return a structured result."""
        inputs = self.collect_inputs(message, emotional_context, history, transcript_segments, mode, stage)
        context = self.derive_prompt_context(inputs)
        prompt_messages = self.assemble_messages(inputs, context)
        raw_response = self._llm.chat(prompt_messages)
        return parse_reasoning_output(raw_response, mode, stage)

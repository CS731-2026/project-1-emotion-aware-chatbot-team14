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
    "\n\n"
    "HIDDEN HEARING-QUALITY CONTEXT — read carefully, never reveal:\n"
    "Each transcript line may carry a tag like (conf NN%). It is a private "
    "signal of how well the audio came through — higher means clear, lower "
    "means hard to make out. You use it silently to decide whether you "
    "understood the user. It is not part of the conversation.\n"
    "\n"
    "ABSOLUTELY DO NOT, under any circumstances, mention or reference:\n"
    "- the words 'confidence', 'threshold', 'percent', 'score', 'signal', "
    "'tag', 'system', 'audio quality', or any number from these tags\n"
    "- the fact that any such metadata exists\n"
    "- phrases like 'that came through at X%', 'I heard you with high "
    "confidence', 'the audio quality was…', 'on my end it sounds…'\n"
    "If you find yourself about to say any of the above, stop and rewrite "
    "your reply as a normal human conversational turn.\n"
    "\n"
    "WHEN TO ACT on a low value:\n"
    "Only when the latest user line is *genuinely hard to make out* — the "
    "words read garbled, fragmentary, or out of context AND the tag value is "
    "low. A clear, sensible sentence is fine even if its tag isn't 100%. "
    "Most turns, you should not act on the tag at all and just respond "
    "naturally to what the user said.\n"
    "\n"
    "HOW to act, when you do:\n"
    "Speak like a person who didn't quite catch what someone said in a "
    "noisy room. Examples of the right tone:\n"
    "  - 'Sorry, I didn't quite catch that — could you say it again?'\n"
    "  - 'I missed that, would you mind repeating?'\n"
    "  - 'It's a bit hard to hear you — could you speak up a touch?'\n"
    "Never explain *why* you missed it. Never reference any number."
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


# Temporarily disabled: we don't want the LLM proposing mode/stage transitions
# yet. When False, the LLM is asked to reply naturally (no JSON envelope) and
# parse_reasoning_output's raw-text fallback returns reply with next_mode /
# next_stage carried over from the caller. Flip back to True once the reasoner
# is wired to drive check-ins through the proper channel.
INCLUDE_OUTPUT_INSTRUCTIONS = False

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


def _system_prompt(
    mode: Mode,
    stage: Stage | None,
    intention: str | None = None,
) -> str:
    """Compose the system prompt.

    When `intention` is provided (the conductor-driven path), the result is
    BASE_PERSONA + intention. The intention is the only state-specific text;
    the LLM gets no other vocabulary about the state machine.

    When `intention` is None (legacy path, while callers are still being
    migrated), fall back to the MODE_PROMPTS / STAGE_PROMPTS composition.
    Iteration 7 deletes the legacy path along with the mode/stage dicts.
    """
    parts = [BASE_PERSONA]
    if intention is not None:
        if intention.strip():
            parts.append(intention)
    else:
        mode_part = MODE_PROMPTS.get(mode, "")
        if mode_part:
            parts.append(mode_part)
        if stage is not None and mode == "qa":
            stage_part = STAGE_PROMPTS.get(stage, "")
            if stage_part:
                parts.append(stage_part)
    if INCLUDE_OUTPUT_INSTRUCTIONS:
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
    # When provided, replaces MODE_PROMPTS/STAGE_PROMPTS composition. Supplied
    # by the conductor each turn via the chat router. Iteration 7 makes this
    # the only path.
    intention: str | None = None


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

import math

# Sigmoid-style remap of raw STT confidence into an LLM-friendly 0-100 scale.
# Raw confidence from whisper for real, accepted speech tends to cluster
# tightly in [0.65, 1.0] — a linear or percent-above-threshold mapping
# squashes that into a narrow range that reads as "low" even for good
# transcripts. The sigmoid below spreads that band out:
#
#   raw 0.65 (barely passed)  → ~46
#   raw 0.80 (typical)        → ~65
#   raw 0.92 (clean)          → ~81
#   raw 0.99 (very clean)     → ~87
#
# Tune via the constants below if the spread feels wrong for the LLM.
_CONF_REMAP_FLOOR = 30      # minimum displayed value
_CONF_REMAP_RANGE = 70      # max displayed = floor + range
_CONF_REMAP_MIDPOINT = 0.80 # raw value that lands near the curve's centre
_CONF_REMAP_STEEPNESS = 8.0


def _confidence_for_llm(raw: float | None) -> int | None:
    """Map raw confidence in [0,1] to a sigmoid-shaped 0-100 score for the LLM.

    Returns None when the engine didn't expose a confidence value.
    """
    if raw is None:
        return None
    value = _CONF_REMAP_FLOOR + _CONF_REMAP_RANGE / (
        1.0 + math.exp(-_CONF_REMAP_STEEPNESS * (raw - _CONF_REMAP_MIDPOINT))
    )
    return max(0, min(100, round(value)))


def _transcript_lines(transcript_segments: list) -> list[str]:
    if not transcript_segments:
        return []
    lines: list[str] = []
    for segment in transcript_segments:
        pct = _confidence_for_llm(getattr(segment, "confidence", None))
        conf_tag = f" (conf {pct}%)" if pct is not None else ""
        lines.append(f"[{segment.timestamp:.1f}s]{conf_tag} {segment.text}")
    return lines


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
        intention: str | None = None,
    ) -> ReasoningInputs:
        """Normalise raw inputs into one explicit turn object."""
        return ReasoningInputs(
            current_message=message,
            emotional_context=emotional_context,
            history=history,
            transcript_segments=transcript_segments or [],
            mode=mode,
            stage=stage,
            intention=intention,
        )

    def derive_prompt_context(self, inputs: ReasoningInputs) -> PromptContext:
        """Turn raw inputs into structured prompt context.

        This is the best place to change future reasoning behavior without
        touching the chat route or provider adapters.
        """
        return PromptContext(
            system_prompt=_system_prompt(inputs.mode, inputs.stage, inputs.intention),
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
        intention: str | None = None,
    ) -> dict:
        """Return a structured snapshot of the current reasoning pipeline."""
        inputs = self.collect_inputs(
            message, emotional_context, history, transcript_segments, mode, stage, intention
        )
        context = self.derive_prompt_context(inputs)
        prompt_messages = self.assemble_messages(inputs, context)

        return {
            "provider": self._llm.provider_name,
            "model": self._llm.model_name,
            "current_message": inputs.current_message,
            "mode": inputs.mode,
            "stage": inputs.stage,
            "intention": inputs.intention,
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
        intention: str | None = None,
    ) -> ReasoningResult:
        """Run the current reasoning pipeline and return a structured result."""
        inputs = self.collect_inputs(
            message, emotional_context, history, transcript_segments, mode, stage, intention
        )
        context = self.derive_prompt_context(inputs)
        prompt_messages = self.assemble_messages(inputs, context)
        raw_response = self._llm.chat(prompt_messages)
        return parse_reasoning_output(raw_response, mode, stage)

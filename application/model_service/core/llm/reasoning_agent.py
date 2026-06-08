"""LLM reasoning agent, lays out the current reasoning pipeline explicitly.

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
from dataclasses import dataclass, field

from core.events import SystemEvent
from core.tool_emissions import ToolEmission, extract_emissions
from core.transcript_render import compose_stream
from ws.session import TranscriptSegment
from .base import LLMProvider, Message

logger = logging.getLogger(__name__)


# Shared persona used by every state. The conductor's intention_prompt
# layers on top per turn.
#
# Persona: EmpathyBot, GP patient feedback assistant.
# Designed for low-digital-literacy older adults who may be confused or
# anxious after an AI-assisted GP appointment. Plain English, short
# replies, warm and reassuring. Never minimise a concern.
BASE_PERSONA = (
    "You are a friendly, patient assistant helping someone understand and "
    "share their experience after a GP visit today. "
    "The person you're speaking with may be confused or anxious, "
    "especially about a computer or AI tool that was used by their doctor. "
    "Always respond in plain, everyday English. No medical or technical "
    "jargon. No words like 'algorithm', 'data processing', or 'AI system'. "
    "If you need to refer to technology, say 'computer' or 'the tool the "
    "doctor used'. "
    "Keep every reply under three sentences. "
    "Be warm, calm, and reassuring at all times. "
    "Never minimise a concern, validate it first, then explain simply. "
    "If the person's emotional cues or their answers show discomfort or "
    "worry, open with empathy before any explanation. "
    "You will receive hidden context about the person's emotional state "
    "derived from their facial expressions. Use it to calibrate your tone "
    "but never mention the webcam, emotion detection, or any internal "
    "system component."
    "\n\n"
    "TRANSCRIPT CONVENTION, read carefully:\n"
    "Lines in the recent transcript context may include fragments inside "
    "{{double braces}}. These are observations from the system, emotion "
    "sensors, form selections, timing signals, not things the person said "
    "aloud. Treat them as hidden background context to inform your reply. "
    "Never quote them back, refer to them as something the person 'said', "
    "or reveal that this metadata exists."
    "\n\n"
    "HIDDEN HEARING-QUALITY CONTEXT, read carefully, never reveal:\n"
    "Each transcript line may carry a tag like (conf NN%). It is a private "
    "signal of how well the audio came through. Use it silently to decide "
    "whether you understood the person. It is not part of the conversation.\n"
    "\n"
    "ABSOLUTELY DO NOT mention or reference:\n"
    "- 'confidence', 'threshold', 'percent', 'score', 'signal', 'tag', "
    "'audio quality', or any number from these tags\n"
    "- phrases like 'that came through at X%' or 'the audio quality was…'\n"
    "If you didn't catch something clearly, respond like a person in a "
    "noisy room:\n"
    "  - 'Sorry, I didn't quite catch that, could you say it again?'\n"
    "  - 'I missed that, would you mind repeating?'\n"
    "Never explain why you missed it. Never reference any number."
)

def _system_prompt(
    intention: str | None = None,
    advance_instruction: str | None = None,
) -> str:
    """Compose the system prompt: BASE_PERSONA + (intention) + (advance_instruction).

    The intention is the conductor's state-specific stance for the LLM.
    advance_instruction is the optional tool-emission directive that lets
    the conductor pick up an [[advance]] marker if the LLM senses a
    natural pause. The LLM sees no state-machine vocabulary, no mode/stage
    enum, no transition JSON.
    """
    parts = [BASE_PERSONA]
    if intention and intention.strip():
        parts.append(intention)
    if advance_instruction:
        parts.append(advance_instruction)
    return "\n\n".join(parts)


@dataclass(frozen=True)
class ReasoningResult:
    """Structured output of one reasoning turn.

    `reply` is the user-facing text with any inline tool-emission markers
    (e.g. `[[advance]]`) stripped. `emissions` contains the parsed
    ToolEmission objects the LLM produced, empty when the model didn't
    emit anything. The conductor reads emissions[*].name to decide
    transitions.
    """

    reply: str
    emissions: list[ToolEmission] = field(default_factory=list)


# Used by extract_json() to strip ```json fences around JSON-mode outputs.
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ReasoningInputs:
    """All raw inputs available to the reasoning layer for one turn."""

    current_message: str
    emotional_context: str
    history: list[Message]
    transcript_segments: list
    # Supplied by the conductor each turn via the chat router. The
    # conductor's per-state intention_prompt.
    intention: str | None = None
    # Typed system events (form answers, emotion windows, segment summaries,
    # etc.) merged with transcript_segments into the LLM-facing stream.
    system_events: list[SystemEvent] | None = None
    # Yarn-state directive appended to the system prompt telling the LLM to
    # append [[advance]] when it senses a natural pause. None for form states.
    advance_instruction: str | None = None


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
    # TODO: decide history strategy, windowed, summarised, or full.
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
# tightly in [0.65, 1.0], a linear or percent-above-threshold mapping
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


def _transcript_lines(
    transcript_segments: list,
    system_events: list[SystemEvent] | None = None,
) -> list[str]:
    """Render the merged speech + system-event stream as transcript lines."""
    return compose_stream(
        transcript_segments,
        system_events or [],
        confidence_remap=_confidence_for_llm,
    )


def _build_transcript_message(
    transcript_segments: list[TranscriptSegment],
    system_events: list[SystemEvent] | None = None,
) -> Message | None:
    """Pack the merged stream into a system-role message for the LLM."""
    transcript_lines = _transcript_lines(transcript_segments, system_events)
    if not transcript_lines:
        return None

    lines = [
        "Recent transcript (user speech + hidden system events in {{…}}):",
        *[f"  {line}" for line in transcript_lines],
    ]
    return {"role": "system", "content": "\n".join(lines)}


class LLMReasoningAgent:
    """Owns the high-level reasoning pipeline for one conversational turn."""

    def __init__(self, llm: LLMProvider, history_window: int = 10) -> None:
        # TODO: history_window default is arbitrary, revisit once history
        # strategy is decided.
        self._llm = llm
        self._history_window = history_window

    def extract_json(
        self,
        instruction: str,
        segment_slice: list[str],
    ) -> dict:
        """One-shot JSON-mode LLM call used by the conductor at state-end.

        `instruction` is the state's facts_extraction_prompt, it describes
        what fields to return. `segment_slice` is the rendered list of
        transcript + event lines from the state we're closing.

        Always returns a dict. On parse failure, returns {_raw: text,
        _error: msg} so the caller can record what came back without
        blocking the transition.
        """
        slice_text = "\n".join(segment_slice) if segment_slice else "(empty)"
        system = (
            "You are a fact-extraction helper. Read the conversation slice "
            "between <slice> tags and return a single JSON object as "
            "instructed. Return ONLY the JSON object, no prose, no "
            "markdown fences, no commentary. If a field can't be determined "
            "from the slice, use null."
        )
        user = (
            f"{instruction}\n\n"
            f"<slice>\n{slice_text}\n</slice>"
        )
        try:
            raw = self._llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
        except Exception as exc:  # noqa: BLE001, never block the transition
            logger.warning("extraction LLM call failed: %s", exc)
            return {"_raw": "", "_error": str(exc)}

        stripped = _JSON_FENCE_RE.sub("", raw).strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning("extraction output not valid JSON; recording raw")
            return {"_raw": raw.strip(), "_error": str(exc)}
        if not isinstance(parsed, dict):
            logger.warning("extraction output JSON was not an object; recording raw")
            return {"_raw": raw.strip(), "_error": "top-level value not an object"}
        return parsed

    def collect_inputs(
        self,
        message: str,
        emotional_context: str,
        history: list[Message],
        transcript_segments: list | None = None,
        intention: str | None = None,
        system_events: list[SystemEvent] | None = None,
        advance_instruction: str | None = None,
    ) -> ReasoningInputs:
        """Normalise raw inputs into one explicit turn object."""
        return ReasoningInputs(
            current_message=message,
            emotional_context=emotional_context,
            history=history,
            transcript_segments=transcript_segments or [],
            intention=intention,
            system_events=system_events,
            advance_instruction=advance_instruction,
        )

    def derive_prompt_context(self, inputs: ReasoningInputs) -> PromptContext:
        """Turn raw inputs into structured prompt context.

        This is the best place to change future reasoning behavior without
        touching the chat route or provider adapters.
        """
        return PromptContext(
            system_prompt=_system_prompt(inputs.intention, inputs.advance_instruction),
            history_messages=_history_window(inputs.history, self._history_window),
            emotional_message=_emotional_message(inputs.emotional_context),
            transcript_message=_build_transcript_message(
                inputs.transcript_segments, inputs.system_events
            ),
            feedback_message=None,
            transcript_lines=_transcript_lines(inputs.transcript_segments, inputs.system_events),
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
        intention: str | None = None,
        system_events: list[SystemEvent] | None = None,
        advance_instruction: str | None = None,
    ) -> dict:
        """Return a structured snapshot of the current reasoning pipeline."""
        inputs = self.collect_inputs(
            message, emotional_context, history, transcript_segments, intention,
            system_events, advance_instruction,
        )
        context = self.derive_prompt_context(inputs)
        prompt_messages = self.assemble_messages(inputs, context)

        return {
            "provider": self._llm.provider_name,
            "model": self._llm.model_name,
            "current_message": inputs.current_message,
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
        intention: str | None = None,
        system_events: list[SystemEvent] | None = None,
        advance_instruction: str | None = None,
    ) -> ReasoningResult:
        """Run the reasoning pipeline and return the parsed result.

        Strips any inline [[…]] tool-emission markers from the raw LLM
        output and collects them into `emissions`. The cleaned text is
        returned verbatim as the reply, the LLM is no longer asked to
        emit JSON, so there's no envelope to parse.
        """
        inputs = self.collect_inputs(
            message, emotional_context, history, transcript_segments, intention,
            system_events, advance_instruction,
        )
        context = self.derive_prompt_context(inputs)
        prompt_messages = self.assemble_messages(inputs, context)
        raw_response = self._llm.chat(prompt_messages)
        cleaned, emissions = extract_emissions(raw_response)
        return ReasoningResult(reply=cleaned.strip(), emissions=emissions)

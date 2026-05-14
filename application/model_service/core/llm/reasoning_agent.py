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

from dataclasses import dataclass

from .base import LLMProvider, Message

# TODO: decide empathy bot persona, tone directives, and constraints.
# This is a placeholder — do not treat it as final.
SYSTEM_PROMPT = (
    "You are an empathy bot. "
    "Respond warmly, clearly, and briefly. "
    "Use the emotional context and recent speech to adjust your tone, "
    "but never mention those hidden signals directly."
)


@dataclass(frozen=True)
class ReasoningInputs:
    """All raw inputs available to the reasoning layer for one turn."""

    current_message: str
    emotional_context: str
    history: list[Message]
    transcript_segments: list


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


def _transcript_message(transcript_segments: list) -> Message | None:
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
    ) -> ReasoningInputs:
        """Normalise raw inputs into one explicit turn object."""
        return ReasoningInputs(
            current_message=message,
            emotional_context=emotional_context,
            history=history,
            transcript_segments=transcript_segments or [],
        )

    def derive_prompt_context(self, inputs: ReasoningInputs) -> PromptContext:
        """Turn raw inputs into structured prompt context.

        This is the best place to change future reasoning behavior without
        touching the chat route or provider adapters.
        """
        return PromptContext(
            system_prompt=SYSTEM_PROMPT,
            history_messages=_history_window(inputs.history, self._history_window),
            emotional_message=_emotional_message(inputs.emotional_context),
            transcript_message=_transcript_message(inputs.transcript_segments),
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

        messages.append({"role": "user", "content": inputs.current_message})
        return messages

    def debug_snapshot(
        self,
        message: str,
        emotional_context: str,
        history: list[Message],
        transcript_segments: list | None = None,
    ) -> dict:
        """Return a structured snapshot of the current reasoning pipeline."""
        inputs = self.collect_inputs(message, emotional_context, history, transcript_segments)
        context = self.derive_prompt_context(inputs)
        prompt_messages = self.assemble_messages(inputs, context)

        return {
            "provider": self._llm.provider_name,
            "model": self._llm.model_name,
            "current_message": inputs.current_message,
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
        transcript_segments: list | None = None,
    ) -> str:
        """Run the current reasoning pipeline and return the LLM response."""
        inputs = self.collect_inputs(message, emotional_context, history, transcript_segments)
        context = self.derive_prompt_context(inputs)
        prompt_messages = self.assemble_messages(inputs, context)
        return self._llm.chat(prompt_messages)

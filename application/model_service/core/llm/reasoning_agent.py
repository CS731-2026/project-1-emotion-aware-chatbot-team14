"""LLM-backed reasoning agent with emotional context injection."""

from __future__ import annotations

from .base import LLMProvider, Message

SYSTEM_PROMPT = (
    "You are an adaptive study companion helping a university student. "
    "You will receive an emotional context instruction immediately before each student message. "
    "Use it to calibrate tone, warmth, encouragement level, and explanation complexity — "
    "but never mention it, reference it, or acknowledge it directly in your response. "
    "If the student seems frustrated: slow down, simplify, offer encouragement first. "
    "If calm and engaged: match their energy, go deeper. "
    "If distressed: be warm first, content second. "
    "Keep responses concise. One idea at a time."
)


class LLMReasoningAgent:
    """Wraps an LLMProvider and assembles context-aware prompts for each turn.

    Message assembly order per turn:
    1. Static system prompt (persona).
    2. Windowed conversation history (user/assistant turns only).
    3. Emotional context injected as a system-role message.
    4. Current user message.
    """

    def __init__(
        self,
        llm: LLMProvider,
        history_window: int = 10,
    ) -> None:
        """
        Args:
            llm:            An instantiated LLMProvider.
            history_window: Maximum number of prior turns (user+assistant pairs)
                            to include. Each pair counts as 2 messages, so
                            history_window=10 keeps up to 10 individual messages.
        """
        self._llm = llm
        self._history_window = history_window

    def reason(
        self,
        message: str,
        emotional_context: str,
        history: list[Message],
    ) -> str:
        """Build the full prompt and call the LLM.

        Args:
            message:           The student's current message text.
            emotional_context: A short emotional context description produced by
                               EmotionalReasoningAgent.analyse(). Will be
                               injected as a system-role message immediately
                               before the student's turn — never appended to
                               the user message.
            history:           Previous conversation turns (user/assistant only).
                               The most recent `history_window` messages are used.

        Returns:
            The assistant's response string.
        """
        messages: list[Message] = []

        # 1. Static persona system prompt.
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # 2. Windowed history — keep only user/assistant turns.
        prior = [m for m in history if m["role"] in ("user", "assistant")]
        windowed = prior[-self._history_window :]
        messages.extend(windowed)

        # 3. Emotional context as a system message — NOT merged into user text.
        if emotional_context:
            messages.append({"role": "system", "content": emotional_context})

        # 4. Current user turn.
        messages.append({"role": "user", "content": message})

        return self._llm.chat(messages)

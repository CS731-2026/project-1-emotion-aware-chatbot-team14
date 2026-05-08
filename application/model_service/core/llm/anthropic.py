"""Anthropic LLM provider stub."""

from __future__ import annotations

from .base import LLMProvider, Message


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider — not yet implemented."""

    def __init__(self, model: str = "claude-3-5-haiku-latest", **kwargs) -> None:
        self._model = model

    def chat(self, messages: list[Message]) -> str:
        raise NotImplementedError("AnthropicProvider not yet implemented")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

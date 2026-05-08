"""Ollama LLM provider stub."""

from __future__ import annotations

from .base import LLMProvider, Message


class OllamaProvider(LLMProvider):
    """Ollama LLM provider — not yet implemented."""

    def __init__(self, model: str = "llama3", **kwargs) -> None:
        self._model = model

    def chat(self, messages: list[Message]) -> str:
        raise NotImplementedError("OllamaProvider not yet implemented")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

"""OpenAI LLM provider."""

from __future__ import annotations

from typing import Any

from .base import LLMProvider, Message


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API.

    Requires the `openai` package and a valid OPENAI_API_KEY.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model:    OpenAI model identifier (default: gpt-4o-mini).
            api_key:  API key. If None, falls back to the OPENAI_API_KEY env var.
            **kwargs: Extra keyword arguments forwarded to openai.OpenAI().
        """
        import openai

        self._model = model
        self._client = openai.OpenAI(api_key=api_key, **kwargs)

    def chat(self, messages: list[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

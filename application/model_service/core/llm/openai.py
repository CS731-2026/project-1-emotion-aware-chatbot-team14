"""OpenAI LLM provider."""

from __future__ import annotations

from .base import LLMProvider, Message


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API.

    Requires the `openai` package and a valid OPENAI_API_KEY.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            model:   OpenAI model identifier (default: gpt-4o-mini).
            api_key: API key. If None, falls back to the OPENAI_API_KEY
                     environment variable (via the openai client default).
            **kwargs: Extra keyword arguments forwarded to openai.OpenAI().
        """
        import openai

        self._model = model
        self._client = openai.OpenAI(api_key=api_key, **kwargs)

    # -------------------------------------------------------------------------
    # LLMProvider interface
    # -------------------------------------------------------------------------

    def chat(self, messages: list[Message]) -> str:
        """Send messages to the OpenAI Chat Completions endpoint.

        Args:
            messages: Ordered list of role/content message dicts.

        Returns:
            The assistant's response text.
        """
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

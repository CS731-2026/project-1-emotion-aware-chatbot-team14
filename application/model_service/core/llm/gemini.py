"""Gemini LLM provider."""

from __future__ import annotations

from .base import LLMProvider, Message


def _flatten_messages(messages: list[Message]) -> str:
    """Serialize chat messages into a simple prompt for Gemini.

    The current reasoning layer is intentionally lightweight. Rather than
    building a richer role-aware content tree here, we keep the prompt in a
    plain transcript-like format so the provider swap stays low-risk.
    """
    lines: list[str] = []
    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


class GeminiProvider(LLMProvider):
    """LLM provider backed by the Google GenAI SDK."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        from google import genai

        self._model = model
        self._client = genai.Client(api_key=api_key, **kwargs)

    def chat(self, messages: list[Message]) -> str:
        prompt = _flatten_messages(messages)
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text or ""

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

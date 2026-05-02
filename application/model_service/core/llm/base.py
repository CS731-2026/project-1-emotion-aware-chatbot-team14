"""Base interfaces for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from typing_extensions import TypedDict


class Message(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: str


class LLMProvider(ABC):
    """Abstract base class for all LLM provider backends."""

    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        """Send a message list and return the assistant's reply.

        Args:
            messages: Ordered list of role/content message dicts.

        Returns:
            The assistant's response text.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'openai', 'anthropic')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier used for this instance (e.g. 'gpt-4o-mini')."""
        ...

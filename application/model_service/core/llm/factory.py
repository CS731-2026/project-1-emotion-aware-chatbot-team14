"""LLM provider factory."""

from __future__ import annotations

from .base import LLMProvider


def create_llm(provider: str, model: str, **kwargs) -> LLMProvider:
    """Create and return an LLMProvider for the requested provider.

    Args:
        provider: Provider name. One of "openai", "anthropic", "ollama".
        model:    Model identifier passed to the chosen provider.
        **kwargs: Additional keyword arguments forwarded to the provider class.

    Returns:
        A ready-to-use LLMProvider instance.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    if provider == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(model=model, **kwargs)

    elif provider == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(model=model, **kwargs)

    elif provider == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(model=model, **kwargs)

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Valid options: 'openai', 'anthropic', 'ollama'."
        )

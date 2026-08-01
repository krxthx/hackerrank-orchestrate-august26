"""Picks the right provider adapter for a given litellm model string.

Order matters: adapters are tried in sequence and the first match wins.
OpenAICompatibleProvider matches unconditionally, so it must stay last -- it's the
catch-all for anything the more specific vendor adapters don't recognize.
"""

from orchestrate.llm.base import LLMProvider
from orchestrate.llm.providers import AnthropicProvider, GeminiProvider, OllamaProvider, OpenAICompatibleProvider

_PROVIDERS: list[type[LLMProvider]] = [
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAICompatibleProvider,  # catch-all -- must be last
]


def get_provider(model: str) -> LLMProvider:
    for provider_cls in _PROVIDERS:
        if provider_cls.matches(model):
            return provider_cls()
    # Unreachable while OpenAICompatibleProvider.matches() is unconditional, but keeps
    # this function honest if that ever changes.
    raise LookupError(f"No provider adapter matches model string: {model!r}")

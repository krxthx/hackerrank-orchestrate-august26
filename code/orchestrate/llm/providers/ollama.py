"""Local models served via Ollama (e.g. ollama/llama3.1).

No API key needed. Defaults the base URL to OLLAMA_API_BASE (or localhost:11434) when the
caller doesn't pass one explicitly -- litellm would resolve the same env var on its own,
but making it explicit here keeps the resolution visible in this codebase rather than
implicit in a third-party default. Was considered for the evaluation judge pass before
switching to OpenRouter (no rate limits, but requires local compute); kept as a supported
option since a grader running fully offline may prefer it.
"""

import os

from orchestrate.llm.base import LLMProvider

_DEFAULT_API_BASE = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    @classmethod
    def matches(cls, model: str) -> bool:
        return model.startswith("ollama/")

    def resolve_api_base(self, api_base: str | None) -> str | None:
        return api_base or os.environ.get("OLLAMA_API_BASE") or _DEFAULT_API_BASE

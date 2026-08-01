"""Common interface every LLM vendor adapter implements.

Callers never instantiate a provider directly -- they call llm.complete(), which uses
llm.registry.get_provider() to pick the right adapter from the model string's prefix. A
provider's only real job is resolving the two things that differ by vendor/endpoint --
where the request goes (`resolve_api_base`) and how it's authenticated
(`resolve_api_key`) -- everything else (pacing, retry, the actual litellm call) is shared
here so subclasses stay tiny.
"""

import time
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential

from orchestrate.config import LLM_CALL_PACING_SECONDS


class LLMProvider(ABC):
    @classmethod
    @abstractmethod
    def matches(cls, model: str) -> bool:
        """Whether this adapter handles the given litellm model string. The registry
        tries adapters in order and uses the first match, so a catch-all adapter (see
        OpenAICompatibleProvider) must return True unconditionally and be registered last.
        """

    def resolve_api_base(self, api_base: str | None) -> str | None:
        """Hook for subclasses with a sensible non-default base URL (e.g. a self-hosted
        endpoint). Default: pass through unchanged -- None lets litellm apply its own
        provider default.
        """
        return api_base

    def resolve_api_key(self, api_key: str | None) -> str | None:
        """Hook for subclasses with a non-standard credential source. Default: pass
        through unchanged -- None lets litellm resolve the provider's standard *_API_KEY
        environment variable on its own.
        """
        return api_key

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        if LLM_CALL_PACING_SECONDS > 0:
            time.sleep(LLM_CALL_PACING_SECONDS)
        return self._call(
            model=model,
            messages=messages,
            tools=tools,
            api_base=self.resolve_api_base(api_base),
            api_key=self.resolve_api_key(api_key),
            **kwargs,
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=90))
    def _call(self, **kwargs):
        import litellm

        return litellm.completion(**kwargs)

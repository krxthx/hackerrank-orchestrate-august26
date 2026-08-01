"""Anything that speaks the OpenAI chat-completions API shape: bare OpenAI models
(gpt-4.1), a self-hosted litellm/OpenAI-compatible proxy, or OpenRouter.

This is the adapter actually in use for this challenge -- both roles this codebase has
(the router's own model, via ORCHESTRATE_MODEL/_API_BASE/_API_KEY, and the evaluation
judge's model, via ORCHESTRATE_JUDGE_MODEL/_API_BASE/_API_KEY) are just differently
configured instances of this one adapter, not different vendors. Neither OpenRouter nor a
custom proxy needs special-casing here: both accept an OpenAI-shaped request, and each
caller (agent.py for the router, evaluate.py for the judge) supplies its own api_base/
api_key from config.py -- see llm/__init__.py's complete() for why those must never leak
between the two roles.

Also the registry's catch-all: any model string the more specific adapters don't
recognize lands here rather than failing with "no provider found", since litellm itself
supports more providers than this codebase has named adapters for.
"""

from orchestrate.llm.base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    @classmethod
    def matches(cls, model: str) -> bool:
        return True

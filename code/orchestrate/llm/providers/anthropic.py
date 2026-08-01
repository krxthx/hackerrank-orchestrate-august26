"""Native Anthropic models (e.g. claude-sonnet-4-5-20250929).

No custom base URL -- litellm talks to Anthropic's API directly and resolves
ANTHROPIC_API_KEY from the environment when no api_key is passed, so this adapter needs
no overrides beyond recognizing the model string. Not the model configured for this
challenge submission, but supported so an evaluator/grader can point ORCHESTRATE_MODEL at
Anthropic directly (set ANTHROPIC_API_KEY in .env) without any code changes.
"""

from orchestrate.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    @classmethod
    def matches(cls, model: str) -> bool:
        return model.startswith("claude") or model.startswith("anthropic/")

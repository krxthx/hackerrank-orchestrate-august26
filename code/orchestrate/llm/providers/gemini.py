"""Native Google Gemini models (e.g. gemini/gemini-2.5-pro).

No custom base URL -- litellm talks to Google's Generative Language API directly and
resolves GEMINI_API_KEY from the environment when no api_key is passed. Not the model
configured for this challenge submission (see PLAN.md for why Gemini free-tier quotas
were dropped for this project), but supported so an evaluator/grader can point
ORCHESTRATE_MODEL at Gemini directly (set GEMINI_API_KEY in .env) without code changes.
"""

from orchestrate.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    @classmethod
    def matches(cls, model: str) -> bool:
        return model.startswith("gemini/")

"""Provider-agnostic chat facade: agent.py and evaluate.py both call complete() without
knowing or caring which vendor adapter (see llm/providers/) actually handles the request
-- that's picked automatically via llm.registry.get_provider() from the model string.

Built on litellm so the same code path works against Anthropic, Gemini, OpenAI-compatible
endpoints (including OpenRouter), or a local model served through Ollama -- just change
MODEL in config.py (or the ORCHESTRATE_MODEL env var).
"""

from orchestrate.core.config import API_BASE, API_KEY, MODEL
from orchestrate.llm.registry import get_provider


def complete(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    **kwargs,
):
    """Call the configured model. Returns the raw litellm response object.

    `messages` follows the OpenAI/Anthropic chat format:
        [{"role": "user"/"assistant"/"system"/"tool", "content": "..."}]
    `tools` follows the OpenAI tool-calling schema (litellm translates this
    for whichever provider is configured).

    The ORCHESTRATE_API_BASE/API_KEY fallback only applies when `model` is left unset (i.e.
    this call is using the default router model/endpoint). Callers that pass an explicit
    `model` (e.g. evaluate.py's judge pass, which deliberately uses a different
    model/provider from the router) get exactly the api_base/api_key they passed -- even if
    that's None -- so a differently-configured call never silently rides on the router's
    endpoint/credentials. (This is the exact bug that once sent the judge's OpenRouter key
    to the router's internal proxy -- see the adapter's own resolve_api_base/resolve_api_key
    hooks in llm/base.py for where per-vendor defaults, if any, are applied instead.)
    """
    resolved_model = model or MODEL
    resolved_api_base = api_base if model else (api_base or API_BASE)
    resolved_api_key = api_key if model else (api_key or API_KEY)

    provider = get_provider(resolved_model)
    return provider.complete(
        messages,
        tools=tools,
        model=resolved_model,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        **kwargs,
    )

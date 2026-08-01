"""Thin, provider-agnostic chat wrapper.

Built on litellm so the same code path works against Anthropic, Gemini,
OpenAI, or a local model served through Ollama -- just change MODEL in
config.py (or the ORCHESTRATE_MODEL env var) once you know what the
challenge needs.
"""

import time

from tenacity import retry, stop_after_attempt, wait_exponential

from orchestrate.config import API_BASE, API_KEY, LLM_CALL_PACING_SECONDS, MODEL


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=90))
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

    Pass `model="openai/<name>"` with `ORCHESTRATE_API_BASE`/`ORCHESTRATE_API_KEY` set (or
    `api_base=`/`api_key=` here) to point at any OpenAI-compatible endpoint instead of a
    named provider.

    The ORCHESTRATE_API_BASE/API_KEY fallback only applies when `model` is left unset (i.e.
    this call is using the default router model/endpoint). Callers that pass an explicit
    `model` (e.g. evaluate.py's judge pass, which deliberately uses a different
    model/provider from the router) get exactly the api_base/api_key they passed -- even if
    that's None -- so a differently-configured call never silently rides on the router's
    endpoint/credentials.
    """
    import litellm

    if LLM_CALL_PACING_SECONDS > 0:
        time.sleep(LLM_CALL_PACING_SECONDS)

    resolved_api_base = api_base if model else (api_base or API_BASE)
    resolved_api_key = api_key if model else (api_key or API_KEY)

    return litellm.completion(
        model=model or MODEL,
        messages=messages,
        tools=tools,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        **kwargs,
    )

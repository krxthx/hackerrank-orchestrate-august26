"""Thin, provider-agnostic chat wrapper.

Built on litellm so the same code path works against Anthropic, Gemini,
OpenAI, or a local model served through Ollama -- just change MODEL in
config.py (or the ORCHESTRATE_MODEL env var) once you know what the
challenge needs.
"""

from tenacity import retry, stop_after_attempt, wait_exponential

from orchestrate.config import MODEL


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def complete(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    model: str | None = None,
    **kwargs,
):
    """Call the configured model. Returns the raw litellm response object.

    `messages` follows the OpenAI/Anthropic chat format:
        [{"role": "user"/"assistant"/"system"/"tool", "content": "..."}]
    `tools` follows the OpenAI tool-calling schema (litellm translates this
    for whichever provider is configured).
    """
    import litellm

    return litellm.completion(
        model=model or MODEL,
        messages=messages,
        tools=tools,
        **kwargs,
    )

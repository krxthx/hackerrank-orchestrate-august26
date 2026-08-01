"""Model-agnostic tool-calling loop.

Give it a system prompt and a user task; it calls the model, executes any
tool calls the model requests, feeds results back, and repeats until the
model stops calling tools or MAX_AGENT_STEPS is hit. Every step is logged
to the transcript.
"""

import json

from orchestrate import tools
from orchestrate.config import MAX_AGENT_STEPS
from orchestrate.constants import ASSISTANT_ROLE, SYSTEM_ROLE, TOOL_ROLE, USER_ROLE
from orchestrate.errors import OrchestrateError
from orchestrate.llm import complete
from orchestrate.transcript import TranscriptLogger
from orchestrate.types import AgentResult


def run_agent(
    system_prompt: str,
    user_input: str | list[dict],
    logger: TranscriptLogger | None = None,
    max_steps: int | None = None,
) -> AgentResult:
    """`user_input` may be a plain string or a list of OpenAI-style content blocks
    (e.g. `[{"type": "text", ...}, {"type": "image_url", ...}]`) for multimodal input.
    """
    logger = logger or TranscriptLogger()
    max_steps = max_steps or MAX_AGENT_STEPS
    messages = [
        {"role": SYSTEM_ROLE, "content": system_prompt},
        {"role": USER_ROLE, "content": user_input},
    ]

    for step in range(1, max_steps + 1):
        response = complete(messages, tools=tools.all_schemas())
        choice = response.choices[0].message
        tool_calls = choice.tool_calls or []

        logger.log_step(
            messages=messages,
            response_text=choice.content or "",
            tool_calls=[tc.model_dump() for tc in tool_calls] if tool_calls else None,
        )

        if not tool_calls:
            return AgentResult(final_text=choice.content or "", steps_used=step)

        messages.append(choice.model_dump())
        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            try:
                result = tools.call(tc.function.name, args)
            except OrchestrateError:
                # Fatal, not a per-call fluke (e.g. a dataset file is missing/malformed) --
                # every future call would fail the same way, so stop the run instead of
                # letting the model burn through MAX_AGENT_STEPS retrying uselessly.
                raise
            except Exception as exc:  # a genuine tool bug/bad args -- surfaced to the model, not fatal
                result = f"ERROR: {exc}"
            messages.append(
                {
                    "role": TOOL_ROLE,
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )

    return AgentResult(
        final_text="ERROR: exceeded MAX_AGENT_STEPS without a final answer",
        steps_used=max_steps,
        hit_step_limit=True,
    )

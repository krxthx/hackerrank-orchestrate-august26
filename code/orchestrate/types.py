"""Shared interfaces/types used across the agent, pipeline, and tools modules.

Kept as plain pydantic models (not tied to any framework) so they're easy to
reshape once the real challenge schema is known.
"""

from typing import Any, Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant", "tool"]


class ToolCallRecord(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ToolSpec(BaseModel):
    """Metadata for one registered tool; `json_schema` is what gets sent to the LLM."""

    name: str
    description: str
    json_schema: dict[str, Any]


class AgentResult(BaseModel):
    final_text: str
    steps_used: int
    hit_step_limit: bool = False


class PipelineRecord(BaseModel):
    """One row in, one row out. Extend with real fields once the input schema is known."""

    row_id: str | int | None = None
    input_text: str
    output_text: str

"""Shared interfaces/types used across the agent, pipeline, and tools modules.

Kept as plain pydantic models (not tied to any framework) so they're easy to
reshape once the real challenge schema is known.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]

Action = Literal["notify", "digest", "mute"]

MessageType = Literal[
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
]


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


class RoutingDecision(BaseModel):
    """One output row for the Message Notification Router -- matches the required
    output.csv columns exactly: message_id,action,message_type,reason,confidence,
    evidence_message_ids.
    """

    message_id: str
    action: Action
    message_type: MessageType
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_message_ids: str = "none"


class JudgeVerdict(BaseModel):
    """One rubric-judge scoring of a RoutingDecision (see evaluate.py, prompts/judge.py).
    Gives the judge's output the same structural validation RoutingDecision gets, rather
    than trusting raw dict access -- a judge hallucinating an out-of-range score or the
    wrong type fails validation and falls back cleanly instead of silently corrupting the
    eval report.
    """

    action_plausible: bool
    message_type_plausible: bool
    reason_score: int = Field(ge=1, le=5)
    evidence_score: int = Field(ge=1, le=5)
    confidence_score: int = Field(ge=1, le=5)
    safety_concern: bool = False
    critique: str = ""

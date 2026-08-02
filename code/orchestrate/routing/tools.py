"""Tool registry. Decorate a function with @tool(...) and it becomes callable
by the agent loop, with the JSON schema auto-generated for the LLM.

These 5 tools are the router agent's only way to see dataset context beyond the raw
incoming message -- it must call them (rather than guess) before deciding how to route.
Each returns a small JSON string, not a raw DataFrame dump.
"""

import inspect
import json
from dataclasses import dataclass
from typing import Callable, get_type_hints

import pandas as pd

from orchestrate.core.types import ToolSpec
from orchestrate.data.dataset import get_dataset

_PY_TO_JSON_TYPE = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def schema(self) -> dict:
        return self.spec.json_schema


REGISTRY: dict[str, Tool] = {}


def tool(description: str, params: dict[str, str] | None = None):
    """`params` is an optional {param_name: description} map for per-argument guidance (e.g.
    "pass empty string if not applicable") that the model sees on each field individually,
    not just buried in the function-level description.
    """
    params = params or {}

    def decorator(fn: Callable):
        hints = get_type_hints(fn)
        sig_params = inspect.signature(fn).parameters
        properties = {}
        required = []
        for name, param in sig_params.items():
            py_type = hints.get(name, str)
            prop: dict = {"type": _PY_TO_JSON_TYPE.get(py_type, "string")}
            if name in params:
                prop["description"] = params[name]
            if param.default is inspect.Parameter.empty:
                required.append(name)
            else:
                prop["default"] = param.default
            properties[name] = prop

        schema = {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
        spec = ToolSpec(name=fn.__name__, description=description, json_schema=schema)
        REGISTRY[fn.__name__] = Tool(spec=spec, fn=fn)
        return fn

    return decorator


def all_schemas() -> list[dict]:
    return [t.schema for t in REGISTRY.values()]


def call(name: str, arguments: dict):
    if name not in REGISTRY:
        raise KeyError(f"Unknown tool: {name}")
    return REGISTRY[name].fn(**arguments)


def _row_to_dict(row: pd.Series) -> dict:
    return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}


@tool(
    "Look up a user's notification behavior: quiet hours (do_not_disturb_window) and their "
    "30-day counts of messages opened, replied to, notifications dismissed, and reports filed. "
    "Call this for the message's receiving user_id before deciding action/confidence.",
    params={"user_id": "The receiving user's user_id (from the incoming message)."},
)
def get_user_profile(user_id: str) -> str:
    ds = get_dataset()
    if user_id not in ds.users.index:
        return json.dumps({"error": f"unknown user_id: {user_id}"})
    return json.dumps(_row_to_dict(ds.users.loc[user_id]))


@tool(
    "Look up group info (type, size, admin_count, recent activity) and this specific user's "
    "membership in it (role, mute state, read/reply/dismiss behavior). Call this when "
    "conversation_type is 'group'.",
    params={
        "group_id": "The incoming message's group_id.",
        "user_id": "The receiving user's user_id.",
    },
)
def get_group_context(group_id: str, user_id: str) -> str:
    ds = get_dataset()
    if group_id not in ds.groups.index:
        return json.dumps({"error": f"unknown group_id: {group_id}"})
    group = _row_to_dict(ds.groups.loc[group_id])
    member_rows = ds.group_members[(ds.group_members.group_id == group_id) & (ds.group_members.user_id == user_id)]
    membership = _row_to_dict(member_rows.iloc[0]) if len(member_rows) else None
    return json.dumps({"group": group, "user_membership": membership})


@tool(
    "Look up a business account (verification, domain match, account age, report count) and "
    "whether this specific user has a real relationship with it (past orders/bookings, "
    "opt-in/opt-out of promotions, reply history). Call this when conversation_type is 'business'.",
    params={
        "business_id": "The incoming message's business_id.",
        "user_id": "The receiving user's user_id.",
    },
)
def get_business_context(business_id: str, user_id: str) -> str:
    ds = get_dataset()
    if business_id not in ds.business_accounts.index:
        return json.dumps({"error": f"unknown business_id: {business_id}"})
    business = _row_to_dict(ds.business_accounts.loc[business_id])
    history_rows = ds.user_business_history[
        (ds.user_business_history.business_id == business_id) & (ds.user_business_history.user_id == user_id)
    ]
    relationship = _row_to_dict(history_rows.iloc[0]) if len(history_rows) else None
    return json.dumps({"business": business, "user_relationship": relationship})


@tool(
    "Fetch the receiving user's most recent PAST messages from the same sender, group, or "
    "business (pass whichever of sender_user_id/group_id/business_id apply, leave others as "
    "empty string), each annotated with how the user reacted (opened/replied/dismissed/muted/"
    "reported). This is the ONLY valid source for evidence_message_ids in your final answer -- "
    "never cite a message_id that was not returned by this tool. Returns 'none' if nothing matches.",
    params={
        "user_id": "The receiving user's user_id.",
        "sender_user_id": "Filter to this sender only, or '' (empty string) to not filter by sender.",
        "group_id": "Filter to this group only, or '' (empty string) to not filter by group.",
        "business_id": "Filter to this business only, or '' (empty string) to not filter by business.",
        "limit": "Max number of past messages to return, most recent first.",
    },
)
def get_message_history(
    user_id: str,
    sender_user_id: str = "",
    group_id: str = "",
    business_id: str = "",
    limit: int = 8,
) -> str:
    ds = get_dataset()
    history = ds.message_history[ds.message_history.user_id == user_id]

    match = pd.Series(False, index=history.index)
    if sender_user_id:
        match |= history.sender_user_id == sender_user_id
    if group_id:
        match |= history.group_id == group_id
    if business_id:
        match |= history.business_id == business_id
    if sender_user_id or group_id or business_id:
        history = history[match]

    history = history.sort_values("created_at", ascending=False).head(limit)
    if history.empty:
        return json.dumps({"history": []})

    records = []
    for _, row in history.iterrows():
        record = _row_to_dict(row)
        key = (user_id, row["message_id"])
        if key in ds.message_events.index:
            record["events"] = _row_to_dict(ds.message_events.loc[key])
        else:
            record["events"] = None
        records.append(record)
    return json.dumps({"history": records})


@tool(
    "Fetch the receiving user's recent daily notification load (notifications sent vs "
    "dismissed per day). Useful to decide whether a borderline case should be batched into "
    "digest instead of interrupting, if the user is already getting a lot of notifications.",
    params={
        "user_id": "The receiving user's user_id.",
        "days": "How many recent days of load to return.",
    },
)
def get_daily_load(user_id: str, days: int = 7) -> str:
    ds = get_dataset()
    rows = ds.daily_notification_summary[ds.daily_notification_summary.user_id == user_id]
    rows = rows.sort_values("date", ascending=False).head(days)
    return json.dumps({"daily_load": [_row_to_dict(r) for _, r in rows.iterrows()]})

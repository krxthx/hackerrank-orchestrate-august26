"""Smoke tests for the dataset lookup tools: registry shape + real lookups against the
actual dataset/ CSVs (no live LLM calls).
"""

import json

from orchestrate import tools


def test_all_five_tools_registered():
    expected = {
        "get_user_profile",
        "get_group_context",
        "get_business_context",
        "get_message_history",
        "get_daily_load",
    }
    assert expected <= tools.REGISTRY.keys()


def test_schema_required_fields():
    schemas = {s["function"]["name"]: s for s in tools.all_schemas()}
    assert schemas["get_user_profile"]["function"]["parameters"]["required"] == ["user_id"]
    history_required = schemas["get_message_history"]["function"]["parameters"]["required"]
    assert history_required == ["user_id"]


def test_schema_carries_per_parameter_descriptions_and_defaults():
    schemas = {s["function"]["name"]: s for s in tools.all_schemas()}
    props = schemas["get_message_history"]["function"]["parameters"]["properties"]
    assert "description" in props["sender_user_id"]
    assert props["limit"]["default"] == 8
    assert schemas["get_message_history"]["function"]["parameters"]["additionalProperties"] is False


def test_get_user_profile_known_user():
    result = json.loads(tools.call("get_user_profile", {"user_id": "u_001"}))
    assert result["user_id"] == "u_001"
    assert "do_not_disturb_window" in result


def test_get_user_profile_unknown_user():
    result = json.loads(tools.call("get_user_profile", {"user_id": "not_a_real_user"}))
    assert "error" in result


def test_get_message_history_returns_only_matching_rows():
    result = json.loads(
        tools.call(
            "get_message_history",
            {"user_id": "u_011", "sender_user_id": "u_043", "group_id": "", "business_id": "", "limit": 5},
        )
    )
    for record in result["history"]:
        assert record["user_id"] == "u_011"
        assert record["sender_user_id"] == "u_043"


def test_get_message_history_no_match_returns_empty():
    result = json.loads(
        tools.call(
            "get_message_history",
            {"user_id": "u_001", "sender_user_id": "no_such_sender", "group_id": "", "business_id": "", "limit": 5},
        )
    )
    assert result["history"] == []

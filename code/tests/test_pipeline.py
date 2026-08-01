"""Offline tests for pipeline.py's parsing/validation logic -- no live LLM calls."""

import json

import pandas as pd
import pytest

from orchestrate.pipeline import (
    _checkpoint_fingerprint,
    _checkpoint_path,
    _content_blocked_decision,
    build_user_content,
    parse_result,
    run_pipeline,
)
from orchestrate.types import AgentResult
from orchestrate.types import RoutingDecision


def test_parse_result_valid_json():
    raw = (
        '{"message_id": "msg_001", "action": "notify", "message_type": "urgent", '
        '"reason": "direct ask", "confidence": 0.9, "evidence_message_ids": "message_0001"}'
    )
    decision = parse_result(raw, "msg_001")
    assert decision.action == "notify"
    assert decision.confidence == 0.9


def test_parse_result_strips_code_fence():
    raw = '```json\n{"action": "digest", "message_type": "greeting", "reason": "ok", "confidence": 0.5}\n```'
    decision = parse_result(raw, "msg_002")
    assert decision.message_id == "msg_002"
    assert decision.action == "digest"


def test_parse_result_falls_back_on_garbage():
    decision = parse_result("not valid json at all", "msg_003")
    assert decision.message_id == "msg_003"
    assert decision.action == "digest"
    assert decision.message_type == "unknown"
    assert decision.confidence == 0.3


def test_parse_result_falls_back_on_invalid_enum():
    raw = '{"action": "not_a_real_action", "message_type": "urgent", "reason": "x", "confidence": 0.5}'
    decision = parse_result(raw, "msg_004")
    assert decision.action == "digest"
    assert decision.message_type == "unknown"


def test_routing_decision_rejects_out_of_range_confidence():
    with pytest.raises(Exception):
        RoutingDecision(
            message_id="x",
            action="notify",
            message_type="urgent",
            reason="x",
            confidence=1.5,
        )


def test_build_user_content_text_only_returns_string():
    row = pd.Series(
        {
            "message_id": "msg_100",
            "user_id": "u_001",
            "conversation_type": "personal",
            "group_id": float("nan"),
            "business_id": float("nan"),
            "sender_user_id": "u_002",
            "created_at": "2026-07-31 10:00",
            "message_text": "hello",
            "media_type": float("nan"),
            "media_id": float("nan"),
            "forwarded_count": 0,
        }
    )
    content = build_user_content(row)
    assert isinstance(content, str)
    assert "msg_100" in content


def test_content_blocked_decision_is_scam_mute():
    decision = _content_blocked_decision("msg_065", "Upstream provider rejected the request.")
    assert decision.action == "mute"
    assert decision.message_type == "scam"
    assert 0 <= decision.confidence <= 1
    assert "Upstream provider rejected the request." in decision.reason


def test_run_pipeline_step_limit_gets_an_honest_fallback_reason(tmp_path, monkeypatch):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(
        "message_id,user_id,conversation_type,group_id,business_id,sender_user_id,created_at,"
        "message_text,media_type,media_id,forwarded_count\n"
        "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hello,,,0\n"
    )
    checkpoint_path = tmp_path / "checkpoint.jsonl"

    monkeypatch.setattr("orchestrate.pipeline._checkpoint_fingerprint", lambda _path: "b" * 64)
    monkeypatch.setattr("orchestrate.pipeline._checkpoint_path", lambda _path, _fingerprint: str(checkpoint_path))
    monkeypatch.setattr("orchestrate.pipeline.build_user_content", lambda _row: "route")
    monkeypatch.setattr(
        "orchestrate.pipeline.run_agent",
        lambda *_args, **_kwargs: AgentResult(
            final_text="ERROR: exceeded MAX_AGENT_STEPS without a final answer",
            steps_used=6,
            hit_step_limit=True,
        ),
    )

    run_pipeline(str(input_path), output_path=str(output_path))

    written = pd.read_csv(output_path).iloc[0]
    assert written["action"] == "digest"
    assert "step-call limit" not in written["reason"]  # sanity: not a copy-paste typo
    assert "step" in written["reason"] and "final decision" in written["reason"]
    assert "could not parse" not in written["reason"]  # must not be mislabeled as a parse failure


def test_checkpoint_fingerprint_changes_with_input_content(tmp_path):
    input_path = tmp_path / "messages.csv"
    input_path.write_text("message_id\nmsg_1\n")
    first = _checkpoint_fingerprint(str(input_path))

    input_path.write_text("message_id\nmsg_2\n")
    second = _checkpoint_fingerprint(str(input_path))

    assert first != second
    assert _checkpoint_path(str(input_path), first).endswith(f"messages_checkpoint_{first[:12]}.jsonl")


def test_checkpoint_fingerprint_changes_with_model(tmp_path, monkeypatch):
    input_path = tmp_path / "messages.csv"
    input_path.write_text("message_id\nmsg_1\n")
    first = _checkpoint_fingerprint(str(input_path))

    monkeypatch.setattr("orchestrate.pipeline.MODEL", "different-provider/different-model")
    second = _checkpoint_fingerprint(str(input_path))

    assert first != second


def test_fresh_run_replaces_matching_checkpoint(tmp_path, monkeypatch):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(
        "message_id,user_id,conversation_type,group_id,business_id,sender_user_id,created_at,"
        "message_text,media_type,media_id,forwarded_count\n"
        "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hello,,,0\n"
    )
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    stale = RoutingDecision(
        message_id="msg_1",
        action="mute",
        message_type="spam",
        reason="stale",
        confidence=0.9,
    )
    checkpoint_path.write_text(json.dumps(stale.model_dump()) + "\n")

    monkeypatch.setattr("orchestrate.pipeline._checkpoint_fingerprint", lambda _path: "a" * 64)
    monkeypatch.setattr("orchestrate.pipeline._checkpoint_path", lambda _path, _fingerprint: str(checkpoint_path))
    monkeypatch.setattr("orchestrate.pipeline.build_user_content", lambda _row: "route")
    monkeypatch.setattr(
        "orchestrate.pipeline.run_agent",
        lambda *_args, **_kwargs: AgentResult(
            final_text=(
                '{"message_id":"msg_1","action":"notify","message_type":"personal",'
                '"reason":"fresh","confidence":0.8,"evidence_message_ids":"none"}'
            ),
            steps_used=1,
        ),
    )

    run_pipeline(str(input_path), output_path=str(output_path), fresh=True)

    written = pd.read_csv(output_path).iloc[0]
    assert written["action"] == "notify"
    checkpoint_lines = checkpoint_path.read_text().splitlines()
    assert len(checkpoint_lines) == 1
    assert json.loads(checkpoint_lines[0])["reason"] == "fresh"

"""Offline tests for pipeline.py's parsing/validation logic -- no live LLM calls."""

import pandas as pd
import pytest
from tenacity import RetryError

from orchestrate.pipeline import _content_blocked_decision, _looks_like_content_block, build_user_content, parse_result
from orchestrate.types import RoutingDecision


class _FakeFuture:
    def __init__(self, exc: BaseException):
        self._exc = exc

    def exception(self):
        return self._exc


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


def test_looks_like_content_block_detects_403_forbidden():
    exc = RetryError(_FakeFuture(Exception("APIError: OpenAIException - 403 Forbidden")))
    assert _looks_like_content_block(exc)


def test_looks_like_content_block_ignores_unrelated_errors():
    exc = RetryError(_FakeFuture(Exception("Connection timed out")))
    assert not _looks_like_content_block(exc)


def test_content_blocked_decision_is_scam_mute():
    decision = _content_blocked_decision("msg_065")
    assert decision.action == "mute"
    assert decision.message_type == "scam"
    assert 0 <= decision.confidence <= 1

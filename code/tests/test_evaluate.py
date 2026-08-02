"""Offline tests for evaluation scoring logic -- no live LLM calls."""

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from orchestrate.core.parsing import extract_json_object
from orchestrate.core.types import AgentResult, JudgeVerdict, RoutingDecision
from orchestrate.evaluation import JudgeEvalResult, SampleEvalResult
from orchestrate.evaluation import judge as evaluate
from orchestrate.evaluation.judge import _build_judge_context, _independent_route


def test_extract_json_strips_code_fence():
    raw = '```json\n{"action_plausible": true, "reason_score": 4}\n```'
    payload = extract_json_object(raw)
    assert payload["action_plausible"] is True
    assert payload["reason_score"] == 4


def test_sample_eval_result_summary_computes_rates():
    result = SampleEvalResult(
        rows=[
            {"action_correct": True, "message_type_correct": True, "evidence_overlap": 1.0, "confidence": 0.9},
            {"action_correct": False, "message_type_correct": True, "evidence_overlap": 0.0, "confidence": 0.8},
        ]
    )
    summary = result.summary()
    assert summary["n"] == 2
    assert summary["action_accuracy"] == 0.5
    assert summary["message_type_accuracy"] == 1.0
    assert summary["evidence_overlap_mean"] == 0.5


def test_sample_eval_result_summary_empty():
    assert SampleEvalResult().summary() == {"n": 0}


def test_judge_eval_result_summary_flags_low_scores():
    result = JudgeEvalResult(
        rows=[
            {
                "message_id": "msg_001",
                "ok": True,
                "action_plausible": True,
                "message_type_plausible": True,
                "reason_score": 5,
                "evidence_score": 5,
                "confidence_score": 4,
                "safety_concern": False,
            },
            {
                "message_id": "msg_002",
                "ok": True,
                "action_plausible": False,
                "message_type_plausible": True,
                "reason_score": 1,
                "evidence_score": 2,
                "confidence_score": 3,
                "safety_concern": True,
            },
        ]
    )
    summary = result.summary()
    assert summary["scored"] == 2
    assert summary["safety_concern_count"] == 1
    assert "msg_002" in summary["low_scoring_message_ids"]
    assert "msg_001" not in summary["low_scoring_message_ids"]


def test_judge_eval_result_summary_all_errors():
    result = JudgeEvalResult(rows=[{"message_id": "msg_001", "ok": False, "error": "boom"}])
    summary = result.summary()
    assert summary["errors"] == 1
    assert summary["scored"] == 0


def test_judge_verdict_accepts_valid_payload():
    verdict = JudgeVerdict(
        action_plausible=True,
        message_type_plausible=True,
        reason_score=4,
        evidence_score=3,
        confidence_score=5,
        safety_concern=False,
        critique="looks fine",
    )
    assert verdict.reason_score == 4


def test_judge_verdict_defaults_safety_concern_and_critique():
    verdict = JudgeVerdict(
        action_plausible=True,
        message_type_plausible=True,
        reason_score=4,
        evidence_score=3,
        confidence_score=5,
    )
    assert verdict.safety_concern is False
    assert verdict.critique == ""


def test_judge_verdict_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        JudgeVerdict(
            action_plausible=True,
            message_type_plausible=True,
            reason_score=7,  # out of the 1-5 range
            evidence_score=3,
            confidence_score=5,
        )


def test_judge_verdict_rejects_missing_required_field():
    with pytest.raises(ValidationError):
        JudgeVerdict(action_plausible=True, message_type_plausible=True, reason_score=4, evidence_score=3)


def _message_row() -> pd.Series:
    return pd.Series(
        {
            "message_id": "msg_1",
            "user_id": "u_001",
            "conversation_type": "personal",
            "group_id": None,
            "business_id": None,
            "sender_user_id": "u_002",
            "created_at": "2026-07-31 10:00",
            "message_text": "hello",
            "media_type": None,
            "media_id": None,
            "forwarded_count": 0,
        }
    )


def test_independent_route_returns_none_on_step_limit(monkeypatch):
    monkeypatch.setattr("orchestrate.evaluation.judge.build_user_content", lambda _row: "route")
    monkeypatch.setattr(
        "orchestrate.evaluation.judge.run_agent",
        lambda *_args, **_kwargs: AgentResult(final_text="ERROR: stuck", steps_used=6, hit_step_limit=True),
    )
    assert _independent_route(_message_row(), "judge-model") is None


def test_independent_route_returns_none_on_unparseable_answer(monkeypatch):
    """A blank/garbled final answer must not silently become a generic fallback decision --
    that would masquerade as a real independent opinion and pollute the agreement comparison
    in run_judge_eval (this is exactly what happened for msg_048 in a live run before this
    was fixed: an empty response became a spurious 'digest' opinion counted as a disagreement).
    """
    monkeypatch.setattr("orchestrate.evaluation.judge.build_user_content", lambda _row: "route")
    monkeypatch.setattr(
        "orchestrate.evaluation.judge.run_agent",
        lambda *_args, **_kwargs: AgentResult(final_text="", steps_used=1),
    )
    assert _independent_route(_message_row(), "judge-model") is None


def test_independent_route_never_sees_router_decision(monkeypatch):
    """The independent opinion must come from a fresh agent run over the raw message alone --
    it must not be handed the router's decision anywhere in its input.
    """
    captured = {}

    def fake_run_agent(system_prompt, user_input, **kwargs):
        captured["user_input"] = user_input
        captured["model"] = kwargs.get("model")
        payload = {
            "message_id": "msg_1",
            "action": "mute",
            "message_type": "spam",
            "reason": "independent call",
            "confidence": 0.7,
            "evidence_message_ids": "none",
        }
        return AgentResult(final_text=json.dumps(payload), steps_used=1)

    monkeypatch.setattr("orchestrate.evaluation.judge.build_user_content", lambda row: f"route:{row['message_id']}")
    monkeypatch.setattr("orchestrate.evaluation.judge.run_agent", fake_run_agent)

    decision = _independent_route(_message_row(), "judge-model")

    assert decision.action == "mute"
    assert captured["model"] == "judge-model"
    assert "notify" not in captured["user_input"] and "digest" not in captured["user_input"]


def test_build_judge_context_includes_independent_opinion(monkeypatch):
    monkeypatch.setattr(evaluate, "get_user_profile", lambda _user_id: json.dumps({"user_id": "u_001"}))
    decision_row = pd.Series(
        {"action": "notify", "message_type": "personal", "reason": "r", "confidence": 0.8, "evidence_message_ids": "none"}
    )
    independent = RoutingDecision(
        message_id="msg_1", action="mute", message_type="spam", reason="r2", confidence=0.6, evidence_message_ids="none"
    )
    context = json.loads(_build_judge_context(_message_row(), decision_row, independent))
    assert context["independent_opinion"]["action"] == "mute"
    assert "independent_opinion_note" in context


def test_build_judge_context_omits_independent_opinion_when_none(monkeypatch):
    monkeypatch.setattr(evaluate, "get_user_profile", lambda _user_id: json.dumps({"user_id": "u_001"}))
    decision_row = pd.Series(
        {"action": "notify", "message_type": "personal", "reason": "r", "confidence": 0.8, "evidence_message_ids": "none"}
    )
    context = json.loads(_build_judge_context(_message_row(), decision_row))
    assert "independent_opinion" not in context


def test_judge_eval_result_summary_computes_independent_agreement_rate():
    result = JudgeEvalResult(
        rows=[
            {
                "message_id": "msg_001",
                "ok": True,
                "action_plausible": True,
                "message_type_plausible": True,
                "reason_score": 5,
                "evidence_score": 5,
                "confidence_score": 4,
                "safety_concern": False,
                "independent_action": "notify",
                "agrees_with_independent": True,
            },
            {
                "message_id": "msg_002",
                "ok": True,
                "action_plausible": True,
                "message_type_plausible": True,
                "reason_score": 4,
                "evidence_score": 4,
                "confidence_score": 4,
                "safety_concern": False,
                "independent_action": "mute",
                "agrees_with_independent": False,
            },
        ]
    )
    summary = result.summary()
    assert summary["independent_opinion_formed"] == 2
    assert summary["independent_agreement_rate"] == 0.5
    assert summary["independent_disagreement_message_ids"] == ["msg_002"]


def test_judge_eval_result_summary_omits_independent_fields_when_absent():
    result = JudgeEvalResult(
        rows=[
            {
                "message_id": "msg_001",
                "ok": True,
                "action_plausible": True,
                "message_type_plausible": True,
                "reason_score": 5,
                "evidence_score": 5,
                "confidence_score": 4,
                "safety_concern": False,
            }
        ]
    )
    summary = result.summary()
    assert "independent_agreement_rate" not in summary

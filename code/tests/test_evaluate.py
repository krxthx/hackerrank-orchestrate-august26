"""Offline tests for evaluate.py's scoring logic -- no live LLM calls."""

import pytest
from pydantic import ValidationError

from orchestrate.evaluate import JudgeEvalResult, SampleEvalResult
from orchestrate.parsing import extract_json_object
from orchestrate.types import JudgeVerdict


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

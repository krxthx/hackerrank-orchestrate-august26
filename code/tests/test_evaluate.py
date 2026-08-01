"""Offline tests for evaluate.py's scoring logic -- no live LLM calls."""

from orchestrate.evaluate import JudgeEvalResult, SampleEvalResult, _evidence_overlap, _evidence_set, _extract_json


def test_evidence_set_handles_none_and_nan():
    assert _evidence_set("none") == set()
    assert _evidence_set("") == set()
    assert _evidence_set(None) == set()


def test_evidence_set_splits_semicolons():
    assert _evidence_set("message_0001;message_0002") == {"message_0001", "message_0002"}


def test_evidence_overlap_both_empty_is_perfect_match():
    assert _evidence_overlap("none", set()) == 1.0


def test_evidence_overlap_one_empty_is_zero():
    assert _evidence_overlap("message_0001", set()) == 0.0
    assert _evidence_overlap("none", {"message_0001"}) == 0.0


def test_evidence_overlap_partial_is_jaccard():
    predicted = "message_0001;message_0002"
    expected = {"message_0001", "message_0003"}
    # intersection={0001} union={0001,0002,0003} -> 1/3
    assert abs(_evidence_overlap(predicted, expected) - (1 / 3)) < 1e-9


def test_evidence_overlap_exact_match_is_one():
    assert _evidence_overlap("message_0001;message_0002", {"message_0001", "message_0002"}) == 1.0


def test_extract_json_strips_code_fence():
    raw = '```json\n{"action_plausible": true, "reason_score": 4}\n```'
    payload = _extract_json(raw)
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

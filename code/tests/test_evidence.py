"""Offline tests for evidence.py -- shared by evaluate.py and validate.py."""

from orchestrate.evidence import evidence_overlap, parse_evidence_ids


def test_parse_evidence_ids_handles_none_and_nan():
    assert parse_evidence_ids("none") == set()
    assert parse_evidence_ids("") == set()
    assert parse_evidence_ids(None) == set()


def test_parse_evidence_ids_splits_semicolons():
    assert parse_evidence_ids("message_0001;message_0002") == {"message_0001", "message_0002"}


def test_parse_evidence_ids_strips_whitespace():
    assert parse_evidence_ids(" message_0001 ; message_0002 ") == {"message_0001", "message_0002"}


def test_evidence_overlap_both_empty_is_perfect_match():
    assert evidence_overlap("none", set()) == 1.0


def test_evidence_overlap_one_empty_is_zero():
    assert evidence_overlap("message_0001", set()) == 0.0
    assert evidence_overlap("none", {"message_0001"}) == 0.0


def test_evidence_overlap_partial_is_jaccard():
    predicted = "message_0001;message_0002"
    expected = {"message_0001", "message_0003"}
    # intersection={0001} union={0001,0002,0003} -> 1/3
    assert abs(evidence_overlap(predicted, expected) - (1 / 3)) < 1e-9


def test_evidence_overlap_exact_match_is_one():
    assert evidence_overlap("message_0001;message_0002", {"message_0001", "message_0002"}) == 1.0

"""Tests for the shared model-output JSON extraction used by both pipeline.py (router
decisions) and evaluate.py (judge verdicts)."""

import pytest

from orchestrate.parsing import extract_json_object


def test_plain_json():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_strips_code_fence():
    raw = '```json\n{"a": 1, "b": 2}\n```'
    assert extract_json_object(raw) == {"a": 1, "b": 2}


def test_ignores_trailing_text_after_object():
    raw = '{"a": 1}\nthanks!'
    assert extract_json_object(raw) == {"a": 1}


def test_handles_nested_braces():
    raw = '{"a": {"b": 1}, "c": 2}'
    assert extract_json_object(raw) == {"a": {"b": 1}, "c": 2}


def test_ignores_braces_inside_string_values():
    raw = '{"reason": "looks like a {scam} attempt", "action": "mute"}'
    assert extract_json_object(raw) == {"reason": "looks like a {scam} attempt", "action": "mute"}


def test_raises_when_no_json_object_found():
    with pytest.raises(ValueError):
        extract_json_object("no json here")

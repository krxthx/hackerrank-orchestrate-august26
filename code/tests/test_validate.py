"""Offline tests for validate.py's batch-level output checks -- no live LLM calls."""

import pandas as pd
import pytest

from orchestrate.core.errors import DatasetError
from orchestrate.data.validation import validate_output

MESSAGES_HEADER = "message_id,user_id,conversation_type,group_id,business_id,sender_user_id,created_at,message_text,media_type,media_id,forwarded_count\n"
OUTPUT_HEADER = "message_id,action,message_type,reason,confidence,evidence_message_ids\n"


class _FakeDataset:
    def __init__(self, message_history: pd.DataFrame):
        self.message_history = message_history


def _history(rows):
    return pd.DataFrame(rows, columns=["message_id", "user_id"])


def test_validate_output_clean_run_has_no_issues(tmp_path):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(MESSAGES_HEADER + "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hi,,,0\n")
    output_path.write_text(OUTPUT_HEADER + "msg_1,notify,personal,ok,0.9,message_0001\n")

    ds = _FakeDataset(_history([("message_0001", "u_001")]))
    report = validate_output(str(input_path), str(output_path), dataset=ds)

    assert report.ok
    assert report.input_count == 1
    assert report.output_count == 1


def test_validate_output_flags_missing_row(tmp_path):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(
        MESSAGES_HEADER
        + "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hi,,,0\n"
        + "msg_2,u_001,personal,,,u_002,2026-07-31 10:01,bye,,,0\n"
    )
    output_path.write_text(OUTPUT_HEADER + "msg_1,notify,personal,ok,0.9,none\n")

    report = validate_output(str(input_path), str(output_path), dataset=_FakeDataset(_history([])))

    assert not report.ok
    assert any(i.message_id == "msg_2" and "missing from output" in i.problem for i in report.issues)


def test_validate_output_flags_duplicate_message_id(tmp_path):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(MESSAGES_HEADER + "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hi,,,0\n")
    output_path.write_text(OUTPUT_HEADER + "msg_1,notify,personal,ok,0.9,none\nmsg_1,digest,personal,ok,0.5,none\n")

    report = validate_output(str(input_path), str(output_path), dataset=_FakeDataset(_history([])))

    assert not report.ok
    assert any("duplicate message_id" in i.problem for i in report.issues)


def test_validate_output_flags_unexpected_row(tmp_path):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(MESSAGES_HEADER + "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hi,,,0\n")
    output_path.write_text(OUTPUT_HEADER + "msg_1,notify,personal,ok,0.9,none\nmsg_ghost,notify,personal,ok,0.9,none\n")

    report = validate_output(str(input_path), str(output_path), dataset=_FakeDataset(_history([])))

    assert not report.ok
    assert any(i.message_id == "msg_ghost" and "not present in input" in i.problem for i in report.issues)


def test_validate_output_flags_invalid_schema_row(tmp_path):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(MESSAGES_HEADER + "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hi,,,0\n")
    output_path.write_text(OUTPUT_HEADER + "msg_1,not_a_real_action,personal,ok,0.9,none\n")

    report = validate_output(str(input_path), str(output_path), dataset=_FakeDataset(_history([])))

    assert not report.ok
    assert any("RoutingDecision schema" in i.problem for i in report.issues)


def test_validate_output_flags_fabricated_evidence_id(tmp_path):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(MESSAGES_HEADER + "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hi,,,0\n")
    output_path.write_text(OUTPUT_HEADER + "msg_1,notify,personal,ok,0.9,message_9999\n")

    ds = _FakeDataset(_history([("message_0001", "u_001")]))  # message_9999 doesn't exist
    report = validate_output(str(input_path), str(output_path), dataset=ds)

    assert not report.ok
    assert any("message_9999" in i.problem for i in report.issues)


def test_validate_output_evidence_must_belong_to_same_user(tmp_path):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(MESSAGES_HEADER + "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hi,,,0\n")
    output_path.write_text(OUTPUT_HEADER + "msg_1,notify,personal,ok,0.9,message_0001\n")

    # message_0001 exists, but belongs to a different user
    ds = _FakeDataset(_history([("message_0001", "u_999")]))
    report = validate_output(str(input_path), str(output_path), dataset=ds)

    assert not report.ok
    assert any("message_0001" in i.problem for i in report.issues)


def test_validate_output_none_evidence_is_never_flagged(tmp_path):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(MESSAGES_HEADER + "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hi,,,0\n")
    output_path.write_text(OUTPUT_HEADER + "msg_1,notify,personal,ok,0.9,none\n")

    report = validate_output(str(input_path), str(output_path), dataset=_FakeDataset(_history([])))

    assert report.ok


def test_validate_output_missing_input_raises_dataset_error(tmp_path):
    output_path = tmp_path / "output.csv"
    output_path.write_text(OUTPUT_HEADER)
    with pytest.raises(DatasetError):
        validate_output(str(tmp_path / "nope.csv"), str(output_path), dataset=_FakeDataset(_history([])))


def test_validation_report_summary_shape(tmp_path):
    input_path = tmp_path / "messages.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(MESSAGES_HEADER + "msg_1,u_001,personal,,,u_002,2026-07-31 10:00,hi,,,0\n")
    output_path.write_text(OUTPUT_HEADER + "msg_1,notify,personal,ok,0.9,none\n")

    report = validate_output(str(input_path), str(output_path), dataset=_FakeDataset(_history([])))
    summary = report.summary()

    assert summary["ok"] is True
    assert summary["input_count"] == 1
    assert summary["issue_count"] == 0

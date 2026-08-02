"""Unit tests for hybrid history ranking (no live LLM calls)."""

import pandas as pd

from orchestrate.data.retrieval import rank_history_hybrid


def _row(message_id, created_at, text):
    return {"message_id": message_id, "created_at": created_at, "message_text": text}


def test_empty_query_text_falls_back_to_pure_recency():
    history = pd.DataFrame(
        [
            _row("m1", "2026-01-01 10:00", "hello there"),
            _row("m2", "2026-02-01 10:00", "unrelated"),
            _row("m3", "2026-03-01 10:00", "also unrelated"),
        ]
    )
    ranked = rank_history_hybrid(history, "", limit=2)
    assert list(ranked["message_id"]) == ["m3", "m2"]


def test_relevant_older_message_can_outrank_irrelevant_recent_one():
    history = pd.DataFrame(
        [
            _row("old_relevant", "2026-01-01 10:00", "your otp verification code expired blocked"),
            _row("recent_irrelevant", "2026-03-01 10:00", "happy birthday see you soon"),
        ]
    )
    ranked = rank_history_hybrid(history, "otp verification code blocked", limit=1)
    assert list(ranked["message_id"]) == ["old_relevant"]


def test_empty_history_returns_empty():
    history = pd.DataFrame(columns=["message_id", "created_at", "message_text"])
    assert rank_history_hybrid(history, "anything", limit=5).empty

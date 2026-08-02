"""Hybrid ranking for get_message_history: blend recency with lexical relevance to the
current message, instead of pure most-recent-first, so the past messages that are actually
informative for this decision (e.g. an earlier near-identical scam, not just whatever was
sent last) are more likely to make the cut when `limit` truncates the history.
"""

import re

import pandas as pd

_WORD_RE = re.compile(r"[a-z0-9]+")

_RECENCY_WEIGHT = 0.5
_RELEVANCE_WEIGHT = 0.5


def _tokens(text) -> set[str]:
    if not text or not isinstance(text, str):
        return set()
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rank_history_hybrid(history: pd.DataFrame, current_message_text: str, limit: int) -> pd.DataFrame:
    """`history` must already be filtered to the relevant sender/group/business and have a
    `created_at` column. Falls back to pure recency (the original behavior) when no current
    message text is given, so passing "" is a no-op change.
    """
    history = history.sort_values("created_at", ascending=False)
    if not current_message_text or history.empty:
        return history.head(limit)

    pool = history.head(max(limit * 4, 20))
    n = len(pool)
    query_tokens = _tokens(current_message_text)

    scored = []
    for rank, (idx, row) in enumerate(pool.iterrows()):
        recency_score = (n - rank) / n
        relevance_score = _jaccard(query_tokens, _tokens(row.get("message_text")))
        scored.append((_RECENCY_WEIGHT * recency_score + _RELEVANCE_WEIGHT * relevance_score, idx))
    scored.sort(key=lambda item: item[0], reverse=True)

    top_idx = [idx for _, idx in scored[:limit]]
    return pool.loc[top_idx]

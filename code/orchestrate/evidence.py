"""Parsing/comparison helpers for the evidence_message_ids column.

Shared by evaluate.py (scoring predicted evidence against sample labels) and validate.py
(checking that every cited ID is a real historical message for that user) so the
semicolon-separated-string convention -- and what "none" means -- is defined in exactly
one place.
"""

import pandas as pd


def parse_evidence_ids(value) -> set[str]:
    if not value or pd.isna(value) or str(value).strip().lower() == "none":
        return set()
    return {v.strip() for v in str(value).split(";") if v.strip()}


def evidence_overlap(predicted, expected: set[str]) -> float:
    """Jaccard similarity between predicted and expected evidence ID sets. Both-empty
    (correctly abstaining) counts as a perfect match.
    """
    predicted_set = parse_evidence_ids(predicted)
    if not predicted_set and not expected:
        return 1.0
    if not predicted_set or not expected:
        return 0.0
    return len(predicted_set & expected) / len(predicted_set | expected)

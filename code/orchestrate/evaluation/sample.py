"""Hard metrics against the labeled sample dataset."""

import logging
import os
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from orchestrate.core.config import DATASET_DIR, ROUTER_MAX_STEPS
from orchestrate.core.errors import DatasetError, classify_llm_error, describe_step_limit
from orchestrate.data.evidence import evidence_overlap, parse_evidence_ids
from orchestrate.prompts import DEFAULT_SYSTEM_PROMPT
from orchestrate.routing.agent import run_agent
from orchestrate.routing.pipeline import build_user_content, parse_result

logger = logging.getLogger(__name__)

INPUT_COLUMNS = [
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "created_at",
    "message_text",
    "media_type",
    "media_id",
    "forwarded_count",
]


def _confusion_matrix(scored: list[dict], predicted_key: str, expected_key: str) -> list[dict]:
    """Counts of (expected -> predicted) pairs, misclassifications only, most frequent first.
    Meant to answer "what does the model most often confuse for what", not to duplicate the
    accuracy figure already in summary().
    """
    pairs = [
        (row[expected_key], row[predicted_key])
        for row in scored
        if expected_key in row and predicted_key in row and row[expected_key] != row[predicted_key]
    ]
    counts = Counter(pairs)
    return [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]


def _calibration_buckets(scored: list[dict]) -> list[dict]:
    """Reliability diagram data: for each confidence decile, mean confidence vs. actual
    action accuracy. A well-calibrated router has mean_confidence ~= accuracy in every
    bucket; a bucket where confidence is consistently higher than accuracy is overconfident.
    """
    edges = [(i / 10, (i + 1) / 10) for i in range(10)]
    buckets = []
    for low, high in edges:
        in_bucket = [
            row for row in scored if low <= row["confidence"] < high or (high == 1.0 and row["confidence"] == 1.0)
        ]
        if not in_bucket:
            continue
        buckets.append(
            {
                "range": f"{low:.1f}-{high:.1f}",
                "n": len(in_bucket),
                "mean_confidence": round(sum(row["confidence"] for row in in_bucket) / len(in_bucket), 3),
                "action_accuracy": round(sum(row["action_correct"] for row in in_bucket) / len(in_bucket), 3),
            }
        )
    return buckets


@dataclass
class SampleEvalResult:
    rows: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    def summary(self) -> dict:
        if not self.rows:
            return {"n": 0}
        scored = [row for row in self.rows if row.get("ok", True)]
        if not scored:
            return {"n": self.n, "scored": 0, "errors": self.n}
        action_accuracy = sum(row["action_correct"] for row in scored) / len(scored)
        type_accuracy = sum(row["message_type_correct"] for row in scored) / len(scored)
        overlap = sum(row["evidence_overlap"] for row in scored) / len(scored)
        brier = sum((row["confidence"] - row["action_correct"]) ** 2 for row in scored) / len(scored)
        return {
            "n": self.n,
            "scored": len(scored),
            "errors": self.n - len(scored),
            "action_accuracy": round(action_accuracy, 3),
            "message_type_accuracy": round(type_accuracy, 3),
            "evidence_overlap_mean": round(overlap, 3),
            "confidence_brier_score": round(brier, 3),
            "action_confusion_matrix": _confusion_matrix(scored, "predicted_action", "expected_action"),
            "message_type_confusion_matrix": _confusion_matrix(
                scored, "predicted_message_type", "expected_message_type"
            ),
            "confidence_calibration": _calibration_buckets(scored),
        }


def run_sample_eval(
    sample_path: str = os.path.join(DATASET_DIR, "sample_messages.csv"),
    max_steps: int = ROUTER_MAX_STEPS,
    limit: int | None = None,
) -> SampleEvalResult:
    """Route labeled sample rows blind to their expected outputs, then score them."""
    if not os.path.exists(sample_path):
        raise DatasetError(f"Sample eval input not found: {sample_path}.")
    try:
        frame = pd.read_csv(sample_path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"Sample eval input is empty or unreadable: {sample_path}.", cause=exc) from exc
    if limit:
        frame = frame.head(limit)

    result = SampleEvalResult()
    for index, row in frame.iterrows():
        message_id = row["message_id"]
        try:
            input_row = row[[column for column in INPUT_COLUMNS if column in row.index]]
            content = build_user_content(input_row)
            agent_result = run_agent(DEFAULT_SYSTEM_PROMPT, content, max_steps=max_steps)
            if agent_result.hit_step_limit:
                error_message = describe_step_limit(max_steps)
                logger.warning("sample eval hit step limit for %s: %s", message_id, error_message)
                record = {"message_id": message_id, "ok": False, "error": error_message}
            else:
                prediction = parse_result(agent_result.final_text, message_id)
                expected_evidence = parse_evidence_ids(row.get("evidence_message_ids"))
                record = {
                    "message_id": message_id,
                    "ok": True,
                    "predicted_action": prediction.action,
                    "expected_action": row["action"],
                    "action_correct": prediction.action == row["action"],
                    "predicted_message_type": prediction.message_type,
                    "expected_message_type": row["message_type"],
                    "message_type_correct": prediction.message_type == row["message_type"],
                    "confidence": prediction.confidence,
                    "evidence_overlap": evidence_overlap(prediction.evidence_message_ids, expected_evidence),
                }
                logger.info("[%d/%d] scored %s", index + 1, len(frame), message_id)
        except DatasetError:
            raise
        except Exception as exc:
            error = classify_llm_error(exc)
            logger.warning("sample eval failed for %s, skipping: %s", message_id, error)
            record = {"message_id": message_id, "ok": False, "error": error.user_message}
        result.rows.append(record)
    return result

"""Hard metrics against the labeled sample dataset."""

import logging
import os
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

"""Evaluation pipeline for the Message Notification Router.

Two independent passes, since no hidden ground truth is available before submission:

1. `run_sample_eval` -- hard metrics. Re-routes every row in dataset/sample_messages.csv
   (a separate, disjoint set of 30 labeled examples -- not part of the 110 rows in
   messages.csv/output.csv) through the real pipeline, blind to the given labels, then
   compares predictions against them: action/message_type accuracy, evidence-set overlap,
   and confidence calibration (Brier score). This is a small, non-representative sample, so
   treat it as a regression check, not a true accuracy estimate.

2. `run_judge_eval` -- rubric-judge pass. For every row already in dataset/output.csv (the
   actual submission, no ground truth exists for it), a separate judge model scores the
   decision against the same five dimensions problem_statement.md says the hidden grader
   uses: action correctness, message_type correctness, reason usefulness/consistency,
   evidence relevance, and confidence calibration. The judge sees the same context the
   router had (user/group/business profile, and the real content behind any cited
   evidence_message_ids) but not a "correct answer" -- it grades plausibility, not exact match.
   Uses a separate JUDGE_MODEL (litellm-agnostic, configurable via env) so grading isn't
   done by the same model/provider that made the decisions.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError

from orchestrate.agent import run_agent
from orchestrate.config import DATASET_DIR, JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL, ROUTER_MAX_STEPS
from orchestrate.constants import SYSTEM_ROLE, USER_ROLE
from orchestrate.data import get_dataset
from orchestrate.llm import complete
from orchestrate.pipeline import build_user_content, parse_result
from orchestrate.prompts import DEFAULT_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT
from orchestrate.tools import get_business_context, get_group_context, get_user_profile

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# 1. Hard metrics against dataset/sample_messages.csv
# ---------------------------------------------------------------------------

LABEL_COLUMNS = ["action", "message_type", "reason", "confidence", "evidence_message_ids"]
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


def _evidence_set(value: str) -> set[str]:
    if not value or pd.isna(value) or str(value).strip().lower() == "none":
        return set()
    return {v.strip() for v in str(value).split(";") if v.strip()}


def _evidence_overlap(predicted: str, expected: set[str]) -> float:
    """Jaccard similarity between predicted and expected evidence ID sets. Both-empty
    (correctly abstaining) counts as a perfect match.
    """
    predicted_set = _evidence_set(predicted)
    if not predicted_set and not expected:
        return 1.0
    if not predicted_set or not expected:
        return 0.0
    return len(predicted_set & expected) / len(predicted_set | expected)


@dataclass
class SampleEvalResult:
    rows: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    def summary(self) -> dict:
        if not self.rows:
            return {"n": 0}
        action_acc = sum(r["action_correct"] for r in self.rows) / self.n
        type_acc = sum(r["message_type_correct"] for r in self.rows) / self.n
        evidence_overlap = sum(r["evidence_overlap"] for r in self.rows) / self.n
        brier = sum((r["confidence"] - r["action_correct"]) ** 2 for r in self.rows) / self.n
        return {
            "n": self.n,
            "action_accuracy": round(action_acc, 3),
            "message_type_accuracy": round(type_acc, 3),
            "evidence_overlap_mean": round(evidence_overlap, 3),
            "confidence_brier_score": round(brier, 3),
        }


def run_sample_eval(
    sample_path: str = os.path.join(DATASET_DIR, "sample_messages.csv"),
    max_steps: int = ROUTER_MAX_STEPS,
    limit: int | None = None,
) -> SampleEvalResult:
    """Routes every row of sample_messages.csv through the real agent (blind to the given
    labels -- only INPUT_COLUMNS are shown to it) and scores the predictions against the
    labels that ship with the file.
    """
    df = pd.read_csv(sample_path)
    if limit:
        df = df.head(limit)

    result = SampleEvalResult()
    for i, row in df.iterrows():
        input_row = row[[c for c in INPUT_COLUMNS if c in row.index]]
        content = build_user_content(input_row)
        agent_result = run_agent(DEFAULT_SYSTEM_PROMPT, content, max_steps=max_steps)
        prediction = parse_result(agent_result.final_text, row["message_id"])

        expected_evidence = _evidence_set(row.get("evidence_message_ids"))
        record = {
            "message_id": row["message_id"],
            "predicted_action": prediction.action,
            "expected_action": row["action"],
            "action_correct": prediction.action == row["action"],
            "predicted_message_type": prediction.message_type,
            "expected_message_type": row["message_type"],
            "message_type_correct": prediction.message_type == row["message_type"],
            "confidence": prediction.confidence,
            "evidence_overlap": _evidence_overlap(prediction.evidence_message_ids, expected_evidence),
        }
        result.rows.append(record)
        logger.info(
            "[%d/%d] %s action=%s(expected %s) type=%s(expected %s)",
            i + 1,
            len(df),
            row["message_id"],
            prediction.action,
            row["action"],
            prediction.message_type,
            row["message_type"],
        )
    return result


# ---------------------------------------------------------------------------
# 2. Rubric-judge pass over dataset/output.csv
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object found in judge output: {text[:200]!r}")
    return json.loads(match.group(0))


def _build_judge_context(message_row: pd.Series, decision_row: pd.Series) -> str:
    """Same context signals the router had access to, plus the real content behind any
    cited evidence_message_ids, so the judge can verify relevance rather than trust it.
    """
    ds = get_dataset()
    user_id = message_row["user_id"]
    context: dict = {
        "message": {c: (None if pd.isna(message_row.get(c)) else message_row.get(c)) for c in INPUT_COLUMNS},
        "decision": {
            "action": decision_row["action"],
            "message_type": decision_row["message_type"],
            "reason": decision_row["reason"],
            "confidence": decision_row["confidence"],
            "evidence_message_ids": decision_row["evidence_message_ids"],
        },
        "user_profile": json.loads(get_user_profile(user_id)),
    }
    if message_row.get("conversation_type") == "group" and not pd.isna(message_row.get("group_id")):
        context["group_context"] = json.loads(get_group_context(message_row["group_id"], user_id))
    if message_row.get("conversation_type") == "business" and not pd.isna(message_row.get("business_id")):
        context["business_context"] = json.loads(get_business_context(message_row["business_id"], user_id))

    evidence_ids = _evidence_set(decision_row["evidence_message_ids"])
    if evidence_ids:
        cited = ds.messages_by_id(user_id, list(evidence_ids))
        context["cited_evidence"] = cited
        context["evidence_ids_verified"] = sorted(r["message_id"] for r in cited) == sorted(evidence_ids)

    return json.dumps(context, indent=2, default=str)


@dataclass
class JudgeEvalResult:
    rows: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    def summary(self) -> dict:
        if not self.rows:
            return {"n": 0}
        scored = [r for r in self.rows if r.get("ok")]
        if not scored:
            return {"n": self.n, "scored": 0, "errors": self.n}
        return {
            "n": self.n,
            "scored": len(scored),
            "errors": self.n - len(scored),
            "action_plausible_rate": round(sum(r["action_plausible"] for r in scored) / len(scored), 3),
            "message_type_plausible_rate": round(sum(r["message_type_plausible"] for r in scored) / len(scored), 3),
            "reason_score_mean": round(sum(r["reason_score"] for r in scored) / len(scored), 2),
            "evidence_score_mean": round(sum(r["evidence_score"] for r in scored) / len(scored), 2),
            "confidence_score_mean": round(sum(r["confidence_score"] for r in scored) / len(scored), 2),
            "safety_concern_count": sum(r["safety_concern"] for r in scored),
            "low_scoring_message_ids": [
                r["message_id"] for r in scored if r["reason_score"] <= 2 or r["evidence_score"] <= 2 or r["safety_concern"]
            ],
        }


def _judge_one(context_json: str, judge_model: str) -> dict:
    messages = [
        {"role": SYSTEM_ROLE, "content": JUDGE_SYSTEM_PROMPT},
        {"role": USER_ROLE, "content": context_json},
    ]
    response = complete(messages, model=judge_model, api_base=JUDGE_API_BASE, api_key=JUDGE_API_KEY)
    text = response.choices[0].message.content or ""
    return _extract_json(text)


def run_judge_eval(
    messages_path: str = os.path.join(DATASET_DIR, "messages.csv"),
    output_path: str = os.path.join(DATASET_DIR, "output.csv"),
    judge_model: str = JUDGE_MODEL,
    limit: int | None = None,
) -> JudgeEvalResult:
    messages_df = pd.read_csv(messages_path).set_index("message_id", drop=False)
    output_df = pd.read_csv(output_path)
    if limit:
        output_df = output_df.head(limit)

    result = JudgeEvalResult()
    for i, decision_row in output_df.iterrows():
        message_id = decision_row["message_id"]
        if message_id not in messages_df.index:
            logger.warning("skipping %s: not found in %s", message_id, messages_path)
            continue
        message_row = messages_df.loc[message_id]
        context_json = _build_judge_context(message_row, decision_row)
        try:
            verdict = _judge_one(context_json, judge_model)
            record = {
                "message_id": message_id,
                "ok": True,
                "action_plausible": bool(verdict["action_plausible"]),
                "message_type_plausible": bool(verdict["message_type_plausible"]),
                "reason_score": int(verdict["reason_score"]),
                "evidence_score": int(verdict["evidence_score"]),
                "confidence_score": int(verdict["confidence_score"]),
                "safety_concern": bool(verdict.get("safety_concern", False)),
                "critique": verdict.get("critique", ""),
            }
        except (ValueError, ValidationError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("judge failed to score %s: %s", message_id, exc)
            record = {"message_id": message_id, "ok": False, "error": str(exc)}
        result.rows.append(record)
        logger.info("[%d/%d] judged %s (ok=%s)", i + 1, len(output_df), message_id, record["ok"])
    return result


def write_report(sample_result: SampleEvalResult | None, judge_result: JudgeEvalResult | None, path: str) -> None:
    report = {}
    if sample_result is not None:
        report["sample_eval"] = {"summary": sample_result.summary(), "rows": sample_result.rows}
    if judge_result is not None:
        report["judge_eval"] = {"summary": judge_result.summary(), "rows": judge_result.rows}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

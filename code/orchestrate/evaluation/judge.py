"""Independent routing and rubric-judge evaluation."""

import json
import logging
import os
from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError

from orchestrate.core.config import DATASET_DIR, JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL, ROUTER_MAX_STEPS
from orchestrate.core.constants import SYSTEM_ROLE, USER_ROLE
from orchestrate.core.errors import (
    DatasetError,
    DecisionParseError,
    OrchestrateError,
    classify_llm_error,
    describe_parse_error,
    describe_step_limit,
)
from orchestrate.core.parsing import extract_json_object
from orchestrate.core.types import JudgeVerdict, RoutingDecision
from orchestrate.data.dataset import get_dataset
from orchestrate.data.evidence import parse_evidence_ids
from orchestrate.evaluation.sample import INPUT_COLUMNS
from orchestrate.llm import complete
from orchestrate.prompts import DEFAULT_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT
from orchestrate.routing.agent import run_agent
from orchestrate.routing.pipeline import build_user_content
from orchestrate.routing.tools import get_business_context, get_group_context, get_user_profile

logger = logging.getLogger(__name__)


def _independent_route(
    message_row: pd.Series,
    judge_model: str,
    max_steps: int = ROUTER_MAX_STEPS,
) -> RoutingDecision | None:
    """Form a second opinion without exposing the router's original decision."""
    input_row = message_row[[column for column in INPUT_COLUMNS if column in message_row.index]]
    content = build_user_content(input_row)
    result = run_agent(
        DEFAULT_SYSTEM_PROMPT,
        content,
        max_steps=max_steps,
        model=judge_model,
        api_base=JUDGE_API_BASE,
        api_key=JUDGE_API_KEY,
    )
    if result.hit_step_limit:
        logger.warning(
            "independent re-route hit step limit for %s: %s",
            message_row["message_id"],
            describe_step_limit(max_steps),
        )
        return None
    try:
        payload = extract_json_object(result.final_text, label="independent opinion")
        payload.setdefault("message_id", message_row["message_id"])
        return RoutingDecision(**payload)
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.warning(
            "independent re-route for %s produced an unparseable answer, discarding: %s",
            message_row["message_id"],
            describe_parse_error(exc),
        )
        return None


def _build_judge_context(
    message_row: pd.Series,
    decision_row: pd.Series,
    independent: RoutingDecision | None = None,
) -> str:
    """Build the decision, profile, relationship, and verified-evidence context."""
    dataset = get_dataset()
    user_id = message_row["user_id"]
    context: dict = {
        "message": {
            column: (None if pd.isna(message_row.get(column)) else message_row.get(column))
            for column in INPUT_COLUMNS
        },
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

    evidence_ids = parse_evidence_ids(decision_row["evidence_message_ids"])
    if evidence_ids:
        cited = dataset.messages_by_id(user_id, list(evidence_ids))
        context["cited_evidence"] = cited
        context["evidence_ids_verified"] = sorted(row["message_id"] for row in cited) == sorted(evidence_ids)

    if independent is not None:
        context["independent_opinion"] = {
            "action": independent.action,
            "message_type": independent.message_type,
            "confidence": independent.confidence,
        }
        context["independent_opinion_note"] = (
            "Formed by an agent that independently re-routed this message from scratch, "
            "blind to the decision above, using the same tools. A second data point, not "
            "ground truth -- it can be wrong too."
        )
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
        scored = [row for row in self.rows if row.get("ok")]
        if not scored:
            return {"n": self.n, "scored": 0, "errors": self.n}
        with_independent = [row for row in scored if row.get("agrees_with_independent") is not None]
        summary = {
            "n": self.n,
            "scored": len(scored),
            "errors": self.n - len(scored),
            "action_plausible_rate": round(sum(row["action_plausible"] for row in scored) / len(scored), 3),
            "message_type_plausible_rate": round(
                sum(row["message_type_plausible"] for row in scored) / len(scored), 3
            ),
            "reason_score_mean": round(sum(row["reason_score"] for row in scored) / len(scored), 2),
            "evidence_score_mean": round(sum(row["evidence_score"] for row in scored) / len(scored), 2),
            "confidence_score_mean": round(sum(row["confidence_score"] for row in scored) / len(scored), 2),
            "safety_concern_count": sum(row["safety_concern"] for row in scored),
            "low_scoring_message_ids": [
                row["message_id"]
                for row in scored
                if row["reason_score"] <= 2 or row["evidence_score"] <= 2 or row["safety_concern"]
            ],
        }
        if with_independent:
            summary["independent_opinion_formed"] = len(with_independent)
            summary["independent_agreement_rate"] = round(
                sum(row["agrees_with_independent"] for row in with_independent) / len(with_independent), 3
            )
            summary["independent_disagreement_message_ids"] = [
                row["message_id"] for row in with_independent if not row["agrees_with_independent"]
            ]
        return summary


def _judge_one(context_json: str, judge_model: str) -> JudgeVerdict:
    messages = [
        {"role": SYSTEM_ROLE, "content": JUDGE_SYSTEM_PROMPT},
        {"role": USER_ROLE, "content": context_json},
    ]
    try:
        response = complete(messages, model=judge_model, api_base=JUDGE_API_BASE, api_key=JUDGE_API_KEY)
    except Exception as exc:
        raise classify_llm_error(exc) from exc
    text = response.choices[0].message.content or ""
    try:
        return JudgeVerdict(**extract_json_object(text, label="judge output"))
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        raise DecisionParseError(describe_parse_error(exc), cause=exc) from exc


def run_judge_eval(
    messages_path: str = os.path.join(DATASET_DIR, "messages.csv"),
    output_path: str = os.path.join(DATASET_DIR, "output.csv"),
    judge_model: str = JUDGE_MODEL,
    limit: int | None = None,
    independent_opinion: bool = True,
) -> JudgeEvalResult:
    """Grade submission decisions, optionally with a blind independent opinion."""
    for path in (messages_path, output_path):
        if not os.path.exists(path):
            raise DatasetError(f"Judge eval input not found: {path}.")
    try:
        messages_frame = pd.read_csv(messages_path).set_index("message_id", drop=False)
        output_frame = pd.read_csv(output_path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"Judge eval input is empty or unreadable: {exc}.", cause=exc) from exc
    if limit:
        output_frame = output_frame.head(limit)

    result = JudgeEvalResult()
    for index, decision_row in output_frame.iterrows():
        message_id = decision_row["message_id"]
        if message_id not in messages_frame.index:
            logger.warning("skipping %s: not found in %s", message_id, messages_path)
            continue
        message_row = messages_frame.loc[message_id]
        independent = None
        if independent_opinion:
            try:
                independent = _independent_route(message_row, judge_model)
            except Exception as exc:
                error = classify_llm_error(exc)
                logger.warning("independent re-route failed for %s, grading without it: %s", message_id, error)

        try:
            verdict = _judge_one(_build_judge_context(message_row, decision_row, independent), judge_model)
            record = {
                "message_id": message_id,
                "ok": True,
                "action_plausible": verdict.action_plausible,
                "message_type_plausible": verdict.message_type_plausible,
                "reason_score": verdict.reason_score,
                "evidence_score": verdict.evidence_score,
                "confidence_score": verdict.confidence_score,
                "safety_concern": verdict.safety_concern,
                "critique": verdict.critique,
                "independent_action": independent.action if independent else None,
                "agrees_with_independent": independent.action == decision_row["action"] if independent else None,
            }
        except OrchestrateError as exc:
            logger.warning("judge failed to score %s: %s", message_id, exc)
            record = {"message_id": message_id, "ok": False, "error": exc.user_message}
        result.rows.append(record)
        logger.info("[%d/%d] judged %s (ok=%s)", index + 1, len(output_frame), message_id, record["ok"])
    return result

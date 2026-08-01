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

   Before grading, `_independent_route` gives the judge model a second, blind pass at the
   *same* message -- its own full tool-calling run through DEFAULT_SYSTEM_PROMPT, never shown
   the router's actual decision. Only the resulting action/message_type/confidence are then
   handed to the rubric-grading call alongside the router's decision. This exists because a
   judge asked to critique an answer already placed in front of it tends to rate it
   charitably (confirmation bias); an independently-formed opinion, computed and compared
   programmatically (`agrees_with_independent`), is a sturdier signal than the judge's own
   self-reported agreement would be.
"""

import json
import logging
import os
from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError

from orchestrate.agent import run_agent
from orchestrate.config import DATASET_DIR, JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL, ROUTER_MAX_STEPS
from orchestrate.constants import SYSTEM_ROLE, USER_ROLE
from orchestrate.data import get_dataset
from orchestrate.errors import (
    DatasetError,
    DecisionParseError,
    OrchestrateError,
    classify_llm_error,
    describe_parse_error,
    describe_step_limit,
)
from orchestrate.evidence import evidence_overlap, parse_evidence_ids
from orchestrate.llm import complete
from orchestrate.parsing import extract_json_object
from orchestrate.pipeline import build_user_content, parse_result
from orchestrate.prompts import DEFAULT_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT
from orchestrate.tools import get_business_context, get_group_context, get_user_profile
from orchestrate.types import JudgeVerdict, RoutingDecision

logger = logging.getLogger(__name__)


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


@dataclass
class SampleEvalResult:
    rows: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    def summary(self) -> dict:
        if not self.rows:
            return {"n": 0}
        scored = [r for r in self.rows if r.get("ok", True)]
        if not scored:
            return {"n": self.n, "scored": 0, "errors": self.n}
        action_acc = sum(r["action_correct"] for r in scored) / len(scored)
        type_acc = sum(r["message_type_correct"] for r in scored) / len(scored)
        evidence_overlap = sum(r["evidence_overlap"] for r in scored) / len(scored)
        brier = sum((r["confidence"] - r["action_correct"]) ** 2 for r in scored) / len(scored)
        return {
            "n": self.n,
            "scored": len(scored),
            "errors": self.n - len(scored),
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
    if not os.path.exists(sample_path):
        raise DatasetError(f"Sample eval input not found: {sample_path}.")
    try:
        df = pd.read_csv(sample_path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"Sample eval input is empty or unreadable: {sample_path}.", cause=exc) from exc
    if limit:
        df = df.head(limit)

    result = SampleEvalResult()
    for i, row in df.iterrows():
        message_id = row["message_id"]
        try:
            input_row = row[[c for c in INPUT_COLUMNS if c in row.index]]
            content = build_user_content(input_row)
            agent_result = run_agent(DEFAULT_SYSTEM_PROMPT, content, max_steps=max_steps)
            if agent_result.hit_step_limit:
                err_message = describe_step_limit(max_steps)
                logger.warning("sample eval hit step limit for %s: %s", message_id, err_message)
                record = {"message_id": message_id, "ok": False, "error": err_message}
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
                logger.info(
                    "[%d/%d] %s action=%s(expected %s) type=%s(expected %s)",
                    i + 1,
                    len(df),
                    message_id,
                    prediction.action,
                    row["action"],
                    prediction.message_type,
                    row["message_type"],
                )
        except DatasetError:
            # Fatal -- would fail identically on every remaining row.
            raise
        except Exception as exc:  # a single sample row's failure shouldn't kill the whole eval run
            err = classify_llm_error(exc)
            logger.warning("sample eval failed for %s, skipping: %s", message_id, err)
            record = {"message_id": message_id, "ok": False, "error": err.user_message}
        result.rows.append(record)
    return result


# ---------------------------------------------------------------------------
# 2. Rubric-judge pass over dataset/output.csv
# ---------------------------------------------------------------------------


def _independent_route(
    message_row: pd.Series,
    judge_model: str,
    max_steps: int = ROUTER_MAX_STEPS,
) -> RoutingDecision | None:
    """A second, blind opinion: re-route this exact message from scratch through the judge
    model, using the same system prompt and tools as the real router, but never shown the
    router's actual decision. Returns None if the run hits its step ceiling, or its final
    answer doesn't parse, rather than parse_result's usual behavior of masking either case
    behind a generic low-confidence decision -- a placeholder opinion here would silently
    count as a real "disagreement" against the router in run_judge_eval's comparison, which
    is worse than just having no opinion.
    """
    input_row = message_row[[c for c in INPUT_COLUMNS if c in message_row.index]]
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

    evidence_ids = parse_evidence_ids(decision_row["evidence_message_ids"])
    if evidence_ids:
        cited = ds.messages_by_id(user_id, list(evidence_ids))
        context["cited_evidence"] = cited
        context["evidence_ids_verified"] = sorted(r["message_id"] for r in cited) == sorted(evidence_ids)

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
        scored = [r for r in self.rows if r.get("ok")]
        if not scored:
            return {"n": self.n, "scored": 0, "errors": self.n}
        with_independent = [r for r in scored if r.get("agrees_with_independent") is not None]
        summary = {
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
        if with_independent:
            summary["independent_opinion_formed"] = len(with_independent)
            summary["independent_agreement_rate"] = round(
                sum(r["agrees_with_independent"] for r in with_independent) / len(with_independent), 3
            )
            summary["independent_disagreement_message_ids"] = [
                r["message_id"] for r in with_independent if not r["agrees_with_independent"]
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
        payload = extract_json_object(text, label="judge output")
        return JudgeVerdict(**payload)
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        raise DecisionParseError(describe_parse_error(exc), cause=exc) from exc


def run_judge_eval(
    messages_path: str = os.path.join(DATASET_DIR, "messages.csv"),
    output_path: str = os.path.join(DATASET_DIR, "output.csv"),
    judge_model: str = JUDGE_MODEL,
    limit: int | None = None,
    independent_opinion: bool = True,
) -> JudgeEvalResult:
    """`independent_opinion` controls whether each row also gets a blind second routing pass
    (see `_independent_route`) before grading. Costs one extra full agent run per row, so it
    can be disabled (e.g. for a fast smoke test) without touching the rubric-grading path.
    """
    for path in (messages_path, output_path):
        if not os.path.exists(path):
            raise DatasetError(f"Judge eval input not found: {path}.")
    try:
        messages_df = pd.read_csv(messages_path).set_index("message_id", drop=False)
        output_df = pd.read_csv(output_path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"Judge eval input is empty or unreadable: {exc}.", cause=exc) from exc
    if limit:
        output_df = output_df.head(limit)

    result = JudgeEvalResult()
    for i, decision_row in output_df.iterrows():
        message_id = decision_row["message_id"]
        if message_id not in messages_df.index:
            logger.warning("skipping %s: not found in %s", message_id, messages_path)
            continue
        message_row = messages_df.loc[message_id]

        independent = None
        if independent_opinion:
            try:
                independent = _independent_route(message_row, judge_model)
            except Exception as exc:  # a failed second opinion shouldn't block grading the first
                err = classify_llm_error(exc)
                logger.warning("independent re-route failed for %s, grading without it: %s", message_id, err)

        context_json = _build_judge_context(message_row, decision_row, independent)
        try:
            verdict = _judge_one(context_json, judge_model)
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
                "agrees_with_independent": (independent.action == decision_row["action"]) if independent else None,
            }
        except OrchestrateError as exc:
            # Covers both LLM-call failures (classify_llm_error) and a malformed verdict
            # that failed JudgeVerdict's schema (DecisionParseError) -- both wrapped the
            # same way by _judge_one, so one handler here is enough.
            logger.warning("judge failed to score %s: %s", message_id, exc)
            record = {"message_id": message_id, "ok": False, "error": exc.user_message}
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

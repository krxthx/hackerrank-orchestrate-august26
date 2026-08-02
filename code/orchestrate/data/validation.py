"""Batch-level validation of a finished output.csv.

Complements RoutingDecision's per-row schema validation (types.py, enforced at write time
by pipeline.parse_result) with checks that only make sense across the whole file: every
input message routed exactly once, no duplicates, and every cited evidence_message_ids
value is a real historical message that actually belongs to that row's user -- the router
is instructed to only cite IDs the get_message_history tool returned it, but nothing
downstream re-checked that until now.

Diagnostic, not enforcement: run_pipeline still writes output.csv regardless of what this
finds -- call validate_output() afterward (main.py does, by default) and surface the
report rather than blocking a run on it.
"""

import os
from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError

from orchestrate.core.errors import DatasetError
from orchestrate.core.types import RoutingDecision
from orchestrate.data.dataset import Dataset, get_dataset
from orchestrate.data.evidence import parse_evidence_ids


@dataclass
class ValidationIssue:
    message_id: str | None
    problem: str


@dataclass
class ValidationReport:
    input_count: int
    output_count: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> dict:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "issue_count": len(self.issues),
            "ok": self.ok,
            "issues": [{"message_id": i.message_id, "problem": i.problem} for i in self.issues],
        }


def validate_output(
    input_path: str,
    output_path: str,
    dataset: Dataset | None = None,
) -> ValidationReport:
    if not os.path.exists(input_path):
        raise DatasetError(f"Validation input not found: {input_path}.")
    if not os.path.exists(output_path):
        raise DatasetError(f"Validation output not found: {output_path}.")

    input_df = pd.read_csv(input_path)
    output_df = pd.read_csv(output_path)
    ds = dataset or get_dataset()

    issues: list[ValidationIssue] = []

    input_ids = input_df["message_id"]
    output_ids = output_df["message_id"]

    for message_id in output_ids[output_ids.duplicated()].unique():
        issues.append(ValidationIssue(message_id, "duplicate message_id in output"))

    for message_id in sorted(set(input_ids) - set(output_ids)):
        issues.append(ValidationIssue(message_id, "present in input but missing from output"))

    for message_id in sorted(set(output_ids) - set(input_ids)):
        issues.append(ValidationIssue(message_id, "not present in input (unexpected row)"))

    user_by_message = input_df.set_index("message_id")["user_id"]
    for _, row in output_df.iterrows():
        message_id = row["message_id"]

        try:
            RoutingDecision(**row.to_dict())
        except ValidationError as exc:
            issues.append(ValidationIssue(message_id, f"row fails RoutingDecision schema: {exc}"))
            continue  # confidence/enum fields are untrustworthy below this point for this row

        evidence_ids = parse_evidence_ids(row.get("evidence_message_ids"))
        if not evidence_ids:
            continue
        user_id = user_by_message.get(message_id)
        if user_id is None:
            continue  # already reported above as "unexpected row"
        valid_ids = set(ds.message_history.loc[ds.message_history.user_id == user_id, "message_id"])
        for bad_id in sorted(evidence_ids - valid_ids):
            issues.append(
                ValidationIssue(
                    message_id,
                    f"cites evidence_message_ids={bad_id!r}, which is not a real historical "
                    f"message for user {user_id!r}",
                )
            )

    return ValidationReport(input_count=len(input_df), output_count=len(output_df), issues=issues)

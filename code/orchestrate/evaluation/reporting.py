"""Serialization for evaluation reports."""

import json
import os

from orchestrate.evaluation.judge import JudgeEvalResult
from orchestrate.evaluation.sample import SampleEvalResult


def write_report(sample_result: SampleEvalResult | None, judge_result: JudgeEvalResult | None, path: str) -> None:
    report = {}
    if sample_result is not None:
        report["sample_eval"] = {"summary": sample_result.summary(), "rows": sample_result.rows}
    if judge_result is not None:
        report["judge_eval"] = {"summary": judge_result.summary(), "rows": judge_result.rows}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as file:
        json.dump(report, file, indent=2, default=str)

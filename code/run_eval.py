#!/usr/bin/env python3
"""CLI for the evaluation pipeline.

Usage:
    python code/run_eval.py sample            # hard metrics vs dataset/sample_messages.csv
    python code/run_eval.py judge             # rubric-judge pass over dataset/output.csv
    python code/run_eval.py all               # both
    python code/run_eval.py judge --limit 10  # smoke test on the first N rows

Writes a JSON report to data/output/eval_report.json (per-row detail + aggregate summary)
and prints the summary to stdout.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from orchestrate.config import DATA_OUTPUT_DIR, JUDGE_MODEL  # noqa: E402
from orchestrate.evaluate import run_judge_eval, run_sample_eval, write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Message Notification Router.")
    parser.add_argument("mode", choices=["sample", "judge", "all"])
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows.")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="litellm model string for the judge pass.")
    parser.add_argument(
        "--report", default=os.path.join(DATA_OUTPUT_DIR, "eval_report.json"), help="Where to write the JSON report."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sample_result = run_sample_eval(limit=args.limit) if args.mode in ("sample", "all") else None
    judge_result = run_judge_eval(judge_model=args.judge_model, limit=args.limit) if args.mode in ("judge", "all") else None

    write_report(sample_result, judge_result, args.report)
    print(f"\nWrote {args.report}\n")

    if sample_result is not None:
        print("== sample_eval (vs dataset/sample_messages.csv) ==")
        print(json.dumps(sample_result.summary(), indent=2))
    if judge_result is not None:
        print("\n== judge_eval (rubric judge over dataset/output.csv) ==")
        print(json.dumps(judge_result.summary(), indent=2))


if __name__ == "__main__":
    main()

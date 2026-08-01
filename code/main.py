#!/usr/bin/env python3
"""CLI entry point for the Message Notification Router.

Usage:
    python code/main.py
    python code/main.py --input dataset/messages.csv --output dataset/output.csv

Reads dataset/messages.csv, runs the router agent over every row, and writes predictions to
dataset/output.csv (the actual submission file) by default. Also drops a copy in
data/output/output.csv, matching the scaffold's original convention, for diffing runs
without touching the submission file.
"""

import argparse
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

from orchestrate.config import DATA_OUTPUT_DIR, DATASET_DIR  # noqa: E402
from orchestrate.constants import DEFAULT_OUTPUT_FILENAME  # noqa: E402
from orchestrate.errors import fatal_error_boundary  # noqa: E402
from orchestrate.pipeline import run_pipeline  # noqa: E402
from orchestrate.validate import validate_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Message Notification Router pipeline.")
    parser.add_argument("--input", default=os.path.join(DATASET_DIR, "messages.csv"))
    parser.add_argument("--output", default=os.path.join(DATASET_DIR, "output.csv"))
    parser.add_argument(
        "--no-scratch-copy",
        action="store_true",
        help="Skip writing the extra copy to data/output/output.csv.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore the matching resumable checkpoint and reroute every input row.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-run output validation pass.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with fatal_error_boundary():
        scratch_path = os.path.join(DATA_OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME)
        written = run_pipeline(input_path=args.input, output_path=scratch_path, fresh=args.fresh)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        shutil.copyfile(written, args.output)
        print(f"Wrote {args.output}" + ("" if args.no_scratch_copy else f" (scratch copy: {written})"))

        if args.no_scratch_copy and os.path.abspath(written) != os.path.abspath(args.output):
            os.remove(written)

        if not args.no_validate:
            report = validate_output(args.input, args.output)
            if report.ok:
                print(f"Validation OK: {report.output_count} rows, no issues found.")
            else:
                print(f"Validation found {len(report.issues)} issue(s) in {report.output_count} rows:")
                for issue in report.issues:
                    print(f"  - {issue.message_id}: {issue.problem}")


if __name__ == "__main__":
    main()

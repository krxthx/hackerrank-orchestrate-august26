"""Batch runner: input file -> agent per record -> output CSV.

Last year's Orchestrate challenge ("support-agent") was graded partly on an
output CSV, so this skeleton assumes the deliverable is "run the agent over
every row of some input and write results out." Adjust `build_user_input`
and `parse_result` once you see the real problem statement -- the loop
itself (read -> agent -> collect -> write) should stay the same.
"""

import os

import pandas as pd

from orchestrate.agent import run_agent
from orchestrate.config import DATA_OUTPUT_DIR
from orchestrate.constants import DEFAULT_OUTPUT_FILENAME
from orchestrate.prompts import DEFAULT_SYSTEM_PROMPT
from orchestrate.transcript import TranscriptLogger
from orchestrate.types import PipelineRecord


def build_user_input(row: pd.Series) -> str:
    """Turn one input row into the text handed to the agent. Customize per-challenge."""
    return row.to_json()


def parse_result(agent_output: str, row: pd.Series) -> PipelineRecord:
    """Turn the agent's final text into an output row. Customize per-challenge."""
    return PipelineRecord(
        row_id=row.get("id"),
        input_text=row.to_json(),
        output_text=agent_output,
    )


def run_pipeline(input_path: str, output_filename: str = DEFAULT_OUTPUT_FILENAME) -> str:
    df = pd.read_csv(input_path)
    logger = TranscriptLogger()

    records: list[PipelineRecord] = []
    for _, row in df.iterrows():
        user_input = build_user_input(row)
        result = run_agent(DEFAULT_SYSTEM_PROMPT, user_input, logger=logger)
        records.append(parse_result(result.final_text, row))

    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(DATA_OUTPUT_DIR, output_filename)
    pd.DataFrame([r.model_dump() for r in records]).to_csv(output_path, index=False)
    return output_path

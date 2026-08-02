"""Auto-logging for every LLM call.

The "chat transcript" is a graded submission artifact, so every run writes a
full record of what was sent/received without any extra effort at hackathon time.
"""

import json
import os
from datetime import datetime, timezone

from orchestrate.core.config import TRANSCRIPTS_DIR
from orchestrate.core.constants import TRANSCRIPT_TIME_FORMAT


class TranscriptLogger:
    def __init__(self, run_name: str | None = None):
        os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
        run_name = run_name or datetime.now(timezone.utc).strftime(TRANSCRIPT_TIME_FORMAT)
        self.jsonl_path = os.path.join(TRANSCRIPTS_DIR, f"{run_name}.jsonl")
        self.md_path = os.path.join(TRANSCRIPTS_DIR, f"{run_name}.md")
        with open(self.md_path, "w") as f:
            f.write(f"# Transcript: {run_name}\n\n")

    def log_step(self, *, messages: list[dict], response_text: str, tool_calls: list[dict] | None = None):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages": messages,
            "response_text": response_text,
            "tool_calls": tool_calls or [],
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        with open(self.md_path, "a") as f:
            last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
            if last_user:
                f.write(f"**User/Tool input:**\n\n```\n{last_user['content']}\n```\n\n")
            f.write(f"**Assistant:**\n\n{response_text}\n\n")
            if tool_calls:
                f.write(f"**Tool calls:** {json.dumps(tool_calls)}\n\n")
            f.write("---\n\n")

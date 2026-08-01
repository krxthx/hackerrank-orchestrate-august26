"""Batch runner: dataset/messages.csv -> router agent per message -> output.csv.

Each message gets its own agent run (agent.run_agent): the model calls the dataset lookup
tools in tools.py as needed, then returns a single JSON routing decision, which is validated
against types.RoutingDecision before being written out.
"""

import base64
import json
import logging
import mimetypes
import os
import re

import pandas as pd
from pydantic import ValidationError

from orchestrate import transcribe
from orchestrate.agent import run_agent
from orchestrate.config import (
    DATA_OUTPUT_DIR,
    DATASET_DIR,
    ROUTER_MAX_STEPS,
    SEND_AUDIO_INLINE,
    TRANSCRIBE_VOICE_NOTES,
)
from orchestrate.constants import DEFAULT_OUTPUT_FILENAME
from orchestrate.data import get_dataset
from orchestrate.prompts import DEFAULT_SYSTEM_PROMPT
from orchestrate.transcript import TranscriptLogger
from orchestrate.types import RoutingDecision

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_content(row: pd.Series) -> str | list[dict]:
    """One incoming message -> the content handed to the router agent.

    Plain text for text-only messages; a multimodal content list (text + inline
    image/audio) when the message carries media, so the model reasons over the actual
    file rather than just its caption/metadata.
    """
    fields = {
        "message_id": row["message_id"],
        "user_id": row["user_id"],
        "conversation_type": row["conversation_type"],
        "group_id": _none_if_nan(row.get("group_id")),
        "business_id": _none_if_nan(row.get("business_id")),
        "sender_user_id": _none_if_nan(row.get("sender_user_id")),
        "created_at": row["created_at"],
        "message_text": _none_if_nan(row.get("message_text")),
        "media_type": _none_if_nan(row.get("media_type")),
        "forwarded_count": int(row["forwarded_count"]),
    }
    media_type = fields["media_type"]
    media_id = row.get("media_id")
    path = get_dataset().media_path(media_type, media_id) if media_type and not pd.isna(media_id) else None
    if media_type and not pd.isna(media_id) and (not path or not os.path.exists(path)):
        logger.warning("media file not found for %s (%s): %s", row["message_id"], media_id, path)
        path = None

    send_audio_inline = SEND_AUDIO_INLINE
    if media_type == "voice" and path:
        if TRANSCRIBE_VOICE_NOTES:
            try:
                fields["voice_transcript"] = transcribe.transcribe(media_id, path)
                send_audio_inline = False  # transcript covers it; no need to also send raw audio
            except Exception as exc:  # local transcription failing shouldn't crash the run
                logger.warning("transcription failed for %s (%s): %s", row["message_id"], media_id, exc)
        if not TRANSCRIBE_VOICE_NOTES and not SEND_AUDIO_INLINE:
            fields["voice_note_content"] = "unavailable -- reason from context (sender/group/history) only"

    text = (
        "Route this incoming message. Use the tools to gather context on the receiving "
        "user (and the group/business/history as relevant) before deciding, then respond "
        "with only the final JSON object.\n\n" + json.dumps(fields, indent=2)
    )

    if not path or (media_type == "voice" and not send_audio_inline):
        return text

    with open(path, "rb") as f:
        data_b64 = base64.b64encode(f.read()).decode("utf-8")

    if media_type == "image":
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data_b64}"}},
        ]
    if media_type == "voice":
        fmt = os.path.splitext(path)[1].lstrip(".") or "mp3"
        return [
            {"type": "text", "text": text},
            {"type": "input_audio", "input_audio": {"data": data_b64, "format": fmt}},
        ]
    return text


def _none_if_nan(value):
    return None if pd.isna(value) else value


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
    return json.loads(match.group(0))


def _fallback_decision(message_id: str, reason: str) -> RoutingDecision:
    return RoutingDecision(
        message_id=message_id,
        action="digest",
        message_type="unknown",
        reason=f"Fallback: {reason}",
        confidence=0.3,
        evidence_message_ids="none",
    )


def parse_result(agent_output: str, message_id: str) -> RoutingDecision:
    """Turn the agent's final text into a validated output row. Falls back to a safe,
    low-confidence 'unknown' decision rather than crashing the batch on one bad row.
    """
    try:
        payload = _extract_json(agent_output)
        payload.setdefault("message_id", message_id)
        return RoutingDecision(**payload)
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.warning("failed to parse routing decision for %s: %s", message_id, exc)
        return _fallback_decision(message_id, f"could not parse a valid routing decision ({exc}).")


def _checkpoint_path(input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(os.path.dirname(DATA_OUTPUT_DIR), "cache", f"{base}_checkpoint.jsonl")


def _load_checkpoint(path: str) -> dict[str, RoutingDecision]:
    if not os.path.exists(path):
        return {}
    decisions = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            decisions[record["message_id"]] = RoutingDecision(**record)
    return decisions


def run_pipeline(
    input_path: str = os.path.join(DATASET_DIR, "messages.csv"),
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    output_path: str | None = None,
) -> str:
    """Runs the router agent over every row of input_path. Resumable: each decision is
    appended to a checkpoint file as soon as it's made, so a crash (network blip, provider
    outage, etc) only costs the one in-flight row, not the whole batch -- rerunning picks up
    where it left off instead of re-calling the LLM for already-decided messages.
    """
    df = pd.read_csv(input_path)
    run_logger = TranscriptLogger()

    checkpoint_file = _checkpoint_path(input_path)
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    checkpoint = _load_checkpoint(checkpoint_file)
    if checkpoint:
        logger.info("resuming from checkpoint: %d/%d already decided", len(checkpoint), len(df))

    decisions: list[RoutingDecision] = []
    with open(checkpoint_file, "a") as ckpt:
        for i, row in df.iterrows():
            message_id = row["message_id"]
            if message_id in checkpoint:
                decision = checkpoint[message_id]
            else:
                try:
                    content = build_user_content(row)
                    result = run_agent(
                        DEFAULT_SYSTEM_PROMPT,
                        content,
                        logger=run_logger,
                        max_steps=ROUTER_MAX_STEPS,
                    )
                    decision = parse_result(result.final_text, message_id)
                except Exception as exc:  # a single message's total failure shouldn't kill the batch
                    logger.warning("agent run failed for %s, using fallback: %s", message_id, exc)
                    decision = _fallback_decision(message_id, f"agent run failed ({exc}).")
                ckpt.write(json.dumps(decision.model_dump()) + "\n")
                ckpt.flush()

            decisions.append(decision)
            logger.info("[%d/%d] %s -> %s/%s", i + 1, len(df), decision.message_id, decision.action, decision.message_type)

    out_df = pd.DataFrame([d.model_dump() for d in decisions])[OUTPUT_COLUMNS]

    final_path = output_path or os.path.join(DATA_OUTPUT_DIR, output_filename)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    out_df.to_csv(final_path, index=False)
    return final_path

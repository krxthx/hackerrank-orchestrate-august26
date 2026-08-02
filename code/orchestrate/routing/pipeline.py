"""Batch runner: dataset/messages.csv -> router agent per message -> output.csv.

Each message gets its own agent run (agent.run_agent): the model calls the dataset lookup
tools in tools.py as needed, then returns a single JSON routing decision, which is validated
against types.RoutingDecision before being written out.
"""

import base64
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from orchestrate.core.config import (
    API_BASE,
    DATA_OUTPUT_DIR,
    DATASET_DIR,
    MODEL,
    ROUTER_MAX_STEPS,
    SEND_AUDIO_INLINE,
    TRANSCRIBE_VOICE_NOTES,
    WHISPER_COMPUTE_TYPE,
    WHISPER_MODEL_SIZE,
)
from orchestrate.core.constants import DEFAULT_OUTPUT_FILENAME
from orchestrate.core.errors import (
    ContentFilterBlockedError,
    DatasetError,
    classify_llm_error,
    describe_parse_error,
    describe_step_limit,
)
from orchestrate.core.parsing import extract_json_object
from orchestrate.core.types import RoutingDecision
from orchestrate.data.dataset import get_dataset
from orchestrate.data.signals import scam_keyword_signals
from orchestrate.prompts import DEFAULT_SYSTEM_PROMPT
from orchestrate.routing.agent import run_agent
from orchestrate.runtime import transcription as transcribe
from orchestrate.runtime.transcript import TranscriptLogger

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

_CHECKPOINT_FINGERPRINT_VERSION = 2
_CONTEXT_FILENAMES = (
    "users.csv",
    "groups.csv",
    "group_members.csv",
    "business_accounts.csv",
    "user_business_history.csv",
    "message_history.csv",
    "message_events.csv",
    "images.csv",
    "voice_notes.csv",
    "daily_notification_summary.csv",
)
_ROUTING_SOURCE_DIRS = ("core", "llm", "routing")
_ROUTING_SOURCE_FILES = (
    os.path.join("data", "dataset.py"),
    os.path.join("data", "evidence.py"),
    os.path.join("prompts", "system.py"),
    os.path.join("runtime", "transcription.py"),
)


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

    scam_text = fields.get("message_text") or fields.get("voice_transcript")
    fields["scam_keyword_signals"] = scam_keyword_signals(scam_text)

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


def _fallback_decision(message_id: str, reason: str) -> RoutingDecision:
    return RoutingDecision(
        message_id=message_id,
        action="digest",
        message_type="unknown",
        reason=f"Fallback: {reason}",
        confidence=0.3,
        evidence_message_ids="none",
    )


def _content_blocked_decision(message_id: str, detail: str) -> RoutingDecision:
    """A request that the upstream provider/proxy itself refuses to process is, in this
    domain, informative: content that trips a safety/content filter is itself a signal of
    phishing/scam-style material. Treat the block as evidence rather than discarding it into
    a generic low-confidence fallback.
    """
    return RoutingDecision(
        message_id=message_id,
        action="mute",
        message_type="scam",
        reason=(
            f"{detail} -- itself a strong signal of phishing or scam-style content, so this "
            "is muted as a precaution rather than left unrouted."
        ),
        confidence=0.6,
        evidence_message_ids="none",
    )


def parse_result(agent_output: str, message_id: str) -> RoutingDecision:
    """Turn the agent's final text into a validated output row. Falls back to a safe,
    low-confidence 'unknown' decision rather than crashing the batch on one bad row.
    """
    try:
        payload = extract_json_object(agent_output, label="model output")
        payload.setdefault("message_id", message_id)
        return RoutingDecision(**payload)
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        logger.warning("failed to parse routing decision for %s: %s", message_id, exc)
        return _fallback_decision(message_id, describe_parse_error(exc))


def _update_hash_from_file(digest, label: str, path: Path) -> None:
    """Add a file to a deterministic fingerprint without storing its path or contents."""
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    if not path.is_file():
        digest.update(b"missing\0")
        return
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")


def _routing_source_paths(package_root: Path | None = None) -> list[Path]:
    """Return every Python source file that can affect a routing decision."""
    root = package_root or Path(__file__).resolve().parents[1]
    paths = [root / relative_path for relative_path in _ROUTING_SOURCE_FILES]
    for directory in _ROUTING_SOURCE_DIRS:
        paths.extend((root / directory).rglob("*.py"))
    return sorted(set(paths))


def _checkpoint_fingerprint(input_path: str) -> str:
    """Identify everything that can materially change routing decisions.

    The same fingerprint is safe to resume after an interruption. Changes to input/context
    data, media, routing code/prompt, model endpoint identity, or media-processing settings
    create a new checkpoint namespace automatically. Secrets, pacing, output paths, and
    logging settings are deliberately excluded.
    """
    digest = hashlib.sha256()
    settings = {
        "fingerprint_version": _CHECKPOINT_FINGERPRINT_VERSION,
        "model": MODEL,
        "api_base": API_BASE,
        "router_max_steps": ROUTER_MAX_STEPS,
        "send_audio_inline": SEND_AUDIO_INLINE,
        "transcribe_voice_notes": TRANSCRIBE_VOICE_NOTES,
        "whisper_model_size": WHISPER_MODEL_SIZE,
        "whisper_compute_type": WHISPER_COMPUTE_TYPE,
    }
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")

    _update_hash_from_file(digest, "input", Path(input_path))
    dataset_root = Path(DATASET_DIR)
    for filename in _CONTEXT_FILENAMES:
        _update_hash_from_file(digest, f"context:{filename}", dataset_root / filename)

    media_root = dataset_root / "media"
    if media_root.is_dir():
        for media_path in sorted(path for path in media_root.rglob("*") if path.is_file()):
            label = media_path.relative_to(dataset_root).as_posix()
            _update_hash_from_file(digest, f"media:{label}", media_path)
    else:
        digest.update(b"media:missing\0")

    package_root = Path(__file__).resolve().parents[1]
    for source_path in _routing_source_paths(package_root):
        label = source_path.relative_to(package_root).as_posix()
        _update_hash_from_file(digest, f"source:{label}", source_path)
    return digest.hexdigest()


def _checkpoint_path(input_path: str, fingerprint: str | None = None) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    fingerprint = fingerprint or _checkpoint_fingerprint(input_path)
    return os.path.join(
        os.path.dirname(DATA_OUTPUT_DIR),
        "cache",
        f"{base}_checkpoint_{fingerprint[:12]}.jsonl",
    )


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
    fresh: bool = False,
) -> str:
    """Runs the router agent over every row of input_path. Resumable: each decision is
    appended to a checkpoint file as soon as it's made, so a crash (network blip, provider
    outage, etc) only costs the one in-flight row, not the whole batch -- rerunning picks up
    where it left off instead of re-calling the LLM for already-decided messages.
    """
    if not os.path.exists(input_path):
        raise DatasetError(f"Input file not found: {input_path}.")
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"Input file is empty or unreadable: {input_path}.", cause=exc) from exc
    run_logger = TranscriptLogger()

    fingerprint = _checkpoint_fingerprint(input_path)
    checkpoint_file = _checkpoint_path(input_path, fingerprint)
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    checkpoint = {} if fresh else _load_checkpoint(checkpoint_file)
    if fresh and os.path.exists(checkpoint_file):
        logger.info("fresh run requested; replacing checkpoint %s", checkpoint_file)
    if checkpoint:
        logger.info(
            "resuming checkpoint %s: %d/%d already decided",
            fingerprint[:12],
            len(checkpoint),
            len(df),
        )

    decisions: list[RoutingDecision] = []
    with open(checkpoint_file, "w" if fresh else "a") as ckpt:
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
                    if result.hit_step_limit:
                        decision = _fallback_decision(message_id, describe_step_limit(ROUTER_MAX_STEPS))
                    else:
                        decision = parse_result(result.final_text, message_id)
                except DatasetError:
                    # Fatal -- a missing/malformed dataset file will fail identically on
                    # every remaining row, so stop the batch instead of silently
                    # fallback-ing through all of it.
                    raise
                except Exception as exc:  # a single message's total failure shouldn't kill the batch
                    err = classify_llm_error(exc)
                    if isinstance(err, ContentFilterBlockedError):
                        logger.warning("agent run for %s blocked upstream, treating as scam signal: %s", message_id, err)
                        decision = _content_blocked_decision(message_id, err.user_message)
                    else:
                        logger.warning("agent run failed for %s, using fallback: %s", message_id, err)
                        decision = _fallback_decision(message_id, err.user_message)
                ckpt.write(json.dumps(decision.model_dump()) + "\n")
                ckpt.flush()

            decisions.append(decision)
            logger.info("[%d/%d] %s -> %s/%s", i + 1, len(df), decision.message_id, decision.action, decision.message_type)

    out_df = pd.DataFrame([d.model_dump() for d in decisions])[OUTPUT_COLUMNS]

    final_path = output_path or os.path.join(DATA_OUTPUT_DIR, output_filename)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    out_df.to_csv(final_path, index=False)
    return final_path

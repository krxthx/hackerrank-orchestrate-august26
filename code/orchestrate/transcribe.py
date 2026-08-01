"""Local speech-to-text for voice notes, via faster-whisper.

Decouples voice-note reasoning from whichever LLM is configured: not every model accepts
raw audio (confirmed: the MiniMax-via-proxy setup rejects it), but every model can read a
text transcript. Runs fully locally/offline once the model weights are cached -- no extra
API key or cost.

Transcripts are cached to disk (keyed by voice_note_id) since the dataset only has ~14 fixed
files and re-transcribing on every pipeline run would be wasted work.
"""

import json
import logging
import os
from functools import lru_cache

from orchestrate.config import DATA_OUTPUT_DIR, WHISPER_COMPUTE_TYPE, WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(os.path.dirname(DATA_OUTPUT_DIR), "cache", "voice_transcripts.json")


@lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel

    logger.info("loading faster-whisper model %s (%s)...", WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE)
    return WhisperModel(WHISPER_MODEL_SIZE, compute_type=WHISPER_COMPUTE_TYPE)


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def transcribe(voice_note_id: str, file_path: str) -> str:
    """Return the transcript for a voice note, using a disk cache keyed by voice_note_id."""
    cache = _load_cache()
    if voice_note_id in cache:
        return cache[voice_note_id]

    model = _get_model()
    segments, _info = model.transcribe(file_path, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments).strip()

    cache[voice_note_id] = text
    _save_cache(cache)
    return text

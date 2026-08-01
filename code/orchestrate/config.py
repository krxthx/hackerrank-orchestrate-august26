import os

from dotenv import load_dotenv

load_dotenv()

# litellm model string. Swap providers by changing this one value (or the
# ORCHESTRATE_MODEL env var) -- everything else in the codebase is provider-agnostic.
#   Anthropic : claude-sonnet-4-5-20250929
#   Gemini    : gemini/gemini-2.5-pro
#   Ollama    : ollama/llama3.1
#   OpenAI    : gpt-4.1
MODEL = os.environ.get("ORCHESTRATE_MODEL", "claude-sonnet-4-5-20250929")

# Optional: point at any OpenAI-compatible endpoint (vLLM, Together, Groq, LM Studio, a
# local proxy, etc). Set MODEL to "openai/<model-name>", this to the endpoint's base URL, and
# API_KEY to whatever that endpoint expects. Leave both unset to use the provider's default
# endpoint/key (Anthropic/Gemini/OpenAI/Ollama as picked by MODEL).
API_BASE = os.environ.get("ORCHESTRATE_API_BASE") or None
API_KEY = os.environ.get("ORCHESTRATE_API_KEY") or None

# Model used for evaluate.py's rubric-judge pass. Deliberately separate from MODEL (the
# router's model) so grading isn't done by the same model/provider that made the decisions --
# reduces self-grading bias. litellm-agnostic like MODEL -- point it at whatever
# provider/endpoint you have access to (set ORCHESTRATE_JUDGE_API_BASE/_KEY below if needed).
JUDGE_MODEL = os.environ.get("ORCHESTRATE_JUDGE_MODEL", "openrouter/openai/gpt-oss-20b:free")
JUDGE_API_BASE = os.environ.get("ORCHESTRATE_JUDGE_API_BASE") or None
JUDGE_API_KEY = os.environ.get("ORCHESTRATE_JUDGE_API_KEY") or None

MAX_AGENT_STEPS = int(os.environ.get("ORCHESTRATE_MAX_STEPS", "12"))

TRANSCRIPTS_DIR = "transcripts"
DATA_INPUT_DIR = "data/input"
DATA_OUTPUT_DIR = "data/output"

# Participant-facing dataset for this challenge (Message Notification Router).
DATASET_DIR = "dataset"

# How many LLM calls (tool-call steps + final answer) the router agent may make per message.
# There are ~4 useful lookups per message (user, group-or-business, evidence history, daily
# load), so this is deliberately tight -- a run hitting the ceiling signals a prompt problem,
# not a legitimate need for more steps.
ROUTER_MAX_STEPS = int(os.environ.get("ORCHESTRATE_ROUTER_MAX_STEPS", "6"))

# Seconds to sleep between LLM calls, to stay under free-tier RPM caps (e.g. Gemini's free
# tier). Applies per LLM call, not per message, since the tool loop makes several calls/message.
LLM_CALL_PACING_SECONDS = float(os.environ.get("ORCHESTRATE_LLM_PACING_SECONDS", "4.0"))

# Whether to attach voice-note audio inline as a native audio content block. Not every model
# accepts audio input (confirmed: Gemini does, the MiniMax-via-proxy setup does not -- it
# raises BadRequestError). When false, voice rows fall back to context-only reasoning (no
# audio content, same as if a transcript were never available) instead of crashing the run.
# Flip back to true once a transcription step (e.g. faster-whisper) replaces raw audio with
# a real text transcript, or when using a model confirmed to accept audio.
SEND_AUDIO_INLINE = os.environ.get("ORCHESTRATE_SEND_AUDIO_INLINE", "true").lower() == "true"

# Local speech-to-text for voice notes (faster-whisper), used instead of/alongside native
# audio input so voice-note reasoning works regardless of which LLM is configured.
TRANSCRIBE_VOICE_NOTES = os.environ.get("ORCHESTRATE_TRANSCRIBE_VOICE_NOTES", "true").lower() == "true"
WHISPER_MODEL_SIZE = os.environ.get("ORCHESTRATE_WHISPER_MODEL_SIZE", "base")
WHISPER_COMPUTE_TYPE = os.environ.get("ORCHESTRATE_WHISPER_COMPUTE_TYPE", "int8")

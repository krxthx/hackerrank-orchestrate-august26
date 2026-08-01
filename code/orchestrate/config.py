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

MAX_AGENT_STEPS = int(os.environ.get("ORCHESTRATE_MAX_STEPS", "12"))

TRANSCRIPTS_DIR = "transcripts"
DATA_INPUT_DIR = "data/input"
DATA_OUTPUT_DIR = "data/output"

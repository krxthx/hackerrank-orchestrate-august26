"""Prompt text lives here, separate from code, so it's fast to iterate on
during the hackathon without touching the loop/pipeline logic.

Replace DEFAULT_SYSTEM_PROMPT with the real task's system prompt once the
problem statement arrives. Keep instructions explicit about output format --
the deterministic, hallucination-free framing in the scoring rubric rewards
agents that refuse to guess rather than produce plausible-looking noise.
"""

DEFAULT_SYSTEM_PROMPT = """\
You are an assistant that processes one record at a time and returns a single,
well-formed answer. Follow the output format exactly as instructed. If you are
uncertain or the input lacks enough information, say so explicitly rather than
guessing.
"""

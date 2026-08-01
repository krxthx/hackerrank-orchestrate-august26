"""Shared parsing for model output: both the router and the judge are told to respond with
"ONLY a single JSON object", possibly still wrapped in a markdown code fence. Used by
pipeline.py (router decisions) and evaluate.py (judge verdicts) so both share one
implementation of "find the JSON object in this text" instead of drifting apart.
"""

import json


def extract_json_object(text: str, *, label: str = "model output") -> dict:
    """Strip an optional markdown code fence, then scan for the first balanced top-level
    `{...}` object and parse it. A balanced scan (rather than a greedy regex spanning the
    first `{` to the last `}`) avoids swallowing unrelated trailing/leading braces if the
    model ever emits stray text around the JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if text.lower().startswith("json") else text

    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in {label}: {text[:200]!r}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError(f"no JSON object found in {label}: {text[:200]!r}")

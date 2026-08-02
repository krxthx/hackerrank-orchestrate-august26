"""Deterministic, code-computed signals for patterns that are common and unambiguous
enough not to leave entirely to model judgement -- the OTP/password/urgent-account-threat
wording that drives most scam mutes, and whether a message landed inside the receiving
user's quiet hours. Surfaced as explicit fields for the model to read, instead of being
re-derived (inconsistently, message to message) purely from raw text.
"""

import re
from datetime import datetime

_SCAM_PATTERNS = {
    "credential_request": re.compile(r"\b(otp|one[- ]time password|cvv|pin code|password)\b", re.I),
    "account_threat": re.compile(
        r"\b(block|blocked|suspend|suspended|expir\w*|deactivat\w*|freeze|frozen)\w*\b"
        r"[^.]{0,40}\b(account|access|profile|card|wallet)\b"
        r"|\b(account|access|profile|card|wallet)\b[^.]{0,40}"
        r"\b(block|blocked|suspend|suspended|expir\w*|deactivat\w*|freeze|frozen)\w*\b",
        re.I,
    ),
    "urgency_pressure": re.compile(
        r"\b(act now|immediately|within \d+\s*(hour|hr|min)|last chance|verify now|confirm now|right away)\b",
        re.I,
    ),
    "prompt_injection": re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)\b", re.I),
}


def scam_keyword_signals(text: str | None) -> list[str]:
    """Which deterministic scam-pattern categories fire on this text, by name. Empty list
    means none matched -- absence of these patterns is not itself evidence of safety, just
    the absence of this particular deterministic signal.
    """
    if not text or not isinstance(text, str):
        return []
    return [name for name, pattern in _SCAM_PATTERNS.items() if pattern.search(text)]


def is_within_quiet_hours(created_at: str, dnd_window: str | None) -> bool | None:
    """Whether `created_at` ('YYYY-MM-DD HH:MM') falls inside `dnd_window` ('HH:MM-HH:MM',
    wrapping past midnight allowed, e.g. '22:00-07:00'). Returns None if either input is
    missing/unparseable rather than guessing.
    """
    if not dnd_window or not isinstance(dnd_window, str) or "-" not in dnd_window:
        return None
    try:
        message_time = datetime.strptime(created_at, "%Y-%m-%d %H:%M").time()
        start_s, end_s = dnd_window.split("-", 1)
        start = datetime.strptime(start_s.strip(), "%H:%M").time()
        end = datetime.strptime(end_s.strip(), "%H:%M").time()
    except ValueError:
        return None
    if start <= end:
        return start <= message_time < end
    return message_time >= start or message_time < end

"""The naive baseline's design choices (``BT_GUARDS=off``), kept in one place.

These are the common tutorial and low-code choices the guards replace (ADR 0007):

- a small hand-written map of zone labels; anything it does not know silently becomes the host's zone;
- tool errors reach the model as plain text, with the calendar's raw answer;
- a CRM meeting is written when the reply contains the word "booked", at the last ``book`` call's time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

#: Label → IANA zone. Deliberately small: it is what a tutorial agent ships with.
NAIVE_ZONES: Mapping[str, str] = {
    "eastern": "America/New_York",
    "est": "America/New_York",
    "edt": "America/New_York",
    "et": "America/New_York",
    "new york": "America/New_York",
    "toronto": "America/Toronto",
    "central": "America/Chicago",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "chicago": "America/Chicago",
    "mountain": "America/Denver",
    "mst": "America/Denver",
    "denver": "America/Denver",
    "pacific": "America/Los_Angeles",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "gmt": "UTC",
    "utc": "UTC",
    "uk": "Europe/London",
    "bst": "Europe/London",
    "london": "Europe/London",
    "cet": "Europe/Berlin",
    "cest": "Europe/Berlin",
    "central european": "Europe/Berlin",
    "berlin": "Europe/Berlin",
    "paris": "Europe/Paris",
    "ist": "Asia/Kolkata",
    "india": "Asia/Kolkata",
    "tokyo": "Asia/Tokyo",
    "jst": "Asia/Tokyo",
    "sydney": "Australia/Sydney",
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
}

_KEYS = sorted(NAIVE_ZONES, key=len, reverse=True)
_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in _KEYS) + r")\b", re.IGNORECASE)
_BOOKED_WORD = re.compile(r"\bbooked\b", re.IGNORECASE)


def naive_zone(text: str) -> str | None:
    """The first label of the map found in ``text`` (longest labels win), else ``None``."""
    match = _PATTERN.search(text)
    return NAIVE_ZONES[match[1].lower()] if match else None


def resolve_naive(text: str, host_zone: str) -> str:
    """The map's zone, or the host's zone when the map does not know the text (silently)."""
    return naive_zone(text) or host_zone


def reply_says_booked(reply: str) -> bool:
    """The naive CRM rule: the prose contains the word "booked"."""
    return _BOOKED_WORD.search(reply) is not None


def error_text(detail: str, *, what: str = "calendar") -> str:
    """A tool error as the naive agent passes it to the model: plain text with the raw detail."""
    detail = detail.strip() or "unknown error"
    return f"Error: {what} returned {detail}"

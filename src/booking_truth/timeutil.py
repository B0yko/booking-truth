"""Time helpers. Every datetime in this package is timezone-aware and compared in UTC."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """A clock frozen at one instant (tests and reproducible runs)."""

    def __init__(self, at: datetime) -> None:
        self._at = ensure_utc(at)

    def now(self) -> datetime:
        return self._at


class MutableClock:
    """A clock that tests can move forward."""

    def __init__(self, at: datetime) -> None:
        self._at = ensure_utc(at)

    def now(self) -> datetime:
        return self._at

    def set(self, at: datetime) -> None:
        self._at = ensure_utc(at)

    def advance(self, delta: timedelta) -> None:
        self._at = self._at + delta


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("naive datetime; an explicit timezone is required")
    return dt.astimezone(UTC)


def iso_z(dt: datetime) -> str:
    """Format as RFC 3339 UTC with a ``Z`` suffix and second precision."""
    return ensure_utc(dt).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_ms_z(dt: datetime) -> str:
    """Format as RFC 3339 UTC with millisecond precision (Cal.com style)."""
    u = ensure_utc(dt)
    return u.strftime("%Y-%m-%dT%H:%M:%S.") + f"{u.microsecond // 1000:03d}Z"


def parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 timestamp that carries an offset or ``Z``; reject naive input."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00") if value.endswith("Z") else value)
    return ensure_utc(dt)

"""Shared helpers for the calendar adapter tests: a real sandbox over HTTP and an adapter aimed at it."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import FastAPI

from booking_truth.calendars.calcom import CalcomAdapter
from booking_truth.sandbox.state import SandboxState
from booking_truth.timeutil import FixedClock

TOKEN = "calendar-test-token"
EVENT_TYPE_ID = 1001
HOST_ZONE = "America/New_York"
# Thursday 1 October 2026, 08:00 in New York (EDT, UTC-4).
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
# Monday 5 October 2026: working hours 09:00-17:00 EDT are 13:00-21:00 UTC.
MON_0900 = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
MON_0930 = MON_0900 + timedelta(minutes=30)
MON_1000 = MON_0900 + timedelta(hours=1)
MON_1700 = MON_0900 + timedelta(hours=8)
MON_DAY = (datetime(2026, 10, 5, tzinfo=UTC), datetime(2026, 10, 6, tzinfo=UTC))
HALF_HOUR = timedelta(minutes=30)
LEAD = "lena@example.com"
OTHER_LEAD = "omar@example.com"
LEAD_NAME = "Lena M"
LEAD_ZONE = "Europe/Berlin"
IDEM = "a" * 64
# A client timeout well below the sandbox's hang, so hangs are observed as client timeouts.
FAST = httpx.Timeout(0.25)
HANG_S = 0.8
NORMAL = httpx.Timeout(5.0)


@dataclass
class CalEnv:
    """One test's view of the shared sandbox server: its fresh state and a control client."""

    app: FastAPI
    state: SandboxState
    url: str
    control: httpx.Client

    def adapter(self, **overrides: Any) -> CalcomAdapter:
        options: dict[str, Any] = {"timeout": NORMAL, **overrides}
        api_key = options.pop("api_key", TOKEN)
        base_url = options.pop("base_url", self.url)
        return CalcomAdapter(base_url, api_key, EVENT_TYPE_ID, str(EVENT_TYPE_ID), HOST_ZONE, **options)

    def faults(self, *rules: dict[str, Any]) -> None:
        response = self.control.post("/_control/faults", json={"rules": list(rules)})
        assert response.status_code == 200, response.text

    def seed(self, **fields: Any) -> None:
        response = self.control.post("/_control/seed", json=fields)
        assert response.status_code == 200, response.text

    def snapshot(self) -> dict[str, Any]:
        response = self.control.get("/_state")
        assert response.status_code == 200, response.text
        data: dict[str, Any] = response.json()
        return data

    def log(self, group: str | None = None) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = self.snapshot()["request_log"]
        return [e for e in entries if group is None or e["group"] == group]

    def bookings(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = self.snapshot()["calcom"]["bookings"]
        return items

    def set_now(self, at: datetime) -> None:
        self.state.clock = FixedClock(at)

    def free_starts(self, start: datetime, end: datetime) -> list[datetime]:
        """The sandbox's own availability, half-open like the adapter's window."""
        return [s for s in self.state.free_starts(start, end) if s < end]

    @staticmethod
    def wait_for(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError("condition not reached in time")
            time.sleep(0.02)


def hang(group: str, mode: str = "timeout", **extra: Any) -> dict[str, Any]:
    """A fault rule that hangs longer than :data:`FAST`."""
    return {"group": group, "mode": mode, "hang_s": HANG_S, **extra}

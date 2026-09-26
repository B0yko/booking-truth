"""Fixtures for sandbox tests: one real HTTP server per session, a fresh state and clock per test."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer
from booking_truth.timeutil import MutableClock

TOKEN = "test-token"
# Thursday 1 October 2026, 08:00 in New York (EDT, UTC-4). Monday 5 October 09:00 EDT is 13:00 UTC.
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
SLOTS_VERSION = {"cal-api-version": "2024-09-04"}
BOOKING_VERSION = {"cal-api-version": "2024-08-13"}
LEAD = "lead@example.com"
MS_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
UID = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{22}$")
REASONS = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    500: "Internal Server Error",
    504: "Gateway Timeout",
}
CODES = {
    400: "BadRequestException",
    401: "UnauthorizedException",
    403: "ForbiddenException",
    404: "NotFoundException",
    500: "InternalServerErrorException",
    504: "GatewayTimeoutException",
}


@dataclass
class Sandbox:
    app: FastAPI
    state: SandboxState
    clock: MutableClock
    url: str
    client: httpx.Client
    token: str = TOKEN

    # Vendor calls ------------------------------------------------------------------------------

    def slots(self, headers: dict[str, str] | None = None, **params: Any) -> httpx.Response:
        query = {"eventTypeId": 1001, **params}
        chosen = SLOTS_VERSION if headers is None else headers
        return self.client.get("/v2/slots", params=query, headers=chosen)

    def book(
        self,
        start: str,
        *,
        email: str = LEAD,
        name: str = "Lena M",
        zone: str = "Europe/Berlin",
        version: str = "2024-08-13",
        timeout: float | None = None,
        **extra: Any,
    ) -> httpx.Response:
        body = {
            "start": start,
            "eventTypeId": 1001,
            "attendee": {"name": name, "email": email, "timeZone": zone},
            **extra,
        }
        kwargs: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        return self.client.post("/v2/bookings", json=body, headers={"cal-api-version": version}, **kwargs)

    def booked(self, start: str, **kwargs: Any) -> dict[str, Any]:
        response = self.book(start, **kwargs)
        assert response.status_code == 201, response.text
        data: dict[str, Any] = response.json()["data"]
        return data

    def get(self, uid: str, version: str = "2024-08-13") -> httpx.Response:
        return self.client.get(f"/v2/bookings/{uid}", headers={"cal-api-version": version})

    def list(self, version: str = "2024-08-13", **params: Any) -> httpx.Response:
        return self.client.get("/v2/bookings", params=params, headers={"cal-api-version": version})

    def reschedule(self, uid: str, body: dict[str, Any], version: str = "2024-08-13") -> httpx.Response:
        return self.client.post(
            f"/v2/bookings/{uid}/reschedule", json=body, headers={"cal-api-version": version}
        )

    def cancel(
        self, uid: str, body: dict[str, Any] | None = None, version: str = "2024-08-13"
    ) -> httpx.Response:
        payload = {} if body is None else body
        headers = {"cal-api-version": version}
        return self.client.post(f"/v2/bookings/{uid}/cancel", json=payload, headers=headers)

    # Assertions ----------------------------------------------------------------------------------

    @staticmethod
    def wait_for(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError("condition not reached in time")
            time.sleep(0.02)

    @staticmethod
    def assert_error(
        response: httpx.Response, status: int, message: str, *, path: str | None = None
    ) -> dict[str, Any]:
        """The standard Cal.com error envelope, with its key order."""
        assert response.status_code == status, response.text
        assert response.headers["content-type"] == "application/json; charset=utf-8"
        body: dict[str, Any] = response.json()
        assert list(body) == ["status", "timestamp", "path", "error"]
        assert body["status"] == "error"
        assert MS_Z.match(body["timestamp"])
        assert list(body["error"]) == ["code", "message", "details"]
        assert body["error"]["code"] == CODES[status]
        assert body["error"]["message"] == message
        assert list(body["error"]["details"]) == ["message", "error", "statusCode"]
        details = {"message": message, "error": REASONS[status], "statusCode": status}
        assert body["error"]["details"] == details
        if path is not None:
            assert body["path"] == path
        return body

    # Control calls -----------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        response = self.client.get("/_state")
        assert response.status_code == 200, response.text
        data: dict[str, Any] = response.json()
        return data

    def faults(self, *rules: dict[str, Any]) -> None:
        response = self.client.post("/_control/faults", json={"rules": list(rules)})
        assert response.status_code == 200, response.text

    def seed(self, **fields: Any) -> dict[str, Any]:
        response = self.client.post("/_control/seed", json=fields)
        assert response.status_code == 200, response.text
        data: dict[str, Any] = response.json()["seed"]
        return data

    def log(self, group: str | None = None) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = self.snapshot()["request_log"]
        return [e for e in entries if group is None or e["group"] == group]


@pytest.fixture(scope="session")
def sandbox_server() -> Iterator[tuple[FastAPI, BackgroundServer]]:
    app = create_sandbox_app(TOKEN, clock=MutableClock(NOW))
    with BackgroundServer(app) as server:
        yield app, server


@pytest.fixture
def sandbox(sandbox_server: tuple[FastAPI, BackgroundServer]) -> Iterator[Sandbox]:
    """A fresh state behind the shared server; the clock stays at ``NOW`` until a test moves it."""
    app, server = sandbox_server
    clock = MutableClock(NOW)
    state = SandboxState(clock=clock)
    app.state.sandbox = state
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with httpx.Client(base_url=server.url, headers=headers, timeout=5.0) as client:
        yield Sandbox(app=app, state=state, clock=clock, url=server.url, client=client)

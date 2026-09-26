"""Shared helpers for the agent tests: a real sandbox over HTTP and the agent app driven in-process."""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from booking_truth.agent.api import create_agent_app
from booking_truth.agent.core import AgentDeps
from booking_truth.agent.scripted import FakeLLM
from booking_truth.agent.tools import ToolExecutor, TurnContext
from booking_truth.calendars.base import CalendarAdapter
from booking_truth.calendars.calcom import CalcomAdapter
from booking_truth.config import Settings, load_settings
from booking_truth.llm.types import LLM
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer
from booking_truth.timeutil import Clock, FixedClock, MutableClock

TOKEN = "agent-test-token"
API_KEY = "agent-test-key"
EVENT_TYPE_ID = 1001
HOST_ZONE = "America/New_York"
# Thursday 1 October 2026, 08:00 in New York (EDT, UTC-4).
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
LEAD = "maya@example.com"
LEAD_NAME = "Maya R"


def agent_settings(sandbox_url: str, db_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "calendar": "calcom",
        "calcom_base_url": sandbox_url,
        "calcom_api_key": TOKEN,
        "calcom_event_type_id": EVENT_TYPE_ID,
        "crm": "none",
        "guards": "all",
        "db_path": db_path,
        "api_key": API_KEY,
        "expose_traces": True,
        "session_secret": "agent-test-secret",
    }
    values.update(overrides)
    return load_settings(**values)


@dataclass
class AgentEnv:
    """One test's agent app (in-process) wired to the shared sandbox (over HTTP)."""

    app: FastAPI
    sandbox_url: str
    control: httpx.Client
    state: SandboxState
    clock: Clock
    llm: LLM
    client: httpx.AsyncClient
    session: str = "s-1"
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1))

    @property
    def deps(self) -> AgentDeps:
        deps: AgentDeps = self.app.state.deps
        return deps

    def next_id(self) -> str:
        return f"m{next(self._ids)}"

    def body(
        self,
        *,
        message: str | None = None,
        action: dict[str, Any] | None = None,
        session: str | None = None,
        email: str = LEAD,
        name: str | None = LEAD_NAME,
        hint: str | None = None,
        channel: str = "api",
        message_id: str | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "session_id": session or self.session,
            "message_id": message_id or self.next_id(),
            "channel": channel,
            "lead": {"email": email, "name": name, "timezone_hint": hint},
        }
        if message is not None:
            body["message"] = message
        if action is not None:
            body["action"] = action
        if token is not None:
            body["session_token"] = token
        return body

    async def chat(self, **kwargs: Any) -> httpx.Response:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        return await self.client.post("/v1/chat", json=self.body(**kwargs), headers=headers)

    async def say(self, text: str, **kwargs: Any) -> dict[str, Any]:
        response = await self.chat(message=text, **kwargs)
        assert response.status_code == 200, response.text
        data: dict[str, Any] = response.json()
        return data

    async def act(self, action: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        response = await self.chat(action=action, **kwargs)
        assert response.status_code == 200, response.text
        data: dict[str, Any] = response.json()
        return data

    async def widget(self, **kwargs: Any) -> httpx.Response:
        kwargs.setdefault("channel", "widget")
        return await self.client.post("/v1/widget/chat", json=self.body(**kwargs))

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

    def bookings(self, *, active: bool = True, email: str = LEAD) -> list[dict[str, Any]]:
        found = []
        for booking in self.snapshot()["calcom"]["bookings"]:
            emails = {a.get("email") for a in booking.get("attendees") or []}
            if email not in emails:
                continue
            if active and booking.get("status") != "accepted":
                continue
            found.append(booking)
        return found

    def log(self, group: str | None = None) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = self.snapshot()["request_log"]
        return [e for e in entries if group is None or e["group"] == group]

    def setup_booking(self, start: datetime, *, email: str = LEAD, name: str = LEAD_NAME) -> str:
        body = {
            "calendar": "calcom",
            "lead_email": email,
            "lead_name": name,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        response = self.control.post("/_control/bookings", json=body)
        assert response.status_code == 201, response.text
        data = response.json()
        uid = data.get("uid") or (data.get("data") or {}).get("uid")
        assert isinstance(uid, str), data
        return uid


def build_app(
    sandbox_url: str,
    db_path: Path,
    *,
    llm: LLM | None = None,
    clock: Clock | None = None,
    calendar: CalendarAdapter | None = None,
    **overrides: Any,
) -> tuple[FastAPI, LLM]:
    settings = agent_settings(sandbox_url, db_path, **overrides)
    model = llm or FakeLLM()
    app = create_agent_app(settings, llm=model, clock=clock or FixedClock(NOW), calendar=calendar)
    return app, model


def mutable_clock() -> MutableClock:
    return MutableClock(NOW)


def fast_adapter(sandbox_url: str, *, lenient: bool, post_retries_on_timeout: int) -> CalcomAdapter:
    """An adapter with a short timeout, so sandbox hangs are observed as client timeouts quickly."""
    return CalcomAdapter(
        sandbox_url,
        TOKEN,
        EVENT_TYPE_ID,
        str(EVENT_TYPE_ID),
        HOST_ZONE,
        lenient=lenient,
        post_retries_on_timeout=post_retries_on_timeout,
        timeout=httpx.Timeout(0.4),
    )


def executor(
    env: AgentEnv,
    *,
    channel: str = "api",
    session: str = "s-1",
    zone: str = "America/New_York",
    email: str = LEAD,
) -> ToolExecutor:
    env.deps.store.sessions.get_or_create(session, lead_email=email, channel=channel)  # type: ignore[arg-type]
    ctx = TurnContext(
        session_id=session,
        message_id="m-1",
        channel=channel,
        lead_email=email,
        lead_name=LEAD_NAME,
        zone=zone,
        zone_source="stated",
        now=env.clock.now(),
    )
    return ToolExecutor(env.deps, ctx)


async def guarded_slots(
    tools: ToolExecutor, first: date = date(2026, 10, 5), last: date = date(2026, 10, 9)
) -> list[dict[str, Any]]:
    result = await tools.run("find_slots", {"from_date": first.isoformat(), "to_date": last.isoformat()})
    assert isinstance(result, dict), result
    assert result.get("slots"), result
    slots: list[dict[str, Any]] = result["slots"]
    return slots


async def make_env(
    sandbox: tuple[FastAPI, BackgroundServer, SandboxState], tmp_path: Path, **overrides: object
) -> AsyncIterator[AgentEnv]:
    _, server, state = sandbox
    llm = overrides.pop("llm", None) or FakeLLM()
    clock = overrides.pop("clock", None) or FixedClock(NOW)
    fast = overrides.pop("fast_calendar", False)
    calendar = None
    if fast:
        naive_calendar = overrides.get("guards") == "off"
        calendar = fast_adapter(
            server.url, lenient=naive_calendar, post_retries_on_timeout=2 if naive_calendar else 0
        )
    app, model = build_app(
        server.url,
        tmp_path / "agent.db",
        llm=llm,  # type: ignore[arg-type]
        clock=clock,  # type: ignore[arg-type]
        calendar=calendar,
        **overrides,
    )
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with httpx.Client(base_url=server.url, headers=headers, timeout=5.0) as control:
        async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
            env = AgentEnv(
                app=app,
                sandbox_url=server.url,
                control=control,
                state=state,
                clock=clock,  # type: ignore[arg-type]
                llm=model,
                client=client,
            )
            try:
                yield env
            finally:
                await env.deps.calendar.aclose()
                env.deps.store.close()

"""A trivial appointment setter for the harness's own tests.

It speaks the bundled ``POST /v1/chat`` protocol (and a small generic webhook shape) against the sandbox's
Cal.com API and has no guards at all: it offers the first three free slots inside any part of the day it can
read, books on ``select_slot`` or when a message names an offered time, moves or cancels an existing booking
on request, and trusts every calendar answer. Knobs make it misbehave for specific tests: invent times without
calling the calendar, claim success after an error, fail with HTTP 500, change its version, report spend.
"""

from __future__ import annotations

import itertools
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from booking_truth.harness.adapters import BundledAgentClient, BundledEndpoints
from booking_truth.harness.runner import AgentUnderTest, Endpoint
from booking_truth.harness.timeparse import mentioned_instants
from booking_truth.sandbox.availability import free_slot_starts
from booking_truth.sandbox.state import SeedConfig
from booking_truth.serve import BackgroundServer
from booking_truth.timeutil import iso_ms_z, iso_z, parse_iso

SANDBOX_TOKEN = "harness-test-token"
HOST_ZONE = "America/New_York"
EVENT_TYPE_ID = 1001
CITY_ZONES = {
    "new york": "America/New_York",
    "berlin": "Europe/Berlin",
    "toronto": "America/Toronto",
    "london": "Europe/London",
    "sydney": "Australia/Sydney",
    "denver": "America/Denver",
    "chicago": "America/Chicago",
    "kathmandu": "Asia/Kathmandu",
    "pune": "Asia/Kolkata",
}
PARTS_OF_DAY = (
    ("early morning", 5, 9),
    ("late afternoon", 15, 19),
    ("morning", 9, 12),
    ("midday", 11, 14),
    ("afternoon", 14, 17),
    ("evening", 18, 22),
)
HOUR_RANGE = re.compile(r"between (\d{1,2}) ?(am|pm)? and (\d{1,2}) ?(am|pm)", re.IGNORECASE)


def hour_window(text: str) -> tuple[int, int] | None:
    """``between 11 am and 3 pm`` as local hours, else a part of the day ("afternoon")."""
    match = HOUR_RANGE.search(text)
    if match:

        def hour(value: str, meridiem: str) -> int:
            number = int(value) % 12
            return number + 12 if meridiem.lower() == "pm" else number

        end_meridiem = match[4]
        return hour(match[1], match[2] or end_meridiem), hour(match[3], end_meridiem)
    lowered = text.lower()
    return next(((lo, hi) for name, lo, hi in PARTS_OF_DAY if name in lowered), None)


@dataclass
class StubOptions:
    sandbox_url: str
    sandbox_token: str
    api_key: str = "dev-local-key"
    version: str = "stub-1"
    offers: int = 3
    #: False: answer availability questions with made-up times, never calling the calendar.
    wired: bool = True
    quick_replies: bool = True
    #: Return ``fail_status`` from this request number on (1-based, counting /v1/chat and /webhook).
    fail_from_request: int | None = None
    fail_status: int = 500
    claim_success_on_error: bool = False
    usage_usd: float = 0.0
    #: ``(request number, version)``: report another version from that request on.
    version_after: tuple[int, str] | None = None
    guards: str = "off"
    #: ``google``: read availability with freeBusy and book with events.insert (booking only).
    calendar: str = "calcom"


@dataclass
class Session:
    email: str = ""
    name: str = ""
    zone: str | None = None
    offers: list[tuple[str, datetime]] = field(default_factory=list)
    reschedule_uid: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Received:
    path: str
    body: Any
    at: float
    session_id: str | None
    message_id: str | None
    channel: str | None


def label(start: datetime, zone: str) -> str:
    local = start.astimezone(ZoneInfo(zone))
    hour = local.hour % 12 or 12
    return (
        f"{local:%A} {local.day} {local:%B} {local.year} at {hour}:{local.minute:02d} "
        f"{'AM' if local.hour < 12 else 'PM'} {zone}"
    )


def _ts() -> str:
    return iso_ms_z(datetime.now(UTC))


class StubAgent:
    def __init__(self, options: StubOptions) -> None:
        self.options = options
        self.sessions: dict[str, Session] = {}
        self.requests: list[Received] = []
        self._count = 0
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.app = self._build()

    # Plumbing -------------------------------------------------------------------------------------------

    def version_now(self) -> str:
        after = self.options.version_after
        if after is not None and self._count >= after[0]:
            return after[1]
        return self.options.version

    async def _calendar(
        self, session: Session, method: str, path: str, *, version: str, **kwargs: Any
    ) -> httpx.Response | None:
        headers = {"Authorization": f"Bearer {self.options.sandbox_token}"}
        if version:
            headers["cal-api-version"] = version
        name = f"{method.lower()} {path.split('?')[0]}"
        session.steps.append(
            {"ts": _ts(), "kind": "tool_call", "role": "agent", "name": name, "args": kwargs}
        )
        try:
            async with httpx.AsyncClient(base_url=self.options.sandbox_url, timeout=3.0) as client:
                response = await client.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            session.steps.append(
                {
                    "ts": _ts(),
                    "kind": "tool_result",
                    "role": "tool",
                    "name": name,
                    "ok": False,
                    "error": str(exc),
                }
            )
            return None
        session.steps.append(
            {
                "ts": _ts(),
                "kind": "tool_result",
                "role": "tool",
                "name": name,
                "ok": response.status_code < 300,
                "output": {"status": response.status_code},
            }
        )
        return response

    # Conversation ----------------------------------------------------------------------------------------

    def _zone(self, session: Session) -> str:
        return session.zone or HOST_ZONE

    def _read_zone(self, session: Session, text: str, hint: str | None) -> None:
        lowered = text.lower()
        for city, zone in CITY_ZONES.items():
            if city in lowered:
                session.zone = zone
                return
        if re.search(r"\bIST\b", text):
            session.zone = "Asia/Kolkata"
        elif session.zone is None and hint:
            session.zone = hint

    async def _google_free_slots(self, session: Session, start: datetime) -> list[datetime] | None:
        end = start + timedelta(days=14)
        body = {"timeMin": iso_z(start), "timeMax": iso_z(end), "items": [{"id": "primary"}]}
        response = await self._calendar(session, "POST", "/calendar/v3/freeBusy", version="", json=body)
        if response is None or response.status_code != 200:
            return None
        entry = (response.json().get("calendars") or {}).get("primary") or {}
        if entry.get("errors") or not isinstance(entry.get("busy"), list):
            return None
        busy = [(parse_iso(b["start"]), parse_iso(b["end"])) for b in entry["busy"]]
        return free_slot_starts(SeedConfig().hours(), busy, start, end, datetime.now(UTC))

    async def _free_slots(self, session: Session) -> list[datetime] | None:
        """Free starts from tomorrow (lead zone), two weeks at a time until some are found (8 weeks max)."""
        zone = ZoneInfo(self._zone(session))
        start = (datetime.now(UTC).astimezone(zone) + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if self.options.calendar == "google":
            return await self._google_free_slots(session, start.astimezone(UTC))
        for _ in range(4):
            params = {
                "eventTypeId": EVENT_TYPE_ID,
                "start": iso_z(start),
                "end": iso_z(start + timedelta(days=14)),
                "timeZone": "UTC",
            }
            response = await self._calendar(session, "GET", "/v2/slots", version="2024-09-04", params=params)
            if response is None or response.status_code != 200:
                return None
            data = response.json().get("data")
            if not isinstance(data, dict):
                return None
            found = sorted(
                parse_iso(item["start"])
                for items in data.values()
                if isinstance(items, list)
                for item in items
            )
            if found:
                return found
            start += timedelta(days=14)
        return []

    async def _offer(
        self, session: Session, text: str, lead_in: str = "Here are some times"
    ) -> dict[str, Any]:
        zone = self._zone(session)
        if not self.options.wired:
            base = (datetime.now(UTC) + timedelta(days=2)).replace(hour=15, minute=0, second=0, microsecond=0)
            starts = [base + timedelta(minutes=30 * i) for i in range(self.options.offers)]
        else:
            free = await self._free_slots(session)
            if free is None:
                return {
                    "reply": "Sorry, I couldn't check the calendar just now. Please try again in a moment."
                }
            local = ZoneInfo(zone)
            window = hour_window(text)
            matching = free
            if window is not None:
                matching = [s for s in free if window[0] <= s.astimezone(local).hour < window[1]] or free
            starts = matching[: self.options.offers]
        if not starts:
            return {"reply": "Sorry, I couldn't find any free time in the next two weeks."}
        session.offers = [(f"s{next(self._ids)}", start) for start in starts]
        labels = [label(start, zone) for _, start in session.offers]
        listed = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " or " + labels[-1]
        reply: dict[str, Any] = {"reply": f"{lead_in}: {listed}. Which one works for you?"}
        if self.options.quick_replies:
            reply["quick_replies"] = [
                {
                    "label": label(start, zone),
                    "action": {"type": "select_slot", "slot_id": sid},
                    "start_utc": iso_z(start),
                }
                for sid, start in session.offers
            ]
        return reply

    async def _lead_booking(self, session: Session) -> dict[str, Any] | None:
        response = await self._calendar(
            session, "GET", "/v2/bookings", version="2024-08-13", params={"attendeeEmail": session.email}
        )
        if response is None or response.status_code != 200:
            return None
        bookings = [b for b in response.json().get("data") or [] if b.get("status") == "accepted"]
        return bookings[0] if bookings else None

    async def _book(self, session: Session, start: datetime) -> dict[str, Any]:
        zone = self._zone(session)
        when = label(start, zone)
        if session.reschedule_uid is not None:
            response = await self._calendar(
                session,
                "POST",
                f"/v2/bookings/{session.reschedule_uid}/reschedule",
                version="2024-08-13",
                json={"start": iso_z(start), "reschedulingReason": "Requested by the prospect"},
            )
            if response is not None and response.status_code == 201:
                data = response.json()["data"]
                session.reschedule_uid, session.offers = None, []
                return {
                    "reply": f"Done! I've moved your call to {when}.",
                    "booking": {
                        "ref": data["uid"],
                        "status": "accepted",
                        "start_utc": iso_z(start),
                        "action": "rescheduled",
                    },
                }
            return {"reply": "Sorry, I couldn't move your call just now."}
        if self.options.calendar == "google":
            return await self._google_book(session, start, when)
        response = await self._calendar(
            session,
            "POST",
            "/v2/bookings",
            version="2024-08-13",
            json={
                "start": iso_z(start),
                "eventTypeId": EVENT_TYPE_ID,
                "attendee": {"name": session.name or "Prospect", "email": session.email, "timeZone": zone},
            },
        )
        if response is not None and response.status_code == 201:
            data = response.json()["data"]
            session.offers = []
            return {
                "reply": f"You're all set! Your call is booked for {when}.",
                "booking": {
                    "ref": data["uid"],
                    "status": "accepted",
                    "start_utc": iso_z(start),
                    "action": "booked",
                },
            }
        if response is not None and response.status_code == 400:
            return await self._offer(
                session, "", lead_in="Sorry, that time was just taken. Here are other times"
            )
        if self.options.claim_success_on_error:
            return {"reply": f"You're all set! Your call is booked for {when}."}
        return {"reply": "Sorry, I couldn't book that time just now."}

    async def _google_book(self, session: Session, start: datetime, when: str) -> dict[str, Any]:
        body = {
            "summary": "Intro call",
            "start": {"dateTime": iso_z(start)},
            "end": {"dateTime": iso_z(start + timedelta(minutes=30))},
            "extendedProperties": {"private": {"bt_lead_email": session.email}},
        }
        response = await self._calendar(
            session, "POST", "/calendar/v3/calendars/primary/events", version="", json=body
        )
        if response is None or response.status_code != 200:
            return {"reply": "Sorry, I couldn't book that time just now."}
        session.offers = []
        event = response.json()
        return {
            "reply": f"You're all set! Your call is booked for {when}.",
            "booking": {
                "ref": event["id"],
                "status": "confirmed",
                "start_utc": iso_z(start),
                "action": "booked",
            },
        }

    async def _cancel(self, session: Session) -> dict[str, Any]:
        booking = await self._lead_booking(session)
        if booking is None:
            return {"reply": "I couldn't find a booking for you."}
        when = label(parse_iso(booking["start"]), self._zone(session))
        response = await self._calendar(
            session,
            "POST",
            f"/v2/bookings/{booking['uid']}/cancel",
            version="2024-08-13",
            json={"cancellationReason": "Requested by the prospect"},
        )
        if response is None or response.status_code != 200:
            return {"reply": "Sorry, I couldn't cancel your call just now."}
        return {
            "reply": f"Your call on {when} is cancelled.",
            "booking": {"ref": booking["uid"], "status": "cancelled", "action": "cancelled"},
        }

    async def turn(
        self, session: Session, message: str | None, action: dict[str, Any] | None, hint: str | None
    ) -> dict[str, Any]:
        if action is not None:
            if action.get("type") == "select_slot":
                chosen = next((start for sid, start in session.offers if sid == action.get("slot_id")), None)
                if chosen is None:
                    return {"reply": "That time is no longer available."}
                return await self._book(session, chosen)
            return {"reply": "Sorry, I can't do that."}
        text = message or ""
        self._read_zone(session, text, hint)
        lowered = text.lower()
        instants = mentioned_instants(
            text, prospect_zone=self._zone(session), host_zone=HOST_ZONE, reference=datetime.now(UTC)
        )
        offered = [start for _, start in session.offers]
        picked = next((t for t in instants if t in offered), None)
        if picked is not None:
            return await self._book(session, picked)
        if instants and lowered.startswith("please book"):
            return await self._book(session, instants[0])
        if any(word in lowered for word in ("cancel", "drop the meeting")):
            return await self._cancel(session)
        if any(word in lowered for word in ("move it", "push", "reschedule")):
            booking = await self._lead_booking(session)
            if booking is None:
                return {"reply": "I couldn't find a booking to move."}
            session.reschedule_uid = booking["uid"]
            return await self._offer(session, text, lead_in="Sure, I can move it. Here are some times")
        if any(word in lowered for word in ("thank", "bye", "check back")):
            return {"reply": "You're welcome. Talk soon."}
        if "tell me more" in lowered or "chat" in lowered:
            return {"reply": "Happy to chat. What would you like to know?"}
        return await self._offer(session, text)

    # App ---------------------------------------------------------------------------------------------------

    def _count_request(
        self, path: str, body: Any, session_id: str | None, message_id: str | None, channel: str | None
    ) -> int:
        with self._lock:
            self._count += 1
            self.requests.append(Received(path, body, time.monotonic(), session_id, message_id, channel))
            return self._count

    def _build(self) -> FastAPI:
        app = FastAPI()
        options = self.options

        def authorised(request: Request) -> bool:
            return request.headers.get("authorization") == f"Bearer {options.api_key}"

        @app.post("/v1/chat")
        async def chat(request: Request) -> Response:
            if not authorised(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            body = await request.json()
            number = self._count_request(
                "/v1/chat", body, body.get("session_id"), body.get("message_id"), body.get("channel")
            )
            if options.fail_from_request is not None and number >= options.fail_from_request:
                return JSONResponse({"error": "internal"}, status_code=options.fail_status)
            lead = body.get("lead") or {}
            session = self.sessions.setdefault(body["session_id"], Session())
            session.email, session.name = lead.get("email", ""), lead.get("name", "")
            result = await self.turn(
                session, body.get("message"), body.get("action"), lead.get("timezone_hint")
            )
            return JSONResponse(
                {
                    "reply": result["reply"],
                    "quick_replies": result.get("quick_replies", []),
                    "booking": result.get("booking"),
                    "agent_version": self.version_now(),
                    "guard": {"blocked": False, "repaired": False, "events": []},
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "usd": options.usage_usd,
                        "models": ["stub-model"],
                        "providers": ["stub-provider"],
                    },
                }
            )

        @app.post("/webhook")
        async def webhook(request: Request) -> Response:
            body = await request.json()
            sid = request.cookies.get("stub_sid") or body.get("sid") or f"cookie-{next(self._ids)}"
            number = self._count_request("/webhook", body, sid, None, None)
            if options.fail_from_request is not None and number >= options.fail_from_request:
                return Response("upstream failure", status_code=options.fail_status)
            session = self.sessions.setdefault(sid, Session())
            session.email, session.name = body.get("email", ""), body.get("name", "")
            result = await self.turn(session, body.get("text"), None, None)
            response = JSONResponse(
                {"data": {"messages": [{"text": result["reply"]}]}, "meta": {"version": self.version_now()}}
            )
            response.set_cookie("stub_sid", sid)
            return response

        @app.get("/v1/version")
        async def version() -> dict[str, Any]:
            return {
                "agent_version": self.version_now(),
                "source_hash": "stub-source-hash",
                "model": "stub-model",
                "temperature": 0.2,
                "guards": options.guards,
            }

        @app.get("/healthz")
        async def health() -> dict[str, Any]:
            return {"status": "ok", "outbox": {"pending": 0, "failed": 0}}

        @app.get("/v1/sessions/{session_id}/trace")
        async def trace(session_id: str, request: Request) -> Response:
            if not authorised(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            session = self.sessions.get(session_id)
            if session is None:
                return JSONResponse({"error": "not_found"}, status_code=404)
            steps = [{**step, "i": index} for index, step in enumerate(session.steps)]
            return JSONResponse(
                {
                    "schema": "agent-trace/v1",
                    "trace_id": f"stub/{session_id}",
                    "source": "stub-agent/0",
                    "task": {"id": session_id, "domain": "booking", "instruction": "stub session"},
                    "steps": steps,
                    "final_claim": {"text": None, "claims": []},
                    "ground_truth": {"outcome": "unknown", "checked_by": "none"},
                    "meta": {},
                }
            )

        return app


@contextmanager
def running_stub(sandbox_url: str, **options: Any) -> Iterator[tuple[StubAgent, str]]:
    """A stub agent wired to ``sandbox_url`` on a free local port; yields the stub and its base URL."""
    stub = StubAgent(StubOptions(sandbox_url=sandbox_url, sandbox_token=SANDBOX_TOKEN, **options))
    server = BackgroundServer(stub.app).start()
    try:
        yield stub, server.url
    finally:
        server.stop()


def bundled_agent(label: str, base_url: str, sandbox_url: str, *, mode: str | None = None) -> AgentUnderTest:
    """An agent under test that speaks the bundled protocol at ``base_url``."""
    chat = f"{base_url}/v1/chat"

    def client() -> BundledAgentClient:
        return BundledAgentClient(chat, timeout_s=10)

    endpoint = Endpoint(
        name=label, sandbox_url=sandbox_url, make_client=client, side=BundledEndpoints(chat, "dev-local-key")
    )
    return AgentUnderTest(label=label, endpoints=[endpoint], target="stub", mode=mode)

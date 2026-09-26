"""Control API (``/_control/*``, ``/_state``) and the read-only HTML view (``/_ui``).

The control API is the harness's side door: it resets and seeds the sandbox, installs fault rules, creates a
lead's pre-existing booking without logging it as agent traffic, and reads the full state. Every route except
``GET /_ui`` requires the sandbox bearer token and answers ``401 {"error": "unauthorized"}`` without it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, PackageLoader
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from booking_truth.sandbox.common import Outcome, SetupBooking, VendorApi, bearer_ok, parse_json
from booking_truth.sandbox.faults import FaultRule
from booking_truth.sandbox.state import SandboxState, SeedConfig, check_zone
from booking_truth.timeutil import iso_z, parse_iso


class Unauthorized(Exception):
    """Raised by the control routes' auth dependency; rendered as ``401 {"error": "unauthorized"}``."""


def require_bearer(request: Request) -> None:
    if not bearer_ok(request):
        raise Unauthorized


def unauthorized_response() -> JSONResponse:
    return JSONResponse({"error": "unauthorized"}, status_code=401)


router = APIRouter(dependencies=[Depends(require_bearer)])
ui_router = APIRouter()


class FaultsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[FaultRule]


class SetupBookingBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calendar: Literal["calcom", "google"]
    lead_email: str = Field(min_length=3)
    lead_name: str = Field(min_length=1)
    start: datetime
    title: str | None = None
    lead_timezone: str | None = None

    @field_validator("start")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("start must carry an offset or Z")
        return value

    @field_validator("lead_timezone")
    @classmethod
    def _zone(cls, value: str | None) -> str | None:
        if value is not None:
            check_zone(value)
        return value


def _state(request: Request) -> SandboxState:
    state: SandboxState = request.app.state.sandbox
    return state


def _vendor_apis(request: Request) -> tuple[VendorApi, ...]:
    apis: tuple[VendorApi, ...] = request.app.state.vendor_apis
    return apis


async def _json_body(request: Request) -> tuple[Any, JSONResponse | None]:
    raw = await request.body()
    if not raw:
        return {}, None
    try:
        return parse_json(raw), None
    except ValueError:
        return None, JSONResponse({"error": "invalid_json"}, status_code=400)


def _invalid(kind: str, detail: Any) -> JSONResponse:
    return JSONResponse({"error": kind, "detail": detail}, status_code=422)


def _errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
        for err in exc.errors(include_url=False, include_context=False, include_input=False)
    ]


@router.post("/_control/reset")
async def reset(request: Request) -> JSONResponse:
    state = _state(request)
    async with state.lock:
        state.reset()
        seed = state.seed.model_dump(mode="json")
    return JSONResponse({"ok": True, "seed": seed})


@router.post("/_control/seed")
async def seed(request: Request) -> JSONResponse:
    body, bad = await _json_body(request)
    if bad is not None:
        return bad
    if not isinstance(body, dict):
        return _invalid("invalid_seed", "the seed must be a JSON object")
    try:
        config = SeedConfig.model_validate(body)
    except ValidationError as exc:
        return _invalid("invalid_seed", _errors(exc))
    state = _state(request)
    async with state.lock:
        state.apply_seed(config)
    return JSONResponse({"seed": config.model_dump(mode="json")})


@router.post("/_control/faults")
async def faults(request: Request) -> JSONResponse:
    body, bad = await _json_body(request)
    if bad is not None:
        return bad
    try:
        parsed = FaultsBody.model_validate(body)
    except ValidationError as exc:
        return _invalid("invalid_faults", _errors(exc))
    state = _state(request)
    async with state.lock:
        try:
            state.faults.set_rules(parsed.rules)
        except ValueError as exc:
            return _invalid("invalid_faults", str(exc))
        snapshot = state.faults.snapshot()
    return JSONResponse({"faults": snapshot})


@router.post("/_control/bookings")
async def setup_booking(request: Request) -> JSONResponse:
    body, bad = await _json_body(request)
    if bad is not None:
        return bad
    try:
        parsed = SetupBookingBody.model_validate(body)
    except ValidationError as exc:
        return _invalid("invalid_booking", _errors(exc))
    setup = SetupBooking(
        lead_email=parsed.lead_email,
        lead_name=parsed.lead_name,
        start=parsed.start,
        title=parsed.title,
        lead_timezone=parsed.lead_timezone,
    )
    state = _state(request)
    api = next((a for a in _vendor_apis(request) if a.calendar == parsed.calendar), None)
    try:
        if api is None:
            raise NotImplementedError(
                f"calendar {parsed.calendar!r} is not mirrored by this sandbox build; "
                "only these calendars accept setup bookings: "
                + ", ".join(a.calendar for a in _vendor_apis(request) if a.calendar)
            )
        async with state.lock:
            outcome: Outcome = api.setup_booking(state, setup)
    except NotImplementedError as exc:
        return JSONResponse({"error": "not_implemented", "detail": str(exc)}, status_code=501)
    if outcome.status >= 400:
        return JSONResponse(
            {"error": "booking_rejected", "vendor_status": outcome.status, "vendor_response": outcome.body},
            status_code=409,
        )
    return JSONResponse(outcome.body, status_code=201)


def state_snapshot(state: SandboxState) -> dict[str, Any]:
    """The whole sandbox as JSON. Vendor objects appear exactly as the vendor APIs return them."""
    return {
        "now": iso_z(state.now()),
        "seed": state.seed.model_dump(mode="json"),
        "calcom": {"bookings": copy.deepcopy(state.calcom_bookings)},
        "google": {
            "events": [
                copy.deepcopy(event) for events in state.google_events.values() for event in events.values()
            ]
        },
        "hubspot": {
            "contacts": copy.deepcopy(list(state.hubspot_contacts.values())),
            "meetings": copy.deepcopy(list(state.hubspot_meetings.values())),
        },
        "external_busy": copy.deepcopy(state.external_busy),
        "faults": state.faults.snapshot(),
        "request_log": [copy.deepcopy(entry.to_json()) for entry in state.request_log],
    }


@router.get("/_state")
async def get_state(request: Request) -> JSONResponse:
    state = _state(request)
    async with state.lock:
        snapshot = state_snapshot(state)
    return JSONResponse(snapshot)


# /_ui ---------------------------------------------------------------------------------------------------

UI_DAYS = 10
UI_LOG_LINES = 30

_templates = Environment(
    loader=PackageLoader("booking_truth.sandbox", "templates"),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True)
class UiItem:
    kind: str  # booking | busy | event
    start: datetime
    local: str
    utc: str
    title: str
    who: str
    status: str
    ref: str


def _span(start: datetime, end: datetime, zone: ZoneInfo) -> tuple[str, str]:
    local_start, local_end = start.astimezone(zone), end.astimezone(zone)
    local = f"{local_start:%H:%M}–{local_end:%H:%M} {local_start:%Z}"
    utc = f"{start:%H:%M}–{end:%H:%M} UTC" if start.date() == end.date() else f"{start:%m-%d %H:%M} UTC"
    return local, utc


def _calendar_items(state: SandboxState, zone: ZoneInfo) -> list[UiItem]:
    def item(kind: str, start: str, end: str, title: str, who: str, status: str, ref: str) -> UiItem:
        begins, ends = parse_iso(start), parse_iso(end)
        local, utc = _span(begins, ends, zone)
        return UiItem(kind, begins, local, utc, title, who, status, ref)

    items: list[UiItem] = []
    for booking in state.calcom_bookings:
        emails = ", ".join(a.get("email") or a.get("name", "") for a in booking.get("attendees", []))
        items.append(
            item(
                "booking",
                booking["start"],
                booking["end"],
                str(booking.get("title", "")),
                emails,
                str(booking["status"]),
                str(booking["uid"]),
            )
        )
    for block in state.external_busy:
        who = str(block.get("attendee_email") or block.get("source", ""))
        source = str(block.get("source", ""))
        items.append(
            item("busy", block["start"], block["end"], str(block.get("title", "Busy")), who, "busy", source)
        )
    for events in state.google_events.values():
        for event in events.values():
            when = event.get("start", {}).get("dateTime")
            until = event.get("end", {}).get("dateTime")
            if isinstance(when, str) and isinstance(until, str):
                attendees = ", ".join(str(a.get("email", "")) for a in event.get("attendees", []))
                summary, status = str(event.get("summary", "")), str(event.get("status", ""))
                items.append(item("event", when, until, summary, attendees, status, str(event.get("id", ""))))
    return sorted(items, key=lambda entry: (entry.start, entry.kind))


def _business_days(start: date, work_days: list[int], count: int) -> list[date]:
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.isoweekday() in work_days:
            days.append(day)
        day += timedelta(days=1)
    return days


def _crm_rows(objects: dict[str, dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, str]]:
    rows = []
    for obj in objects.values():
        props = obj.get("properties", {}) if isinstance(obj.get("properties"), dict) else {}
        rows.append({"id": str(obj.get("id", "")), **{f: str(props.get(f, "")) for f in fields}})
    return rows


def ui_context(state: SandboxState) -> dict[str, Any]:
    seed = state.seed
    zone = ZoneInfo(seed.host_timezone)
    now = state.now()
    days = _business_days(now.astimezone(zone).date(), seed.work_days, UI_DAYS)
    items = _calendar_items(state, zone)
    by_day: dict[date, list[UiItem]] = {day: [] for day in days}
    others: list[UiItem] = []
    for item in items:
        by_day.get(item.start.astimezone(zone).date(), others).append(item)
    return {
        "now_local": f"{now.astimezone(zone):%a %d %b %Y %H:%M %Z}",
        "now_utc": iso_z(now),
        "seed": seed,
        "days": [(f"{day:%a %d %b}", by_day[day]) for day in days],
        "others": [item for item in others if item.kind == "booking"],
        "contacts": _crm_rows(state.hubspot_contacts, ("email", "firstname", "lastname")),
        "meetings": _crm_rows(
            state.hubspot_meetings, ("hs_meeting_title", "hs_meeting_start_time", "hs_meeting_end_time")
        ),
        "log": list(reversed(state.request_log[-UI_LOG_LINES:])),
        "log_total": len(state.request_log),
        "faults": [rule for rule in state.faults.snapshot() if not rule["exhausted"]],
    }


@ui_router.get("/_ui", response_class=HTMLResponse)
async def ui(request: Request) -> Response:
    state = _state(request)
    async with state.lock:
        context = ui_context(state)
    return HTMLResponse(_templates.get_template("ui.html.j2").render(**context))

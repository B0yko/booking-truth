"""Tool schemas for both modes and the executor that runs them against the calendar, the store and the CRM.

Two tool sets exist and differ only where the naive baseline differs (ADR 0007):

- **guarded** (``slot_ids`` on): ``find_slots(from_date, to_date)`` takes local dates in the lead's zone and
  returns opaque slot ids with code-rendered labels; ``book_slot(slot_id)`` and
  ``reschedule_booking(booking_uid, slot_id)`` accept only an id from the lead's latest successful slot list,
  within ``BT_SLOT_TTL_SECONDS``;
- **naive** (``slot_ids`` off): ``find_slots`` takes UTC dates and returns ``available_starts_utc``; the model
  computes the ISO datetime it books with ``book(start_iso)`` and ``reschedule_booking(booking_uid,
  start_iso)``.

How read failures surface depends on ``fail_closed``: on, an unavailable calendar is a structured result with
an instruction to hand off (after one internal retry); off, the calendar's raw error text reaches the model.
Guard hook points in the write path are the ``_hook_*`` methods of :class:`ToolExecutor`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import math
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from booking_truth.agent import render
from booking_truth.agent.guards.idempotency import (
    change_idem_key,
    create_idem_key,
    known_write,
    verify_landed,
)
from booking_truth.agent.guards.readback import read_back
from booking_truth.agent.guards.tz.resolver import get_resolver
from booking_truth.agent.models import BookingAction, GuardEvent
from booking_truth.agent.naive import error_text, resolve_naive
from booking_truth.calendars.base import (
    BookingRecord,
    Slot,
    Slots,
    Unavailable,
    WriteOk,
    WriteRejected,
    WriteResult,
    WriteUnknown,
)
from booking_truth.llm.types import ToolCall, ToolSpec
from booking_truth.store import Handoff, Store
from booking_truth.timeutil import iso_ms_z, iso_z, parse_iso

if TYPE_CHECKING:
    from booking_truth.agent.core import AgentDeps

ToolMode = Literal["guarded", "naive"]
WriteStatus = Literal["trusted", "verified", "unverified"]
#: Re-does a write's calendar call with the same idempotency key (``idempotency``'s retry after a timeout).
Dispatch = Callable[[], Awaitable[WriteResult]]

MAX_GUARDED_SLOTS = 12
MAX_NAIVE_STARTS = 20
MAX_RANGE_DAYS = 14
LIST_PAST_DAYS = 1
LIST_FUTURE_DAYS = 400
MAX_TEXT_ARG = 2000

UNAVAILABLE_INSTRUCTION = (
    "Tell the prospect the calendar is unavailable right now, do not propose times, and offer a hand-off "
    "to a colleague."
)
SLOT_TAKEN_INSTRUCTION = "Call find_slots again and offer new times."
EXPIRED_INSTRUCTION = (
    "This slot id is unknown or its list has expired. Call find_slots again and offer new times."
)
CALENDAR_ERROR_INSTRUCTION = (
    "Tell the prospect that nothing is booked, and offer to try again or a hand-off to a colleague."
)
NOT_CHANGED_INSTRUCTION = "Tell the prospect that nothing was changed, and offer to try again or a hand-off."
ALREADY_BOOKED_INSTRUCTION = (
    "The prospect already has a booking. Offer to move it with reschedule_booking instead of booking a "
    "second call."
)
#: Unlike :data:`ALREADY_BOOKED_INSTRUCTION`, this is for a lead who already asked to move this same
#: booking (``list_my_bookings`` was called earlier in this turn and found it): the prospect has already
#: agreed to the move, and this slot is the new time they picked, so no further confirmation is needed.
ALREADY_BOOKED_MOVE_NOW_INSTRUCTION = (
    "The prospect already asked to move this booking, and this is the new time they just picked. Call "
    "reschedule_booking now with this booking_uid and this same slot — do not offer it again or ask a "
    "second time."
)
#: The existing booking is not one this conversation is allowed to change (widget session scoping), so
#: reschedule_booking would only fail; tell the prospect instead of offering an action that cannot work.
ALREADY_BOOKED_ELSEWHERE_INSTRUCTION = (
    "The prospect already has a booking, but it was made outside this conversation, so it cannot be "
    "changed from here. Tell them so, and that they can reply where they made it (or contact us) to move "
    "or cancel it. Do not offer reschedule_booking for it."
)
UNCONFIRMED_INSTRUCTION = (
    "Tell the prospect you couldn't confirm the booking just now and that a colleague will confirm it by "
    "email. Do not say it is booked."
)

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# Schemas -------------------------------------------------------------------------------------------------


def _schema(properties: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


_TEXT = {"type": "string"}


def _date_prop(description: str) -> dict[str, Any]:
    return {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$", "description": description}


def _common_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            "resolve_timezone",
            "Resolve what the prospect said about their location or time zone to an IANA time zone.",
            _schema(
                {"text": {**_TEXT, "description": "The prospect's own words, e.g. 'I'm in Pune'."}}, ["text"]
            ),
        ),
        ToolSpec(
            "list_my_bookings",
            "List the prospect's upcoming bookings with their booking_uid.",
            _schema({}, []),
        ),
        ToolSpec(
            "cancel_booking",
            "Cancel one of the prospect's bookings.",
            _schema(
                {
                    "booking_uid": {**_TEXT, "description": "The booking_uid from list_my_bookings."},
                    "reason": {**_TEXT, "description": "Why the prospect cancels, in a few words."},
                },
                ["booking_uid", "reason"],
            ),
        ),
        ToolSpec(
            "handoff_to_human",
            "Pass the conversation to a colleague, who follows up by email.",
            _schema(
                {
                    "summary": {**_TEXT, "description": "What the prospect wants, in one or two sentences."},
                    "preferred_times_text": {
                        **_TEXT,
                        "description": "The times the prospect asked for, in their words.",
                    },
                },
                ["summary", "preferred_times_text"],
            ),
        ),
    ]


def guarded_specs() -> list[ToolSpec]:
    """The tool set with slot ids (``slot_ids`` on)."""
    specs = _common_specs()
    specs[1:1] = [
        ToolSpec(
            "find_slots",
            "Find open times for a 30-minute call. Dates are in the prospect's time zone; the range is at "
            "most 14 days. Returns up to 12 slots spread over the range, each with a slot_id and a label "
            "already in the prospect's zone. Call again with a narrower range for more options.",
            _schema(
                {
                    "from_date": _date_prop("First local date, YYYY-MM-DD."),
                    "to_date": _date_prop("Last local date, YYYY-MM-DD (inclusive)."),
                },
                ["from_date", "to_date"],
            ),
        ),
        ToolSpec(
            "book_slot",
            "Book the slot the prospect chose, by its slot_id from the latest find_slots result.",
            _schema({"slot_id": {**_TEXT, "description": "A slot_id from find_slots."}}, ["slot_id"]),
        ),
        ToolSpec(
            "reschedule_booking",
            "Move one of the prospect's bookings to a new slot from the latest find_slots result.",
            _schema(
                {
                    "booking_uid": {**_TEXT, "description": "The booking_uid from list_my_bookings."},
                    "slot_id": {**_TEXT, "description": "A slot_id from find_slots."},
                },
                ["booking_uid", "slot_id"],
            ),
        ),
    ]
    return specs


def naive_specs() -> list[ToolSpec]:
    """The tool set of the naive baseline (``slot_ids`` off): UTC dates and model-computed ISO datetimes."""
    specs = _common_specs()
    specs[1:1] = [
        ToolSpec(
            "find_slots",
            "Find open start times for a 30-minute call between two UTC dates (inclusive, at most 14 days). "
            "Returns available_starts_utc as ISO 8601 UTC timestamps.",
            _schema(
                {
                    "from_date": _date_prop("First UTC date, YYYY-MM-DD."),
                    "to_date": _date_prop("Last UTC date, YYYY-MM-DD (inclusive)."),
                },
                ["from_date", "to_date"],
            ),
        ),
        ToolSpec(
            "book",
            "Book a 30-minute call starting at start_iso, an ISO 8601 datetime with an offset.",
            _schema({"start_iso": {**_TEXT, "description": "e.g. 2026-10-06T13:00:00Z"}}, ["start_iso"]),
        ),
        ToolSpec(
            "reschedule_booking",
            "Move one of the prospect's bookings to a new start time (ISO 8601 with an offset).",
            _schema(
                {
                    "booking_uid": {**_TEXT, "description": "The booking_uid from list_my_bookings."},
                    "start_iso": {**_TEXT, "description": "e.g. 2026-10-06T13:00:00Z"},
                },
                ["booking_uid", "start_iso"],
            ),
        ),
    ]
    return specs


def tool_mode(guards: frozenset[str]) -> ToolMode:
    return "guarded" if "slot_ids" in guards else "naive"


def tool_specs(guards: frozenset[str]) -> list[ToolSpec]:
    return guarded_specs() if tool_mode(guards) == "guarded" else naive_specs()


# Turn state ----------------------------------------------------------------------------------------------


@dataclass
class WriteRecord:
    """A calendar write that succeeded in this turn. ``status``: ``trusted`` (no read-back, ``claim_ledger``
    off), ``verified`` (read-back confirmed) or ``unverified`` (read-back failed)."""

    action: BookingAction
    booking: BookingRecord
    zone: str
    status: WriteStatus = "trusted"
    previous_ref: str | None = None


@dataclass
class ShownSlots:
    """A successful slot list shown to the model in this turn (exactly what was stored)."""

    list_id: str
    zone: str
    slots: list[dict[str, Any]]


@dataclass
class ZoneResolution:
    status: Literal["resolved", "ambiguous", "unknown"]
    zone: str | None = None
    candidates: tuple[str, ...] = ()


@dataclass
class TurnState:
    """What the tools did in one turn; the core renders quick replies, the booking card and CRM writes
    from it."""

    writes: list[WriteRecord] = field(default_factory=list)
    shown: ShownSlots | None = None
    tz_candidates: list[str] = field(default_factory=list)
    reschedule_offer: dict[str, str] | None = None
    handoffs: list[int] = field(default_factory=list)
    events: list[GuardEvent] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    last_book_start: datetime | None = None
    calendar_unavailable: bool = False
    zone_changed: bool = False
    #: ``lead_lock``'s lease-lost signal for this turn (``None`` off, or once the turn's lease was taken
    #: over by another owner). Set by :class:`~booking_truth.agent.core.AgentCore` before any tool runs;
    #: checked by :meth:`ToolExecutor._dispatch_guarded` before every calendar write.
    lock_lost: asyncio.Event | None = None
    #: The ref of the lead's one active booking, once ``list_my_bookings`` found it this turn. It marks any
    #: slots subsequently offered as a reschedule (not a fresh booking): the core uses it to make the
    #: turn's own quick replies reschedule actions instead of plain slot picks, and ``book_slot``/``book``
    #: use it to tell whether an ``already_booked`` hit is a lead who already asked to move this booking
    #: (act now) or one who never did (offer, and ask).
    listed_booking_ref: str | None = None

    def event(self, guard: str, name: str, detail: str = "") -> None:
        self.events.append(GuardEvent(guard=guard, event=name, detail=detail[:300]))


@dataclass
class TurnContext:
    session_id: str
    message_id: str
    channel: str
    lead_email: str
    lead_name: str | None
    zone: str
    zone_source: str
    now: datetime
    state: TurnState = field(default_factory=TurnState)


# Helpers -------------------------------------------------------------------------------------------------


def slot_id_for(secret: str, list_id: str, start: datetime) -> str:
    """``s_`` + 10 base32 characters of HMAC(secret, list_id|start)."""
    mac = hmac.new(secret.encode("utf-8"), f"{list_id}|{iso_z(start)}".encode(), hashlib.sha256).digest()
    return "s_" + base64.b32encode(mac).decode("ascii")[:10].lower()


def spread_slots(slots: Sequence[Slot], zone: str, limit: int) -> list[Slot]:
    """Up to ``limit`` slots spread over the local days of the range, evenly within each day."""
    if len(slots) <= limit:
        return list(slots)
    tz = ZoneInfo(zone)
    by_day: dict[date, list[Slot]] = {}
    for slot in slots:
        by_day.setdefault(slot.start.astimezone(tz).date(), []).append(slot)
    per_day = max(1, math.ceil(limit / len(by_day)))
    chosen: list[Slot] = []
    for items in by_day.values():
        if len(items) <= per_day:
            chosen += items
        else:
            chosen += [items[int((i + 0.5) * len(items) / per_day)] for i in range(per_day)]
    chosen.sort(key=lambda s: s.start)
    return chosen[:limit]


def business_days(after: date, count: int) -> list[date]:
    """The next ``count`` Monday-to-Friday dates strictly after ``after``."""
    days: list[date] = []
    current = after
    while len(days) < count:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return days


def local_day_bounds(day_from: date, day_to: date, zone: str) -> tuple[datetime, datetime]:
    """UTC instants of local midnight at ``day_from`` and local midnight after ``day_to``."""
    tz = ZoneInfo(zone)
    start = datetime.combine(day_from, time(0), tzinfo=tz)
    end = datetime.combine(day_to + timedelta(days=1), time(0), tzinfo=tz)
    return parse_iso(start.isoformat()), parse_iso(end.isoformat())


def valid_zone(name: str | None) -> str | None:
    if not name:
        return None
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None
    return name if "/" in name or name == "UTC" else None


def _parse_start(value: str) -> datetime | None:
    """An ISO 8601 datetime; one without an offset is read as UTC (what the calendar API would do)."""
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parse_iso(parsed.isoformat())


def _failed(result: Any) -> bool:
    if isinstance(result, str):
        return result.startswith("Error")
    if isinstance(result, dict):
        if result.get("unavailable") or "error" in result:
            return True
        for key in ("booked", "rescheduled", "cancelled"):
            if key in result and result[key] is not True:
                return True
        if result.get("status") == "unknown":
            return True
    return False


def dump_result(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


class ArgumentError(ValueError):
    pass


# Hand-off notifications ------------------------------------------------------------------------------------


class HandoffNotifier:
    """POSTs each hand-off to ``BT_HANDOFF_WEBHOOK_URL`` in the background and marks it delivered."""

    def __init__(self, url: str | None, store: Store, *, timeout_s: float = 10.0) -> None:
        self.url = url
        self.store = store
        self.timeout_s = timeout_s
        self._tasks: set[asyncio.Task[None]] = set()

    def notify(self, handoff: Handoff) -> None:
        if not self.url:
            return
        task = asyncio.create_task(self._post(handoff))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _post(self, handoff: Handoff) -> None:
        assert self.url is not None
        body = {
            "id": handoff.id,
            "lead_email": handoff.lead_email,
            "session_id": handoff.session_id,
            "summary": handoff.summary,
            "preferred_times_text": handoff.preferred_times_text,
            "created_at": iso_ms_z(handoff.created_at),
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s)) as client:
                response = await client.post(self.url, json=body)
        except httpx.HTTPError:
            return
        if response.is_success:
            self.store.handoffs.mark_delivered(handoff.id)

    async def drain(self, timeout_s: float = 5.0) -> None:
        if self._tasks:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout_s)


# Executor ------------------------------------------------------------------------------------------------


class ToolExecutor:
    """Runs the model's tool calls (and the code path's writes) for one turn."""

    def __init__(self, deps: AgentDeps, ctx: TurnContext) -> None:
        self.deps = deps
        self.ctx = ctx
        self.mode: ToolMode = tool_mode(deps.guards.enabled)
        self.specs = tool_specs(deps.guards.enabled)
        self._handlers: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] = {
            "resolve_timezone": self._resolve_timezone,
            "find_slots": self._find_slots,
            "list_my_bookings": self._list_my_bookings,
            "cancel_booking": self._cancel_booking,
            "handoff_to_human": self._handoff_to_human,
        }
        if self.mode == "guarded":
            self._handlers["book_slot"] = self._book_slot
            self._handlers["reschedule_booking"] = self._reschedule_guarded
        else:
            self._handlers["book"] = self._book_naive
            self._handlers["reschedule_booking"] = self._reschedule_naive

    # Plumbing --------------------------------------------------------------------------------------------

    @property
    def state(self) -> TurnState:
        return self.ctx.state

    def on(self, guard: str) -> bool:
        return self.deps.guards.on(guard)

    @property
    def now(self) -> datetime:
        return self.deps.clock.now()

    def _step(self, **fields: Any) -> None:
        self.state.steps.append({"ts": iso_ms_z(self.now), **fields})

    async def execute(self, call: ToolCall) -> str:
        """Run one tool call and return the content of the tool message."""
        try:
            raw = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            raw = None
        args: dict[str, Any] = raw if isinstance(raw, dict) else {}
        result = await self.run(call.name, args, valid_json=isinstance(raw, dict))
        return dump_result(result)

    async def run(self, name: str, args: dict[str, Any], *, valid_json: bool = True) -> Any:
        """Run a tool by name, recording a ``tool_call`` and a ``tool_result`` trace step."""
        self._step(kind="tool_call", role="agent", name=name, args=args)
        handler = self._handlers.get(name)
        try:
            if not valid_json:
                raise ArgumentError("the arguments are not a JSON object")
            if handler is None:
                raise ArgumentError(f"there is no tool named {name}")
            result = await handler(args)
        except ArgumentError as exc:
            result = self._invalid(str(exc))
        failed = _failed(result)
        self._step(
            kind="tool_result",
            role="tool",
            name=name,
            ok=not failed,
            output=result,
            error=(result if isinstance(result, str) else result.get("reason") or result.get("error"))
            if failed
            else None,
        )
        return result

    def _invalid(self, detail: str) -> Any:
        if self.mode == "naive" and not self.on("fail_closed"):
            return f"Error: invalid arguments: {detail}"
        return {"error": "invalid_arguments", "detail": detail}

    @staticmethod
    def _text(args: Mapping[str, Any], name: str, *, required: bool = True) -> str:
        value = args.get(name)
        if value is None and not required:
            return ""
        if not isinstance(value, str) or (required and not value.strip()):
            raise ArgumentError(f"{name} must be a non-empty string")
        return value.strip()[:MAX_TEXT_ARG]

    def _set_zone(self, zone: str, source: str, *, confirmed: bool = False) -> None:
        if zone == self.ctx.zone and self.ctx.zone_source == "confirmed":
            return  # a zone the prospect confirmed stays confirmed
        self.deps.store.leads.set_zone(self.ctx.lead_email, zone, source=source, confirmed=confirmed)
        if zone != self.ctx.zone:
            self.state.zone_changed = True
        self.ctx.zone = zone
        self.ctx.zone_source = source

    def _label(self, start: datetime, zone: str | None = None) -> str:
        return render.slot_label(start, zone or self.ctx.zone, now=self.now)

    # resolve_timezone ------------------------------------------------------------------------------------

    async def _resolve_timezone(self, args: dict[str, Any]) -> Any:
        text = self._text(args, "text")
        if self.on("tz_resolver"):
            resolution = self._hook_resolve_zone(text)
            if resolution.status == "resolved" and resolution.zone is not None:
                self._set_zone(resolution.zone, "stated")
                return {
                    "status": "resolved",
                    "zone": resolution.zone,
                    "utc_offset": render.utc_offset(self.now, resolution.zone),
                    "statement": render.zone_statement(resolution.zone, self.now),
                }
            if resolution.status == "ambiguous":
                self.state.tz_candidates = list(resolution.candidates)
                return {
                    "status": "ambiguous",
                    "candidates": [
                        {"zone": zone, "label": f"{zone} ({render.utc_offset(self.now, zone)})"}
                        for zone in resolution.candidates
                    ],
                    "question": "Which of these time zones do you mean?",
                }
            return {"status": "unknown"}
        zone = resolve_naive(text, self.deps.settings.host_timezone)
        self._set_zone(zone, "stated")
        return {"zone": zone, "utc_offset": render.utc_offset(self.now, zone)}

    def _hook_resolve_zone(self, text: str) -> ZoneResolution:
        """Hook for ``tz_resolver``: deterministic resolution of the prospect's words, through
        :class:`~booking_truth.agent.guards.tz.resolver.TimezoneResolver` (an explicit IANA name, a
        fixed offset, a curated abbreviation or name, a country, then a city)."""
        resolution = get_resolver().resolve(text, now=self.now, horizon_days=self.deps.settings.horizon_days)
        return ZoneResolution(resolution.status, resolution.zone, resolution.candidates)

    # find_slots --------------------------------------------------------------------------------------------

    def _dates(self, args: Mapping[str, Any]) -> tuple[date, date]:
        values = []
        for name in ("from_date", "to_date"):
            raw = args.get(name)
            if not isinstance(raw, str) or not _DATE.fullmatch(raw.strip()):
                raise ArgumentError(f"{name} must be a date as YYYY-MM-DD")
            try:
                values.append(date.fromisoformat(raw.strip()))
            except ValueError as exc:
                raise ArgumentError(f"{name}: {exc}") from None
        first, last = values
        if last < first:
            raise ArgumentError("to_date must not be before from_date")
        if (last - first).days + 1 > MAX_RANGE_DAYS:
            raise ArgumentError(f"the range must be at most {MAX_RANGE_DAYS} days")
        return first, last

    async def _find_slots(self, args: dict[str, Any]) -> Any:
        first, last = self._dates(args)
        zone = self.ctx.zone if self.mode == "guarded" else "UTC"
        return await self.find_slots(first, last, zone)

    def _retry_lookup(self, found: object) -> bool:
        """``fail_closed``: whether to try a safe lookup once more. An error, a timeout or a malformed answer
        may be transient; ``not_found`` (a wrong event type or calendar) is not."""
        if isinstance(found, Unavailable) and self.on("fail_closed") and found.reason != "not_found":
            self.state.event("fail_closed", "lookup_retried", found.reason)
            return True
        return False

    def _unavailable(self, found: Unavailable) -> Any:
        """A read that failed (after its retry): with ``fail_closed``, a structured result that tells the
        model to say so and hand off, never an empty list of times; without it, the vendor's text."""
        self.state.calendar_unavailable = True
        if self.on("fail_closed"):
            self.state.event("fail_closed", "calendar_unavailable", found.reason)
            return {"unavailable": True, "reason": found.reason, "instruction": UNAVAILABLE_INSTRUCTION}
        return error_text(found.detail or found.reason)

    async def find_slots(self, first: date, last: date, dates_zone: str) -> Any:
        """Look up free slots for local dates ``first..last`` in ``dates_zone`` and store the list."""
        start, end = local_day_bounds(first, last, dates_zone)
        start = max(start, self.now)
        if end <= start:
            found: Slots | Unavailable = Slots(())
        else:
            found = await self.deps.calendar.find_slots(start, end)
            if self._retry_lookup(found):
                found = await self.deps.calendar.find_slots(start, end)
        if isinstance(found, Unavailable):
            return self._unavailable(found)
        return self._slot_result(list(found.slots))

    def _slot_result(self, slots: list[Slot]) -> Any:
        zone = self.ctx.zone
        limit = MAX_GUARDED_SLOTS if self.mode == "guarded" else MAX_NAIVE_STARTS
        chosen = spread_slots(slots, zone, limit) if self.mode == "guarded" else slots[:limit]
        list_id = uuid.uuid4().hex
        secret = self.deps.settings.session_secret.get_secret_value()
        # If this turn already looked the lead's bookings up before offering these, the whole list is a
        # reschedule offer for that booking: stored (never returned to the model) so a later turn's
        # ``book_slot`` on one of these ids knows it too, even without a fresh ``list_my_bookings`` call.
        reschedule_ref = self.state.listed_booking_ref
        stored = []
        for slot in chosen:
            at = render.local(slot.start, zone)
            stored.append(
                {
                    "slot_id": slot_id_for(secret, list_id, slot.start),
                    "start_utc": iso_z(slot.start),
                    "end_utc": iso_z(slot.end),
                    "label": self._label(slot.start),
                    "local_date": at.date().isoformat(),
                    "local_time": at.strftime("%H:%M"),
                    "reschedule_ref": reschedule_ref,
                }
            )
        if stored:
            self.deps.store.slot_lists.save(
                self.ctx.lead_email, stored, zone=zone, session_id=self.ctx.session_id, list_id=list_id
            )
            self.state.shown = ShownSlots(list_id, zone, stored)
        if self.mode == "naive":
            return {"available_starts_utc": [s["start_utc"] for s in stored]}
        return {
            "zone": zone,
            "slots": [{k: s[k] for k in ("slot_id", "label", "local_date", "local_time")} for s in stored],
            "more_available": len(slots) > len(chosen),
        }

    # Writes: shared --------------------------------------------------------------------------------------

    def _lead_name(self) -> str:
        return self.ctx.lead_name or self.ctx.lead_email.split("@", 1)[0]

    def _slot(self, slot_id: str) -> dict[str, Any] | None:
        found = self.deps.store.slot_lists.find_slot(
            self.ctx.lead_email, slot_id, ttl_s=float(self.deps.settings.slot_ttl_seconds)
        )
        if found is None:
            self.state.event("slot_ids", "unknown_or_expired_slot", slot_id)
        return found

    def _allowed(self, booking_uid: str) -> bool:
        """On the widget channel, only bookings made in the same widget session can be changed or listed."""
        if self.ctx.channel != "widget":
            return True
        return self.deps.store.widget_bookings.owns(self.ctx.session_id, booking_uid)

    def _record(self, action: BookingAction, result: WriteOk, status: WriteStatus) -> WriteRecord:
        record = WriteRecord(action, result.booking, self.ctx.zone, status, result.previous_ref)
        self.state.writes.append(record)
        if self.ctx.channel == "widget" and action != "cancelled":
            self.deps.store.widget_bookings.add(self.ctx.session_id, result.booking.ref)
        return record

    # Guard hooks in the write path -------------------------------------------------------------------------

    async def _hook_existing_booking(self) -> BookingRecord | None:
        """Hook for ``lead_lock``'s policy: one active booking per lead and event key. A booking returned here
        turns a new booking into a reschedule offer (``already_booked``); the calendar is the source, since a
        booking made outside the agent counts too. ``None``: book as asked.

        The calendar, not the ledger, because a booking made outside the agent (a setup booking, one from
        another channel this session's ledger never recorded) is not in the ledger but is still real. The
        lookup is not scoped to this session or channel: the policy is a per-lead lock across all of them.
        A calendar that cannot be read right now answers ``None`` (book as asked) rather than blocking a
        booking on a read failure; the write itself still goes through ``claim_ledger``'s read-back."""
        if not self.on("lead_lock"):
            return None
        now = self.now
        start, end = now - timedelta(days=LIST_PAST_DAYS), now + timedelta(days=LIST_FUTURE_DAYS)
        found = await self.deps.calendar.list_bookings(lead_email=self.ctx.lead_email, start=start, end=end)
        if self._retry_lookup(found):
            found = await self.deps.calendar.list_bookings(
                lead_email=self.ctx.lead_email, start=start, end=end
            )
        if isinstance(found, Unavailable):
            return None
        active = next((b for b in found if b.active), None)
        if active is not None:
            self.state.event("lead_lock", "reschedule_offered", f"existing={active.ref}")
        return active

    async def _hook_write_key(
        self, kind: Literal["create", "reschedule", "cancel"], *, start: datetime | None, ref: str | None
    ) -> tuple[str | None, WriteOk | None]:
        """Hook for ``idempotency``: the key registered as ``pending`` before dispatch (sent with a create as
        ``metadata.bt_idem``).

        A **create** whose key is already ``committed`` or ``adopted`` — the same lead booking the same slot
        again, with no cancel in between — is not dispatched a second time: the booking it already produced
        is read back and returned instead, so a repeated ``book_slot`` call (a retried tool call, a scripted
        misbehaviour) cannot land a genuine duplicate. A reschedule or cancel whose key is already finished
        is dispatched again regardless: Cal.com itself rejects it as ``duplicate`` (the booking it targeted
        already moved or is already cancelled), which is the more informative outcome for a deliberate
        repeat, such as cancelling an already-cancelled booking.

        Any kind whose key is still ``pending`` — an earlier attempt for the same intent may be in flight,
        elsewhere, or was interrupted before it could finish — is checked the same way as an unknown outcome
        (:func:`~booking_truth.agent.guards.idempotency.verify_landed`) before this attempt dispatches.

        The second element of the result carries an already-found write; the caller skips its calendar call
        for it. ``(None, None)``: no key (guard off)."""
        if not self.on("idempotency"):
            return None, None
        store, lead, event_key = self.deps.store, self.ctx.lead_email, self.deps.calendar.event_key
        if kind == "create":
            assert start is not None
            generation = store.generations.current(lead, event_key)
            key = create_idem_key(lead, event_key, start, generation)
            record, previous = store.idem.begin(
                key,
                kind="create",
                lead_email=lead,
                event_key=event_key,
                slot_start_utc=start,
                generation=generation,
            )
        else:
            assert ref is not None
            key = change_idem_key(ref, kind, start)
            record, previous = store.idem.begin(
                key, kind=kind, lead_email=lead, event_key=event_key, slot_start_utc=start
            )
        if kind == "create" and previous in ("committed", "adopted") and record.booking_ref:
            found = await known_write(self.deps.calendar, record.booking_ref)
            if found is not None:
                self.state.event("idempotency", "replayed", f"{kind} {found.ref}")
                return key, WriteOk(found)
        if previous == "pending":
            landed = await verify_landed(
                self.deps.calendar, kind, lead_email=lead, key=key, start=start, ref=ref
            )
            if landed is not None:
                store.idem.adopt(key, landed.booking.ref)
                self.state.event("idempotency", "replayed", f"{kind} {landed.booking.ref}")
                return key, landed
        return key, None

    async def _dispatch_guarded(self, dispatch: Dispatch) -> WriteResult:
        """Hook for ``lead_lock``: the one choke point every calendar write (a fresh dispatch, and a retry
        of one) goes through. Once the turn's lease has been taken over by another owner
        (``ctx.state.lock_lost`` set — see :meth:`~booking_truth.agent.core.AgentCore._renew_lead_lock``),
        this turn is no longer serialised against whatever that new owner is doing for the same lead, so
        sending the write now could double-book with nothing left to catch it; the write is refused instead
        of dispatched. With the guard off, or the lease still held, dispatches exactly as asked."""
        lost = self.ctx.state.lock_lost
        if lost is not None and lost.is_set():
            self.state.event(
                "lead_lock", "lease_lost", "the lease was taken over mid-turn; the write was not sent"
            )
            return WriteUnknown("lease_lost", "the lead's lease was lost while this turn was still running")
        return await dispatch()

    async def _hook_after_unknown(
        self,
        kind: Literal["create", "reschedule", "cancel"],
        result: WriteUnknown,
        *,
        key: str | None,
        start: datetime | None,
        ref: str | None,
        dispatch: Dispatch | None = None,
    ) -> WriteResult:
        """Hook for ``idempotency``: on a timeout (never for a malformed or server-error answer, which are
        not ambiguous about having committed), verify before retrying: adopt a booking that already landed,
        else retry once with the same key and check again if that retry is itself unknown. Without the guard,
        or for any other reason, the unknown result stands."""
        if not self.on("idempotency") or key is None or result.reason != "timeout":
            return result
        lead = self.ctx.lead_email
        current: WriteResult = result
        reason = result.reason
        for attempt in range(2):
            landed = await verify_landed(
                self.deps.calendar, kind, lead_email=lead, key=key, start=start, ref=ref
            )
            if landed is not None:
                self.deps.store.idem.adopt(key, landed.booking.ref)
                self.state.event("idempotency", "adopted", f"{kind} {landed.booking.ref}")
                return landed
            if attempt == 1 or dispatch is None:
                break
            self.state.event("idempotency", "retry", f"{kind}: {reason}")
            current = await self._dispatch_guarded(dispatch)
            if not isinstance(current, WriteUnknown):
                if not isinstance(current, WriteOk):
                    self.deps.store.idem.fail(key)
                return current
            reason = current.reason
        self.deps.store.idem.fail(key)
        self.state.event("idempotency", "gave_up", f"{kind}: {reason}")
        return current

    async def _hook_verify(self, action: BookingAction, result: WriteOk) -> WriteStatus:
        """``claim_ledger``: read the write back (retrying for up to 5 seconds) and record it in the ledger as
        ``verified`` or ``unverified``. Without the guard the write result is ``trusted`` as it is, with no
        read-back and no ledger entry."""
        if not self.on("claim_ledger"):
            return "trusted"
        booking = result.booking
        checked = await read_back(self.deps.calendar, action, booking, self.ctx.lead_email)
        status: Literal["verified", "unverified"] = "verified" if checked.confirmed else "unverified"
        self.deps.store.claims.record(
            lead_email=self.ctx.lead_email,
            event_key=self.deps.calendar.event_key,
            action=action,
            booking_ref=booking.ref,
            start_utc=booking.start,
            end_utc=booking.end,
            status=status,
            zone=self.ctx.zone,
            session_id=self.ctx.session_id,
            channel=self.ctx.channel,
            previous_ref=result.previous_ref,
        )
        reads = f"{checked.attempts} read{'s' if checked.attempts != 1 else ''}"
        self.state.event("claim_ledger", status, f"{action} {booking.ref}: {checked.detail} ({reads})")
        return status

    def _hook_booking_gone(self, ref: str, why: str) -> None:
        """``claim_ledger``: the calendar says a booking no longer exists or is already cancelled, so its
        ledger entries stop supporting claims."""
        if self.on("claim_ledger") and self.deps.store.claims.void(ref):
            self.state.event("claim_ledger", "entry_voided", f"{ref}: {why}")

    async def _hook_after_write(self, record: WriteRecord, *, key: str | None, replay: bool = False) -> None:
        """Hook for ``idempotency`` bookkeeping after a successful write: commit the key to the booking it
        produced, unless the row was already finished before this call began (``replay``: the result is a
        repeat of an earlier write, adopted by :meth:`_hook_write_key` without dispatching); and bump the
        lead's generation after a cancel that is newly resolved in this call, so a later booking of the same
        slot gets a fresh key rather than replaying the cancelled one."""
        if not self.on("idempotency") or key is None or replay:
            return
        stored = self.deps.store.idem.get(key)
        if stored is not None and stored.status == "pending":
            self.deps.store.idem.commit(key, record.booking.ref)
        if record.action == "cancelled":
            self.deps.store.generations.increment(self.ctx.lead_email, self.deps.calendar.event_key)

    # Create ----------------------------------------------------------------------------------------------

    async def create(self, start: datetime) -> tuple[str, WriteRecord | None, BookingRecord | None, str]:
        """Book ``start``. Returns ``(outcome, record, existing, detail)`` with outcome ``booked``,
        ``unconfirmed``, ``already_booked``, ``slot_taken``, ``rejected`` or ``unknown``."""
        self.state.last_book_start = start
        existing = await self._hook_existing_booking()
        if existing is not None:
            return "already_booked", None, existing, ""
        key, adopted = await self._hook_write_key("create", start=start, ref=None)
        replay = adopted is not None

        async def dispatch() -> WriteResult:
            return await self.deps.calendar.create_booking(
                start=start,
                lead_email=self.ctx.lead_email,
                lead_name=self._lead_name(),
                lead_zone=self.ctx.zone,
                idem_key=key,
            )

        result: WriteResult = adopted if adopted is not None else await self._dispatch_guarded(dispatch)
        if isinstance(result, WriteUnknown):
            result = await self._hook_after_unknown(
                "create", result, key=key, start=start, ref=None, dispatch=dispatch
            )
        if isinstance(result, WriteOk):
            status = await self._hook_verify("booked", result)
            record = self._record("booked", result, status)
            await self._hook_after_write(record, key=key, replay=replay)
            return ("unconfirmed" if status == "unverified" else "booked"), record, None, ""
        if isinstance(result, WriteRejected):
            return ("slot_taken" if result.reason == "slot_taken" else "rejected"), None, None, result.detail
        return "unknown", None, None, result.detail or result.reason

    def _existing_view(self, existing: BookingRecord) -> dict[str, str]:
        return {
            "booking_uid": existing.ref,
            "label": self._label(existing.start),
            "start_utc": iso_z(existing.start),
        }

    async def _book_slot(self, args: dict[str, Any]) -> Any:
        slot_id = self._text(args, "slot_id")
        slot = self._slot(slot_id)
        if slot is None:
            return {"booked": False, "reason": "unknown_or_expired_slot", "instruction": EXPIRED_INSTRUCTION}
        start = parse_iso(str(slot["start_utc"]))
        outcome, record, existing, _ = await self.create(start)
        if outcome == "already_booked" and existing is not None:
            changeable = self._allowed(existing.ref)
            # ``listed_booking_ref`` covers a list_my_bookings call earlier in this same turn; the slot's
            # own ``reschedule_ref`` (stamped when its list was offered, in ``_slot_result``) covers the
            # far more common case, a pick on a *later* turn than the one that offered it, with no fresh
            # list_my_bookings call in between (the run-1 pattern: offer turn, then a separate pick turn).
            asked_to_move = changeable and existing.ref in {
                self.state.listed_booking_ref,
                slot.get("reschedule_ref"),
            }
            if changeable and not asked_to_move:
                self.state.reschedule_offer = {"booking_uid": existing.ref, "slot_id": slot_id}
            if asked_to_move:
                instruction = ALREADY_BOOKED_MOVE_NOW_INSTRUCTION
            elif changeable:
                instruction = ALREADY_BOOKED_INSTRUCTION
            else:
                instruction = ALREADY_BOOKED_ELSEWHERE_INSTRUCTION
            return {
                "booked": False,
                "reason": "already_booked",
                "existing": self._existing_view(existing),
                "instruction": instruction,
            }
        if outcome == "booked" and record is not None:
            return {
                "booked": True,
                "booking_uid": record.booking.ref,
                "label": self._label(record.booking.start),
                "zone": self.ctx.zone,
            }
        if outcome == "unconfirmed":
            return {"booked": "unconfirmed", "instruction": UNCONFIRMED_INSTRUCTION}
        if outcome == "slot_taken":
            return {"booked": False, "reason": "slot_taken", "instruction": SLOT_TAKEN_INSTRUCTION}
        return {"booked": False, "reason": "calendar_error", "instruction": CALENDAR_ERROR_INSTRUCTION}

    async def _book_naive(self, args: dict[str, Any]) -> Any:
        raw = self._text(args, "start_iso")
        start = _parse_start(raw)
        if start is None:
            return f"Error: could not parse start_iso {raw!r}"
        outcome, record, existing, detail = await self.create(start)
        if outcome == "booked" and record is not None:
            return {"booked": True, "booking_uid": record.booking.ref, "start": iso_z(record.booking.start)}
        if outcome == "already_booked" and existing is not None:
            asked_to_move = self.state.listed_booking_ref == existing.ref
            instruction = ALREADY_BOOKED_MOVE_NOW_INSTRUCTION if asked_to_move else ALREADY_BOOKED_INSTRUCTION
            return {
                "booked": False,
                "reason": "already_booked",
                "existing": {"booking_uid": existing.ref, "start": iso_z(existing.start)},
                "instruction": instruction,
            }
        if outcome == "unconfirmed":
            return {"booked": "unconfirmed", "instruction": UNCONFIRMED_INSTRUCTION}
        return error_text(detail or outcome)

    # Reschedule ------------------------------------------------------------------------------------------

    async def reschedule(self, ref: str, start: datetime) -> tuple[str, WriteRecord | None, str]:
        """Move ``ref`` to ``start``. Outcome: ``rescheduled``, ``unconfirmed``, ``not_allowed``,
        ``slot_taken``, ``not_found``, ``rejected`` or ``unknown``."""
        if not self._allowed(ref):
            return "not_allowed", None, ""
        key, adopted = await self._hook_write_key("reschedule", start=start, ref=ref)
        replay = adopted is not None

        async def dispatch() -> WriteResult:
            return await self.deps.calendar.reschedule(
                ref=ref, new_start=start, idem_key=key, reason="Rescheduled by the prospect"
            )

        result: WriteResult = adopted if adopted is not None else await self._dispatch_guarded(dispatch)
        if isinstance(result, WriteUnknown):
            result = await self._hook_after_unknown(
                "reschedule", result, key=key, start=start, ref=ref, dispatch=dispatch
            )
        if isinstance(result, WriteOk):
            status = await self._hook_verify("rescheduled", result)
            record = self._record("rescheduled", result, status)
            await self._hook_after_write(record, key=key, replay=replay)
            return ("unconfirmed" if status == "unverified" else "rescheduled"), record, ""
        if isinstance(result, WriteRejected):
            if result.reason == "not_found":
                self._hook_booking_gone(ref, "the calendar has no such booking")
            if result.reason in ("slot_taken", "not_found"):
                return result.reason, None, result.detail
            return "rejected", None, result.detail
        return "unknown", None, result.detail or result.reason

    async def _reschedule_guarded(self, args: dict[str, Any]) -> Any:
        ref = self._text(args, "booking_uid")
        slot_id = self._text(args, "slot_id")
        slot = self._slot(slot_id)
        if slot is None:
            return {
                "rescheduled": False,
                "reason": "unknown_or_expired_slot",
                "instruction": EXPIRED_INSTRUCTION,
            }
        outcome, record, _ = await self.reschedule(ref, parse_iso(str(slot["start_utc"])))
        if outcome == "rescheduled" and record is not None:
            return {
                "rescheduled": True,
                "booking_uid": record.booking.ref,
                "label": self._label(record.booking.start),
            }
        return self._reschedule_failure(outcome)

    def _reschedule_failure(self, outcome: str) -> dict[str, Any]:
        if outcome == "unconfirmed":
            return {"rescheduled": "unconfirmed", "instruction": UNCONFIRMED_INSTRUCTION}
        if outcome == "slot_taken":
            return {"rescheduled": False, "reason": "slot_taken", "instruction": SLOT_TAKEN_INSTRUCTION}
        if outcome in ("not_found", "not_allowed"):
            return {"rescheduled": False, "reason": outcome, "instruction": NOT_CHANGED_INSTRUCTION}
        return {"rescheduled": False, "reason": "calendar_error", "instruction": NOT_CHANGED_INSTRUCTION}

    async def _reschedule_naive(self, args: dict[str, Any]) -> Any:
        ref = self._text(args, "booking_uid")
        raw = self._text(args, "start_iso")
        start = _parse_start(raw)
        if start is None:
            return f"Error: could not parse start_iso {raw!r}"
        outcome, record, detail = await self.reschedule(ref, start)
        if outcome == "rescheduled" and record is not None:
            return {
                "rescheduled": True,
                "booking_uid": record.booking.ref,
                "start": iso_z(record.booking.start),
            }
        if outcome == "unconfirmed":
            return {"rescheduled": "unconfirmed", "instruction": UNCONFIRMED_INSTRUCTION}
        if outcome == "not_allowed":
            return "Error: this booking was not made in this conversation"
        return error_text(detail or outcome)

    # Cancel ----------------------------------------------------------------------------------------------

    async def cancel(self, ref: str, reason: str) -> tuple[str, WriteRecord | None, str]:
        """Cancel ``ref``. Outcome: ``cancelled``, ``unconfirmed``, ``not_allowed``, ``not_found``,
        ``already_cancelled``, ``rejected`` or ``unknown``."""
        if not self._allowed(ref):
            return "not_allowed", None, ""
        cancel_reason = reason or "Cancelled by the prospect"
        key, adopted = await self._hook_write_key("cancel", start=None, ref=ref)
        replay = adopted is not None

        async def dispatch() -> WriteResult:
            return await self.deps.calendar.cancel(ref=ref, reason=cancel_reason, idem_key=key)

        result: WriteResult = adopted if adopted is not None else await self._dispatch_guarded(dispatch)
        if isinstance(result, WriteUnknown):
            result = await self._hook_after_unknown(
                "cancel", result, key=key, start=None, ref=ref, dispatch=dispatch
            )
        if isinstance(result, WriteOk):
            status = await self._hook_verify("cancelled", result)
            record = self._record("cancelled", result, status)
            await self._hook_after_write(record, key=key, replay=replay)
            return ("unconfirmed" if status == "unverified" else "cancelled"), record, ""
        if isinstance(result, WriteRejected):
            if result.reason == "not_found":
                self._hook_booking_gone(ref, "the calendar has no such booking")
                return "not_found", None, result.detail
            if result.reason == "duplicate":
                self._hook_booking_gone(ref, "the booking was already cancelled")
                return "already_cancelled", None, result.detail
            return "rejected", None, result.detail
        return "unknown", None, result.detail or result.reason

    async def _cancel_booking(self, args: dict[str, Any]) -> Any:
        ref = self._text(args, "booking_uid")
        reason = self._text(args, "reason", required=False)
        outcome, record, detail = await self.cancel(ref, reason)
        if outcome == "cancelled" and record is not None:
            if self.mode == "naive":
                return {"cancelled": True, "booking_uid": ref, "start": iso_z(record.booking.start)}
            return {"cancelled": True, "booking_uid": ref, "label": self._label(record.booking.start)}
        if outcome == "unconfirmed":
            return {"cancelled": "unconfirmed", "instruction": UNCONFIRMED_INSTRUCTION}
        if self.mode == "naive" and outcome in ("rejected", "unknown"):
            return error_text(detail or outcome)
        reason_out = (
            outcome if outcome in ("not_found", "not_allowed", "already_cancelled") else "calendar_error"
        )
        return {"cancelled": False, "reason": reason_out}

    # list_my_bookings ------------------------------------------------------------------------------------

    async def bookings(self) -> list[BookingRecord] | Unavailable:
        """The lead's active bookings (on the widget channel, only this session's); with ``fail_closed`` a
        failed lookup is tried once more before it counts as unavailable."""
        now = self.now
        start, end = now - timedelta(days=LIST_PAST_DAYS), now + timedelta(days=LIST_FUTURE_DAYS)
        found = await self.deps.calendar.list_bookings(lead_email=self.ctx.lead_email, start=start, end=end)
        if self._retry_lookup(found):
            found = await self.deps.calendar.list_bookings(
                lead_email=self.ctx.lead_email, start=start, end=end
            )
        if isinstance(found, Unavailable):
            return found
        return [b for b in found if b.active and self._allowed(b.ref)]

    async def _list_my_bookings(self, args: dict[str, Any]) -> Any:
        found = await self.bookings()
        if isinstance(found, Unavailable):
            return self._unavailable(found)
        if found and self.state.listed_booking_ref is None:
            # One active booking per lead per event key (the ``lead_lock`` policy), so the first is the one.
            self.state.listed_booking_ref = found[0].ref
        if self.mode == "naive":
            return {
                "bookings": [
                    {"booking_uid": b.ref, "start": iso_z(b.start), "end": iso_z(b.end), "status": b.status}
                    for b in found
                ]
            }
        return {
            "bookings": [
                {
                    "booking_uid": b.ref,
                    "label": self._label(b.start),
                    "status": b.status,
                    "start_utc": iso_z(b.start),
                }
                for b in found
            ]
        }

    # handoff_to_human ------------------------------------------------------------------------------------

    async def handoff(self, summary: str, preferred_times_text: str) -> Handoff:
        handoff = self.deps.store.handoffs.create(
            lead_email=self.ctx.lead_email,
            summary=summary,
            preferred_times_text=preferred_times_text,
            session_id=self.ctx.session_id,
        )
        self.state.handoffs.append(handoff.id)
        self.deps.handoffs.notify(handoff)
        return handoff

    async def _handoff_to_human(self, args: dict[str, Any]) -> Any:
        summary = self._text(args, "summary")
        preferred = self._text(args, "preferred_times_text", required=False)
        handoff = await self.handoff(summary, preferred)
        return {"handoff": "created", "reference": f"H{handoff.id}"}

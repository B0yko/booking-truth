"""The calendar adapter contract: sealed result types and the ``CalendarAdapter`` protocol.

Adapter methods never raise for vendor failures. Each one returns a member of a closed union, so a caller has
to handle every outcome explicitly (``match`` over the union):

- ``find_slots`` returns ``Slots | Unavailable``. ``Unavailable.reason`` is one of ``error`` (an HTTP error or
  a network failure), ``timeout``, ``not_found`` (the event type or calendar does not exist), ``malformed``
  (a response that does not have the documented shape) or ``missing_calendar`` (Google: the calendar is absent
  from a ``freeBusy`` answer or carries errors). A failed lookup is never an empty ``Slots``.
- The writes (``create_booking``, ``reschedule``, ``cancel``) return
  ``WriteOk | WriteRejected | WriteUnknown``. ``WriteRejected`` means the vendor refused the write and nothing
  changed: ``slot_taken``, ``invalid``, ``not_found`` (the booking to change does not exist) or ``duplicate``
  (the change was already applied, e.g. a second cancel). ``WriteUnknown`` means the outcome is not known and
  the write **may have committed**: ``timeout``, ``server_error`` or ``malformed``. Callers verify before they
  retry.
- ``get_booking`` returns ``BookingRecord | NotFound | Unavailable`` and ``list_bookings`` returns
  ``tuple[BookingRecord, ...] | Unavailable``.

Every datetime is timezone-aware and stored in UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from booking_truth.timeutil import ensure_utc

BookingStatus = Literal["active", "cancelled"]


def _interval(start: datetime, end: datetime, what: str) -> tuple[datetime, datetime]:
    start_utc, end_utc = ensure_utc(start), ensure_utc(end)
    if end_utc <= start_utc:
        raise ValueError(
            f"{what} must end after it starts ({start_utc.isoformat()} .. {end_utc.isoformat()})"
        )
    return start_utc, end_utc


@dataclass(frozen=True)
class Slot:
    """A free interval on the host calendar, in UTC."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start, end = _interval(self.start, self.end, "a slot")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class Slots:
    """A successful availability lookup, sorted by start. An empty tuple means the calendar has no free time
    in the window, never that the lookup failed."""

    slots: tuple[Slot, ...]


@dataclass(frozen=True)
class Unavailable:
    """The calendar could not be read.

    ``reason``: error | timeout | not_found | malformed | missing_calendar.
    """

    reason: str
    detail: str = ""


SlotsResult = Slots | Unavailable


@dataclass(frozen=True)
class BookingRecord:
    """One booking as the vendor reports it."""

    #: Cal.com booking uid or Google event id.
    ref: str
    start: datetime
    end: datetime
    status: BookingStatus
    #: The lead's email: the attendee (Cal.com) or ``bt_lead_email`` (Google).
    lead_email: str | None
    #: The idempotency key: ``metadata.bt_idem`` (Cal.com) or the event id (Google).
    idem_key: str | None
    #: The vendor object exactly as received.
    raw: dict[str, Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        start, end = _interval(self.start, self.end, f"booking {self.ref!r}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if self.status not in ("active", "cancelled"):
            raise ValueError(f"unknown booking status {self.status!r}")

    @property
    def active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class WriteOk:
    """The write happened. ``booking`` is the booking after the write; for a reschedule it is the new booking
    and ``previous_ref`` the booking it replaced."""

    booking: BookingRecord
    previous_ref: str | None = None


@dataclass(frozen=True)
class WriteRejected:
    """The vendor refused the write; nothing changed.

    ``reason``: slot_taken | invalid | not_found | duplicate.
    """

    reason: str
    detail: str = ""


@dataclass(frozen=True)
class WriteUnknown:
    """The outcome is unknown and the write may have committed.

    ``reason``: timeout | server_error | malformed.
    """

    reason: str
    detail: str = ""


WriteResult = WriteOk | WriteRejected | WriteUnknown


@dataclass(frozen=True)
class NotFound:
    """The vendor says the booking does not exist."""

    detail: str = ""


ReadResult = BookingRecord | NotFound | Unavailable
ListResult = tuple[BookingRecord, ...] | Unavailable


@runtime_checkable
class CalendarAdapter(Protocol):
    """One host calendar behind one event type (Cal.com) or one calendar id (Google)."""

    kind: Literal["calcom", "google"]
    #: The event key used in idempotency keys: the Cal.com event type id, or ``BT_EVENT_KEY`` for Google.
    event_key: str

    async def find_slots(self, start: datetime, end: datetime) -> SlotsResult:
        """Free slots with ``start <= slot.start < end``."""
        ...

    async def create_booking(
        self, *, start: datetime, lead_email: str, lead_name: str, lead_zone: str, idem_key: str | None
    ) -> WriteResult: ...

    async def get_booking(self, ref: str) -> ReadResult: ...

    async def list_bookings(self, *, lead_email: str, start: datetime, end: datetime) -> ListResult:
        """The lead's bookings that lie inside ``[start, end]``."""
        ...

    async def reschedule(
        self, *, ref: str, new_start: datetime, idem_key: str | None, reason: str
    ) -> WriteResult:
        """Move a booking. On success ``WriteOk.booking`` is the booking at the new time and
        ``WriteOk.previous_ref`` the booking that was moved (Cal.com creates a new booking with a new uid)."""
        ...

    async def cancel(self, *, ref: str, reason: str, idem_key: str | None) -> WriteResult: ...

    async def aclose(self) -> None: ...

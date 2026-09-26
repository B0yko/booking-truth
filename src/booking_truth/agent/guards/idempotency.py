"""Idempotency keys and verify-before-retry (``idempotency``).

A calendar write is dispatched under a key computed before it, so a duplicate attempt — a retried tool
call, a duplicate message, a client timeout whose write may have committed anyway — can be told from a
genuinely new intent instead of producing a second booking:

- **create**: ``sha256(normalised email | event key | slot start UTC | generation)``. ``generation`` is a
  per (lead, event key) counter that starts at 0 and moves up by one after every cancel
  (:class:`~booking_truth.store.repos.GenerationsRepo`), so rebooking the same slot after a cancel gets a
  fresh key instead of replaying the cancelled booking.
- **reschedule** / **cancel**: ``sha256(booking_uid | kind | new slot start UTC or "")``, empty for a
  cancel.

Each key is written to SQLite as ``pending`` before the calendar call (``store.idem``), and Cal.com carries
it as ``metadata.bt_idem`` on a create (``calendars/calcom.py``). Three situations call for a look before a
write is dispatched (or dispatched again), all answered by :func:`verify_landed`:

- a **create** whose key is already ``committed`` or ``adopted`` — the same lead booking the same slot
  again, with no cancel in between — is read back and adopted instead of dispatched, so a retried tool
  call cannot land a genuine duplicate (``ToolExecutor._hook_write_key``). A reschedule or cancel whose key
  is already finished dispatches again regardless: Cal.com itself rejects it as ``duplicate`` (the booking
  it targeted already moved or is already cancelled), which is the more informative outcome for a
  deliberate repeat, such as cancelling an already-cancelled booking;
- any kind whose key is still ``pending`` — an earlier attempt for the same intent may be in flight, or was
  interrupted before it could finish — is checked the same way, also before dispatch;
- a write whose outcome came back ``WriteUnknown("timeout")`` — the calendar may have committed it before
  the client gave up — is checked *after* the attempt, before any retry
  (``ToolExecutor._hook_after_unknown``). A ``WriteUnknown`` for another reason (a malformed answer, a
  server error with no ambiguity about it having committed) is not retried here.

A write that already landed is adopted: the booking it produced becomes the result, and the calendar is
never asked to repeat it. One not found after a timeout is retried once with the same key; an unknown
outcome after that retry is checked once more before the write gives up (the row becomes ``failed``, so a
later attempt with the same key dispatches fresh rather than waiting on it forever).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Literal

from booking_truth.calendars.base import BookingRecord, CalendarAdapter, Unavailable, WriteOk
from booking_truth.store import normalize_email
from booking_truth.timeutil import iso_z

WriteKind = Literal["create", "reschedule", "cancel"]

#: How far before a create or reschedule's target start :func:`verify_landed` lists the lead's bookings
#: from (the slot window the design describes).
VERIFY_MARGIN_S = 60.0
#: Comfortably longer than any plausible meeting length. A vendor listing bounds itself by a booking's own
#: *end*, not its start, so the upper bound of the search has to clear the booking's length, not just the
#: margin around its start.
_MAX_MEETING_SPAN = timedelta(hours=6)


def create_idem_key(lead_email: str, event_key: str, start: datetime, generation: int) -> str:
    """The key of a booking attempt: stable across a retry of the same intent, fresh after a cancel."""
    raw = f"{normalize_email(lead_email)}|{event_key}|{iso_z(start)}|{generation}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def change_idem_key(booking_uid: str, kind: Literal["reschedule", "cancel"], start: datetime | None) -> str:
    """The key of a reschedule or cancel attempt; a cancel carries no slot."""
    raw = f"{booking_uid}|{kind}|{iso_z(start) if start is not None else ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def known_write(calendar: CalendarAdapter, ref: str) -> BookingRecord | None:
    """``ref`` as the calendar has it now, or ``None`` when it cannot be read (the caller falls back to
    :func:`verify_landed` or, failing that, dispatches fresh)."""
    found = await calendar.get_booking(ref)
    return found if isinstance(found, BookingRecord) else None


async def verify_landed(
    calendar: CalendarAdapter,
    kind: WriteKind,
    *,
    lead_email: str,
    key: str,
    start: datetime | None,
    ref: str | None,
    margin_s: float = VERIFY_MARGIN_S,
) -> WriteOk | None:
    """A booking that already reflects this write's intent, found without dispatching it again.

    ``cancel``: ``ref`` read back and found cancelled. ``create`` / ``reschedule``: an active booking of
    the lead starting exactly at ``start``, listed from ``margin_s`` before it, preferring one whose own
    idempotency key matches ``key`` (two attempts could in principle land in the same minute); ``None``
    when nothing matches or the calendar cannot be read right now.
    """
    if kind == "cancel":
        if not ref:
            return None
        found = await calendar.get_booking(ref)
        if isinstance(found, BookingRecord) and found.status == "cancelled":
            return WriteOk(found)
        return None
    if start is None:
        return None
    margin = timedelta(seconds=margin_s)
    found_list = await calendar.list_bookings(
        lead_email=lead_email, start=start - margin, end=start + margin + _MAX_MEETING_SPAN
    )
    if isinstance(found_list, Unavailable):
        return None
    candidates = [b for b in found_list if b.active and b.start == start]
    if not candidates:
        return None
    matched = next((b for b in candidates if b.idem_key == key), candidates[0])
    return WriteOk(matched, previous_ref=ref if kind == "reschedule" else None)


__all__ = [
    "VERIFY_MARGIN_S",
    "WriteKind",
    "change_idem_key",
    "create_idem_key",
    "known_write",
    "verify_landed",
]

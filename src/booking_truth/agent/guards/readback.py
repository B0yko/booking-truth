"""Read-back verification of calendar writes (``claim_ledger``).

A write counts only after the calendar confirms it when asked again: ``get_booking(ref)`` must show the
booking active at the written start for the lead (a create or a reschedule), or cancelled (a cancel). A read
that fails or disagrees is retried every 0.5 seconds for up to 5 seconds; a write still unconfirmed after
that is ``unverified``, and the agent tells the prospect so and hands off instead of claiming it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from booking_truth.agent.models import BookingAction
from booking_truth.calendars.base import BookingRecord, CalendarAdapter, NotFound, ReadResult, Unavailable
from booking_truth.store import normalize_email
from booking_truth.timeutil import iso_z

READBACK_WINDOW_S = 5.0
READBACK_STEP_S = 0.5
#: A single read is never given less time than this, even at the end of the window.
MIN_READ_S = 0.2


@dataclass(frozen=True)
class ReadBack:
    """The result of reading a write back: ``confirmed`` or not, why, and how many reads it took."""

    confirmed: bool
    detail: str
    attempts: int


def _minute(record: BookingRecord) -> str:
    return iso_z(record.start)[:16]


def confirms(
    action: BookingAction, written: BookingRecord, found: ReadResult, lead_email: str
) -> tuple[bool, str]:
    """Whether one read-back confirms the write, and a short reason."""
    if isinstance(found, Unavailable):
        return False, f"read-back failed ({found.reason})"
    if isinstance(found, NotFound):
        return False, "read-back found no such booking"
    if found.ref != written.ref:
        return False, f"read-back returned booking {found.ref}, not {written.ref}"
    if normalize_email(found.lead_email or "") != normalize_email(lead_email):
        return False, "the booking belongs to another attendee"
    if action == "cancelled":
        return (True, "cancelled") if not found.active else (False, "the booking is still active")
    if not found.active:
        return False, "the booking is cancelled"
    if _minute(found) != _minute(written):
        return False, f"the booking starts at {iso_z(found.start)}, not {iso_z(written.start)}"
    return True, "active at the written time"


async def read_back(
    calendar: CalendarAdapter,
    action: BookingAction,
    written: BookingRecord,
    lead_email: str,
    *,
    window_s: float | None = None,
    step_s: float | None = None,
) -> ReadBack:
    """Read ``written`` back until a read confirms it or ``window_s`` (default :data:`READBACK_WINDOW_S`)
    has passed, waiting ``step_s`` (default :data:`READBACK_STEP_S`) between reads; monotonic time."""
    window = READBACK_WINDOW_S if window_s is None else window_s
    step = READBACK_STEP_S if step_s is None else step_s
    loop = asyncio.get_running_loop()
    deadline = loop.time() + window
    attempts = 0
    while True:
        attempts += 1
        remaining = max(deadline - loop.time(), MIN_READ_S)
        try:
            found: ReadResult = await asyncio.wait_for(calendar.get_booking(written.ref), timeout=remaining)
        except TimeoutError:
            found = Unavailable("timeout", "the read-back did not answer in time")
        confirmed, detail = confirms(action, written, found, lead_email)
        if confirmed:
            return ReadBack(True, detail, attempts)
        left = deadline - loop.time()
        if left <= 0:
            return ReadBack(False, detail, attempts)
        await asyncio.sleep(min(step, left))


__all__ = ["MIN_READ_S", "READBACK_STEP_S", "READBACK_WINDOW_S", "ReadBack", "confirms", "read_back"]

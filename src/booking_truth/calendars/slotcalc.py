"""Working-hours slot computation from busy intervals.

Google Calendar's ``freeBusy`` answers with busy intervals only, so on the Google path free slots are
computed client-side from them plus ``BT_WORK_HOURS``, ``BT_WORK_DAYS``, ``BT_SLOT_MINUTES``,
``BT_MIN_NOTICE_MINUTES`` and ``BT_HORIZON_DAYS`` in ``BT_HOST_TIMEZONE``. The math is the one the sandbox
uses for its Cal.com slots (:mod:`booking_truth.sandbox.availability`), so both calendars offer the same grid,
and it also serves as the reference computation for tests and the harness.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from booking_truth.calendars.base import Slot, Slots
from booking_truth.config import Settings
from booking_truth.sandbox.availability import Hours, free_slot_starts, local_to_utc, overlaps

__all__ = [
    "Hours",
    "compute_slots",
    "free_slot_starts",
    "hours_from_settings",
    "local_to_utc",
    "overlaps",
]


def hours_from_settings(settings: Settings) -> Hours:
    """The host's working hours as configured through ``BT_*`` settings."""
    start, end = settings.work_hours_range
    return Hours(
        zone=settings.host_timezone,
        start=start,
        end=end,
        days=settings.work_days_set,
        slot_minutes=settings.slot_minutes,
        min_notice_minutes=settings.min_notice_minutes,
        horizon_days=settings.horizon_days,
    )


def compute_slots(
    hours: Hours,
    busy: Iterable[tuple[datetime, datetime]],
    start: datetime,
    end: datetime,
    now: datetime,
) -> Slots:
    """Free slots with ``start <= slot.start < end``: whole slots inside working hours on working days in the
    host zone, at least the minimum notice after ``now``, within the horizon, and clear of every busy
    interval. A local start that does not exist (DST gap) is skipped."""
    length = timedelta(minutes=hours.slot_minutes)
    starts = free_slot_starts(hours, busy, start, end, now)
    return Slots(tuple(Slot(slot_start, slot_start + length) for slot_start in starts))

"""Host availability: working-hours slot starts minus busy intervals.

The same function serves the Cal.com slots endpoint in the sandbox and, through
``booking_truth.calendars.slotcalc``, the Google path, where free time is computed client-side from
``freeBusy``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from booking_truth.timeutil import ensure_utc


@dataclass(frozen=True)
class Hours:
    """Working hours of the host in its own zone."""

    zone: str
    start: time
    end: time
    days: tuple[int, ...]  # ISO weekdays, Monday = 1
    slot_minutes: int
    min_notice_minutes: int
    horizon_days: int


def local_to_utc(day: date, at: time, zone: ZoneInfo) -> datetime | None:
    """Convert a local wall-clock time to UTC; ``None`` when that local time does not exist (DST gap)."""
    local = datetime.combine(day, at, tzinfo=zone)
    utc = local.astimezone(ZoneInfo("UTC"))
    if utc.astimezone(zone).replace(tzinfo=None) != local.replace(tzinfo=None):
        return None
    return utc


def overlaps(start: datetime, end: datetime, busy: Iterable[tuple[datetime, datetime]]) -> bool:
    return any(start < b_end and b_start < end for b_start, b_end in busy)


def free_slot_starts(
    hours: Hours,
    busy: Iterable[tuple[datetime, datetime]],
    window_start: datetime,
    window_end: datetime,
    now: datetime,
) -> list[datetime]:
    """Every free slot start (UTC) with ``window_start <= start < window_end``.

    A slot is offered when it lies wholly inside working hours on a working day in the host zone,
    starts at least ``min_notice`` after ``now`` and no later than ``horizon`` after ``now``, and does
    not overlap a busy interval.
    """
    zone = ZoneInfo(hours.zone)
    window_start, window_end, now = ensure_utc(window_start), ensure_utc(window_end), ensure_utc(now)
    earliest = max(window_start, now + timedelta(minutes=hours.min_notice_minutes))
    latest = min(window_end, now + timedelta(days=hours.horizon_days))
    if earliest >= latest:
        return []
    busy_list = [(ensure_utc(s), ensure_utc(e)) for s, e in busy]
    length = timedelta(minutes=hours.slot_minutes)
    day = earliest.astimezone(zone).date() - timedelta(days=1)
    last_day = latest.astimezone(zone).date() + timedelta(days=1)
    result: list[datetime] = []
    while day <= last_day:
        if day.isoweekday() in hours.days:
            cursor = datetime.combine(day, hours.start)
            day_end = datetime.combine(day, hours.end)
            while cursor + length <= day_end:
                start = local_to_utc(day, cursor.time(), zone)
                end_local = cursor + length
                end = local_to_utc(end_local.date(), end_local.time(), zone)
                if (
                    start is not None
                    and end is not None
                    and earliest <= start < latest
                    and not overlaps(start, end, busy_list)
                ):
                    result.append(start)
                cursor += length
        day += timedelta(days=1)
    return sorted(set(result))

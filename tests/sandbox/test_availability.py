from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from booking_truth.sandbox.availability import Hours, free_slot_starts, local_to_utc

NY = "America/New_York"
HOURS = Hours(
    zone=NY,
    start=time(9),
    end=time(17),
    days=(1, 2, 3, 4, 5),
    slot_minutes=30,
    min_notice_minutes=120,
    horizon_days=400,
)


def test_full_free_weekday_has_16_slots_in_host_zone() -> None:
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)  # Thursday
    start = datetime(2026, 10, 5, 0, 0, tzinfo=ZoneInfo(NY))  # Monday
    slots = free_slot_starts(HOURS, [], start, start + timedelta(days=1), now)
    assert len(slots) == 16
    assert slots[0] == datetime(2026, 10, 5, 13, 0, tzinfo=UTC)  # 09:00 EDT
    assert slots[-1] == datetime(2026, 10, 5, 20, 30, tzinfo=UTC)  # 16:30 EDT


def test_weekend_is_closed_and_busy_blocks_are_removed() -> None:
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    sat = datetime(2026, 10, 3, 0, 0, tzinfo=ZoneInfo(NY))
    assert free_slot_starts(HOURS, [], sat, sat + timedelta(days=2), now) == []
    mon = datetime(2026, 10, 5, 0, 0, tzinfo=ZoneInfo(NY))
    busy = [(datetime(2026, 10, 5, 13, 15, tzinfo=UTC), datetime(2026, 10, 5, 14, 0, tzinfo=UTC))]
    slots = free_slot_starts(HOURS, busy, mon, mon + timedelta(days=1), now)
    assert datetime(2026, 10, 5, 13, 0, tzinfo=UTC) not in slots
    assert datetime(2026, 10, 5, 13, 30, tzinfo=UTC) not in slots
    assert datetime(2026, 10, 5, 14, 0, tzinfo=UTC) in slots


def test_minimum_notice_and_horizon() -> None:
    now = datetime(2026, 10, 5, 14, 10, tzinfo=UTC)  # 10:10 EDT Monday
    mon = datetime(2026, 10, 5, 0, 0, tzinfo=ZoneInfo(NY))
    slots = free_slot_starts(HOURS, [], mon, mon + timedelta(days=1), now)
    assert slots[0] == datetime(2026, 10, 5, 16, 30, tzinfo=UTC)  # first start >= 12:10 EDT is 12:30
    short = Hours(**{**HOURS.__dict__, "horizon_days": 1})
    far = free_slot_starts(short, [], mon, mon + timedelta(days=30), now)
    assert all(s <= now + timedelta(days=1) for s in far)


def test_dst_end_week_uses_new_offset() -> None:
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    mon = datetime(2026, 11, 2, 0, 0, tzinfo=ZoneInfo(NY))  # first Monday after US DST ends (Nov 1)
    slots = free_slot_starts(HOURS, [], mon, mon + timedelta(days=1), now)
    assert slots[0] == datetime(2026, 11, 2, 14, 0, tzinfo=UTC)  # 09:00 EST = 14:00 UTC


def test_nonexistent_local_time_is_skipped() -> None:
    zone = ZoneInfo(NY)
    assert local_to_utc(date(2027, 3, 14), time(2, 30), zone) is None
    assert local_to_utc(date(2027, 3, 14), time(3, 30), zone) == datetime(2027, 3, 14, 7, 30, tzinfo=UTC)
    night = Hours(**{**HOURS.__dict__, "start": time(1), "end": time(4), "days": (7,)})
    now = datetime(2027, 3, 1, tzinfo=UTC)
    day = datetime(2027, 3, 14, 0, 0, tzinfo=zone)
    starts = [
        s.astimezone(zone).time() for s in free_slot_starts(night, [], day, day + timedelta(days=1), now)
    ]
    assert time(2, 0) not in starts
    assert time(2, 30) not in starts
    assert time(3, 0) in starts

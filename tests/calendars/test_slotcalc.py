"""Working-hours slot computation for calendars that report busy time only."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from calendar_env import NOW

from booking_truth.calendars.base import Slots
from booking_truth.calendars.slotcalc import Hours, compute_slots, hours_from_settings
from booking_truth.config import Settings
from booking_truth.sandbox.state import SandboxState, SeedConfig
from booking_truth.timeutil import FixedClock

NEW_YORK = Hours(
    zone="America/New_York",
    start=time(9),
    end=time(17),
    days=(1, 2, 3, 4, 5),
    slot_minutes=30,
    min_notice_minutes=120,
    horizon_days=400,
)


def test_slots_skip_busy_time_notice_and_weekends() -> None:
    monday = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
    busy = [(monday + timedelta(minutes=30), monday + timedelta(minutes=90))]
    result = compute_slots(NEW_YORK, busy, NOW, NOW + timedelta(days=5), NOW)
    starts = [s.start for s in result.slots]
    assert starts[0] == NOW + timedelta(hours=2)  # Thursday 10:00 EDT, after the two-hour notice
    assert monday in starts
    assert monday + timedelta(minutes=30) not in starts
    assert monday + timedelta(minutes=60) not in starts
    assert monday + timedelta(minutes=90) in starts
    assert not any(
        s.start.date() in (datetime(2026, 10, 3).date(), datetime(2026, 10, 4).date()) for s in result.slots
    )
    assert all(s.end - s.start == timedelta(minutes=30) for s in result.slots)


def test_slots_match_the_sandbox_grid() -> None:
    state = SandboxState(clock=FixedClock(NOW))
    state.apply_seed(
        SeedConfig(existing_bookings=[{"start": "2026-10-06T14:00:00Z", "end": "2026-10-06T16:00:00Z"}])
    )
    end = NOW + timedelta(days=14)
    result = compute_slots(state.seed.hours(), state.busy_intervals(), NOW, end, NOW)
    assert [s.start for s in result.slots] == state.free_starts(NOW, end)


def test_hours_from_settings() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        host_timezone="Asia/Kolkata",
        work_hours="10:00-13:00",
        work_days="mon,wed",
        slot_minutes=45,
        min_notice_minutes=0,
        horizon_days=30,
    )
    hours = hours_from_settings(settings)
    assert hours == Hours("Asia/Kolkata", time(10), time(13), (1, 3), 45, 0, 30)
    wednesday = datetime(2026, 10, 7, tzinfo=UTC)
    result = compute_slots(hours, [], wednesday - timedelta(hours=6), wednesday + timedelta(days=1), NOW)
    # 10:00, 10:45, 11:30 and 12:15 IST; 13:00 would end past working hours.
    assert [s.start.strftime("%H:%M") for s in result.slots] == ["04:30", "05:15", "06:00", "06:45"]


def test_an_empty_window_is_an_empty_calendar() -> None:
    assert compute_slots(NEW_YORK, [], NOW, NOW, NOW) == Slots(())

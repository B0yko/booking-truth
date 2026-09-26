from datetime import UTC, datetime, timedelta, timezone

import pytest

from booking_truth.timeutil import FixedClock, MutableClock, ensure_utc, iso_ms_z, iso_z, parse_iso


def test_iso_z_formats_utc_with_z_suffix() -> None:
    dt = datetime(2026, 10, 6, 15, 30, 12, 999000, tzinfo=timezone(timedelta(hours=2)))
    assert iso_z(dt) == "2026-10-06T13:30:12Z"
    assert iso_ms_z(dt) == "2026-10-06T13:30:12.999Z"


def test_parse_iso_accepts_z_and_offsets_and_rejects_naive() -> None:
    assert parse_iso("2026-10-06T13:30:00Z") == datetime(2026, 10, 6, 13, 30, tzinfo=UTC)
    assert parse_iso("2026-10-06T15:30:00+02:00") == datetime(2026, 10, 6, 13, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="naive"):
        parse_iso("2026-10-06T13:30:00")


def test_ensure_utc_rejects_naive() -> None:
    with pytest.raises(ValueError, match="naive"):
        ensure_utc(datetime(2026, 1, 1))


def test_clocks() -> None:
    start = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    assert FixedClock(start).now() == start
    clock = MutableClock(start)
    clock.advance(timedelta(minutes=5))
    assert clock.now() == start + timedelta(minutes=5)

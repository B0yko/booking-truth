"""The sealed result types and the adapter protocol."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from booking_truth.calendars import (
    BookingRecord,
    CalcomAdapter,
    CalendarAdapter,
    NotFound,
    Slot,
    Slots,
    Unavailable,
    WriteOk,
    WriteRejected,
    WriteUnknown,
)

BERLIN_SUMMER = timezone(timedelta(hours=2))


def test_slot_is_stored_in_utc() -> None:
    slot = Slot(datetime(2026, 10, 5, 15, 0, tzinfo=BERLIN_SUMMER), datetime(2026, 10, 5, 13, 30, tzinfo=UTC))
    assert slot.start == datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
    assert slot.start.tzinfo == UTC
    assert slot.end.tzinfo == UTC


def test_slot_rejects_naive_and_empty_intervals() -> None:
    with pytest.raises(ValueError, match="naive"):
        Slot(datetime(2026, 10, 5, 13, 0), datetime(2026, 10, 5, 13, 30, tzinfo=UTC))
    at = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="end after it starts"):
        Slot(at, at)


def test_booking_record_equality_ignores_the_raw_vendor_object() -> None:
    start = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
    fields = {
        "ref": "uid1",
        "start": start,
        "end": start + timedelta(minutes=30),
        "status": "active",
        "lead_email": "lena@example.com",
        "idem_key": None,
    }
    one = BookingRecord(**fields, raw={"uid": "uid1"})  # type: ignore[arg-type]
    two = BookingRecord(**fields, raw={"uid": "uid1", "extra": True})  # type: ignore[arg-type]
    assert one == two
    assert one.active
    assert "raw" not in repr(one)


def test_booking_record_rejects_unknown_status() -> None:
    start = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="unknown booking status"):
        BookingRecord("uid1", start, start + timedelta(minutes=30), "accepted", None, None, {})  # type: ignore[arg-type]


def test_results_are_immutable_values() -> None:
    assert Unavailable("timeout") == Unavailable("timeout", "")
    assert WriteRejected("slot_taken", "x") != WriteRejected("invalid", "x")
    assert WriteUnknown("timeout").detail == ""
    assert NotFound().detail == ""
    assert Slots(()) != Unavailable("error")
    with pytest.raises(AttributeError):
        Unavailable("timeout").reason = "error"  # type: ignore[misc]


def test_write_ok_carries_the_previous_ref_only_for_a_reschedule() -> None:
    start = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
    record = BookingRecord("new", start, start + timedelta(minutes=30), "active", None, None, {})
    assert WriteOk(record).previous_ref is None
    assert WriteOk(record, "old").previous_ref == "old"


async def test_protocol_is_checkable_at_runtime() -> None:
    async with CalcomAdapter("http://127.0.0.1:9", "key", 1001, "1001", "America/New_York") as adapter:
        assert isinstance(adapter, CalendarAdapter)
        assert adapter.kind == "calcom"
        assert adapter.event_key == "1001"
    assert not isinstance(object(), CalendarAdapter)


def test_adapter_rejects_bad_options() -> None:
    with pytest.raises(ValueError, match="negative"):
        CalcomAdapter("http://127.0.0.1:9", "key", 1001, "1001", "UTC", post_retries_on_timeout=-1)
    with pytest.raises(ValueError, match="positive"):
        CalcomAdapter("http://127.0.0.1:9", "key", 1001, "1001", "UTC", slot_minutes=0)
    with pytest.raises(ValueError, match="Mars"):
        CalcomAdapter("http://127.0.0.1:9", "key", 1001, "1001", "Mars/Base")

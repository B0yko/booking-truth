"""Every sandbox fault mode, seen through the adapter: the result type it maps to, strict and lenient, and the
naive client's POST re-sends after a timeout, counted in the sandbox request log."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, get_args

import pytest
from calendar_env import (
    FAST,
    IDEM,
    LEAD,
    LEAD_NAME,
    LEAD_ZONE,
    MON_0900,
    MON_0930,
    MON_1000,
    MON_1700,
    MON_DAY,
    CalEnv,
    hang,
)

from booking_truth.calendars import (
    BookingRecord,
    CalcomAdapter,
    NotFound,
    ReadResult,
    Slots,
    Unavailable,
    WriteOk,
    WriteRejected,
    WriteResult,
    WriteUnknown,
)
from booking_truth.sandbox.faults import FaultMode
from booking_truth.timeutil import parse_iso

ISO = re.compile(r"\d{4}-\d{2}-\d{2}T[0-9:.]+(?:Z|[+-]\d{2}:\d{2})")


async def create(adapter: CalcomAdapter, start: datetime = MON_0900, key: str | None = IDEM) -> WriteResult:
    return await adapter.create_booking(
        start=start, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=key
    )


async def booked(adapter: CalcomAdapter, start: datetime = MON_0900) -> BookingRecord:
    result = await create(adapter, start)
    assert isinstance(result, WriteOk), result
    return result.booking


def statuses(env: CalEnv, group: str) -> list[tuple[int, str | None]]:
    return [(e["status"], e["fault"]) for e in env.log(group)]


# Slots -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "status", "reason", "message"),
    [
        ("error_500", 500, "error", "Internal server error"),
        ("not_found", 404, "not_found", "Event Type not found"),
    ],
)
async def test_slot_errors_are_unavailable(
    env: CalEnv, mode: str, status: int, reason: str, message: str
) -> None:
    env.faults({"group": "slots", "mode": mode, "times": None})
    async with env.adapter() as strict, env.adapter(lenient=True) as lenient:
        assert await strict.find_slots(MON_0900, MON_1700) == Unavailable(reason, f"HTTP {status}: {message}")
        loose = await lenient.find_slots(MON_0900, MON_1700)
    assert isinstance(loose, Unavailable)
    assert loose.reason == reason
    # The naive tool layer shows the model the vendor's raw answer.
    assert loose.detail.startswith(f'HTTP {status}: {{"status":"error","timestamp":')
    assert f'"message":"{message}"' in loose.detail


async def test_slot_timeout_is_unavailable(env: CalEnv) -> None:
    env.faults(hang("slots"))
    async with env.adapter(timeout=FAST, lenient=True) as adapter:
        result = await adapter.find_slots(MON_0900, MON_1700)
    assert isinstance(result, Unavailable)
    assert result.reason == "timeout"
    assert result.detail.startswith("ReadTimeout")
    assert statuses(env, "slots") == [(0, "timeout")]  # still hanging on the sandbox side


async def test_malformed_slots_are_unavailable_when_strict(env: CalEnv, calcom: CalcomAdapter) -> None:
    env.faults({"group": "slots", "mode": "malformed"})
    result = await calcom.find_slots(MON_0900, MON_1700)
    assert result == Unavailable("malformed", "'data' key 'busy' is not a YYYY-MM-DD date")


async def test_malformed_slots_fail_open_when_lenient(env: CalEnv) -> None:
    env.faults({"group": "slots", "mode": "malformed"})
    async with env.adapter(lenient=True) as adapter:
        result = await adapter.find_slots(MON_0900, MON_1700)
    assert isinstance(result, Slots)
    body: dict[str, Any] = env.log("slots")[0]["response"]
    assert list(body["data"]) == ["busy"]
    stamps = sorted({parse_iso(s) for s in ISO.findall(str(body))})
    assert stamps
    assert [s.start for s in result.slots] == stamps
    # The scraped "slots" are the edges of the sandbox's busy periods, among them times that were never free.
    assert any(not env.state.is_free(s.start) for s in result.slots)


async def test_slow_slots_still_answer(env: CalEnv, calcom: CalcomAdapter) -> None:
    env.faults({"group": "slots", "mode": "slow", "latency_ms": 100})
    result = await calcom.find_slots(MON_0900, MON_1000)
    assert isinstance(result, Slots)
    assert len(result.slots) == 2
    assert statuses(env, "slots") == [(200, "slow")]


async def test_slot_taken_after_offer_rejects_the_next_create(env: CalEnv, calcom: CalcomAdapter) -> None:
    env.faults({"group": "bookings.create", "mode": "slot_taken_after_offer"})
    offered = await calcom.find_slots(MON_0900, MON_1700)
    assert isinstance(offered, Slots)
    result = await create(calcom, offered.slots[0].start)
    assert isinstance(result, WriteRejected)
    assert result.reason == "slot_taken"
    assert env.bookings() == []
    after = await calcom.find_slots(MON_0900, MON_1700)
    assert after == Slots(())  # every offered slot went to a third party


# Create ----------------------------------------------------------------------------------------------------


async def test_create_error_500_is_unknown_and_never_resent(env: CalEnv) -> None:
    env.faults({"group": "bookings.create", "mode": "error_500"})
    async with env.adapter(post_retries_on_timeout=2) as adapter:
        result = await create(adapter)
    assert result == WriteUnknown("server_error", "HTTP 500: Internal server error")
    assert statuses(env, "bookings.create") == [(500, "error_500")]
    assert env.bookings() == []


async def test_create_timeout_is_unknown_without_retries(env: CalEnv) -> None:
    env.faults(hang("bookings.create"))
    async with env.adapter(timeout=FAST) as adapter:
        result = await create(adapter)
    assert isinstance(result, WriteUnknown)
    assert result.reason == "timeout"
    assert len(env.log("bookings.create")) == 1
    assert env.bookings() == []


async def test_naive_client_resends_a_timed_out_create(env: CalEnv) -> None:
    env.faults(hang("bookings.create"))
    async with env.adapter(timeout=FAST, post_retries_on_timeout=2) as adapter:
        result = await create(adapter)
    assert isinstance(result, WriteOk)
    log = env.log("bookings.create")
    assert [e["fault"] for e in log] == ["timeout", None]
    assert log[0]["body"] == log[1]["body"]
    assert [b["uid"] for b in env.bookings()] == [result.booking.ref]


async def test_naive_client_gives_up_after_its_retries(env: CalEnv) -> None:
    env.faults(hang("bookings.create", times=None))
    async with env.adapter(timeout=FAST, post_retries_on_timeout=2) as adapter:
        result = await create(adapter)
    assert isinstance(result, WriteUnknown)
    assert result.reason == "timeout"
    assert [e["fault"] for e in env.log("bookings.create")] == ["timeout"] * 3


async def test_commit_then_timeout_leaves_a_booking_the_caller_can_find(env: CalEnv) -> None:
    env.faults(hang("bookings.create", "commit_then_timeout"))
    async with env.adapter(timeout=FAST) as adapter:
        result = await create(adapter)
        assert isinstance(result, WriteUnknown)
        assert result.reason == "timeout"
        [stored] = env.bookings()
        # Verify-before-retry: the lead's bookings around the slot show the committed one, with its key.
        found = await adapter.list_bookings(lead_email=LEAD, start=MON_0900, end=MON_0930)
    assert isinstance(found, tuple)
    assert [(r.ref, r.idem_key) for r in found] == [(stored["uid"], IDEM)]
    assert len(env.log("bookings.create")) == 1


async def test_naive_resend_after_commit_is_told_the_slot_is_taken(env: CalEnv) -> None:
    env.faults(hang("bookings.create", "commit_then_timeout"))
    async with env.adapter(timeout=FAST, post_retries_on_timeout=2) as adapter:
        result = await create(adapter)
    # The first attempt committed; the re-send collides with it, so the caller believes nothing was booked.
    assert isinstance(result, WriteRejected)
    assert result.reason == "slot_taken"
    assert [e["fault"] for e in env.log("bookings.create")] == ["commit_then_timeout", None]
    assert [b["attendees"][0]["email"] for b in env.bookings()] == [LEAD]


async def test_create_not_found_is_invalid(env: CalEnv, calcom: CalcomAdapter) -> None:
    env.faults({"group": "bookings.create", "mode": "not_found"})
    result = await create(calcom)
    assert result == WriteRejected("invalid", "HTTP 404: Event type with id 1001 not found.")


@pytest.mark.parametrize("lenient", [False, True])
async def test_malformed_create_is_unknown_and_did_commit(env: CalEnv, lenient: bool) -> None:
    env.faults({"group": "bookings.create", "mode": "malformed"})
    async with env.adapter(lenient=lenient) as adapter:
        result = await create(adapter)
    assert isinstance(result, WriteUnknown)
    assert result.reason == "malformed"
    if lenient:
        assert result.detail.startswith('HTTP 200: {"status":"success","data":{"booking":')
    else:
        assert result.detail == "the booking has no uid"
    assert len(env.bookings()) == 1


async def test_slow_create_still_books(env: CalEnv, calcom: CalcomAdapter) -> None:
    env.faults({"group": "bookings.create", "mode": "slow", "latency_ms": 100})
    assert isinstance(await create(calcom), WriteOk)


# Get and list -----------------------------------------------------------------------------------------------


def _read_expectation(result: ReadResult, expected: str) -> None:
    if expected == "record":
        assert isinstance(result, BookingRecord)
    elif expected == "not_found":
        assert isinstance(result, NotFound)
        assert "was not found in the database" in result.detail
    else:
        assert isinstance(result, Unavailable)
        assert result.reason == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("error_500", "error"),
        ("timeout", "timeout"),
        ("commit_then_timeout", "timeout"),
        ("not_found", "not_found"),
        ("malformed", "malformed"),
        ("slow", "record"),
    ],
)
async def test_get_booking_faults(env: CalEnv, mode: str, expected: str) -> None:
    async with env.adapter(timeout=FAST) as adapter:
        record = await booked(adapter)
        env.faults(hang("bookings.get", mode, latency_ms=50))
        _read_expectation(await adapter.get_booking(record.ref), expected)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("error_500", "error"),
        ("timeout", "timeout"),
        ("not_found", "not_found"),
        ("malformed", "malformed"),
        ("slow", "records"),
    ],
)
async def test_list_bookings_faults(env: CalEnv, mode: str, expected: str) -> None:
    async with env.adapter(timeout=FAST) as adapter:
        record = await booked(adapter)
        env.faults(hang("bookings.list", mode, latency_ms=50))
        result = await adapter.list_bookings(lead_email=LEAD, start=MON_DAY[0], end=MON_DAY[1])
    if expected == "records":
        assert result == (record,)
    else:
        assert isinstance(result, Unavailable)
        assert result.reason == expected


async def test_list_fault_on_a_later_page_fails_the_whole_list(env: CalEnv, calcom: CalcomAdapter) -> None:
    for start in (MON_0900, MON_0930, MON_1000):
        await booked(calcom, start)
    calcom.list_page_size = 2
    env.faults({"group": "bookings.list", "mode": "malformed", "after_calls": 1})
    result = await calcom.list_bookings(lead_email=LEAD, start=MON_DAY[0], end=MON_DAY[1])
    assert isinstance(result, Unavailable)
    assert result.reason == "malformed"
    assert statuses(env, "bookings.list") == [(200, None), (200, "malformed")]


# Reschedule and cancel -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected", "committed"),
    [
        ("error_500", WriteUnknown, False),
        ("timeout", WriteUnknown, False),
        ("commit_then_timeout", WriteUnknown, True),
        ("not_found", WriteRejected, False),
        ("malformed", WriteUnknown, True),
        ("slow", WriteOk, True),
    ],
)
async def test_reschedule_faults(env: CalEnv, mode: str, expected: type, committed: bool) -> None:
    async with env.adapter(timeout=FAST) as adapter:
        old = await booked(adapter)
        env.faults(hang("bookings.reschedule", mode, latency_ms=50))
        result = await adapter.reschedule(ref=old.ref, new_start=MON_1000, idem_key=None, reason="")
    assert isinstance(result, expected)
    if isinstance(result, WriteRejected):
        assert result.reason == "not_found"
    starts = sorted(b["start"] for b in env.bookings() if b["status"] == "accepted")
    assert starts == (["2026-10-05T14:00:00.000Z"] if committed else ["2026-10-05T13:00:00.000Z"])


@pytest.mark.parametrize(
    ("mode", "expected", "committed"),
    [
        ("error_500", WriteUnknown, False),
        ("timeout", WriteUnknown, False),
        ("commit_then_timeout", WriteUnknown, True),
        ("not_found", WriteRejected, False),
        ("malformed", WriteUnknown, True),
        ("slow", WriteOk, True),
    ],
)
async def test_cancel_faults(env: CalEnv, mode: str, expected: type, committed: bool) -> None:
    async with env.adapter(timeout=FAST) as adapter:
        record = await booked(adapter)
        env.faults(hang("bookings.cancel", mode, latency_ms=50))
        result = await adapter.cancel(ref=record.ref, reason="", idem_key=None)
    assert isinstance(result, expected)
    if isinstance(result, WriteRejected):
        assert result.reason == "not_found"
    [stored] = env.bookings()
    assert stored["status"] == ("cancelled" if committed else "accepted")


async def test_naive_reschedule_resend_after_commit_is_a_duplicate(env: CalEnv) -> None:
    async with env.adapter(timeout=FAST, post_retries_on_timeout=2) as adapter:
        old = await booked(adapter)
        env.faults(hang("bookings.reschedule", "commit_then_timeout"))
        result = await adapter.reschedule(ref=old.ref, new_start=MON_1000, idem_key=None, reason="")
    assert isinstance(result, WriteRejected)
    assert result.reason == "duplicate"
    assert [e["fault"] for e in env.log("bookings.reschedule")] == ["commit_then_timeout", None]


async def test_naive_client_resends_a_timed_out_reschedule_and_cancel(env: CalEnv) -> None:
    async with env.adapter(timeout=FAST, post_retries_on_timeout=1) as adapter:
        old = await booked(adapter)
        env.faults(hang("bookings.reschedule"), hang("bookings.cancel"))
        moved = await adapter.reschedule(ref=old.ref, new_start=MON_1000, idem_key=None, reason="")
        assert isinstance(moved, WriteOk)
        cancelled = await adapter.cancel(ref=moved.booking.ref, reason="", idem_key=None)
    assert isinstance(cancelled, WriteOk)
    assert [e["fault"] for e in env.log("bookings.reschedule")] == ["timeout", None]
    assert [e["fault"] for e in env.log("bookings.cancel")] == ["timeout", None]


async def test_naive_cancel_resend_after_commit_is_a_duplicate(env: CalEnv) -> None:
    async with env.adapter(timeout=FAST, post_retries_on_timeout=2) as adapter:
        record = await booked(adapter)
        env.faults(hang("bookings.cancel", "commit_then_timeout"))
        result = await adapter.cancel(ref=record.ref, reason="", idem_key=None)
    assert isinstance(result, WriteRejected)
    assert result.reason == "duplicate"
    assert len(env.log("bookings.cancel")) == 2


def test_every_fault_mode_is_covered_here() -> None:
    covered = {
        "error_500",
        "timeout",
        "commit_then_timeout",
        "not_found",
        "malformed",
        "slot_taken_after_offer",
        "slow",
    }
    assert set(get_args(FaultMode)) == covered

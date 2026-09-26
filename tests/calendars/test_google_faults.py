"""Every sandbox fault mode, seen through the Google adapter: the result type it maps to, strict and
lenient, and the pre-insert conflict check's own use of ``freeBusy``."""

from __future__ import annotations

from datetime import datetime

from calendar_env import FAST, IDEM, LEAD, LEAD_NAME, LEAD_ZONE, MON_0900, MON_0930, MON_1700, CalEnv, hang
from google_env import google_adapter

from booking_truth.calendars import (
    BookingRecord,
    Slots,
    Unavailable,
    WriteOk,
    WriteRejected,
    WriteResult,
    WriteUnknown,
)
from booking_truth.calendars.google import GoogleAdapter, encode_event_id


async def create(adapter: GoogleAdapter, start: datetime = MON_0900, key: str | None = IDEM) -> WriteResult:
    return await adapter.create_booking(
        start=start, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=key
    )


def statuses(env: CalEnv, group: str) -> list[tuple[int, str | None]]:
    return [(e["status"], e["fault"]) for e in env.log(group)]


# freeBusy --------------------------------------------------------------------------------------------------


async def test_freebusy_error_500_is_unavailable(env: CalEnv, google: GoogleAdapter) -> None:
    env.faults({"group": "freebusy", "mode": "error_500"})
    result = await google.find_slots(MON_0900, MON_1700)
    assert isinstance(result, Unavailable)
    assert result.reason == "error"
    assert statuses(env, "freebusy") == [(500, "error_500")]


async def test_freebusy_not_found_is_missing_calendar_when_strict(env: CalEnv, google: GoogleAdapter) -> None:
    env.faults({"group": "freebusy", "mode": "not_found"})
    result = await google.find_slots(MON_0900, MON_1700)
    assert isinstance(result, Unavailable)
    assert result.reason == "missing_calendar"


async def test_freebusy_not_found_fails_open_when_lenient(env: CalEnv) -> None:
    env.faults({"group": "freebusy", "mode": "not_found"})
    async with google_adapter(env, lenient=True) as adapter:
        result = await adapter.find_slots(MON_0900, MON_1700)
    # No busy time at all, the fail-open bug: every working-hours slot in the window comes back free.
    assert isinstance(result, Slots)
    assert len(result.slots) == 16


async def test_freebusy_timeout_is_unavailable(env: CalEnv) -> None:
    env.faults(hang("freebusy"))
    async with google_adapter(env, timeout=FAST) as adapter:
        result = await adapter.find_slots(MON_0900, MON_1700)
    assert isinstance(result, Unavailable)
    assert result.reason == "timeout"
    assert statuses(env, "freebusy") == [(0, "timeout")]


async def test_malformed_freebusy_is_unavailable_when_strict(env: CalEnv, google: GoogleAdapter) -> None:
    env.faults({"group": "freebusy", "mode": "malformed"})
    result = await google.find_slots(MON_0900, MON_1700)
    assert result == Unavailable("malformed", "'busy' is not a list")


async def test_malformed_freebusy_fails_open_to_no_busy_when_lenient(env: CalEnv) -> None:
    env.faults({"group": "freebusy", "mode": "malformed"})
    async with google_adapter(env, lenient=True) as adapter:
        result = await adapter.find_slots(MON_0900, MON_1700)
    assert isinstance(result, Slots)
    assert len(result.slots) == 16


async def test_slot_taken_after_offer_rejects_the_next_create(env: CalEnv, google: GoogleAdapter) -> None:
    """Google's twin of the fault targets ``freebusy`` (insert itself does no conflict check): the harness
    seeds it there, and the adapter's own pre-insert re-check is what notices the third-party take."""
    env.faults({"group": "freebusy", "mode": "slot_taken_after_offer"})
    offered = await google.find_slots(MON_0900, MON_1700)
    assert isinstance(offered, Slots)
    result = await create(google, offered.slots[0].start, key=IDEM)
    assert result == WriteRejected("slot_taken", "the slot is no longer free")
    assert env.snapshot()["google"]["events"] == []
    after = await google.find_slots(MON_0900, MON_1700)
    assert after == Slots(())  # every offered slot went to a third party


# events.insert -----------------------------------------------------------------------------------------


async def test_insert_error_500_is_unknown_and_never_resent(env: CalEnv) -> None:
    env.faults({"group": "events.insert", "mode": "error_500"})
    async with google_adapter(env, post_retries_on_timeout=2) as adapter:
        result = await create(adapter)
    assert result == WriteUnknown("server_error", "HTTP 500: Backend Error")
    assert statuses(env, "events.insert") == [(500, "error_500")]
    assert env.snapshot()["google"]["events"] == []


async def test_insert_timeout_is_unknown_without_retries(env: CalEnv) -> None:
    env.faults(hang("events.insert"))
    async with google_adapter(env, timeout=FAST) as adapter:
        result = await create(adapter)
    assert isinstance(result, WriteUnknown)
    assert result.reason == "timeout"
    assert len(env.log("events.insert")) == 1
    assert env.snapshot()["google"]["events"] == []


async def test_naive_client_resends_a_timed_out_create_and_can_double_book(env: CalEnv) -> None:
    env.faults(hang("events.insert", "commit_then_timeout"))
    env.seed(google_sa_can_invite=True)
    async with google_adapter(env, timeout=FAST, post_retries_on_timeout=2, lenient=True) as adapter:
        result = await create(adapter, key=None)
    assert isinstance(result, WriteOk)
    log = env.log("events.insert")
    assert [e["fault"] for e in log] == ["commit_then_timeout", None]
    # No id was sent (idempotency off), so the resend is a brand new event: two bookings, same slot.
    events = env.snapshot()["google"]["events"]
    assert len({e["id"] for e in events}) == 2
    assert result.booking.ref in {e["id"] for e in events}


async def test_commit_then_timeout_leaves_a_booking_the_caller_can_find(env: CalEnv) -> None:
    env.seed(google_sa_can_invite=True)  # so the commit this fault leaves behind is a real insert
    env.faults(hang("events.insert", "commit_then_timeout"))
    async with google_adapter(env, timeout=FAST) as adapter:
        result = await create(adapter, key=IDEM)
        assert isinstance(result, WriteUnknown)
        assert result.reason == "timeout"
        found = await adapter.list_bookings(lead_email=LEAD, start=MON_0900, end=MON_0930)
    assert isinstance(found, tuple)
    [record] = found
    assert isinstance(record, BookingRecord)
    assert record.ref == encode_event_id(IDEM)
    assert len(env.log("events.insert")) == 1


async def test_create_not_found_is_invalid(env: CalEnv, google: GoogleAdapter) -> None:
    env.faults({"group": "events.insert", "mode": "not_found"})
    result = await create(google)
    assert isinstance(result, WriteRejected)
    assert result.reason == "invalid"


async def test_malformed_insert_is_unknown_and_did_commit(env: CalEnv, google: GoogleAdapter) -> None:
    env.seed(google_sa_can_invite=True)  # so the real insert this fault reshapes actually commits
    env.faults({"group": "events.insert", "mode": "malformed"})
    result = await create(google)
    assert isinstance(result, WriteUnknown)
    assert result.reason == "malformed"
    assert len(env.snapshot()["google"]["events"]) == 1


# events.get ---------------------------------------------------------------------------------------------


async def test_get_error_500_is_unavailable(env: CalEnv, google: GoogleAdapter) -> None:
    created = await create(google)
    assert isinstance(created, WriteOk)
    env.faults({"group": "events.get", "mode": "error_500"})
    result = await google.get_booking(created.booking.ref)
    assert isinstance(result, Unavailable)
    assert result.reason == "error"

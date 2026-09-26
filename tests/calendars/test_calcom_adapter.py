"""Every CalcomAdapter method over real HTTP against the sandbox's Cal.com mirror."""

from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from calendar_env import (
    EVENT_TYPE_ID,
    HALF_HOUR,
    HOST_ZONE,
    IDEM,
    LEAD,
    LEAD_NAME,
    LEAD_ZONE,
    MON_0900,
    MON_0930,
    MON_1000,
    MON_1700,
    MON_DAY,
    NOW,
    OTHER_LEAD,
    TOKEN,
    CalEnv,
)

from booking_truth.calendars import (
    BookingRecord,
    CalcomAdapter,
    CalendarAdapter,
    NotFound,
    Slots,
    Unavailable,
    WriteOk,
    WriteRejected,
    WriteUnknown,
)
from booking_truth.timeutil import iso_z

MSG_TAKEN = "User either already has booking at this time or is not available"


async def book(
    adapter: CalcomAdapter, start: datetime, email: str = LEAD, key: str | None = None
) -> BookingRecord:
    result = await adapter.create_booking(
        start=start, lead_email=email, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=key
    )
    assert isinstance(result, WriteOk), result
    return result.booking


# Slots -----------------------------------------------------------------------------------------------------


async def test_adapter_satisfies_the_protocol(calcom: CalcomAdapter) -> None:
    assert isinstance(calcom, CalendarAdapter)
    assert (calcom.kind, calcom.event_key, calcom.host_zone) == ("calcom", str(EVENT_TYPE_ID), HOST_ZONE)


async def test_find_slots_returns_the_working_day_in_utc(env: CalEnv, calcom: CalcomAdapter) -> None:
    result = await calcom.find_slots(MON_0900, MON_1700)
    assert isinstance(result, Slots)
    assert [s.start for s in result.slots] == [MON_0900 + i * HALF_HOUR for i in range(16)]
    assert all(s.end - s.start == HALF_HOUR and s.start.tzinfo == UTC for s in result.slots)
    entry = env.log("slots")[0]
    assert entry["query"] == {
        "eventTypeId": str(EVENT_TYPE_ID),
        "start": "2026-10-05T13:00:00Z",
        "end": "2026-10-05T21:00:00Z",
        "timeZone": HOST_ZONE,
        "format": "range",
    }
    assert entry["status"] == 200


async def test_find_slots_window_is_half_open(env: CalEnv, calcom: CalcomAdapter) -> None:
    result = await calcom.find_slots(MON_0900, MON_1000)
    assert isinstance(result, Slots)
    assert [s.start for s in result.slots] == [MON_0900, MON_0930]
    # Cal.com reads the end as inclusive and returned the 10:00 slot too; the adapter drops it.
    assert "2026-10-05T10:00:00.000-04:00" in str(env.log("slots")[0]["response"])


async def test_find_slots_agrees_with_the_sandbox_availability(env: CalEnv, calcom: CalcomAdapter) -> None:
    env.seed(existing_bookings=[{"start": "2026-10-05T14:00:00Z", "end": "2026-10-05T15:30:00Z"}])
    await book(calcom, MON_0930, email=OTHER_LEAD)
    start, end = NOW, NOW + timedelta(days=8)
    result = await calcom.find_slots(start, end)
    assert isinstance(result, Slots)
    starts = [s.start for s in result.slots]
    assert starts == env.free_starts(start, end)
    assert MON_0900 in starts
    assert MON_0930 not in starts
    assert MON_1000 not in starts


async def test_a_day_without_free_time_is_empty_slots_not_unavailable(calcom: CalcomAdapter) -> None:
    saturday = datetime(2026, 10, 3, tzinfo=UTC)
    assert await calcom.find_slots(saturday, saturday + timedelta(days=1)) == Slots(())


async def test_version_headers_and_bearer_on_a_shared_client(env: CalEnv) -> None:
    seen: list[httpx.Request] = []

    async def capture(request: httpx.Request) -> None:
        seen.append(request)

    async with httpx.AsyncClient(event_hooks={"request": [capture]}) as client:
        async with env.adapter(client=client) as adapter:
            assert isinstance(await adapter.find_slots(MON_0900, MON_1000), Slots)
            record = await book(adapter, MON_0900)
            await adapter.get_booking(record.ref)
            await adapter.list_bookings(lead_email=LEAD, start=MON_DAY[0], end=MON_DAY[1])
            await adapter.reschedule(ref=record.ref, new_start=MON_1000, idem_key=None, reason="")
            await adapter.cancel(ref=record.ref, reason="", idem_key=None)
        assert not client.is_closed  # a shared client stays open
    versions = [(r.method, r.url.path, r.headers["cal-api-version"]) for r in seen]
    assert versions[0] == ("GET", "/v2/slots", "2024-09-04")
    assert all(version == "2024-08-13" for _, path, version in versions[1:])
    assert [path for _, path, _ in versions[1:]] == [
        "/v2/bookings",
        f"/v2/bookings/{record.ref}",
        "/v2/bookings",
        f"/v2/bookings/{record.ref}/reschedule",
        f"/v2/bookings/{record.ref}/cancel",
    ]
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in seen)


async def test_a_wrong_api_key_is_an_error_not_an_empty_calendar(env: CalEnv) -> None:
    async with env.adapter(api_key="cal_wrong") as adapter:
        slots = await adapter.find_slots(MON_0900, MON_1700)
        assert slots == Unavailable(
            "error", "HTTP 401: ApiAuthStrategy - api key - Your api key is not valid"
        )
        created = await adapter.create_booking(
            start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=None
        )
        assert isinstance(created, WriteRejected)
        assert created.reason == "invalid"
        listed = await adapter.list_bookings(lead_email=LEAD, start=MON_DAY[0], end=MON_DAY[1])
        assert isinstance(listed, Unavailable)
        assert listed.reason == "error"
    assert env.bookings() == []


async def test_network_failures_map_to_error_and_unknown(env: CalEnv) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    async with env.adapter(base_url=f"http://127.0.0.1:{port}", post_retries_on_timeout=2) as adapter:
        slots = await adapter.find_slots(MON_0900, MON_1700)
        assert isinstance(slots, Unavailable)
        assert slots.reason == "error"
        assert slots.detail.startswith("ConnectError")
        got = await adapter.get_booking("abc")
        assert isinstance(got, Unavailable)
        assert got.reason == "error"
        created = await adapter.create_booking(
            start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=None
        )
        assert isinstance(created, WriteUnknown)
        assert created.reason == "server_error"


# Create ----------------------------------------------------------------------------------------------------


async def test_create_booking_sends_the_documented_body(env: CalEnv, calcom: CalcomAdapter) -> None:
    result = await calcom.create_booking(
        start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=IDEM
    )
    assert isinstance(result, WriteOk)
    assert result.previous_ref is None
    record = result.booking
    assert (record.start, record.end, record.status) == (MON_0900, MON_0900 + HALF_HOUR, "active")
    assert (record.lead_email, record.idem_key) == (LEAD, IDEM)
    assert record.raw["uid"] == record.ref
    entry = env.log("bookings.create")[0]
    assert entry["body"] == {
        "start": "2026-10-05T13:00:00Z",
        "eventTypeId": EVENT_TYPE_ID,
        "attendee": {"name": LEAD_NAME, "email": LEAD, "timeZone": LEAD_ZONE},
        "metadata": {"bt_idem": IDEM},
    }
    [stored] = env.bookings()
    assert stored["uid"] == record.ref
    assert stored["metadata"] == {"bt_idem": IDEM}


async def test_create_without_a_key_sends_no_metadata(env: CalEnv, calcom: CalcomAdapter) -> None:
    record = await book(calcom, MON_0900)
    assert record.idem_key is None
    body = env.log("bookings.create")[0]["body"]
    assert "metadata" not in body
    assert "lengthInMinutes" not in body


async def test_create_on_a_taken_slot_is_slot_taken(env: CalEnv, calcom: CalcomAdapter) -> None:
    await book(calcom, MON_0900, email=OTHER_LEAD)
    result = await calcom.create_booking(
        start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=IDEM
    )
    assert result == WriteRejected("slot_taken", f"HTTP 400: {MSG_TAKEN}")
    assert len(env.bookings()) == 1


@pytest.mark.parametrize(
    ("start", "zone", "message"),
    [
        (NOW + timedelta(hours=1), LEAD_ZONE, "too soon"),
        (NOW - timedelta(days=1), LEAD_ZONE, "in the past"),
        (MON_0900, "Mars/Base", "timeZone must be a valid IANA time-zone"),
    ],
)
async def test_other_create_refusals_are_invalid(
    env: CalEnv, calcom: CalcomAdapter, start: datetime, zone: str, message: str
) -> None:
    result = await calcom.create_booking(
        start=start, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=zone, idem_key=None
    )
    assert isinstance(result, WriteRejected)
    assert result.reason == "invalid"
    assert message in result.detail
    assert env.bookings() == []


async def test_create_outside_working_hours_is_slot_taken(calcom: CalcomAdapter) -> None:
    result = await calcom.create_booking(
        start=MON_0900 - timedelta(hours=2),
        lead_email=LEAD,
        lead_name=LEAD_NAME,
        lead_zone=LEAD_ZONE,
        idem_key=None,
    )
    assert isinstance(result, WriteRejected)
    assert result.reason == "slot_taken"


async def test_lenient_create_keeps_the_raw_error_text(env: CalEnv) -> None:
    async with env.adapter(lenient=True) as adapter:
        await book(adapter, MON_0900, email=OTHER_LEAD)
        result = await adapter.create_booking(
            start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=None
        )
    assert isinstance(result, WriteRejected)
    assert result.reason == "slot_taken"
    assert result.detail.startswith('HTTP 400: {"status":"error","timestamp":')
    assert '"code":"BadRequestException"' in result.detail


# Get --------------------------------------------------------------------------------------------------------


async def test_get_booking_reads_back_what_was_created(calcom: CalcomAdapter) -> None:
    created = await book(calcom, MON_0900, key=IDEM)
    fetched = await calcom.get_booking(created.ref)
    assert fetched == created
    assert isinstance(fetched, BookingRecord)
    assert "isPlatformManagedUserBooking" not in fetched.raw


async def test_get_unknown_booking_is_not_found(calcom: CalcomAdapter) -> None:
    result = await calcom.get_booking("doesNotExist123abc")
    assert result == NotFound("HTTP 404: Booking with uid=doesNotExist123abc was not found in the database")
    assert await calcom.get_booking("") == NotFound("empty booking reference")


async def test_get_a_cancelled_booking_reports_it_cancelled(calcom: CalcomAdapter) -> None:
    created = await book(calcom, MON_0900)
    assert isinstance(await calcom.cancel(ref=created.ref, reason="", idem_key=None), WriteOk)
    fetched = await calcom.get_booking(created.ref)
    assert isinstance(fetched, BookingRecord)
    assert fetched.status == "cancelled"
    assert not fetched.active


# List -------------------------------------------------------------------------------------------------------


async def test_list_bookings_filters_by_lead_window_and_status(env: CalEnv, calcom: CalcomAdapter) -> None:
    later = await book(calcom, MON_1000, key="k2")
    first = await book(calcom, MON_0900, key="k1")
    await book(calcom, MON_0930, email=OTHER_LEAD)
    dropped = await book(calcom, MON_1000 + HALF_HOUR)
    assert isinstance(await calcom.cancel(ref=dropped.ref, reason="", idem_key=None), WriteOk)
    tuesday = await book(calcom, MON_0900 + timedelta(days=1))

    result = await calcom.list_bookings(lead_email=LEAD, start=MON_DAY[0], end=MON_DAY[1])
    assert result == (first, later)
    assert isinstance(result, tuple)
    assert [r.idem_key for r in result] == ["k1", "k2"]

    narrow = await calcom.list_bookings(
        lead_email=LEAD,
        start=MON_0900 - timedelta(minutes=1),
        end=MON_0900 + HALF_HOUR + timedelta(minutes=1),
    )
    assert narrow == (first,)
    week = await calcom.list_bookings(lead_email=LEAD, start=NOW, end=NOW + timedelta(days=7))
    assert week == (first, later, tuesday)
    assert await calcom.list_bookings(lead_email="nobody@example.com", start=NOW, end=MON_DAY[1]) == ()

    query = env.log("bookings.list")[0]["query"]
    assert query == {
        "attendeeEmail": LEAD,
        "afterStart": "2026-10-05T00:00:00Z",
        "beforeEnd": "2026-10-06T00:00:00Z",
        "eventTypeId": str(EVENT_TYPE_ID),
        "status": "upcoming,past",
        "sortStart": "asc",
        "take": "100",
        "skip": "0",
    }


async def test_list_bookings_follows_pagination(env: CalEnv, calcom: CalcomAdapter) -> None:
    created = [await book(calcom, MON_0900 + i * HALF_HOUR) for i in range(5)]
    calcom.list_page_size = 2
    result = await calcom.list_bookings(lead_email=LEAD, start=MON_DAY[0], end=MON_DAY[1])
    assert result == tuple(created)
    pages = [(e["query"]["take"], e["query"]["skip"]) for e in env.log("bookings.list")]
    assert pages == [("2", "0"), ("2", "2"), ("2", "4")]


async def test_list_bookings_includes_past_ones(env: CalEnv, calcom: CalcomAdapter) -> None:
    record = await book(calcom, MON_0900)
    env.set_now(MON_1700)
    assert await calcom.list_bookings(lead_email=LEAD, start=MON_DAY[0], end=MON_DAY[1]) == (record,)


# Reschedule ------------------------------------------------------------------------------------------------


async def test_reschedule_returns_the_new_booking_and_the_previous_ref(
    env: CalEnv, calcom: CalcomAdapter
) -> None:
    old = await book(calcom, MON_0900, key=IDEM)
    result = await calcom.reschedule(ref=old.ref, new_start=MON_1000, idem_key="r1", reason="Clash")
    assert isinstance(result, WriteOk)
    assert result.previous_ref == old.ref
    new = result.booking
    assert new.ref != old.ref
    assert (new.start, new.end, new.status, new.lead_email) == (
        MON_1000,
        MON_1000 + HALF_HOUR,
        "active",
        LEAD,
    )
    assert new.idem_key == IDEM  # Cal.com copies the old booking's metadata
    assert new.raw["rescheduledFromUid"] == old.ref
    assert env.log("bookings.reschedule")[0]["body"] == {
        "start": "2026-10-05T14:00:00Z",
        "reschedulingReason": "Clash",
    }
    moved = await calcom.get_booking(old.ref)
    assert isinstance(moved, BookingRecord)
    assert moved.status == "cancelled"
    assert moved.raw["rescheduledToUid"] == new.ref


async def test_reschedule_without_a_reason_sends_only_the_start(env: CalEnv, calcom: CalcomAdapter) -> None:
    old = await book(calcom, MON_0900)
    assert isinstance(
        await calcom.reschedule(ref=old.ref, new_start=MON_1000, idem_key=None, reason=""), WriteOk
    )
    assert env.log("bookings.reschedule")[0]["body"] == {"start": iso_z(MON_1000)}


async def test_reschedule_refusals(calcom: CalcomAdapter) -> None:
    old = await book(calcom, MON_0900)
    await book(calcom, MON_1000, email=OTHER_LEAD)
    taken = await calcom.reschedule(ref=old.ref, new_start=MON_1000, idem_key=None, reason="")
    assert taken == WriteRejected("slot_taken", f"HTTP 400: {MSG_TAKEN}")

    missing = await calcom.reschedule(ref="doesNotExist123abc", new_start=MON_0930, idem_key=None, reason="")
    assert isinstance(missing, WriteRejected)
    assert missing.reason == "not_found"
    empty = await calcom.reschedule(ref="", new_start=MON_0930, idem_key=None, reason="")
    assert empty == WriteRejected("not_found", "empty booking reference")

    moved = await calcom.reschedule(ref=old.ref, new_start=MON_0930, idem_key=None, reason="")
    assert isinstance(moved, WriteOk)
    again = await calcom.reschedule(ref=old.ref, new_start=MON_1000 + HALF_HOUR, idem_key=None, reason="")
    assert isinstance(again, WriteRejected)
    assert again.reason == "duplicate"
    assert "rescheduled already" in again.detail

    assert isinstance(await calcom.cancel(ref=moved.booking.ref, reason="", idem_key=None), WriteOk)
    cancelled = await calcom.reschedule(ref=moved.booking.ref, new_start=MON_0900, idem_key=None, reason="")
    assert isinstance(cancelled, WriteRejected)
    assert cancelled.reason == "invalid"
    assert "has been cancelled" in cancelled.detail


# Cancel -----------------------------------------------------------------------------------------------------


async def test_cancel_returns_the_cancelled_booking(env: CalEnv, calcom: CalcomAdapter) -> None:
    record = await book(calcom, MON_0900)
    result = await calcom.cancel(ref=record.ref, reason="Plans changed", idem_key="c1")
    assert isinstance(result, WriteOk)
    assert result.previous_ref is None
    assert (result.booking.ref, result.booking.status) == (record.ref, "cancelled")
    assert result.booking.raw["cancellationReason"] == "Plans changed"
    assert env.log("bookings.cancel")[0]["body"] == {"cancellationReason": "Plans changed"}


async def test_cancel_refusals(env: CalEnv, calcom: CalcomAdapter) -> None:
    record = await book(calcom, MON_0900)
    assert isinstance(await calcom.cancel(ref=record.ref, reason="", idem_key=None), WriteOk)
    assert env.log("bookings.cancel")[0]["body"] == {}
    again = await calcom.cancel(ref=record.ref, reason="", idem_key=None)
    assert isinstance(again, WriteRejected)
    assert again.reason == "duplicate"
    assert "cancelled already" in again.detail

    missing = await calcom.cancel(ref="doesNotExist123abc", reason="", idem_key=None)
    assert missing == WriteRejected("not_found", "HTTP 404: Booking with uid=doesNotExist123abc not found")
    assert await calcom.cancel(ref="", reason="", idem_key=None) == WriteRejected(
        "not_found", "empty booking reference"
    )

    ended = await book(calcom, MON_1000)
    env.set_now(MON_1700)
    result = await calcom.cancel(ref=ended.ref, reason="", idem_key=None)
    assert result == WriteRejected("invalid", "HTTP 400: Cannot cancel a booking that has already ended")

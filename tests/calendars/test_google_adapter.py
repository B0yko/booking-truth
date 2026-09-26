"""Every ``GoogleAdapter`` method over real HTTP against the sandbox's Google Calendar mirror and fake
OAuth token endpoint."""

from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta

from calendar_env import (
    HALF_HOUR,
    IDEM,
    LEAD,
    LEAD_NAME,
    LEAD_ZONE,
    MON_0900,
    MON_0930,
    MON_1000,
    MON_1700,
    CalEnv,
)
from google_env import CALENDAR_ID, EVENT_KEY, google_adapter

from booking_truth.calendars import (
    BookingRecord,
    CalendarAdapter,
    NotFound,
    Slots,
    Unavailable,
    WriteOk,
    WriteRejected,
    WriteUnknown,
)
from booking_truth.calendars.google import GoogleAdapter, encode_event_id


async def book(
    adapter: GoogleAdapter, start: datetime = MON_0900, email: str = LEAD, key: str | None = None
) -> BookingRecord:
    result = await adapter.create_booking(
        start=start, lead_email=email, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=key
    )
    assert isinstance(result, WriteOk), result
    return result.booking


# Protocol and auth --------------------------------------------------------------------------------------


async def test_adapter_satisfies_the_protocol(google: GoogleAdapter) -> None:
    assert isinstance(google, CalendarAdapter)
    assert (google.kind, google.event_key, google.calendar_id) == ("google", EVENT_KEY, CALENDAR_ID)


async def test_the_token_is_fetched_once_and_reused(env: CalEnv, google: GoogleAdapter) -> None:
    await google.find_slots(MON_0900, MON_1700)
    await google.find_slots(MON_0900, MON_1700)
    assert len(env.log("oauth.token")) == 1


async def test_a_second_adapter_gets_its_own_token(env: CalEnv, google: GoogleAdapter) -> None:
    async with google_adapter(env) as other:
        await google.find_slots(MON_0900, MON_1700)
        await other.find_slots(MON_0900, MON_1700)
    assert len(env.log("oauth.token")) == 2


# Slots ---------------------------------------------------------------------------------------------------


async def test_find_slots_matches_the_calcom_grid(env: CalEnv, google: GoogleAdapter) -> None:
    result = await google.find_slots(MON_0900, MON_1700)
    assert isinstance(result, Slots)
    assert [s.start for s in result.slots] == [MON_0900 + i * HALF_HOUR for i in range(16)]
    assert all(s.end - s.start == HALF_HOUR and s.start.tzinfo == UTC for s in result.slots)
    entry = env.log("freebusy")[0]
    assert entry["body"] == {
        "timeMin": "2026-10-05T13:00:00Z",
        "timeMax": "2026-10-05T21:00:00Z",
        "items": [{"id": CALENDAR_ID}],
    }


async def test_find_slots_window_is_half_open(google: GoogleAdapter) -> None:
    result = await google.find_slots(MON_0900, MON_1000)
    assert isinstance(result, Slots)
    assert [s.start for s in result.slots] == [MON_0900, MON_0930]


async def test_find_slots_excludes_a_google_event(google: GoogleAdapter) -> None:
    await book(google, MON_0930, email="other@example.com")
    result = await google.find_slots(MON_0900, MON_1700)
    assert isinstance(result, Slots)
    assert MON_0930 not in [s.start for s in result.slots]
    assert MON_0900 in [s.start for s in result.slots]


async def test_network_failures_map_to_error_and_unknown(env: CalEnv) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    async with google_adapter(
        env, base_url=f"http://127.0.0.1:{port}", token_uri=f"http://127.0.0.1:{port}/token"
    ) as adapter:
        slots = await adapter.find_slots(MON_0900, MON_1700)
        assert isinstance(slots, Unavailable)
        assert slots.reason == "error"
        created = await adapter.create_booking(
            start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=None
        )
        assert isinstance(created, WriteUnknown)
        assert created.reason == "server_error"


# Create ----------------------------------------------------------------------------------------------------


async def test_create_booking_sends_the_documented_body(env: CalEnv, google: GoogleAdapter) -> None:
    record = await book(google, MON_0900, key=IDEM)
    assert (record.start, record.end, record.status) == (MON_0900, MON_0900 + HALF_HOUR, "active")
    assert record.lead_email == LEAD
    assert record.ref == encode_event_id(IDEM)
    assert record.idem_key == record.ref
    entry = env.log("events.insert")[-1]  # the attempt that actually succeeded (see the next test)
    assert entry["body"]["id"] == encode_event_id(IDEM)
    assert entry["body"]["start"] == {"dateTime": "2026-10-05T13:00:00Z", "timeZone": "UTC"}
    private = entry["body"]["extendedProperties"]["private"]
    assert private == {"bt_lead_email": LEAD, "bt_event_key": EVENT_KEY, "bt_idem": IDEM}
    assert "attendees" not in entry["body"]  # the default seed forbids service-account invitations
    [stored] = env.snapshot()["google"]["events"]
    assert stored["id"] == record.ref


async def test_create_without_a_key_gets_a_server_assigned_id(env: CalEnv, google: GoogleAdapter) -> None:
    record = await book(google, MON_0900)
    assert record.idem_key == record.ref
    assert "id" not in env.log("events.insert")[0]["body"]


async def test_create_tries_attendees_then_remembers_not_to(env: CalEnv, google: GoogleAdapter) -> None:
    await book(google, MON_0900)
    await book(google, MON_0930, email="second@example.com")
    calls = env.log("events.insert")
    assert len(calls) == 3  # first attempt (with attendees, 403), its retry, and the second booking
    assert calls[0]["body"].get("attendees") == [{"email": LEAD, "displayName": LEAD_NAME}]
    assert calls[0]["status"] == 403
    assert "attendees" not in calls[1]["body"]
    assert calls[1]["status"] == 200
    assert "attendees" not in calls[2]["body"]  # never retried on the second call


async def test_create_when_the_service_account_may_invite(env: CalEnv, google: GoogleAdapter) -> None:
    env.seed(google_sa_can_invite=True)
    record = await book(google, MON_0900)
    assert record.raw["attendees"] == [
        {"email": LEAD, "displayName": LEAD_NAME, "responseStatus": "needsAction"}
    ]
    assert len(env.log("events.insert")) == 1


async def test_create_with_the_same_key_twice_adopts_the_first(env: CalEnv, google: GoogleAdapter) -> None:
    first = await book(google, MON_0900, key=IDEM)
    before = len(env.log("events.insert"))
    result = await google.create_booking(
        start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=IDEM
    )
    assert result == WriteOk(first)
    assert len(env.log("events.insert")) == before  # adopted before ever trying to insert again


async def test_create_with_a_colliding_key_for_a_different_write_is_duplicate(
    env: CalEnv, google: GoogleAdapter
) -> None:
    """A found event whose start or lead does not match this call's is not adopted (an astronomically
    unlikely hash collision with someone else's key), so the insert runs and Google's own ``409`` on the
    reused id is reported as an unmatched duplicate."""
    other = await book(google, MON_0930, email="other@example.com", key=IDEM)
    before = len(env.log("events.insert"))
    result = await google.create_booking(
        start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=IDEM
    )
    assert result == WriteRejected("duplicate", "HTTP 409: The requested identifier already exists.")
    assert other.ref == encode_event_id(IDEM)
    assert len(env.log("events.insert")) == before + 1  # the insert ran once and was refused


async def test_guarded_create_rechecks_freebusy_and_rejects_a_taken_slot(
    env: CalEnv, google: GoogleAdapter
) -> None:
    env.seed(existing_bookings=[{"start": "2026-10-05T13:00:00Z", "end": "2026-10-05T13:30:00Z"}])
    result = await google.create_booking(
        start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=None
    )
    assert result == WriteRejected("slot_taken", "the slot is no longer free")
    assert env.log("events.insert") == []
    assert len(env.log("freebusy")) == 1


async def test_naive_create_skips_the_precheck_and_can_double_book(env: CalEnv) -> None:
    env.seed(
        google_sa_can_invite=True,  # keep this test's insert count free of the attendee-retry mechanic
        existing_bookings=[{"start": "2026-10-05T13:00:00Z", "end": "2026-10-05T13:30:00Z"}],
    )
    async with google_adapter(env, lenient=True) as naive:
        result = await naive.create_booking(
            start=MON_0900, lead_email=LEAD, lead_name=LEAD_NAME, lead_zone=LEAD_ZONE, idem_key=None
        )
    assert isinstance(result, WriteOk)
    assert env.log("freebusy") == []
    assert len(env.log("events.insert")) == 1


# Get / list ----------------------------------------------------------------------------------------------


async def test_get_booking_reads_back_what_was_created(google: GoogleAdapter) -> None:
    created = await book(google, MON_0900, key=IDEM)
    fetched = await google.get_booking(created.ref)
    assert fetched == created


async def test_get_booking_unknown_ref_is_not_found(google: GoogleAdapter) -> None:
    result = await google.get_booking("unknownref00000000000000")
    assert isinstance(result, NotFound)


async def test_get_booking_empty_ref_is_not_found_without_a_call(env: CalEnv, google: GoogleAdapter) -> None:
    result = await google.get_booking("")
    assert result == NotFound("empty booking reference")
    assert env.log("events.get") == []


async def test_list_bookings_filters_by_lead_and_active_status(env: CalEnv, google: GoogleAdapter) -> None:
    mine = await book(google, MON_0900)
    await book(google, MON_0930, email="other@example.com")
    result = await google.list_bookings(lead_email=LEAD, start=MON_0900 - timedelta(days=1), end=MON_1700)
    assert isinstance(result, tuple)
    assert [b.ref for b in result] == [mine.ref]
    entry = env.log("events.list")[0]
    assert entry["query"]["privateExtendedProperty"] == f"bt_lead_email={LEAD}"


async def test_list_bookings_excludes_a_cancelled_booking(google: GoogleAdapter) -> None:
    created = await book(google, MON_0900)
    cancelled = await google.cancel(ref=created.ref, reason="", idem_key=None)
    assert isinstance(cancelled, WriteOk)
    result = await google.list_bookings(lead_email=LEAD, start=MON_0900 - timedelta(days=1), end=MON_1700)
    assert result == ()


# Reschedule ------------------------------------------------------------------------------------------------


async def test_reschedule_moves_the_same_event_id(google: GoogleAdapter) -> None:
    created = await book(google, MON_0900)
    result = await google.reschedule(ref=created.ref, new_start=MON_0930, idem_key=None, reason="moved")
    assert isinstance(result, WriteOk)
    assert result.previous_ref == created.ref
    assert result.booking.ref == created.ref
    assert result.booking.start == MON_0930


async def test_reschedule_unknown_ref_is_not_found(google: GoogleAdapter) -> None:
    result = await google.reschedule(
        ref="unknownref00000000000000", new_start=MON_0930, idem_key=None, reason=""
    )
    assert result == WriteRejected("not_found", "HTTP 404: Not Found")


async def test_reschedule_a_cancelled_booking_is_not_found(google: GoogleAdapter) -> None:
    created = await book(google, MON_0900)
    await google.cancel(ref=created.ref, reason="", idem_key=None)
    result = await google.reschedule(ref=created.ref, new_start=MON_0930, idem_key=None, reason="")
    assert result == WriteRejected("not_found", f"booking {created.ref} no longer exists")


# Cancel ----------------------------------------------------------------------------------------------------


async def test_cancel_reads_the_tombstone_back(google: GoogleAdapter) -> None:
    created = await book(google, MON_0900)
    result = await google.cancel(ref=created.ref, reason="no longer needed", idem_key=None)
    assert isinstance(result, WriteOk)
    assert result.booking.status == "cancelled"
    assert result.booking.start == created.start


async def test_cancel_twice_is_duplicate(google: GoogleAdapter) -> None:
    created = await book(google, MON_0900)
    await google.cancel(ref=created.ref, reason="", idem_key=None)
    again = await google.cancel(ref=created.ref, reason="", idem_key=None)
    assert isinstance(again, WriteRejected)
    assert again.reason == "duplicate"


async def test_cancel_unknown_ref_is_not_found(google: GoogleAdapter) -> None:
    result = await google.cancel(ref="unknownref00000000000000", reason="", idem_key=None)
    assert result == WriteRejected("not_found", "HTTP 404: Not Found")

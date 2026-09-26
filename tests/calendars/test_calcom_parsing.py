"""Strict parsing of Cal.com bodies, the lenient scrape, and error message extraction (no network)."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from booking_truth.calendars.calcom import (
    MalformedResponse,
    parse_booking,
    parse_booking_page,
    parse_slots,
    scrape_slots,
    vendor_message,
)

WINDOW = (datetime(2026, 10, 5, tzinfo=UTC), datetime(2026, 10, 6, tzinfo=UTC))
MON_0900 = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
HALF_HOUR = timedelta(minutes=30)

SLOTS_BODY: dict[str, Any] = {
    "data": {
        "2026-10-05": [
            {"start": "2026-10-05T09:30:00.000-04:00", "end": "2026-10-05T10:00:00.000-04:00"},
            {"start": "2026-10-05T09:00:00.000-04:00", "end": "2026-10-05T09:30:00.000-04:00"},
        ]
    },
    "status": "success",
}

BOOKING: dict[str, Any] = {
    "id": 7,
    "uid": "cnM2dpP3aKbFjXXNrc3g8C",
    "title": "Intro call between Sandbox Host and Lena M",
    "status": "accepted",
    "start": "2026-10-05T13:00:00.000Z",
    "end": "2026-10-05T13:30:00.000Z",
    "metadata": {"bt_idem": "k1"},
    "attendees": [{"name": "Lena M", "email": "lena@example.com", "timeZone": "Europe/Berlin"}],
    "isPlatformManagedUserBooking": False,
}


def booking(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(BOOKING)
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


# Slots -----------------------------------------------------------------------------------------------------


def test_slots_are_converted_to_utc_sorted_and_deduplicated() -> None:
    body = copy.deepcopy(SLOTS_BODY)
    body["data"]["2026-10-05"].append(dict(body["data"]["2026-10-05"][0], seatsRemaining=3))
    result = parse_slots(body, *WINDOW)
    assert [(s.start, s.end) for s in result.slots] == [
        (MON_0900, MON_0900 + HALF_HOUR),
        (MON_0900 + HALF_HOUR, MON_0900 + 2 * HALF_HOUR),
    ]


def test_an_empty_data_object_is_a_calendar_without_free_time() -> None:
    assert parse_slots({"data": {}, "status": "success"}, *WINDOW).slots == ()


def test_slots_outside_the_half_open_window_are_dropped() -> None:
    result = parse_slots(SLOTS_BODY, MON_0900, MON_0900 + HALF_HOUR)
    assert [s.start for s in result.slots] == [MON_0900]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (None, "not a JSON object"),
        ([], "not a JSON object"),
        ({"data": {}, "status": "error"}, "not 'success'"),
        ({"status": "success"}, "no 'data'"),
        ({"data": [], "status": "success"}, "keyed by date"),
        ({"data": {"busy": []}, "status": "success"}, "not a YYYY-MM-DD date"),
        ({"data": {"2026-02-30": []}, "status": "success"}, "not a valid date"),
        ({"data": {"2026-10-05": {}}, "status": "success"}, "not a list"),
        ({"data": {"2026-10-05": ["2026-10-05T13:00:00Z"]}, "status": "success"}, "not an object"),
        ({"data": {"2026-10-05": [{"start": "2026-10-05T13:00:00Z"}]}, "status": "success"}, "slot end"),
        (
            {"data": {"2026-10-05": [{"time": "2026-10-05T13:00:00Z"}]}, "status": "success"},
            "slot start",
        ),
        (
            {"data": {"2026-10-05": [{"start": "2026-10-05T13:00:00", "end": "2026-10-05T13:30:00"}]}},
            "not 'success'",
        ),
        (
            {
                "data": {"2026-10-05": [{"start": "2026-10-05T13:00:00", "end": "2026-10-05T13:30:00"}]},
                "status": "success",
            },
            "with an offset",
        ),
        (
            {
                "data": {"2026-10-05": [{"start": "2026-10-05T13:30:00Z", "end": "2026-10-05T13:00:00Z"}]},
                "status": "success",
            },
            "ends before it starts",
        ),
        ({"data": {"2026-10-05": [{"start": 1790000000, "end": 1790001800}]}, "status": "success"}, "start"),
    ],
)
def test_any_deviation_from_the_documented_slots_shape_is_malformed(body: object, message: str) -> None:
    with pytest.raises(MalformedResponse, match=message):
        parse_slots(body, *WINDOW)


# Lenient scrape --------------------------------------------------------------------------------------------


def test_scrape_turns_every_timestamp_into_a_slot() -> None:
    text = (
        '{"status":"success","data":{"busy":[{"start":"2026-10-05T13:00:00.000Z",'
        '"end":"2026-10-05T15:30:00+02:00"},{"from":"2026-10-06T09:00:00","day":"2026-10-07"}]},'
        '"again":"2026-10-05T13:00:00Z"}'
    )
    result = scrape_slots(text, HALF_HOUR)
    assert [s.start for s in result.slots] == [
        MON_0900,
        MON_0900 + HALF_HOUR,
        datetime(2026, 10, 6, 9, 0, tzinfo=UTC),  # no offset: read as UTC
    ]
    assert all(s.end - s.start == HALF_HOUR for s in result.slots)


def test_scrape_of_a_body_without_timestamps_is_an_empty_calendar() -> None:
    assert scrape_slots('{"status":"success","data":{"slots":"none"}}', HALF_HOUR).slots == ()


# Bookings --------------------------------------------------------------------------------------------------


def test_booking_is_parsed_into_a_record() -> None:
    record = parse_booking(booking())
    assert record.ref == "cnM2dpP3aKbFjXXNrc3g8C"
    assert (record.start, record.end) == (MON_0900, MON_0900 + HALF_HOUR)
    assert record.status == "active"
    assert record.lead_email == "lena@example.com"
    assert record.idem_key == "k1"
    assert record.raw["isPlatformManagedUserBooking"] is False


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("accepted", "active"),
        ("pending", "active"),
        ("awaiting_host", "active"),
        ("cancelled", "cancelled"),
        ("rejected", "cancelled"),
    ],
)
def test_vendor_statuses_map_to_active_or_cancelled(status: str, expected: str) -> None:
    assert parse_booking(booking(status=status)).status == expected


def test_booking_without_metadata_or_attendee_email() -> None:
    record = parse_booking(booking(metadata=None, attendees=[{"name": "Lena M", "phoneNumber": "+4930"}]))
    assert record.idem_key is None
    assert record.lead_email is None
    assert parse_booking(booking(metadata={"bt_idem": 5})).idem_key is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([BOOKING], "not a JSON object"),
        (booking(uid=None), "no uid"),
        (booking(uid=""), "no uid"),
        (booking(start="2026-10-05"), "booking start"),
        (booking(end="2026-10-05T12:00:00Z"), "ends before it starts"),
        (booking(status="ACCEPTED"), "unknown booking status"),
        (booking(status=None), "unknown booking status"),
        (booking(attendees=None), "no attendee list"),
        (booking(attendees=["lena@example.com"]), "no attendee list"),
        (booking(attendees=[{"email": 5}]), "not a string"),
        (booking(metadata=["bt_idem"]), "not an object"),
    ],
)
def test_any_deviation_from_the_booking_shape_is_malformed(data: object, message: str) -> None:
    with pytest.raises(MalformedResponse, match=message):
        parse_booking(data)


def test_booking_page_needs_its_pagination() -> None:
    page = {"status": "success", "data": [BOOKING], "pagination": {"hasNextPage": False, "totalItems": 1}}
    records, has_next = parse_booking_page(page)
    assert [r.ref for r in records] == [BOOKING["uid"]]
    assert has_next is False
    with pytest.raises(MalformedResponse, match="hasNextPage"):
        parse_booking_page({"status": "success", "data": []})
    with pytest.raises(MalformedResponse, match="list of bookings"):
        parse_booking_page({"status": "success", "data": {"items": []}, "pagination": {"hasNextPage": False}})


# Error messages --------------------------------------------------------------------------------------------


def test_vendor_message_reads_every_cal_com_error_shape() -> None:
    envelope = {"status": "error", "error": {"code": "BadRequestException", "message": "User either ..."}}
    assert vendor_message(envelope) == "User either ..."
    assert (
        vendor_message({"statusCode": 409, "message": "booking_conflict_error"}) == "booking_conflict_error"
    )
    pipe = {
        "status": "error",
        "error": {
            "code": "BadRequestException",
            "message": "Bad Request Exception",
            "details": {
                "errors": [{"property": "start", "constraints": {"isString": "start must be a string"}}]
            },
        },
    }
    assert vendor_message(pipe) == "start must be a string"
    assert vendor_message({"message": ["a", "b"]}) == "a; b"
    assert vendor_message("Bad Gateway") == ""
    assert vendor_message({"error": "boom"}) == ""

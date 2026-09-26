"""Contract tests: the Cal.com API v2 subset answers with the shapes, statuses and texts of the real API."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from booking_truth.sandbox.calcom import FieldError, format_errors, short_uuid

if TYPE_CHECKING:
    from conftest import Sandbox

LEAD = "lead@example.com"
MON_0900 = "2026-10-05T13:00:00Z"  # Monday 09:00 in New York
MON_0930 = "2026-10-05T13:30:00Z"
MON_1000 = "2026-10-05T14:00:00Z"
TUE_0900 = "2026-10-06T13:00:00Z"
SLOTS_V = {"cal-api-version": "2024-09-04"}
UID = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{22}$")
MS_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

MSG_TAKEN = "User either already has booking at this time or is not available"
MSG_OUT_OF_BOUNDS = (
    "The event type can't be booked at the \"start\" time provided. This could be because it's too soon "
    "(violating the minimum booking notice) or too far in the future (outside the event's scheduling "
    "window). Try fetching available slots first using the GET /v2/slots endpoint and then make a booking "
    'with "start" time equal to one of the available slots.'
)
MSG_METADATA = (
    "metadata property is wrong,Metadata must have at most 50 keys, each key up to 40 characters, and string "
    "values up to 500 characters. "
)
BOOKING_KEYS = [
    "id",
    "uid",
    "title",
    "description",
    "hosts",
    "status",
    "cancellationReason",
    "cancelledByEmail",
    "rescheduledByEmail",
    "start",
    "end",
    "duration",
    "eventTypeId",
    "eventType",
    "meetingUrl",
    "location",
    "absentHost",
    "createdAt",
    "updatedAt",
    "metadata",
    "rating",
    "icsUid",
    "attendees",
    "bookingFieldsResponses",
]
LANGUAGE_LIST = (
    "ar, ca, de, es, eu, he, id, ja, lv, pl, ro, sr, th, vi, az, cs, el, es-419, fi, hr, it, km, nl, pt, ru, "
    "sv, tr, zh-CN, bg, da, en, et, fr, hu, iw, ko, no, pt-BR, sk, sl, ta, uk, zh-TW, bn"
)


def starts(body: dict[str, Any]) -> list[str]:
    return [slot["start"] for day in body["data"].values() for slot in day]


def assert_pipe_error(response: Any, errors: list[dict[str, Any]], *, path: str | None = None) -> None:
    """Cal.com's global ValidationPipe 400: a generic message and class-validator errors in ``details``."""
    assert response.status_code == 400, response.text
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    body = response.json()
    assert list(body) == ["status", "timestamp", "path", "error"]
    assert body["status"] == "error"
    assert MS_Z.match(body["timestamp"])
    assert body["error"] == {
        "code": "BadRequestException",
        "message": "Bad Request Exception",
        "details": {"errors": errors},
    }
    for entry in body["error"]["details"]["errors"]:
        assert list(entry) == ["property", "children", "constraints"]
    if path is not None:
        assert body["path"] == path


def constraint(prop: str, **constraints: str) -> dict[str, Any]:
    return {"property": prop, "children": [], "constraints": constraints}


# Slots ---------------------------------------------------------------------------------------------


def test_slots_need_the_2024_09_04_version_header(sandbox: Sandbox) -> None:
    query = "eventTypeId=1001&start=2026-10-05&end=2026-10-05"
    for headers in ({}, {"cal-api-version": "2024-08-13"}, {"cal-api-version": "2026-02-25"}):
        response = sandbox.client.get(f"/v2/slots?{query}", headers=headers)
        sandbox.assert_error(response, 404, f"Cannot GET /v2/slots?{query}", path=f"/v2/slots?{query}")
    assert sandbox.slots(start="2026-10-05", end="2026-10-05").status_code == 200


def test_slots_default_format_is_utc_objects_keyed_by_date(sandbox: Sandbox) -> None:
    response = sandbox.slots(start="2026-10-03", end="2026-10-05")  # Saturday to Monday
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    body = response.json()
    assert list(body) == ["data", "status"]  # slots put data first, bookings put status first
    assert body["status"] == "success"
    assert list(body["data"]) == ["2026-10-05"]  # days without slots are omitted
    monday = body["data"]["2026-10-05"]
    assert len(monday) == 16
    assert monday[0] == {"start": "2026-10-05T13:00:00.000Z"}
    assert monday[-1] == {"start": "2026-10-05T20:30:00.000Z"}


def test_slots_are_grouped_by_date_in_the_requested_zone(sandbox: Sandbox) -> None:
    body = sandbox.slots(start="2026-10-05", end="2026-10-05", timeZone="Asia/Kolkata").json()
    assert list(body["data"]) == ["2026-10-05", "2026-10-06"]
    assert len(body["data"]["2026-10-05"]) == 11
    assert len(body["data"]["2026-10-06"]) == 5
    assert body["data"]["2026-10-05"][0] == {"start": "2026-10-05T18:30:00.000+05:30"}
    assert body["data"]["2026-10-06"][-1] == {"start": "2026-10-06T02:00:00.000+05:30"}
    rome = sandbox.slots(start="2026-10-05", end="2026-10-05", timeZone="Europe/Rome").json()
    assert rome["data"]["2026-10-05"][0] == {"start": "2026-10-05T15:00:00.000+02:00"}


def test_slots_range_format_adds_end(sandbox: Sandbox) -> None:
    body = sandbox.slots(start="2026-10-05", end="2026-10-05", format="range").json()
    assert body["data"]["2026-10-05"][0] == {
        "start": "2026-10-05T13:00:00.000Z",
        "end": "2026-10-05T13:30:00.000Z",
    }


def test_slots_respect_minimum_notice_and_busy_time(sandbox: Sandbox) -> None:
    today = sandbox.slots(start="2026-10-01", end="2026-10-01").json()
    assert starts(today)[0] == "2026-10-01T14:00:00.000Z"  # 08:00 EDT now + 2 h notice
    assert len(starts(today)) == 14
    sandbox.seed(existing_bookings=[{"start": MON_1000}])
    sandbox.booked(MON_0900)
    monday = starts(sandbox.slots(start="2026-10-05", end="2026-10-05").json())
    assert "2026-10-05T13:00:00.000Z" not in monday
    assert "2026-10-05T14:00:00.000Z" not in monday
    assert len(monday) == 14


def test_slots_datetime_window_is_inclusive_and_empty_windows_give_an_empty_object(sandbox: Sandbox) -> None:
    window = sandbox.slots(start="2026-10-05T13:00:00Z", end="2026-10-05T14:00:00Z").json()
    assert starts(window) == [
        "2026-10-05T13:00:00.000Z",
        "2026-10-05T13:30:00.000Z",
        "2026-10-05T14:00:00.000Z",
    ]
    assert sandbox.slots(start="2026-10-03", end="2026-10-04").json() == {"data": {}, "status": "success"}
    assert sandbox.slots(start="2026-10-06", end="2026-10-05").json() == {"data": {}, "status": "success"}


def test_slots_unknown_event_type_is_the_verbatim_404(sandbox: Sandbox) -> None:
    response = sandbox.slots(eventTypeId=999999999, start="2033-09-05", end="2033-09-06")
    path = "/v2/slots?eventTypeId=999999999&start=2033-09-05&end=2033-09-06"
    sandbox.assert_error(response, 404, "Event Type not found", path=path)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"end": "2026-10-05"}, "start property is wrong,start must be a valid ISO 8601 date string "),
        (
            {"start": "2026-10-05", "end": "2026-10-05", "timeZone": "Mars/Base"},
            "timeZone property is wrong,timeZone must be a valid IANA time-zone ",
        ),
        (
            {"start": "2026-10-05", "end": "2026-10-05", "format": "list"},
            "format property is wrong,slotFormat must be either 'range' or 'time' ",
        ),
        (
            {"start": "2026-10-05", "end": "2026-10-05", "foo": "1"},
            "foo property is wrong,property foo should not exist ",
        ),
        (
            {"start": "2026-10-05", "end": "2026-10-05", "username": "sandbox-host"},
            "username property is wrong,property username should not exist ",
        ),
    ],
)
def test_slots_validation_messages(sandbox: Sandbox, params: dict[str, str], message: str) -> None:
    sandbox.assert_error(sandbox.slots(**params), 400, message)


def test_slots_input_classes_as_the_pipe_picks_them(sandbox: Sandbox) -> None:
    # By id: unknown keys first, then the class's own eventTypeId, then the shared start, end, duration.
    response = sandbox.slots(eventTypeId="x", duration="y", organizationSlug="acme")
    sandbox.assert_error(
        response,
        400,
        "organizationSlug property is wrong,property organizationSlug should not exist , "
        "eventTypeId property is wrong,eventTypeId must be a number conforming to the specified "
        "constraints , "
        "start property is wrong,start must be a valid ISO 8601 date string , "
        "end property is wrong,end must be a valid ISO 8601 date string , "
        "duration property is wrong,duration must be a number conforming to the specified constraints ",
    )
    # No id and no slug pair: the usernames class, which needs two usernames and an organization.
    no_lookup = sandbox.client.get(
        "/v2/slots", params={"start": "2026-10-05", "end": "2026-10-05"}, headers=SLOTS_V
    )
    sandbox.assert_error(
        no_lookup,
        400,
        "usernames property is wrong,each value in usernames must be a string, The array must contain at "
        "least 2 elements., usernames must be an array , organizationSlug property is wrong,organizationSlug "
        "must be a string ",
    )
    one = sandbox.client.get(
        "/v2/slots",
        params={
            "start": "2026-10-05",
            "end": "2026-10-05",
            "usernames": "sandbox-host",
            "organizationSlug": "o",
        },
        headers=SLOTS_V,
    )
    sandbox.assert_error(one, 400, "usernames property is wrong,The array must contain at least 2 elements. ")
    # format is lower-cased first; an empty format is no format.
    upper = sandbox.slots(start="2026-10-05", end="2026-10-05", format="RANGE").json()
    assert "end" in upper["data"]["2026-10-05"][0]
    assert (
        "end"
        not in sandbox.slots(start="2026-10-05", end="2026-10-05", format="").json()["data"]["2026-10-05"][0]
    )


def test_slots_lookup_by_slug_and_username(sandbox: Sandbox) -> None:
    params = {"start": "2026-10-05", "end": "2026-10-05", "eventTypeSlug": "intro-call"}
    ok = sandbox.client.get("/v2/slots", params={**params, "username": "sandbox-host"}, headers=SLOTS_V)
    assert len(starts(ok.json())) == 16
    missing = sandbox.client.get("/v2/slots", params={**params, "username": "nobody"}, headers=SLOTS_V)
    sandbox.assert_error(missing, 404, "User with username nobody not found")


def test_slots_booking_uid_to_reschedule_frees_that_booking(sandbox: Sandbox) -> None:
    booking = sandbox.booked(MON_0900)
    plain = starts(sandbox.slots(start="2026-10-05", end="2026-10-05").json())
    assert "2026-10-05T13:00:00.000Z" not in plain
    freed = sandbox.slots(start="2026-10-05", end="2026-10-05", bookingUidToReschedule=booking["uid"]).json()
    assert "2026-10-05T13:00:00.000Z" in starts(freed)


# Create --------------------------------------------------------------------------------------------


def test_create_returns_201_and_a_booking_in_cal_com_key_order(sandbox: Sandbox) -> None:
    response = sandbox.book(MON_0900, metadata={"bt_idem": "a" * 64})
    assert response.status_code == 201
    body = response.json()
    assert list(body) == ["status", "data"]
    data = body["data"]
    assert list(data) == [*BOOKING_KEYS, "isPlatformManagedUserBooking"]
    assert data["isPlatformManagedUserBooking"] is False
    assert UID.match(data["uid"])
    assert data["icsUid"] == f"{data['uid']}@Cal.com"
    assert data["title"] == "Intro call between Sandbox Host and Lena M"
    assert data["status"] == "accepted"
    assert (data["start"], data["end"], data["duration"]) == (
        "2026-10-05T13:00:00.000Z",
        "2026-10-05T13:30:00.000Z",
        30,
    )
    assert data["eventTypeId"] == 1001
    assert data["eventType"] == {"id": 1001, "slug": "intro-call"}
    assert (data["cancellationReason"], data["cancelledByEmail"], data["rescheduledByEmail"]) == (
        "",
        "",
        None,
    )
    assert MS_Z.match(data["createdAt"])
    assert MS_Z.match(data["updatedAt"])
    assert data["metadata"] == {"bt_idem": "a" * 64}
    assert data["hosts"] == [
        {
            "id": 1,
            "name": "Sandbox Host",
            "email": "host@example.com",
            "displayEmail": "host@example.com",
            "username": "sandbox-host",
            "timeZone": "America/New_York",
        }
    ]
    assert data["attendees"] == [
        {
            "name": "Lena M",
            "email": LEAD,
            "displayEmail": LEAD,
            "timeZone": "Europe/Berlin",
            "language": "en",
            "absent": False,
        }
    ]
    # No guests sent: neither bookingFieldsResponses.guests nor the top-level guests appear.
    assert data["bookingFieldsResponses"] == {"email": LEAD, "name": "Lena M", "displayEmail": LEAD}
    # The deprecated meetingUrl always mirrors location, even when it is not a URL.
    assert data["meetingUrl"] == data["location"] == "integrations:daily"


def test_create_with_guests_and_a_meeting_link(sandbox: Sandbox) -> None:
    data = sandbox.booked(
        MON_0900,
        guests=["guest@example.com"],
        location={"type": "link", "link": "https://meet.example.com/intro"},
    )
    assert list(data)[-3:] == ["guests", "bookingFieldsResponses", "isPlatformManagedUserBooking"]
    assert data["guests"] == ["guest@example.com"]
    assert data["bookingFieldsResponses"] == {
        "email": LEAD,
        "name": "Lena M",
        "guests": ["guest@example.com"],
        "displayEmail": LEAD,
        "displayGuests": ["guest@example.com"],
    }
    assert data["meetingUrl"] == data["location"] == "https://meet.example.com/intro"
    empty = sandbox.booked(MON_1000, guests=[])
    assert empty["guests"] == []
    assert empty["bookingFieldsResponses"]["displayGuests"] == []


def test_create_takes_the_deprecated_meeting_url_as_the_location(sandbox: Sandbox) -> None:
    data = sandbox.booked(MON_0900, meetingUrl="https://meet.example.com/x")
    assert data["meetingUrl"] == data["location"] == "https://meet.example.com/x"
    sandbox.assert_error(
        sandbox.book(MON_0930, meetingUrl="not a url"),
        400,
        "meetingUrl property is wrong,meetingUrl must be a URL address ",
    )


@pytest.mark.parametrize("version", ["2024-08-13", "2026-02-25", "2026-05-01"])
def test_create_works_under_every_booking_version(sandbox: Sandbox, version: str) -> None:
    assert sandbox.book(MON_0900, version=version).status_code == 201


def test_create_with_the_slots_version_is_a_route_404(sandbox: Sandbox) -> None:
    response = sandbox.book(MON_0900, version="2024-09-04")
    sandbox.assert_error(response, 404, "Cannot POST /v2/bookings", path="/v2/bookings")


def test_create_without_version_reaches_legacy_validation(sandbox: Sandbox) -> None:
    response = sandbox.client.post("/v2/bookings", json={})
    assert response.status_code == 400
    body = response.json()
    assert list(body) == ["status", "timestamp", "path", "error"]
    assert body["error"]["code"] == "BadRequestException"
    assert body["error"]["message"] == "Bad Request Exception"
    assert body["error"]["details"] == {
        "errors": [
            {"property": "start", "children": [], "constraints": {"isString": "start must be a string"}},
            {
                "property": "eventTypeId",
                "children": [],
                "constraints": {
                    "isNumber": "eventTypeId must be a number conforming to the specified constraints"
                },
            },
            {
                "property": "timeZone",
                "children": [],
                "constraints": {"isTimeZone": "timeZone must be a valid IANA time-zone"},
            },
            {
                "property": "language",
                "children": [],
                "constraints": {"isString": "language must be a string"},
            },
            {
                "property": "metadata",
                "children": [],
                "constraints": {"isObject": "metadata must be an object"},
            },
        ]
    }
    assert sandbox.state.calcom_bookings == []


def test_create_empty_body_message_is_verbatim(sandbox: Sandbox) -> None:
    response = sandbox.client.post("/v2/bookings", json={}, headers={"cal-api-version": "2026-02-25"})
    sandbox.assert_error(
        response,
        400,
        "start property is wrong,start must be a valid ISO 8601 date string , attendee property is wrong,"
        "attendee should not be null or undefined , eventTypeId or eventTypeSlug + username property is "
        "wrong,Either eventTypeId or eventTypeSlug + username or eventTypeSlug + teamSlug must be provided ",
        path="/v2/bookings",
    )


def test_create_nested_attendee_errors_are_verbatim(sandbox: Sandbox) -> None:
    body = {
        "start": "not-a-date",
        "eventTypeId": 1001,
        "attendee": {"name": "Lena M", "email": "bad", "timeZone": "Mars/Base", "language": "xx"},
    }
    response = sandbox.client.post("/v2/bookings", json=body, headers={"cal-api-version": "2024-08-13"})
    sandbox.assert_error(
        response,
        400,
        "start property is wrong,start must be a valid ISO 8601 date string , attendee property is wrong, "
        "timeZone property is wrong,timeZone must be a valid IANA time-zone , language property is wrong,"
        f"language must be one of the following values: {LANGUAGE_LIST} ",
    )


def test_create_attendee_needs_an_email_or_a_phone(sandbox: Sandbox) -> None:
    body = {
        "start": MON_0900,
        "eventTypeId": 1001,
        "attendee": {"name": "Lena M", "timeZone": "Europe/Berlin"},
    }
    response = sandbox.client.post("/v2/bookings", json=body, headers={"cal-api-version": "2024-08-13"})
    sandbox.assert_error(
        response,
        400,
        "attendee property is wrong, attendee email or phone property is wrong,Attendee must have at "
        "least one contact method (email or phone number) ",
    )
    body["attendee"]["phoneNumber"] = "+15555550100"  # type: ignore[index]
    phone_only = sandbox.client.post("/v2/bookings", json=body, headers={"cal-api-version": "2024-08-13"})
    assert phone_only.status_code == 201
    assert phone_only.json()["data"]["attendees"][0]["phoneNumber"] == "+15555550100"


def test_create_rejects_unknown_properties(sandbox: Sandbox) -> None:
    response = sandbox.book(MON_0900, timeZone="Europe/Berlin")
    sandbox.assert_error(response, 400, "timeZone property is wrong,property timeZone should not exist ")


@pytest.mark.parametrize(
    "metadata",
    [
        {"k" * 41: "v"},
        {"k": "v" * 501},
        {f"k{i}": "v" for i in range(51)},
        {"nested": {"a": "b"}},
        {"list": ["a"]},
    ],
)
def test_create_metadata_limits(sandbox: Sandbox, metadata: object) -> None:
    sandbox.assert_error(sandbox.book(MON_0900, metadata=metadata), 400, MSG_METADATA)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        # The metadata rule walks any JS object, so a short array passes it and fails only IsObject.
        (["not", "an", "object"], "metadata property is wrong,metadata must be an object "),
        ("text", MSG_METADATA[:-1] + ", metadata must be an object "),
    ],
)
def test_create_metadata_that_is_not_an_object(sandbox: Sandbox, metadata: object, message: str) -> None:
    sandbox.assert_error(sandbox.book(MON_0900, metadata=metadata), 400, message)


def test_create_skips_optional_fields_sent_as_null(sandbox: Sandbox) -> None:
    fields = {"metadata": None, "guests": None, "bookingFieldsResponses": None, "location": None}
    data = sandbox.booked(MON_0900, **fields)
    assert data["metadata"] == {}
    assert "guests" not in data


def test_create_validation_follows_the_input_class_order(sandbox: Sandbox) -> None:
    body = {
        "lengthInMinutes": 0,
        "metadata": ["a"],
        "guests": "guest@example.com",
        "eventTypeId": "1001",
        "attendee": {"name": "Lena M", "timeZone": "Mars/Base"},
        "start": "soon",
        "extra": 1,
    }
    response = sandbox.client.post("/v2/bookings", json=body, headers={"cal-api-version": "2026-02-25"})
    sandbox.assert_error(
        response,
        400,
        "extra property is wrong,property extra should not exist , "
        "start property is wrong,start must be a valid ISO 8601 date string , "
        "attendee property is wrong, attendee email or phone property is wrong,Attendee must have at least "
        "one contact method (email or phone number) , timeZone property is wrong,timeZone must be a valid "
        "IANA time-zone , "
        "eventTypeId property is wrong,eventTypeId must be an integer number , "
        "guests property is wrong,guests must be an array , "
        "metadata property is wrong,metadata must be an object , "
        "lengthInMinutes property is wrong,lengthInMinutes must not be less than 1 ",
    )


@pytest.mark.parametrize(
    ("guests", "message"),
    [
        ([1], "guests property is wrong,each value in guests must be a string "),
        (5, "guests property is wrong,each value in guests must be a string, guests must be an array "),
    ],
)
def test_create_guest_list_rules(sandbox: Sandbox, guests: object, message: str) -> None:
    sandbox.assert_error(sandbox.book(MON_0900, guests=guests), 400, message)
    assert sandbox.book(MON_0900, guests=["no-at-sign"]).status_code == 201  # no email check in Cal.com


def test_create_length_in_minutes_on_a_single_length_event_type(sandbox: Sandbox) -> None:
    sandbox.assert_error(
        sandbox.book(MON_0900, lengthInMinutes=30),
        400,
        "Can't specify 'lengthInMinutes' because event type does not have multiple possible lengths. Please, "
        "remove the 'lengthInMinutes' field from the request.",
    )
    sandbox.assert_error(
        sandbox.book(MON_0900, lengthInMinutes="30"),
        400,
        "lengthInMinutes property is wrong,lengthInMinutes must not be less than 1, "
        "lengthInMinutes must be an integer number ",
    )


def test_create_metadata_accepts_strings_numbers_and_booleans_at_the_limits(sandbox: Sandbox) -> None:
    metadata: dict[str, Any] = {f"k{i}": "v" for i in range(47)}
    metadata.update({"k" * 40: "v" * 500, "n": 5, "b": True})
    data = sandbox.booked(MON_0900, metadata=metadata)
    assert data["metadata"] == metadata


def test_create_unknown_event_type_is_the_verbatim_404(sandbox: Sandbox) -> None:
    response = sandbox.book(MON_0900, eventTypeId=999999999)
    sandbox.assert_error(response, 404, "Event type with id 999999999 not found.", path="/v2/bookings")


def test_create_on_a_taken_slot_is_the_documented_400(sandbox: Sandbox) -> None:
    sandbox.booked(MON_0900)
    response = sandbox.book(MON_0900, email="other@example.com")
    body = sandbox.assert_error(response, 400, MSG_TAKEN, path="/v2/bookings")
    assert body == {
        "status": "error",
        "timestamp": body["timestamp"],
        "path": "/v2/bookings",
        "error": {
            "code": "BadRequestException",
            "message": MSG_TAKEN,
            "details": {"message": MSG_TAKEN, "error": "Bad Request", "statusCode": 400},
        },
    }
    assert len(sandbox.state.calcom_bookings) == 1


def test_create_rejects_overlaps_with_seeded_and_third_party_busy_time(sandbox: Sandbox) -> None:
    sandbox.seed(existing_bookings=[{"start": "2026-10-05T13:15:00Z", "end": "2026-10-05T13:45:00Z"}])
    sandbox.assert_error(sandbox.book(MON_0900), 400, MSG_TAKEN)
    sandbox.assert_error(sandbox.book(MON_0930), 400, MSG_TAKEN)
    assert sandbox.book(MON_1000).status_code == 201


@pytest.mark.parametrize(
    ("start", "message"),
    [
        ("2026-10-01T13:00:00Z", MSG_OUT_OF_BOUNDS),  # 09:00 today, inside the 2-hour notice
        ("2027-11-08T14:00:00Z", MSG_OUT_OF_BOUNDS),  # beyond the 400-day horizon
        ("2026-09-30T13:00:00Z", "Attempting to book a meeting in the past."),
        ("2026-10-05T21:00:00Z", MSG_TAKEN),  # 17:00 New York: outside working hours
        ("2026-10-05T20:45:00Z", MSG_TAKEN),  # 16:45 New York: would end after 17:00
        ("2026-10-03T14:00:00Z", MSG_TAKEN),  # Saturday
    ],
)
def test_create_bounds_and_hours(sandbox: Sandbox, start: str, message: str) -> None:
    sandbox.assert_error(sandbox.book(start), 400, message)


def test_create_accepts_an_off_grid_start_when_the_host_is_free(sandbox: Sandbox) -> None:
    data = sandbox.booked("2026-10-05T13:10:00Z")
    assert (data["start"], data["end"]) == ("2026-10-05T13:10:00.000Z", "2026-10-05T13:40:00.000Z")


def test_create_by_event_type_slug_and_username(sandbox: Sandbox) -> None:
    body = {
        "start": MON_0900,
        "eventTypeSlug": "intro-call",
        "username": "sandbox-host",
        "attendee": {"name": "Lena M", "email": LEAD, "timeZone": "UTC"},
    }
    assert (
        sandbox.client.post("/v2/bookings", json=body, headers={"cal-api-version": "2024-08-13"}).status_code
        == 201
    )
    body["username"] = "nobody"
    response = sandbox.client.post("/v2/bookings", json=body, headers={"cal-api-version": "2024-08-13"})
    sandbox.assert_error(response, 404, "Event type with slug intro-call belonging to user nobody not found.")


# Get -----------------------------------------------------------------------------------------------


def test_get_returns_the_stored_booking(sandbox: Sandbox) -> None:
    created = sandbox.booked(MON_0900)
    for version in ("2024-08-13", "2026-02-25", "2026-05-01"):
        response = sandbox.get(created["uid"], version=version)
        assert response.status_code == 200
        body = response.json()
        assert list(body) == ["status", "data"]
        assert list(body["data"]) == BOOKING_KEYS
        assert body["data"] == {k: v for k, v in created.items() if k != "isPlatformManagedUserBooking"}


def test_get_unknown_uid_is_the_verbatim_404(sandbox: Sandbox) -> None:
    sandbox.assert_error(
        sandbox.get("doesNotExist123abc"),
        404,
        "Booking with uid=doesNotExist123abc was not found in the database",
        path="/v2/bookings/doesNotExist123abc",
    )


# List ----------------------------------------------------------------------------------------------


def test_list_filters_by_exact_attendee_email_with_offset_pagination(sandbox: Sandbox) -> None:
    first = sandbox.booked(MON_0900)
    sandbox.booked(MON_0930, email="other@example.com")
    second = sandbox.booked(TUE_0900)
    response = sandbox.list(attendeeEmail=LEAD)
    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["status", "data", "pagination"]
    assert [b["uid"] for b in body["data"]] == [first["uid"], second["uid"]]
    assert list(body["pagination"]) == [
        "returnedItems",
        "totalItems",
        "itemsPerPage",
        "remainingItems",
        "currentPage",
        "totalPages",
        "hasNextPage",
        "hasPreviousPage",
    ]
    assert body["pagination"] == {
        "returnedItems": 2,
        "totalItems": 2,
        "itemsPerPage": 100,
        "remainingItems": 0,
        "currentPage": 1,
        "totalPages": 1,
        "hasNextPage": False,
        "hasPreviousPage": False,
    }
    assert sandbox.list(attendeeEmail="LEAD@example.com").json()["data"] == []  # exact match
    assert len(sandbox.list(attendeeEmail=f" {LEAD} ").json()["data"]) == 2  # trimmed


def test_list_take_and_skip(sandbox: Sandbox) -> None:
    for hour in range(13, 18):
        sandbox.booked(f"2026-10-05T{hour}:00:00Z")
    body = sandbox.list(take=2, skip=2).json()
    assert [b["start"] for b in body["data"]] == ["2026-10-05T15:00:00.000Z", "2026-10-05T16:00:00.000Z"]
    assert body["pagination"] == {
        "returnedItems": 2,
        "totalItems": 5,
        "itemsPerPage": 2,
        "remainingItems": 1,
        "currentPage": 2,
        "totalPages": 3,
        "hasNextPage": True,
        "hasPreviousPage": True,
    }
    # getPagination() clamps skip to the total: past the end of an empty result there is no previous page.
    beyond = sandbox.list(take=2, skip=9, attendeeEmail="nobody@example.com").json()["pagination"]
    assert (beyond["currentPage"], beyond["hasPreviousPage"]) == (0, False)
    # The list query goes through the global ValidationPipe: unknown parameters are dropped, not rejected.
    assert len(sandbox.list(limit=1, cursor="x").json()["data"]) == 5
    assert_pipe_error(
        sandbox.list(take=251, skip=-1),
        [
            constraint("take", max="take must not be greater than 250"),
            constraint("skip", min="skip must not be less than 0"),
        ],
        path="/v2/bookings?take=251&skip=-1",
    )
    assert_pipe_error(
        sandbox.list(take="many"),
        [
            constraint(
                "take",
                max="take must not be greater than 250",
                min="take must not be less than 1",
                isNumber="take must be a number conforming to the specified constraints",
            )
        ],
    )
    assert len(sandbox.list(take="2abc").json()["data"]) == 2  # parseInt reads the leading digits


def test_list_status_filters(sandbox: Sandbox) -> None:
    past = sandbox.booked("2026-10-01T15:00:00Z")
    cancelled = sandbox.booked(MON_0900)
    upcoming = sandbox.booked(MON_0930)
    assert sandbox.cancel(cancelled["uid"]).status_code == 200
    sandbox.clock.set(datetime(2026, 10, 1, 16, 0, tzinfo=UTC))  # the first booking has ended

    def uids(**params: str) -> list[str]:
        return [b["uid"] for b in sandbox.list(**params).json()["data"]]

    assert uids() == [upcoming["uid"]]  # 2024-08-13 defaults to upcoming
    assert uids(status="past") == [past["uid"]]
    assert uids(status="cancelled") == [cancelled["uid"]]
    assert uids(status="upcoming,past") == [past["uid"], upcoming["uid"]]
    assert uids(status="unconfirmed") == []
    assert_pipe_error(
        sandbox.list(status="upcoming,booked"),
        [
            constraint(
                "status",
                isEnum="Invalid status. Allowed are upcoming, recurring, past, cancelled, unconfirmed",
            )
        ],
    )


def test_list_time_window_filters(sandbox: Sandbox) -> None:
    sandbox.booked(MON_0900)
    tuesday = sandbox.booked(TUE_0900)
    body = sandbox.list(afterStart="2026-10-06T00:00:00Z", beforeEnd="2026-10-06T23:59:59Z").json()
    assert [b["uid"] for b in body["data"]] == [tuesday["uid"]]
    assert_pipe_error(
        sandbox.list(afterStart="tomorrow", beforeEnd="later", sortStart="up", sortUpdatedAt="down"),
        [
            constraint("afterStart", isIso8601="fromDate must be a valid ISO 8601 date."),
            constraint("beforeEnd", isIso8601="toDate must be a valid ISO 8601 date."),
            constraint("sortStart", isEnum='SortStart must be either "asc" or "desc".'),
            constraint("sortUpdatedAt", isEnum='SortCreated must be either "asc" or "desc".'),
        ],
    )


def test_list_filter_precedence_and_transforms(sandbox: Sandbox) -> None:
    plus = sandbox.booked(MON_0900, email="lead+tag@example.com")
    other = sandbox.booked(TUE_0900)
    # An unencoded "+" arrives as a space; Cal.com turns inner whitespace back into "+".
    response = sandbox.client.get(
        "/v2/bookings?attendeeEmail=lead+tag@example.com", headers={"cal-api-version": "2024-08-13"}
    )
    assert [b["uid"] for b in response.json()["data"]] == [plus["uid"]]
    # eventTypeIds wins over eventTypeId; an empty filter value filters nothing.
    assert len(sandbox.list(eventTypeIds="1001", eventTypeId=5).json()["data"]) == 2
    assert len(sandbox.list(attendeeEmail="").json()["data"]) == 2
    assert [b["uid"] for b in sandbox.list(bookingUid=f" {other['uid']} ").json()["data"]] == [other["uid"]]
    # Only the first sort parameter (sortStart, sortEnd, sortCreated, sortUpdatedAt) applies.
    ordered = sandbox.list(sortStart="desc", sortEnd="asc").json()["data"]
    assert [b["uid"] for b in ordered] == [other["uid"], plus["uid"]]
    assert_pipe_error(
        sandbox.list(eventTypeIds="1001,x", eventTypeId="1.5", teamsIds="y", teamId="z"),
        [
            constraint(
                "eventTypeIds",
                isNumber="each value in eventTypeIds must be a number conforming to the specified "
                "constraints",
            ),
            constraint("eventTypeId", isInt="eventTypeId must be an integer number"),
            constraint(
                "teamsIds",
                isNumber="each value in teamsIds must be a number conforming to the specified constraints",
            ),
            constraint("teamId", isInt="teamId must be an integer number"),
        ],
    )


def test_list_2026_05_01_uses_cursor_pagination(sandbox: Sandbox) -> None:
    created = [sandbox.booked(f"2026-10-05T{hour}:00:00Z")["uid"] for hour in range(13, 18)]
    first = sandbox.list(version="2026-05-01", status="upcoming", limit=2).json()
    assert list(first) == ["status", "data", "pagination"]
    assert [b["uid"] for b in first["data"]] == created[:2]
    assert list(first["pagination"]) == ["nextCursor", "hasMore"]
    assert first["pagination"]["hasMore"] is True
    assert first["pagination"]["nextCursor"].startswith("eyJ2Ijox")
    second = sandbox.list(
        version="2026-05-01", status="upcoming", limit=2, cursor=first["pagination"]["nextCursor"]
    )
    third = sandbox.list(
        version="2026-05-01", status="upcoming", limit=2, cursor=second.json()["pagination"]["nextCursor"]
    ).json()
    assert [b["uid"] for b in second.json()["data"]] == created[2:4]
    assert [b["uid"] for b in third["data"]] == created[4:]
    assert third["pagination"] == {"nextCursor": None, "hasMore": False}
    # Without a status the walk covers every status, newest start first.
    everything = sandbox.list(version="2026-05-01").json()
    assert [b["uid"] for b in everything["data"]] == created[::-1]
    assert len(sandbox.list(version="2026-05-01", take=2).json()["data"]) == 5  # take is dropped here
    assert_pipe_error(
        sandbox.list(version="2026-05-01", status="upcoming,past", limit=101),
        [
            constraint(
                "status",
                isEnum="status must be one of the following values: upcoming, recurring, past, cancelled, "
                "unconfirmed",
            ),
            constraint("limit", max="limit must not be greater than 100"),
        ],
    )


# Reschedule ----------------------------------------------------------------------------------------


def test_reschedule_creates_a_new_booking_and_cancels_the_old_one(sandbox: Sandbox) -> None:
    old = sandbox.booked(MON_0900, metadata={"bt_idem": "k1"})
    response = sandbox.reschedule(
        old["uid"], {"start": TUE_0900, "reschedulingReason": "Something came up", "rescheduledBy": LEAD}
    )
    assert response.status_code == 201
    body = response.json()
    assert list(body) == ["status", "data"]
    new = body["data"]
    assert new["uid"] != old["uid"]
    assert UID.match(new["uid"])
    assert new["rescheduledFromUid"] == old["uid"]
    assert new["reschedulingReason"] == "Something came up"
    assert new["rescheduledByEmail"] == LEAD
    assert new["icsUid"] == old["icsUid"]  # the iCal UID keeps the original uid
    assert new["status"] == "accepted"
    assert (new["start"], new["end"]) == ("2026-10-06T13:00:00.000Z", "2026-10-06T13:30:00.000Z")
    assert new["metadata"] == {"bt_idem": "k1"}
    assert new["attendees"] == old["attendees"]
    assert new["bookingFieldsResponses"]["rescheduledReason"] == "Something came up"
    assert new["isPlatformManagedUserBooking"] is False
    assert list(new)[:13] == [
        "id",
        "uid",
        "title",
        "description",
        "hosts",
        "status",
        "cancellationReason",
        "cancelledByEmail",
        "reschedulingReason",
        "rescheduledByEmail",
        "rescheduledFromUid",
        "start",
        "end",
    ]
    moved = sandbox.get(old["uid"]).json()["data"]
    assert moved["status"] == "cancelled"
    assert moved["rescheduledToUid"] == new["uid"]
    assert moved["rescheduledByEmail"] == LEAD
    assert list(moved).index("rescheduledToUid") == list(moved).index("start") - 1
    assert "2026-10-05T13:00:00.000Z" in starts(sandbox.slots(start="2026-10-05", end="2026-10-05").json())


def test_reschedule_may_overlap_the_booking_being_moved(sandbox: Sandbox) -> None:
    old = sandbox.booked(MON_0900)
    assert sandbox.reschedule(old["uid"], {"start": "2026-10-05T13:15:00Z"}).status_code == 201


def test_reschedule_errors(sandbox: Sandbox) -> None:
    sandbox.assert_error(
        sandbox.reschedule("doesNotExist123abc", {"start": TUE_0900}),
        404,
        "Booking with uid=doesNotExist123abc was not found in the database",
    )
    old = sandbox.booked(MON_0900)
    sandbox.assert_error(
        sandbox.reschedule(old["uid"], {}),
        400,
        "start property is wrong,start must be a valid ISO 8601 date string ",
    )
    sandbox.assert_error(
        sandbox.reschedule(old["uid"], {"start": TUE_0900, "rescheduleReason": "x"}),
        400,
        "rescheduleReason property is wrong,property rescheduleReason should not exist ",
    )
    other = sandbox.booked(TUE_0900, email="other@example.com")
    sandbox.assert_error(sandbox.reschedule(old["uid"], {"start": TUE_0900}), 400, MSG_TAKEN)
    new = sandbox.reschedule(old["uid"], {"start": MON_1000}).json()["data"]
    sandbox.assert_error(
        sandbox.reschedule(old["uid"], {"start": "2026-10-07T13:00:00Z"}),
        400,
        f"Can't reschedule booking with uid={old['uid']} because it has been cancelled and rescheduled "
        f"already to booking with uid={new['uid']}. You probably want to reschedule {new['uid']} instead by "
        "passing it within the request URL.",
    )
    assert sandbox.cancel(other["uid"]).status_code == 200
    sandbox.assert_error(
        sandbox.reschedule(other["uid"], {"start": "2026-10-07T13:00:00Z"}),
        400,
        f"Can't reschedule booking with uid={other['uid']} because it has been cancelled. Please provide "
        "uid of a booking that is not cancelled.",
    )


# Cancel --------------------------------------------------------------------------------------------


def test_cancel_returns_200_with_the_same_uid_cancelled(sandbox: Sandbox) -> None:
    booking = sandbox.booked(MON_0900)
    response = sandbox.cancel(booking["uid"], {"cancellationReason": "No longer needed"})
    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["status", "data"]
    data = body["data"]
    assert list(data) == BOOKING_KEYS
    assert data["uid"] == booking["uid"]
    assert data["status"] == "cancelled"
    assert data["cancellationReason"] == "No longer needed"
    assert data["cancelledByEmail"] == "host@example.com"
    assert sandbox.get(booking["uid"]).json()["data"] == data
    assert "2026-10-05T13:00:00.000Z" in starts(sandbox.slots(start="2026-10-05", end="2026-10-05").json())


def test_cancel_errors(sandbox: Sandbox) -> None:
    sandbox.assert_error(
        sandbox.cancel("doesNotExist123abc"),
        404,
        "Booking with uid=doesNotExist123abc not found",
        path="/v2/bookings/doesNotExist123abc/cancel",
    )
    booking = sandbox.booked(MON_0900)
    sandbox.assert_error(
        sandbox.cancel(booking["uid"], {"reason": "x"}),
        400,
        "reason property is wrong,property reason should not exist ",
    )
    assert sandbox.cancel(booking["uid"]).status_code == 200
    sandbox.assert_error(
        sandbox.cancel(booking["uid"]),
        400,
        f"Can't cancel booking with uid={booking['uid']} because it has been cancelled already. Please "
        "provide uid of a booking that is not cancelled.",
    )


def test_cancel_of_an_ended_booking_is_the_raw_core_error(sandbox: Sandbox) -> None:
    booking = sandbox.booked(MON_0900)
    sandbox.clock.set(datetime(2026, 10, 5, 14, 0, tzinfo=UTC))
    response = sandbox.cancel(booking["uid"])
    assert response.status_code == 400
    assert response.json() == {"statusCode": 400, "message": "Cannot cancel a booking that has already ended"}


# Round trip and routing ----------------------------------------------------------------------------


def test_round_trip_slots_create_get_list_reschedule_cancel(sandbox: Sandbox) -> None:
    offered = starts(sandbox.slots(start="2026-10-05", end="2026-10-06", timeZone="Europe/Berlin").json())
    assert offered[0] == "2026-10-05T15:00:00.000+02:00"
    created = sandbox.booked(offered[0])
    assert created["start"] == "2026-10-05T13:00:00.000Z"
    assert sandbox.get(created["uid"]).json()["data"]["status"] == "accepted"
    listed = sandbox.list(attendeeEmail=LEAD).json()["data"]
    assert [b["uid"] for b in listed] == [created["uid"]]
    moved = sandbox.reschedule(created["uid"], {"start": offered[1]}).json()["data"]
    assert moved["start"] == "2026-10-05T13:30:00.000Z"
    listed = sandbox.list(attendeeEmail=LEAD, status="upcoming,cancelled").json()["data"]
    assert {b["uid"]: b["status"] for b in listed} == {created["uid"]: "cancelled", moved["uid"]: "accepted"}
    cancelled = sandbox.cancel(moved["uid"], {"cancellationReason": "Changed plans"}).json()["data"]
    assert cancelled["status"] == "cancelled"
    assert sandbox.list(attendeeEmail=LEAD).json()["data"] == []
    assert [b["status"] for b in sandbox.snapshot()["calcom"]["bookings"]] == ["cancelled", "cancelled"]


def test_unknown_routes_and_methods_under_v2_are_vendor_404s(sandbox: Sandbox) -> None:
    sandbox.assert_error(
        sandbox.client.get("/v2/nope?x=1"), 404, "Cannot GET /v2/nope?x=1", path="/v2/nope?x=1"
    )
    sandbox.assert_error(sandbox.client.delete("/v2/bookings/abc"), 404, "Cannot DELETE /v2/bookings/abc")
    legacy = sandbox.client.get("/v2/slots/available", params={"eventTypeId": 1001})
    sandbox.assert_error(legacy, 404, "Cannot GET /v2/slots/available?eventTypeId=1001")
    # Unknown routes are logged too, so a wiring check can see an agent calling an unmirrored endpoint.
    log = sandbox.log()
    assert [(e["method"], e["path"], e["group"], e["status"]) for e in log] == [
        ("GET", "/v2/nope", "unrouted", 404),
        ("DELETE", "/v2/bookings/abc", "unrouted", 404),
        ("GET", "/v2/slots/available", "unrouted", 404),
    ]
    assert log[2]["query"] == {"eventTypeId": "1001"}
    assert log[2]["response"] == legacy.json()
    assert all(e["completed"] and e["fault"] is None for e in log)


def test_booking_routes_without_a_known_version_are_not_served(sandbox: Sandbox) -> None:
    booking = sandbox.booked(MON_0900)
    for response in (
        sandbox.client.get(f"/v2/bookings/{booking['uid']}"),
        sandbox.client.get("/v2/bookings", headers={"cal-api-version": "2099-01-01"}),
        sandbox.client.post(f"/v2/bookings/{booking['uid']}/cancel", json={}),
    ):
        assert response.status_code == 400
        assert response.json()["error"]["message"] == "Bad Request Exception"
    sandbox.assert_error(
        sandbox.get(booking["uid"], version="2024-09-04"), 404, f"Cannot GET /v2/bookings/{booking['uid']}"
    )
    # The legacy controller has no POST /:uid/reschedule, so that route is a 404 without a known version.
    path = f"/v2/bookings/{booking['uid']}/reschedule"
    for headers in ({}, {"cal-api-version": "2024-06-14"}):
        response = sandbox.client.post(path, json={"start": TUE_0900}, headers=headers)
        sandbox.assert_error(response, 404, f"Cannot POST {path}", path=path)
    assert sandbox.get(booking["uid"]).json()["data"]["status"] == "accepted"


def test_version_routing_is_answered_before_auth(sandbox: Sandbox) -> None:
    """Nest's router and the unguarded legacy create answer before any auth guard runs."""
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:
        query = "eventTypeId=1001&start=2026-10-05&end=2026-10-05"
        response = anonymous.get(f"/v2/slots?{query}")
        sandbox.assert_error(response, 404, f"Cannot GET /v2/slots?{query}")
        response = anonymous.post("/v2/bookings", json={}, headers={"cal-api-version": "2024-09-04"})
        sandbox.assert_error(response, 404, "Cannot POST /v2/bookings")
        legacy = anonymous.post("/v2/bookings", json={})  # the live probe without credentials: legacy 400
        assert legacy.status_code == 400
        assert legacy.json()["error"]["message"] == "Bad Request Exception"
        # Routes that exist for the version still need the token.
        assert anonymous.get("/v2/bookings/abc", headers={"cal-api-version": "2024-08-13"}).status_code == 401
        assert anonymous.get("/v2/bookings/abc").status_code == 401  # the legacy get runs after auth here
    assert [e["status"] for e in sandbox.log()] == [404, 404, 400, 401, 401]


def test_error_path_keeps_the_raw_percent_encoding(sandbox: Sandbox) -> None:
    response = sandbox.client.get("/v2/bookings/abc%20def", headers={"cal-api-version": "2024-08-13"})
    sandbox.assert_error(
        response,
        404,
        "Booking with uid=abc def was not found in the database",
        path="/v2/bookings/abc%20def",
    )


# Units ---------------------------------------------------------------------------------------------


def test_short_uuid_is_22_flickr_base58_characters() -> None:
    assert short_uuid(uuid.UUID(int=0)) == "1" * 22
    assert short_uuid(uuid.UUID(int=57)) == "1" * 21 + "Z"
    for _ in range(50):
        assert UID.match(short_uuid(uuid.uuid4()))


def test_format_errors_matches_the_nested_pipe_format() -> None:
    errors = [
        FieldError("start", ("start must be a valid ISO 8601 date string",)),
        FieldError("attendee", (), (FieldError("timeZone", ("timeZone must be a valid IANA time-zone",)),)),
    ]
    assert format_errors(errors) == (
        "start property is wrong,start must be a valid ISO 8601 date string , attendee property is wrong, "
        "timeZone property is wrong,timeZone must be a valid IANA time-zone "
    )


def test_timestamps_follow_the_clock(sandbox: Sandbox) -> None:
    sandbox.clock.advance(timedelta(minutes=5, milliseconds=250))
    data = sandbox.booked(MON_0900)
    assert data["createdAt"] == "2026-10-01T12:05:00.250Z"


def test_list_team_filters_use_the_teams_ids_spelling(sandbox: Sandbox) -> None:
    sandbox.booked(MON_0900)
    assert sandbox.list(teamsIds="50,60").json()["data"] == []  # the event type belongs to a user
    assert len(sandbox.list(eventTypeIds="1001,1002").json()["data"]) == 1
    # "teamIds" (the docs' example spelling) is not a parameter: the global pipe drops it, so nothing filters.
    assert len(sandbox.list(teamIds="50").json()["data"]) == 1


# Hostile input -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b'{"start":"2026-10-05T13:00:00Z","eventTypeId":1001,'
        b'"attendee":{"name":"Lena M","email":"lead@example.com","timeZone":"UTC"},"metadata":{"n":NaN}}',
        b'{"start":"2026-10-05T13:00:00Z","eventTypeId":1001,'
        b'"attendee":{"name":"Lena M","email":"lead@example.com","timeZone":"UTC"},"metadata":{"n":1e999}}',
        b"[" * 50_000 + b"]" * 50_000,
    ],
)
def test_non_json_bodies_are_rejected_and_state_stays_readable(sandbox: Sandbox, raw: bytes) -> None:
    headers = {"cal-api-version": "2024-08-13", "content-type": "application/json"}
    response = sandbox.client.post("/v2/bookings", content=raw, headers=headers)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BadRequestException"
    state = sandbox.snapshot()  # a non-finite number in stored state would break GET /_state
    assert state["calcom"]["bookings"] == []
    assert state["request_log"][0]["completed"] is True


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"start": "2026-10-05", "end": "2026-10-05", "duration": "²"},
            "duration must be a number conforming to the specified constraints",
        ),
        (
            {"start": "2026-10-05", "end": "2026-10-05", "eventTypeId": "¹"},
            "eventTypeId must be a number conforming to the specified constraints",
        ),
        (
            {"start": "0001-01-01T00:00:00+05:00", "end": "2026-10-05"},
            "start must be a valid ISO 8601 date string",
        ),
        ({"start": "2026-10-05", "end": "9999-12-31"}, "end must be a valid ISO 8601 date string"),
    ],
)
def test_slots_out_of_range_input_is_a_validation_error(
    sandbox: Sandbox, params: dict[str, str], message: str
) -> None:
    response = sandbox.slots(**params)
    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert sandbox.log("slots")[0]["completed"] is True


@pytest.mark.parametrize("start", ["9999-12-31T23:59:00Z", "0001-01-01T00:00:00+05:00"])
def test_create_with_an_unrepresentable_start_is_a_validation_error(sandbox: Sandbox, start: str) -> None:
    sandbox.assert_error(
        sandbox.book(start), 400, "start property is wrong,start must be a valid ISO 8601 date string "
    )
    assert_pipe_error(
        sandbox.list(afterStart=start),
        [constraint("afterStart", isIso8601="fromDate must be a valid ISO 8601 date.")],
    )

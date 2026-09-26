"""Contract tests: the Google Calendar v3 subset answers with the real API's shapes, statuses and texts."""

from __future__ import annotations

import json
import re
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from booking_truth.timeutil import iso_z, parse_iso

if TYPE_CHECKING:
    from conftest import Sandbox

LEAD = "lead@example.com"
CAL = "primary"
EVENTS = f"/calendar/v3/calendars/{CAL}/events"
MON_0900 = "2026-10-05T13:00:00Z"  # Monday 09:00 in New York
MON_0930 = "2026-10-05T13:30:00Z"
MON_1000 = "2026-10-05T14:00:00Z"
MONDAY = ("2026-10-05T00:00:00Z", "2026-10-06T00:00:00Z")
EVENT_ID = "bt0k1d3m4q"
HANG_S = 0.4
CLIENT_TIMEOUT_S = 0.15
MSG_SA = "Service accounts cannot invite attendees without Domain-Wide Delegation of Authority."
MSG_AUTH_INVALID = (
    "Request had invalid authentication credentials. Expected OAuth 2 access token, login cookie or other "
    "valid authentication credential. See https://developers.google.com/identity/sign-in/web/devconsole-project."
)
MSG_AUTH_MISSING = (
    "Request is missing required authentication credential. Expected OAuth 2 access token, login cookie or "
    "other valid authentication credential. See https://developers.google.com/identity/sign-in/web/devconsole-project."
)
EVENT_KEYS = [
    "kind",
    "etag",
    "id",
    "status",
    "htmlLink",
    "created",
    "updated",
    "summary",
    "description",
    "creator",
    "organizer",
    "start",
    "end",
    "iCalUID",
    "sequence",
    "extendedProperties",
    "reminders",
    "eventType",
]
SERVER_ID = re.compile(r"^[a-v0-9]{26}$")
ETAG = re.compile(r'^"\d+"$')


def plus(start: str, minutes: int) -> str:
    return iso_z(parse_iso(start) + timedelta(minutes=minutes))


def event_body(at: str = MON_0900, minutes: int = 30, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "summary": "Intro call with Lena M",
        "description": "Lead: Lena M <lead@example.com>",
        "start": {"dateTime": at, "timeZone": "America/New_York"},
        "end": {"dateTime": plus(at, minutes), "timeZone": "America/New_York"},
        "extendedProperties": {"private": {"bt_lead_email": LEAD}},
    }
    body.update(extra)
    return body


def insert(sandbox: Sandbox, body: dict[str, Any] | None = None, **params: Any) -> httpx.Response:
    return sandbox.client.post(EVENTS, json=event_body() if body is None else body, params=params)


def inserted(sandbox: Sandbox, body: dict[str, Any] | None = None) -> dict[str, Any]:
    response = insert(sandbox, body)
    assert response.status_code == 200, response.text
    data: dict[str, Any] = response.json()
    return data


def freebusy(
    sandbox: Sandbox, window: tuple[str, str] = MONDAY, items: tuple[str, ...] = (CAL,), **extra: Any
) -> httpx.Response:
    body = {"timeMin": window[0], "timeMax": window[1], "items": [{"id": i} for i in items], **extra}
    return sandbox.client.post("/calendar/v3/freeBusy", json=body)


def busy(sandbox: Sandbox, window: tuple[str, str] = MONDAY) -> list[dict[str, str]]:
    response = freebusy(sandbox, window)
    assert response.status_code == 200, response.text
    periods: list[dict[str, str]] = response.json()["calendars"][CAL]["busy"]
    return periods


def assert_error(
    response: httpx.Response,
    status: int,
    reason: str,
    message: str,
    *,
    domain: str = "global",
    location: tuple[str, str] | None = None,
) -> None:
    """The Calendar backend envelope, its key order and its bytes (1-space indent, trailing newline)."""
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == "application/json; charset=UTF-8"
    item: dict[str, str] = {"domain": domain, "reason": reason, "message": message}
    if location is not None:
        item.update(locationType=location[0], location=location[1])
    expected = {"error": {"errors": [item], "code": status, "message": message}}
    assert response.json() == expected
    assert list(response.json()["error"]) == ["errors", "code", "message"]
    assert list(response.json()["error"]["errors"][0]) == list(item)
    assert response.text == json.dumps(expected, indent=1) + "\n"


# freeBusy ----------------------------------------------------------------------------------------------


def test_freebusy_reports_the_shared_host_calendar(sandbox: Sandbox) -> None:
    sandbox.seed(existing_bookings=[{"start": "2026-10-05T15:00:00Z", "title": "Dentist"}])
    sandbox.booked(MON_0900)  # a Cal.com booking occupies the same calendar
    inserted(sandbox, event_body("2026-10-05T18:00:00Z", 60))
    response = freebusy(sandbox)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=UTF-8"
    body = response.json()
    assert list(body) == ["kind", "timeMin", "timeMax", "calendars"]  # no "groups" without group ids
    assert body["kind"] == "calendar#freeBusy"
    assert (body["timeMin"], body["timeMax"]) == ("2026-10-05T00:00:00.000Z", "2026-10-06T00:00:00.000Z")
    assert body["calendars"] == {
        CAL: {
            "busy": [
                {"start": "2026-10-05T13:00:00Z", "end": "2026-10-05T13:30:00Z"},
                {"start": "2026-10-05T15:00:00Z", "end": "2026-10-05T15:30:00Z"},
                {"start": "2026-10-05T18:00:00Z", "end": "2026-10-05T19:00:00Z"},
            ]
        }
    }
    assert response.text == json.dumps(body, indent=1) + "\n"


def test_freebusy_merges_touching_blocks_clips_to_the_window_and_honours_time_zone(sandbox: Sandbox) -> None:
    inserted(sandbox, event_body(MON_0900))
    inserted(sandbox, event_body(MON_0930))  # touches the first one
    window = ("2026-10-05T13:15:00Z", "2026-10-05T20:00:00Z")
    assert busy(sandbox, window) == [{"start": "2026-10-05T13:15:00Z", "end": "2026-10-05T14:00:00Z"}]
    response = freebusy(sandbox, window, timeZone="Europe/Berlin")
    assert response.json()["calendars"][CAL]["busy"] == [
        {"start": "2026-10-05T15:15:00+02:00", "end": "2026-10-05T16:00:00+02:00"}
    ]


def test_freebusy_for_an_unknown_calendar_is_a_200_with_a_not_found_entry(sandbox: Sandbox) -> None:
    inserted(sandbox)
    response = freebusy(sandbox, items=("someone@example.com", CAL))
    assert response.status_code == 200
    calendars = response.json()["calendars"]
    assert calendars["someone@example.com"] == {
        "errors": [{"domain": "global", "reason": "notFound"}],
        "busy": [],
    }
    assert list(calendars["someone@example.com"]) == ["errors", "busy"]  # errors first, busy still present
    assert calendars[CAL] == {"busy": [{"start": MON_0900, "end": MON_0930}]}
    empty = sandbox.client.post(
        "/calendar/v3/freeBusy", json={"timeMin": MONDAY[0], "timeMax": MONDAY[1], "items": [{}]}
    )
    assert list(empty.json()["calendars"]) == [""]  # an item without an id becomes the key ""
    no_items = sandbox.client.post("/calendar/v3/freeBusy", json={"timeMin": MONDAY[0], "timeMax": MONDAY[1]})
    assert no_items.json()["calendars"] == {}


def test_freebusy_request_errors(sandbox: Sandbox) -> None:
    post = sandbox.client.post
    url = "/calendar/v3/freeBusy"
    assert_error(post(url, json={"timeMax": MONDAY[1]}), 400, "required", "Required")
    assert_error(
        post(url, json={"timeMin": "2026-10-05", "timeMax": MONDAY[1]}), 400, "badRequest", "Bad Request"
    )
    assert_error(
        post(url, json={"timeMin": "2026-10-05T00:00:00", "timeMax": MONDAY[1]}),
        400,
        "badRequest",
        "Bad Request",
    )
    assert_error(
        post(url, json={"timeMin": MONDAY[1], "timeMax": MONDAY[0]}),
        400,
        "timeRangeEmpty",
        "The specified time range is empty.",
        domain="calendar",
        location=("parameter", "timeMax"),
    )
    raw = post(url, content=b"{not json", headers={"content-type": "application/json"})
    assert_error(raw, 400, "parseError", "Parse Error")
    bad_zone = post(url, json={"timeMin": MONDAY[0], "timeMax": MONDAY[1], "timeZone": "Mars/Base"})
    assert_error(bad_zone, 400, "invalid", "Invalid value for: timeZone")


def test_request_log_records_the_full_freebusy_request_and_response(sandbox: Sandbox) -> None:
    response = freebusy(sandbox, timeZone="UTC")
    entry = sandbox.log("freebusy")[0]
    assert (entry["method"], entry["path"], entry["status"], entry["fault"]) == (
        "POST",
        "/calendar/v3/freeBusy",
        200,
        None,
    )
    assert entry["body"] == {
        "timeMin": MONDAY[0],
        "timeMax": MONDAY[1],
        "items": [{"id": CAL}],
        "timeZone": "UTC",
    }
    assert entry["response"] == response.json()


# events.insert -------------------------------------------------------------------------------------------


def test_insert_with_a_client_id_returns_the_event_resource(sandbox: Sandbox) -> None:
    response = insert(sandbox, event_body(id=EVENT_ID))
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=UTF-8"
    event = response.json()
    assert list(event) == EVENT_KEYS
    assert response.text == json.dumps(event, indent=1) + "\n"
    assert event["kind"] == "calendar#event"
    assert ETAG.match(event["etag"])
    assert event["id"] == EVENT_ID
    assert event["status"] == "confirmed"
    assert event["htmlLink"].startswith("https://www.google.com/calendar/event?eid=")
    assert event["created"] == "2026-10-01T12:00:00Z"  # no milliseconds
    assert event["updated"] == "2026-10-01T12:00:00.000Z"  # milliseconds
    assert event["creator"] == {"email": "booking-agent@example.com"}
    assert event["organizer"] == {"email": "host@example.com", "self": True}
    # Times are rendered in the calendar's zone; the time zone sent is echoed.
    assert event["start"] == {"dateTime": "2026-10-05T09:00:00-04:00", "timeZone": "America/New_York"}
    assert event["end"] == {"dateTime": "2026-10-05T09:30:00-04:00", "timeZone": "America/New_York"}
    assert event["iCalUID"] == f"{EVENT_ID}@google.com"
    assert event["sequence"] == 0
    assert event["extendedProperties"] == {"private": {"bt_lead_email": LEAD}}
    assert event["reminders"] == {"useDefault": True}
    assert event["eventType"] == "default"
    assert "attendees" not in event
    assert sandbox.snapshot()["google"]["events"] == [event]


def test_insert_without_an_id_gets_a_server_id(sandbox: Sandbox) -> None:
    body = event_body()
    del body["extendedProperties"]
    event = inserted(sandbox, body)
    assert SERVER_ID.match(event["id"])
    assert "extendedProperties" not in event


def test_duplicate_id_is_a_409_with_the_verbatim_body(sandbox: Sandbox) -> None:
    inserted(sandbox, event_body(id=EVENT_ID))
    response = insert(sandbox, event_body(MON_1000, id=EVENT_ID))
    assert_error(response, 409, "duplicate", "The requested identifier already exists.")
    # Byte for byte what Google sent in a raw 2020 capture.
    assert response.text == (
        '{\n "error": {\n  "errors": [\n   {\n    "domain": "global",\n    "reason": "duplicate",\n'
        '    "message": "The requested identifier already exists."\n   }\n  ],\n  "code": 409,\n'
        '  "message": "The requested identifier already exists."\n }\n}\n'
    )
    assert len(sandbox.snapshot()["google"]["events"]) == 1


@pytest.mark.parametrize("event_id", ["abc", "asv2-3265", "UPPERCASE1", "wxyz12345", "a" * 1025, 12345])
def test_invalid_ids_are_rejected(sandbox: Sandbox, event_id: object) -> None:
    assert_error(insert(sandbox, event_body(id=event_id)), 400, "invalid", "Invalid resource id value.")


@pytest.mark.parametrize("event_id", ["asv2t3265", "0" * 5, "v" * 1024])
def test_valid_ids_are_base32hex_of_5_to_1024_characters(sandbox: Sandbox, event_id: str) -> None:
    assert inserted(sandbox, event_body(id=event_id))["id"] == event_id


def test_insert_does_no_conflict_checking(sandbox: Sandbox) -> None:
    sandbox.booked(MON_0900)
    sandbox.seed(existing_bookings=[{"start": MON_0900}])
    first = inserted(sandbox, event_body(MON_0900))
    second = inserted(sandbox, event_body(MON_0900))
    assert first["id"] != second["id"]
    assert len(sandbox.snapshot()["google"]["events"]) == 2


def test_insert_validation(sandbox: Sandbox) -> None:
    body = event_body()
    del body["end"]
    assert_error(insert(sandbox, body), 400, "required", "Missing end time.")
    body = event_body()
    del body["start"]
    assert_error(insert(sandbox, body), 400, "required", "Missing start time.")
    backwards = event_body(MON_1000)
    backwards["end"] = {"dateTime": MON_0900}
    assert_error(
        insert(sandbox, backwards),
        400,
        "timeRangeEmpty",
        "The specified time range is empty.",
        domain="calendar",
    )
    local = event_body(start={"dateTime": "2026-10-05T09:00:00"}, end={"dateTime": "2026-10-05T09:30:00"})
    assert_error(insert(sandbox, local), 400, "required", "Missing time zone definition for end time.")
    bad_zone = event_body(end={"dateTime": MON_0930, "timeZone": "Mars/Base"})
    assert_error(insert(sandbox, bad_zone), 400, "invalid", "Invalid time zone definition for end time.")
    all_day = event_body(start={"date": "2026-10-05"}, end={"date": "2026-10-06"})
    assert (
        insert(sandbox, all_day)
        .json()["error"]["message"]
        .startswith("The sandbox mirrors timed events only")
    )
    assert_error(insert(sandbox, event_body(status="done")), 400, "invalid", "Invalid value for: status")
    assert_error(
        insert(sandbox, sendUpdates="everyone"),
        400,
        "invalidParameter",
        "Invalid value 'everyone'. Values must match the following regular expression: "
        "'all|externalOnly|none'",
        location=("parameter", "sendUpdates"),
    )
    raw = sandbox.client.post(EVENTS, content=b"[1,", headers={"content-type": "application/json"})
    assert_error(raw, 400, "parseError", "Parse Error")
    assert sandbox.snapshot()["google"]["events"] == []


def test_local_date_time_is_read_in_the_time_zone_sent(sandbox: Sandbox) -> None:
    body = event_body(
        start={"dateTime": "2026-10-05T15:00:00", "timeZone": "Europe/Berlin"},
        end={"dateTime": "2026-10-05T15:30:00", "timeZone": "Europe/Berlin"},
    )
    event = inserted(sandbox, body)
    assert event["start"] == {"dateTime": "2026-10-05T09:00:00-04:00", "timeZone": "Europe/Berlin"}
    assert busy(sandbox) == [{"start": MON_0900, "end": MON_0930}]


def test_extended_property_limits(sandbox: Sandbox) -> None:
    props = {"k" * 44: "kept", "k" * 45: "dropped", "long": "x" * 1100, "n": 7}
    event = inserted(sandbox, event_body(extendedProperties={"private": props}))
    private = event["extendedProperties"]["private"]
    assert private == {"k" * 44: "kept", "long": "x" * 1024, "n": "7"}


def test_insert_on_an_unknown_calendar_is_a_404(sandbox: Sandbox) -> None:
    response = sandbox.client.post("/calendar/v3/calendars/other%40example.com/events", json=event_body())
    assert_error(response, 404, "notFound", "Not Found")


def test_seeded_calendar_id_replaces_primary(sandbox: Sandbox) -> None:
    sandbox.seed(google_calendar_id="team@example.com")
    path = "/calendar/v3/calendars/team%40example.com/events"
    event = sandbox.client.post(path, json=event_body()).json()
    assert event["organizer"] == {"email": "team@example.com", "self": True}
    assert_error(insert(sandbox), 404, "notFound", "Not Found")  # "primary" no longer exists
    response = freebusy(sandbox, items=("team@example.com",))
    assert response.json()["calendars"] == {
        "team@example.com": {"busy": [{"start": MON_0900, "end": MON_0930}]}
    }


# Service account and attendees ----------------------------------------------------------------------------


def test_a_service_account_cannot_add_attendees(sandbox: Sandbox) -> None:
    body = event_body(id=EVENT_ID, attendees=[{"email": LEAD}])
    response = insert(sandbox, body, sendUpdates="none")  # sendUpdates=none does not help
    assert_error(response, 403, "forbiddenForServiceAccounts", MSG_SA, domain="calendar")
    assert sandbox.snapshot()["google"]["events"] == []
    event = inserted(sandbox, event_body(id=EVENT_ID, attendees=[]))  # an empty list is fine
    assert "attendees" not in event
    patch = sandbox.client.patch(f"{EVENTS}/{EVENT_ID}", json={"attendees": [{"email": LEAD}]})
    assert_error(patch, 403, "forbiddenForServiceAccounts", MSG_SA, domain="calendar")


def test_domain_wide_delegation_allows_attendees(sandbox: Sandbox) -> None:
    sandbox.seed(google_sa_can_invite=True)
    event = inserted(sandbox, event_body(attendees=[{"email": LEAD, "displayName": "Lena M"}]))
    assert event["attendees"] == [{"email": LEAD, "displayName": "Lena M", "responseStatus": "needsAction"}]
    assert list(event).index("attendees") == list(event).index("sequence") + 1
    missing = event_body(attendees=[{"displayName": "No email"}])
    assert_error(insert(sandbox, missing), 400, "required", "Missing attendee email.")


# Tombstones ---------------------------------------------------------------------------------------------


def test_deleted_events_stay_as_cancelled_tombstones(sandbox: Sandbox) -> None:
    created = inserted(sandbox, event_body(id=EVENT_ID))
    url = f"{EVENTS}/{EVENT_ID}"
    deleted = sandbox.client.delete(url)
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert sandbox.log("events.delete")[0]["response"] is None
    # get always returns the tombstone
    tombstone = sandbox.client.get(url)
    assert tombstone.status_code == 200
    assert tombstone.json()["status"] == "cancelled"
    assert tombstone.json()["summary"] == created["summary"]
    assert tombstone.json()["etag"] != created["etag"]
    # list hides it unless showDeleted=true
    assert sandbox.client.get(EVENTS).json()["items"] == []
    shown = sandbox.client.get(EVENTS, params={"showDeleted": "true"}).json()["items"]
    assert [(e["id"], e["status"]) for e in shown] == [(EVENT_ID, "cancelled")]
    # the id stays reserved, a second delete is 410, and the time is free again
    assert_error(
        insert(sandbox, event_body(id=EVENT_ID)), 409, "duplicate", "The requested identifier already exists."
    )
    assert_error(sandbox.client.delete(url), 410, "deleted", "Resource has been deleted")
    assert busy(sandbox) == []
    # patch with status confirmed revives it
    revived = sandbox.client.patch(
        url,
        json={
            "status": "confirmed",
            "start": {"dateTime": MON_1000},
            "end": {"dateTime": plus(MON_1000, 30)},
        },
    )
    assert revived.status_code == 200
    assert revived.json()["status"] == "confirmed"
    assert sandbox.client.get(EVENTS).json()["items"][0]["id"] == EVENT_ID
    assert busy(sandbox) == [{"start": MON_1000, "end": plus(MON_1000, 30)}]


def test_patch_to_cancelled_deletes_and_a_tombstone_stays_cancelled_when_patched(sandbox: Sandbox) -> None:
    inserted(sandbox, event_body(id=EVENT_ID))
    url = f"{EVENTS}/{EVENT_ID}"
    assert sandbox.client.patch(url, json={"status": "cancelled"}).json()["status"] == "cancelled"
    assert busy(sandbox) == []
    renamed = sandbox.client.patch(url, json={"summary": "Renamed"}).json()
    assert (renamed["summary"], renamed["status"]) == ("Renamed", "cancelled")


# events.get and events.patch -------------------------------------------------------------------------------


def test_get_renders_times_in_the_requested_zone(sandbox: Sandbox) -> None:
    created = inserted(sandbox, event_body(id=EVENT_ID))
    url = f"{EVENTS}/{EVENT_ID}"
    assert sandbox.client.get(url).json() == created
    tokyo = sandbox.client.get(url, params={"timeZone": "Asia/Tokyo"}).json()
    assert tokyo["start"] == {"dateTime": "2026-10-05T22:00:00+09:00", "timeZone": "America/New_York"}
    assert_error(sandbox.client.get(f"{EVENTS}/nosuchevent"), 404, "notFound", "Not Found")
    assert_error(
        sandbox.client.get(f"/calendar/v3/calendars/other/events/{EVENT_ID}"), 404, "notFound", "Not Found"
    )


def test_patch_merges_objects_replaces_arrays_and_bumps_sequence_on_a_move(sandbox: Sandbox) -> None:
    sandbox.seed(google_sa_can_invite=True)
    props = {"private": {"bt_lead_email": LEAD, "bt_idem": "k1"}}
    created = inserted(
        sandbox,
        event_body(
            id=EVENT_ID, extendedProperties=props, attendees=[{"email": LEAD}, {"email": "b@example.com"}]
        ),
    )
    url = f"{EVENTS}/{EVENT_ID}"
    sandbox.clock.advance(timedelta(minutes=5))
    renamed = sandbox.client.patch(url, json={"summary": "Renamed"}).json()
    assert renamed["summary"] == "Renamed"
    assert renamed["description"] == created["description"]
    assert renamed["sequence"] == 0
    assert renamed["etag"] != created["etag"]
    assert renamed["updated"] == "2026-10-01T12:05:00.000Z"
    assert renamed["created"] == created["created"]
    patch = {
        "extendedProperties": {"private": {"bt_idem": None, "bt_moved": "yes"}},
        "attendees": [{"email": LEAD}],
        "start": {"dateTime": MON_1000},
        "end": {"dateTime": plus(MON_1000, 30)},
    }
    moved = sandbox.client.patch(url, json=patch).json()
    assert moved["extendedProperties"] == {"private": {"bt_lead_email": LEAD, "bt_moved": "yes"}}
    assert [a["email"] for a in moved["attendees"]] == [LEAD]
    assert moved["start"] == {"dateTime": "2026-10-05T10:00:00-04:00", "timeZone": "America/New_York"}
    assert moved["sequence"] == 1
    assert sandbox.snapshot()["google"]["events"] == [moved]


def test_patch_preconditions_and_errors(sandbox: Sandbox) -> None:
    created = inserted(sandbox, event_body(id=EVENT_ID))
    url = f"{EVENTS}/{EVENT_ID}"
    stale = sandbox.client.patch(url, json={"summary": "x"}, headers={"If-Match": '"1"'})
    assert_error(stale, 412, "conditionNotMet", "Precondition Failed", location=("header", "If-Match"))
    fresh = sandbox.client.patch(url, json={"summary": "x"}, headers={"If-Match": created["etag"]})
    assert fresh.status_code == 200
    assert_error(sandbox.client.patch(f"{EVENTS}/missing123", json={}), 404, "notFound", "Not Found")
    backwards = {"end": {"dateTime": "2026-10-05T12:00:00Z"}}
    assert_error(
        sandbox.client.patch(url, json=backwards),
        400,
        "timeRangeEmpty",
        "The specified time range is empty.",
        domain="calendar",
    )
    assert_error(sandbox.client.patch(url, json={"end": None}), 400, "required", "Missing end time.")


# events.list -------------------------------------------------------------------------------------------


def test_list_by_private_extended_property(sandbox: Sandbox) -> None:
    mine = inserted(sandbox, event_body(MON_0900))
    other = event_body(MON_1000, extendedProperties={"private": {"bt_lead_email": "other@example.com"}})
    inserted(sandbox, other)
    response = sandbox.client.get(
        f"{EVENTS}?privateExtendedProperty=bt_lead_email%3Dlead%40example.com&timeMin=2026-10-01T00%3A00%3A00Z"
    )
    assert response.status_code == 200
    body = response.json()
    assert list(body) == [
        "kind",
        "etag",
        "summary",
        "updated",
        "timeZone",
        "accessRole",
        "defaultReminders",
        "items",
    ]
    assert body["kind"] == "calendar#events"
    assert (body["summary"], body["timeZone"], body["accessRole"]) == (
        "host@example.com",
        "America/New_York",
        "writer",
    )
    assert body["items"] == [mine]
    both = sandbox.client.get(
        EVENTS,
        params=[
            ("privateExtendedProperty", "bt_lead_email=lead@example.com"),
            ("privateExtendedProperty", "x=y"),
        ],
    )
    assert both.json()["items"] == []  # repeated constraints must all match


def test_list_time_bounds_are_exclusive(sandbox: Sandbox) -> None:
    event = inserted(sandbox, event_body(MON_0900))

    def ids(**params: str) -> list[str]:
        return [e["id"] for e in sandbox.client.get(EVENTS, params=params).json()["items"]]

    assert ids(timeMin=MON_0930) == []  # the end must be after timeMin
    assert ids(timeMin="2026-10-05T13:29:59Z") == [event["id"]]
    assert ids(timeMax=MON_0900) == []  # the start must be before timeMax
    assert ids(timeMax="2026-10-05T13:00:01Z") == [event["id"]]
    assert ids(q="lena") == [event["id"]]
    assert ids(q="nobody") == []


def test_list_ordering_paging_and_errors(sandbox: Sandbox) -> None:
    late = inserted(sandbox, event_body(MON_1000))
    early = inserted(sandbox, event_body(MON_0900))
    assert [e["id"] for e in sandbox.client.get(EVENTS).json()["items"]] == [late["id"], early["id"]]
    ordered = sandbox.client.get(EVENTS, params={"singleEvents": "true", "orderBy": "startTime"}).json()
    assert [e["id"] for e in ordered["items"]] == [early["id"], late["id"]]
    assert_error(
        sandbox.client.get(EVENTS, params={"orderBy": "startTime"}),
        400,
        "badRequest",
        "The requested ordering is not available for the particular query.",
    )
    first = sandbox.client.get(EVENTS, params={"maxResults": "1"}).json()
    assert [e["id"] for e in first["items"]] == [late["id"]]
    second = sandbox.client.get(
        EVENTS, params={"maxResults": "1", "pageToken": first["nextPageToken"]}
    ).json()
    assert [e["id"] for e in second["items"]] == [early["id"]]
    assert "nextPageToken" not in second
    assert_error(
        sandbox.client.get(EVENTS, params={"timeMin": "2026-10-05"}), 400, "badRequest", "Bad Request"
    )
    assert_error(
        sandbox.client.get(EVENTS, params={"timeMin": MON_1000, "timeMax": MON_0900}),
        400,
        "timeRangeEmpty",
        "The specified time range is empty.",
        domain="calendar",
        location=("parameter", "timeMax"),
    )
    assert_error(
        sandbox.client.get(EVENTS, params={"syncToken": "abc"}),
        410,
        "fullSyncRequired",
        "Sync token is no longer valid, a full sync is required.",
        domain="calendar",
        location=("parameter", "syncToken"),
    )
    assert_error(
        sandbox.client.get(EVENTS, params={"privateExtendedProperty": "novalue"}),
        400,
        "badRequest",
        "Bad Request",
    )
    assert_error(sandbox.client.get("/calendar/v3/calendars/x/events"), 404, "notFound", "Not Found")


# Auth and routing ----------------------------------------------------------------------------------------


GOOGLE_ROUTES: list[tuple[str, str, dict[str, Any] | None, str]] = [
    ("POST", "/calendar/v3/freeBusy", {"timeMin": MONDAY[0], "timeMax": MONDAY[1]}, "freebusy"),
    ("POST", EVENTS, {}, "events.insert"),
    ("GET", EVENTS, None, "events.list"),
    ("GET", f"{EVENTS}/{EVENT_ID}", None, "events.get"),
    ("PATCH", f"{EVENTS}/{EVENT_ID}", {}, "events.patch"),
    ("DELETE", f"{EVENTS}/{EVENT_ID}", None, "events.delete"),
]


@pytest.mark.parametrize(("method", "path", "body", "group"), GOOGLE_ROUTES)
def test_auth_errors_are_the_front_end_401s(
    sandbox: Sandbox, method: str, path: str, body: Any, group: str
) -> None:
    sandbox.faults({"group": group, "mode": "error_500"})
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:
        missing = anonymous.request(method, path, json=body)
        wrong = anonymous.request(method, path, json=body, headers={"Authorization": "Bearer ya29.wrong"})
    for response, message, item_message, reason in (
        (missing, MSG_AUTH_MISSING, "Login Required.", "required"),
        (wrong, MSG_AUTH_INVALID, "Invalid Credentials", "authError"),
    ):
        assert response.status_code == 401
        error = response.json()["error"]
        assert list(error) == ["code", "message", "errors", "status"]
        assert error == {
            "code": 401,
            "message": message,
            "errors": [
                {
                    "message": item_message,
                    "domain": "global",
                    "reason": reason,
                    "location": "Authorization",
                    "locationType": "header",
                }
            ],
            "status": "UNAUTHENTICATED",
        }
        assert response.text == json.dumps(response.json(), indent=2) + "\n"  # the front end indents by 2
    assert [e["status"] for e in sandbox.log(group)] == [401, 401]
    assert sandbox.snapshot()["faults"][0]["matched"] == 0  # auth failures never meet fault rules


def test_unknown_routes_are_logged_as_unrouted(sandbox: Sandbox) -> None:
    assert_error(sandbox.client.get("/calendar/v3/users/me/calendarList"), 404, "notFound", "Not Found")
    assert_error(sandbox.client.put(f"{EVENTS}/{EVENT_ID}", json={}), 404, "notFound", "Not Found")
    assert [e["path"] for e in sandbox.log("unrouted")] == [
        "/calendar/v3/users/me/calendarList",
        f"{EVENTS}/{EVENT_ID}",
    ]


def test_google_events_block_cal_com_slots(sandbox: Sandbox) -> None:
    inserted(sandbox, event_body(MON_0900))
    monday = sandbox.slots(start="2026-10-05", end="2026-10-05").json()["data"]["2026-10-05"]
    assert {"start": "2026-10-05T13:00:00.000Z"} not in monday
    assert len(monday) == 15
    transparent = inserted(sandbox, event_body(MON_1000, transparency="transparent"))
    assert transparent["transparency"] == "transparent"
    assert len(sandbox.slots(start="2026-10-05", end="2026-10-05").json()["data"]["2026-10-05"]) == 15


# Fault modes ------------------------------------------------------------------------------------------


def test_not_found_on_freebusy_is_a_200_with_not_found_entries(sandbox: Sandbox) -> None:
    inserted(sandbox)
    sandbox.faults({"group": "freebusy", "mode": "not_found"})
    response = freebusy(sandbox)
    assert response.status_code == 200
    assert response.json() == {
        "kind": "calendar#freeBusy",
        "timeMin": "2026-10-05T00:00:00.000Z",
        "timeMax": "2026-10-06T00:00:00.000Z",
        "calendars": {CAL: {"errors": [{"domain": "global", "reason": "notFound"}], "busy": []}},
    }
    assert sandbox.log("freebusy")[0]["fault"] == "not_found"


@pytest.mark.parametrize(
    "group", ["events.insert", "events.get", "events.list", "events.patch", "events.delete"]
)
def test_not_found_on_events_is_the_404(sandbox: Sandbox, group: str) -> None:
    inserted(sandbox, event_body(id=EVENT_ID))
    sandbox.faults({"group": group, "mode": "not_found"})
    url = f"{EVENTS}/{EVENT_ID}"
    calls = {
        "events.insert": lambda: insert(sandbox, event_body(MON_1000)),
        "events.get": lambda: sandbox.client.get(url),
        "events.list": lambda: sandbox.client.get(EVENTS),
        "events.patch": lambda: sandbox.client.patch(url, json={"summary": "x"}),
        "events.delete": lambda: sandbox.client.delete(url),
    }
    assert_error(calls[group](), 404, "notFound", "Not Found")
    events = sandbox.snapshot()["google"]["events"]
    assert [(e["id"], e["status"], e["summary"]) for e in events] == [
        (EVENT_ID, "confirmed", "Intro call with Lena M")
    ]


def test_malformed_freebusy_hides_busy_periods_under_another_key(sandbox: Sandbox) -> None:
    inserted(sandbox)
    sandbox.faults({"group": "freebusy", "mode": "malformed"})
    response = freebusy(sandbox)
    assert response.status_code == 200
    body = response.json()
    assert body["calendars"] == {CAL: {"busyPeriods": [{"start": MON_0900, "end": MON_0930}]}}
    assert body["calendars"][CAL].get("busy", []) == []  # a lenient client reading only "busy" sees free time
    assert "errors" not in body["calendars"][CAL]
    assert sandbox.log("freebusy")[0]["response"] == body


def test_malformed_event_writes_commit(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "events.insert", "mode": "malformed"})
    response = insert(sandbox, event_body(id=EVENT_ID))
    assert response.status_code == 200
    assert response.json() == {
        "event": {
            "eventId": EVENT_ID,
            "startTime": "2026-10-05T09:00:00-04:00",
            "endTime": "2026-10-05T09:30:00-04:00",
            "eventStatus": "CONFIRMED",
        }
    }
    assert [e["id"] for e in sandbox.snapshot()["google"]["events"]] == [EVENT_ID]
    sandbox.faults(
        {"group": "events.list", "mode": "malformed"}, {"group": "events.delete", "mode": "malformed"}
    )
    listed = sandbox.client.get(EVENTS).json()
    assert listed["count"] == 1
    assert "items" not in listed
    deleted = sandbox.client.delete(f"{EVENTS}/{EVENT_ID}")
    assert (deleted.status_code, deleted.json()) == (200, {"deleted": {"eventId": EVENT_ID}})
    assert sandbox.snapshot()["google"]["events"][0]["status"] == "cancelled"


def test_error_500_is_the_backend_error(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "events.*", "mode": "error_500", "times": 2})
    assert_error(insert(sandbox), 500, "backendError", "Backend Error")
    assert_error(sandbox.client.get(EVENTS), 500, "backendError", "Backend Error")
    assert insert(sandbox).status_code == 200


def test_timeout_does_not_commit_and_a_patient_client_gets_a_503(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "events.insert", "mode": "timeout", "hang_s": HANG_S})
    with pytest.raises(httpx.ReadTimeout):
        sandbox.client.post(EVENTS, json=event_body(id=EVENT_ID), timeout=CLIENT_TIMEOUT_S)
    sandbox.wait_for(lambda: sandbox.log("events.insert")[0]["completed"])
    assert sandbox.snapshot()["google"]["events"] == []
    sandbox.faults({"group": "freebusy", "mode": "timeout", "hang_s": 0.2})
    assert_error(freebusy(sandbox), 503, "backendError", "Backend Error")


def test_commit_then_timeout_then_a_retry_with_the_same_id_is_a_duplicate(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "events.insert", "mode": "commit_then_timeout", "hang_s": HANG_S})
    with pytest.raises(httpx.ReadTimeout):
        sandbox.client.post(EVENTS, json=event_body(id=EVENT_ID), timeout=CLIENT_TIMEOUT_S)
    assert [e["id"] for e in sandbox.snapshot()["google"]["events"]] == [EVENT_ID]  # committed while hanging
    assert_error(
        insert(sandbox, event_body(id=EVENT_ID)), 409, "duplicate", "The requested identifier already exists."
    )
    sandbox.wait_for(lambda: sandbox.log("events.insert")[0]["completed"])
    assert sandbox.log("events.insert")[0]["status"] == 200


def test_slot_taken_after_offer_takes_the_last_window_and_the_insert_still_succeeds(sandbox: Sandbox) -> None:
    assert busy(sandbox) == []  # the offer: Monday is free
    sandbox.faults({"group": "events.insert", "mode": "slot_taken_after_offer"})
    response = insert(sandbox, event_body(MON_0900, id=EVENT_ID))
    assert response.status_code == 200  # Google does no conflict checking
    state = sandbox.snapshot()
    taken = state["external_busy"]
    assert len(taken) == 16  # every free working-hours slot of the window
    assert {b["attendee_email"] for b in taken} == {"third-party@example.com"}
    assert {b["source"] for b in taken} == {"slot_taken_after_offer"}
    assert [e["id"] for e in state["google"]["events"]] == [EVENT_ID]
    assert busy(sandbox) == [{"start": MON_0900, "end": "2026-10-05T21:00:00Z"}]
    tuesday = ("2026-10-06T00:00:00Z", "2026-10-07T00:00:00Z")
    assert busy(sandbox, tuesday) == []  # only the offered window was taken
    entry = sandbox.log("events.insert")[0]
    assert (entry["status"], entry["fault"]) == (200, "slot_taken_after_offer")


def test_slot_taken_after_offer_skips_windows_the_client_did_not_receive_intact(sandbox: Sandbox) -> None:
    freebusy(sandbox)  # Monday, intact
    sandbox.faults(
        {"group": "freebusy", "mode": "malformed"},
        {"group": "events.insert", "mode": "slot_taken_after_offer"},
    )
    freebusy(sandbox, ("2026-10-06T00:00:00Z", "2026-10-07T00:00:00Z"))  # Tuesday, malformed
    freebusy(sandbox, items=("someone@example.com",))  # not the host calendar
    assert insert(sandbox).status_code == 200
    taken = [parse_iso(b["start"]) for b in sandbox.snapshot()["external_busy"]]
    assert len(taken) == 16
    assert all(t.day == 5 for t in taken)


def test_slot_taken_after_offer_on_freebusy_takes_that_window_after_answering(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "freebusy", "mode": "slot_taken_after_offer"})
    assert busy(sandbox) == []  # answered before the take
    assert busy(sandbox) == [{"start": MON_0900, "end": "2026-10-05T21:00:00Z"}]


def test_slow_adds_latency_on_google_groups(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "events.get", "mode": "slow", "latency_ms": 250})
    inserted(sandbox, event_body(id=EVENT_ID))
    started = time.monotonic()
    assert sandbox.client.get(f"{EVENTS}/{EVENT_ID}").status_code == 200
    assert time.monotonic() - started >= 0.25
    assert sandbox.log("events.get")[0]["fault"] == "slow"

"""Contract tests: the HubSpot CRM v3 subset answers with the real API's shapes, statuses and texts."""

from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest

if TYPE_CHECKING:
    from conftest import Sandbox

LEAD = "lead@example.com"
NOW_ISO = "2026-10-01T12:00:00Z"  # Instant.toString(): no fraction when the milliseconds are zero
CONTACTS = "/crm/v3/objects/contacts"
MEETINGS = "/crm/v3/objects/meetings"
MSG_AUTH = (
    "Authentication credentials not found. This API supports OAuth 2.0 authentication and you can find more "
    "details at https://developers.hubspot.com/docs/methods/auth/oauth-overview"
)
UUID7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
OBJECT_KEYS = ["id", "properties", "createdAt", "updatedAt", "archived"]
OUTCOME_OPTIONS = (
    '[label: "Scheduled"\nvalue: "SCHEDULED"\ndisplay_order: 0\nhidden: false\nread_only: false\n, '
    'label: "Completed"\nvalue: "COMPLETED"\ndisplay_order: 1\nhidden: false\nread_only: false\n, '
    'label: "Rescheduled"\nvalue: "RESCHEDULED"\ndisplay_order: 2\nhidden: false\nread_only: false\n, '
    'label: "No Show"\nvalue: "NO_SHOW"\ndisplay_order: 3\nhidden: false\nread_only: false\n, '
    'label: "Canceled"\nvalue: "CANCELED"\ndisplay_order: 4\nhidden: false\nread_only: false\n]'
)


def create_contact(sandbox: Sandbox, **props: Any) -> httpx.Response:
    properties = {"email": LEAD, "firstname": "Lena", "lastname": "M", **props}
    return sandbox.client.post(CONTACTS, json={"properties": properties})


def contact(sandbox: Sandbox, **props: Any) -> dict[str, Any]:
    response = create_contact(sandbox, **props)
    assert response.status_code == 201, response.text
    data: dict[str, Any] = response.json()
    return data


def search(sandbox: Sandbox, email: str = LEAD, **extra: Any) -> httpx.Response:
    body = {
        "filterGroups": [{"filters": [{"propertyName": "email", "operator": "EQ", "value": email}]}],
        **extra,
    }
    return sandbox.client.post(f"{CONTACTS}/search", json=body)


def meeting_body(contact_id: str | int | None = None, **props: Any) -> dict[str, Any]:
    properties = {
        "hs_timestamp": "2026-10-05T13:00:00.000Z",
        "hs_meeting_title": "Intro call",
        "hs_meeting_body": "Booked by the agent",
        "hs_meeting_start_time": "2026-10-05T13:00:00Z",
        "hs_meeting_end_time": "2026-10-05T13:30:00Z",
        "hs_meeting_outcome": "SCHEDULED",
        **props,
    }
    body: dict[str, Any] = {"properties": properties}
    if contact_id is not None:
        body["associations"] = [
            {
                "to": {"id": contact_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 200}],
            }
        ]
    return body


def meeting(sandbox: Sandbox, body: dict[str, Any]) -> dict[str, Any]:
    response = sandbox.client.post(MEETINGS, json=body)
    assert response.status_code == 201, response.text
    data: dict[str, Any] = response.json()
    return data


def assert_error(
    response: httpx.Response,
    status: int,
    message: str,
    category: str | None,
    *,
    keys: list[str] | None = None,
) -> dict[str, Any]:
    """HubSpot's compact error body, key order, correlation id and header."""
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == "application/json;charset=utf-8"
    body: dict[str, Any] = response.json()
    expected_keys = keys or ["status", "message", "correlationId", *(["category"] if category else [])]
    assert list(body) == expected_keys
    assert body["status"] == "error"
    assert body["message"] == message
    assert UUID7.match(body["correlationId"])
    assert response.headers["x-hubspot-correlation-id"] == body["correlationId"]
    assert response.headers["x-request-id"] == body["correlationId"]
    if category is not None:
        assert body["category"] == category
    assert response.text == json.dumps(body, separators=(",", ":"))
    return body


def validation_message(*items: dict[str, Any]) -> str:
    embedded = [
        {
            "isValid": False,
            "message": item["message"],
            "error": item["error"],
            "name": item["name"],
            "localizedErrorMessage": item["message"],
            "propertyValue": item["value"],
            "portalId": 20000001,
        }
        for item in items
    ]
    return "Property values were not valid: " + json.dumps(embedded, separators=(",", ":"))


# Contacts ----------------------------------------------------------------------------------------------


def test_create_contact(sandbox: Sandbox) -> None:
    response = create_contact(sandbox)
    assert response.status_code == 201
    assert response.headers["content-type"] == "application/json;charset=utf-8"
    assert UUID7.match(response.headers["x-hubspot-correlation-id"])
    assert response.headers["x-request-id"] == response.headers["x-hubspot-correlation-id"]
    body = response.json()
    assert list(body) == OBJECT_KEYS
    assert response.text == json.dumps(body, separators=(",", ":"))
    contact_id = body["id"]
    assert contact_id.isdigit()
    assert response.headers["location"] == f"{sandbox.url}{CONTACTS}/{contact_id}"
    assert body["properties"] == {
        "createdate": NOW_ISO,
        "email": LEAD,
        "firstname": "Lena",
        "hs_all_contact_vids": contact_id,
        "hs_email_domain": "example.com",
        "hs_is_contact": "true",
        "hs_is_unworked": "true",
        "hs_lifecyclestage_lead_date": NOW_ISO,
        "hs_marketable_status": "false",
        "hs_marketable_until_renewal": "false",
        "hs_object_id": contact_id,
        "hs_object_source": "INTEGRATION",
        "hs_object_source_label": "INTEGRATION",
        "hs_pipeline": "contacts-lifecycle-pipeline",
        "lastmodifieddate": NOW_ISO,
        "lastname": "M",
        "lifecyclestage": "lead",
    }
    assert list(body["properties"]) == sorted(body["properties"])
    assert (body["createdAt"], body["updatedAt"], body["archived"]) == (NOW_ISO, NOW_ISO, False)
    assert sandbox.snapshot()["hubspot"]["contacts"] == [body]


def test_existing_email_is_the_409_conflict_with_the_existing_id(sandbox: Sandbox) -> None:
    first = contact(sandbox)
    response = create_contact(sandbox, email="LEAD@example.com")  # emails match case-insensitively
    assert_error(response, 409, f"Contact already exists. Existing ID: {first['id']}", "CONFLICT")
    assert len(sandbox.snapshot()["hubspot"]["contacts"]) == 1


def test_contact_validation_errors(sandbox: Sandbox) -> None:
    response = sandbox.client.post(
        CONTACTS,
        json={"properties": {"email": "test222", "color": "red", "hs_object_id": "5", "phone": 5551234}},
    )
    body = assert_error(
        response,
        400,
        validation_message(
            {
                "message": "Email address test222 is invalid",
                "error": "INVALID_EMAIL",
                "name": "email",
                "value": "test222",
            },
            {
                "message": 'Property "color" does not exist',
                "error": "PROPERTY_DOESNT_EXIST",
                "name": "color",
                "value": "red",
            },
            {
                "message": '"hs_object_id" is a calculated property; its value cannot be set.',
                "error": "READ_ONLY_VALUE",
                "name": "hs_object_id",
                "value": "5",
            },
        ),
        "VALIDATION_ERROR",
        keys=["status", "message", "correlationId", "errors", "category"],
    )
    assert body["errors"] == [
        {
            "message": "Email address test222 is invalid",
            "code": "INVALID_EMAIL",
            "context": {"propertyName": ["email"]},
        },
        {
            "message": 'Property "color" does not exist',
            "code": "PROPERTY_DOESNT_EXIST",
            "context": {"propertyName": ["color"]},
        },
        {
            "message": '"hs_object_id" is a calculated property; its value cannot be set.',
            "code": "READ_ONLY_VALUE",
            "context": {"propertyName": ["hs_object_id"]},
        },
    ]
    assert sandbox.snapshot()["hubspot"]["contacts"] == []


def test_malformed_bodies(sandbox: Sandbox) -> None:
    raw = sandbox.client.post(
        CONTACTS, content=b'{"properties": {', headers={"content-type": "application/json"}
    )
    assert_error(
        raw,
        400,
        "Invalid input JSON on line 1, column 17: Expecting property name enclosed in double quotes",
        None,
    )
    missing = sandbox.client.post(CONTACTS, json={})
    message = "Invalid input JSON: some of required attributes are not set [properties]"
    assert_error(missing, 400, message, "VALIDATION_ERROR")


def test_update_contact(sandbox: Sandbox) -> None:
    created = contact(sandbox, phone="555")
    sandbox.clock.advance(timedelta(seconds=90, milliseconds=384))
    response = sandbox.client.patch(
        f"{CONTACTS}/{created['id']}", json={"properties": {"firstname": "Elena", "phone": ""}}
    )
    assert response.status_code == 200
    body = response.json()
    assert list(body) == OBJECT_KEYS
    assert body["properties"]["firstname"] == "Elena"
    assert "phone" not in body["properties"]  # an empty string clears the value
    assert body["properties"]["lastmodifieddate"] == "2026-10-01T12:01:30.384Z"
    assert (body["createdAt"], body["updatedAt"]) == (NOW_ISO, "2026-10-01T12:01:30.384Z")
    by_email = sandbox.client.patch(
        f"{CONTACTS}/{LEAD}", params={"idProperty": "email"}, json={"properties": {"lastname": "Q"}}
    )
    assert by_email.json()["properties"]["lastname"] == "Q"
    other = contact(sandbox, email="other@example.com")
    conflict = sandbox.client.patch(f"{CONTACTS}/{other['id']}", json={"properties": {"email": LEAD}})
    assert_error(conflict, 409, f"Contact already exists. Existing ID: {created['id']}", "CONFLICT")
    cleared = sandbox.client.patch(f"{CONTACTS}/{other['id']}", json={"properties": {"email": ""}}).json()
    assert "email" not in cleared["properties"]
    assert "hs_email_domain" not in cleared["properties"]  # the calculated domain goes with the email


def test_not_found_bodies(sandbox: Sandbox) -> None:
    numeric = sandbox.client.patch(f"{CONTACTS}/999999", json={"properties": {"firstname": "x"}})
    assert_error(numeric, 404, "resource not found", None)  # no category
    word = sandbox.client.patch(f"{CONTACTS}/{LEAD}", json={"properties": {"firstname": "x"}})
    body = assert_error(
        word,
        404,
        "Object not found.  objectId are usually numeric.",  # two spaces, as HubSpot sends it
        "OBJECT_NOT_FOUND",
        keys=["status", "message", "correlationId", "context", "category"],
    )
    assert body["context"] == {"id": [LEAD]}
    assert_error(sandbox.client.get(f"{MEETINGS}/12345"), 404, "resource not found", None)


# Search ------------------------------------------------------------------------------------------------


def test_search_by_email_is_case_insensitive_and_immediately_consistent(sandbox: Sandbox) -> None:
    assert search(sandbox).json() == {"total": 0, "results": []}
    created = contact(sandbox)
    contact(sandbox, email="other@example.com")
    response = search(sandbox, "Lead@Example.com")
    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["total", "results"]
    assert body["total"] == 1
    result = body["results"][0]
    assert list(result) == OBJECT_KEYS
    assert result["id"] == created["id"]
    assert result["properties"] == {
        "createdate": NOW_ISO,
        "email": LEAD,
        "firstname": "Lena",
        "hs_object_id": created["id"],
        "lastmodifieddate": NOW_ISO,
        "lastname": "M",
    }


def test_search_properties_paging_and_sorts(sandbox: Sandbox) -> None:
    ids = [contact(sandbox, email=f"lead{i}@example.com", firstname=f"Lead {i}")["id"] for i in range(3)]
    requested = sandbox.client.post(
        f"{CONTACTS}/search",
        json={"properties": ["email", "phone", "nonexistent"], "limit": 2, "sorts": ["hs_object_id"]},
    ).json()
    assert requested["total"] == 3
    assert [r["id"] for r in requested["results"]] == ids[:2]
    assert requested["results"][0]["properties"] == {
        "createdate": NOW_ISO,
        "email": "lead0@example.com",
        "hs_object_id": ids[0],
        "lastmodifieddate": NOW_ISO,
        "phone": None,  # requested but unset
    }
    assert requested["paging"] == {"next": {"after": "2"}}
    rest = sandbox.client.post(f"{CONTACTS}/search", json={"limit": 2, "after": 2}).json()
    assert [r["id"] for r in rest["results"]] == ids[2:]
    assert "paging" not in rest
    descending = sandbox.client.post(
        f"{CONTACTS}/search", json={"sorts": [{"propertyName": "firstname", "direction": "DESCENDING"}]}
    ).json()
    assert [r["id"] for r in descending["results"]] == ids[::-1]
    top_level = sandbox.client.post(
        f"{CONTACTS}/search",
        json={"filters": [{"propertyName": "firstname", "operator": "CONTAINS_TOKEN", "value": "lead 1"}]},
    ).json()
    assert [r["id"] for r in top_level["results"]] == [ids[1]]
    queried = sandbox.client.post(f"{CONTACTS}/search", json={"query": "LEAD2"}).json()
    assert [r["id"] for r in queried["results"]] == [ids[2]]
    either = sandbox.client.post(
        f"{CONTACTS}/search",
        json={
            "filterGroups": [
                {"filters": [{"propertyName": "email", "operator": "EQ", "value": "lead0@example.com"}]},
                {"filters": [{"propertyName": "email", "operator": "IN", "values": ["lead2@example.com"]}]},
            ]
        },
    ).json()
    assert [r["id"] for r in either["results"]] == [ids[0], ids[2]]
    null_limit = sandbox.client.post(f"{CONTACTS}/search", json={"limit": None, "after": None}).json()
    assert null_limit["total"] == 3
    assert "paging" not in null_limit  # the default page of 10 holds all three


def test_search_in_wants_lowercase_values(sandbox: Sandbox) -> None:
    """The search guide: with ``IN`` and ``NOT_IN`` on a string property, the searched values must be
    lowercase; ``EQ`` is case-insensitive."""
    created = contact(sandbox, email="Mixed.Case@example.com")

    def ids(operator: str, values: list[str]) -> list[str]:
        body = {"filters": [{"propertyName": "email", "operator": operator, "values": values}]}
        response = sandbox.client.post(f"{CONTACTS}/search", json=body)
        assert response.status_code == 200, response.text
        return [r["id"] for r in response.json()["results"]]

    assert ids("IN", ["mixed.case@example.com"]) == [created["id"]]
    assert ids("IN", ["Mixed.Case@example.com"]) == []
    assert ids("NOT_IN", ["Mixed.Case@example.com"]) == [created["id"]]
    assert ids("NOT_IN", ["mixed.case@example.com"]) == []
    assert [r["id"] for r in search(sandbox, "MIXED.case@EXAMPLE.com").json()["results"]] == [created["id"]]


def test_query_searches_the_default_searchable_properties(sandbox: Sandbox) -> None:
    """``query`` covers the documented list, e.g. ``mobilephone``, not only names and email."""
    created = contact(sandbox, mobilephone="+1 555 0199")
    contact(sandbox, email="other@example.com")
    found = sandbox.client.post(f"{CONTACTS}/search", json={"query": "555 0199"}).json()
    assert [r["id"] for r in found["results"]] == [created["id"]]


def test_search_validation(sandbox: Sandbox) -> None:
    group = {"filters": [{"propertyName": "email", "operator": "EQ", "value": LEAD}]}
    too_many_groups = sandbox.client.post(f"{CONTACTS}/search", json={"filterGroups": [group] * 6})
    message = "Invalid input JSON: too many filterGroups (count: 6, max allowed: 5)"
    assert_error(too_many_groups, 400, message, "VALIDATION_ERROR")
    wide = {"filters": [group["filters"][0]] * 7}
    too_many = sandbox.client.post(f"{CONTACTS}/search", json={"filterGroups": [wide]})
    message = "Invalid input JSON: too many filters per filterGroup (count: 7, max allowed: 6)"
    assert_error(too_many, 400, message, "VALIDATION_ERROR")
    bad_operator = sandbox.client.post(
        f"{CONTACTS}/search", json={"filters": [{"propertyName": "email", "operator": "LIKE", "value": "x"}]}
    )
    assert bad_operator.status_code == 400


# Meetings ----------------------------------------------------------------------------------------------


def test_create_meeting_with_an_inline_contact_association(sandbox: Sandbox) -> None:
    lead = contact(sandbox)
    body = meeting_body(int(lead["id"]), hs_timestamp="1791205200000")  # epoch millis, sent as a string
    response = sandbox.client.post(MEETINGS, json=body)
    assert response.status_code == 201
    created = response.json()
    assert list(created) == OBJECT_KEYS  # no associations in the create response
    meeting_id = created["id"]
    assert response.headers["location"] == f"{sandbox.url}{MEETINGS}/{meeting_id}"
    assert created["properties"] == {
        "hs_createdate": NOW_ISO,
        "hs_lastmodifieddate": NOW_ISO,
        "hs_meeting_body": "Booked by the agent",
        "hs_meeting_end_time": "2026-10-05T13:30:00Z",
        "hs_meeting_outcome": "SCHEDULED",
        "hs_meeting_start_time": "2026-10-05T13:00:00Z",
        "hs_meeting_title": "Intro call",
        "hs_object_id": meeting_id,
        "hs_timestamp": "2026-10-05T13:00:00Z",  # millis in, ISO out
    }
    stored = sandbox.snapshot()["hubspot"]["meetings"]
    assert stored == [
        {
            **created,
            "associations": {
                "contacts": {"results": [{"id": lead["id"], "type": "meeting_event_to_contact"}]}
            },
        }
    ]


def test_timestamps_keep_milliseconds_and_hs_timestamp_defaults_to_the_start(sandbox: Sandbox) -> None:
    props = {"hs_meeting_start_time": "2026-10-05T09:00:00.250-04:00", "hs_timestamp": None}
    created = meeting(sandbox, meeting_body(**props))
    assert created["properties"]["hs_meeting_start_time"] == "2026-10-05T13:00:00.250Z"
    assert created["properties"]["hs_timestamp"] == "2026-10-05T13:00:00.250Z"
    as_number = meeting(sandbox, meeting_body(hs_timestamp=1791205200000))
    assert as_number["properties"]["hs_timestamp"] == "2026-10-05T13:00:00Z"
    both_missing = sandbox.client.post(MEETINGS, json={"properties": {"hs_meeting_title": "x"}})
    assert both_missing.status_code == 400
    assert both_missing.json()["category"] == "VALIDATION_ERROR"
    garbage = sandbox.client.post(MEETINGS, json=meeting_body(hs_timestamp="tomorrow"))
    assert garbage.json()["errors"] == [
        {
            "message": "tomorrow was not a valid long.",
            "code": "INVALID_LONG",
            "context": {"propertyName": ["hs_timestamp"]},
        }
    ]


@pytest.mark.parametrize("outcome", ["SCHEDULED", "COMPLETED", "RESCHEDULED", "NO_SHOW", "CANCELED"])
def test_meeting_outcomes_accept_the_five_default_values(sandbox: Sandbox, outcome: str) -> None:
    assert (
        meeting(sandbox, meeting_body(hs_meeting_outcome=outcome))["properties"]["hs_meeting_outcome"]
        == outcome
    )


def test_an_unknown_meeting_outcome_is_invalid_option(sandbox: Sandbox) -> None:
    response = sandbox.client.post(MEETINGS, json=meeting_body(hs_meeting_outcome="CANCELLED"))
    message = f"CANCELLED was not one of the allowed options: {OUTCOME_OPTIONS}"
    body = assert_error(
        response,
        400,
        validation_message(
            {
                "message": message,
                "error": "INVALID_OPTION",
                "name": "hs_meeting_outcome",
                "value": "CANCELLED",
            }
        ),
        "VALIDATION_ERROR",
        keys=["status", "message", "correlationId", "errors", "category"],
    )
    assert body["errors"][0]["code"] == "INVALID_OPTION"
    assert sandbox.snapshot()["hubspot"]["meetings"] == []


def test_association_errors(sandbox: Sandbox) -> None:
    missing = sandbox.client.post(MEETINGS, json=meeting_body("424242"))
    body = assert_error(
        missing,
        400,
        "One or more associations are invalid",
        "VALIDATION_ERROR",
        keys=["status", "message", "correlationId", "context", "category"],
    )
    assert body["context"] == {
        "INVALID_OBJECT_IDS": ["CONTACT=424242 is not valid"],
        "objectId": ["424242"],
        "objectType": ["CONTACT"],
    }
    lead = contact(sandbox)
    wrong_direction = meeting_body(lead["id"])
    wrong_direction["associations"][0]["types"][0]["associationTypeId"] = 199
    response = sandbox.client.post(MEETINGS, json=wrong_direction)
    message = "invalid from object type 0-47 for associations to be created. expected: 0-1"
    assert_error(response, 400, message, "VALIDATION_ERROR")
    untyped = meeting_body(lead["id"])
    untyped["associations"][0]["types"] = []
    response = sandbox.client.post(MEETINGS, json=untyped)
    message = "Invalid input JSON: each association needs at least one type"
    assert_error(response, 400, message, "VALIDATION_ERROR")
    assert sandbox.snapshot()["hubspot"]["meetings"] == []


def test_get_meeting_with_properties_and_associations(sandbox: Sandbox) -> None:
    lead = contact(sandbox)
    created = meeting(sandbox, meeting_body(lead["id"]))
    url = f"{MEETINGS}/{created['id']}"
    plain = sandbox.client.get(url).json()
    assert list(plain) == OBJECT_KEYS
    assert plain["properties"] == {
        "hs_createdate": NOW_ISO,
        "hs_lastmodifieddate": NOW_ISO,
        "hs_object_id": created["id"],
    }
    full = sandbox.client.get(
        f"{url}?properties=hs_meeting_outcome,hs_meeting_start_time,hs_meeting_location,nonexistent"
        "&associations=contacts"
    ).json()
    assert list(full) == [*OBJECT_KEYS, "associations"]
    assert full["properties"] == {
        "hs_createdate": NOW_ISO,
        "hs_lastmodifieddate": NOW_ISO,
        "hs_meeting_location": None,
        "hs_meeting_outcome": "SCHEDULED",
        "hs_meeting_start_time": "2026-10-05T13:00:00Z",
        "hs_object_id": created["id"],
    }
    assert full["associations"] == {
        "contacts": {"results": [{"id": lead["id"], "type": "meeting_event_to_contact"}]}
    }
    unassociated = meeting(sandbox, meeting_body())
    no_links = sandbox.client.get(
        f"{MEETINGS}/{unassociated['id']}", params={"associations": "contacts"}
    ).json()
    assert "associations" not in no_links


def test_update_meeting(sandbox: Sandbox) -> None:
    created = meeting(sandbox, meeting_body())
    url = f"{MEETINGS}/{created['id']}"
    cancelled = sandbox.client.patch(url, json={"properties": {"hs_meeting_outcome": "CANCELED"}})
    assert cancelled.status_code == 200
    assert cancelled.json()["properties"]["hs_meeting_outcome"] == "CANCELED"
    moved = sandbox.client.patch(
        url,
        json={
            "properties": {
                "hs_timestamp": "2026-10-06T13:00:00Z",
                "hs_meeting_start_time": "2026-10-06T13:00:00Z",
                "hs_meeting_end_time": "2026-10-06T13:30:00Z",
                "hs_meeting_outcome": "RESCHEDULED",
            }
        },
    ).json()
    assert moved["properties"]["hs_meeting_start_time"] == "2026-10-06T13:00:00Z"
    invalid = sandbox.client.patch(url, json={"properties": {"hs_meeting_outcome": "LOST"}})
    assert invalid.json()["errors"][0]["code"] == "INVALID_OPTION"
    assert_error(
        sandbox.client.patch(f"{MEETINGS}/777", json={"properties": {}}), 404, "resource not found", None
    )


# Auth, routing and faults -----------------------------------------------------------------------------


HUBSPOT_ROUTES: list[tuple[str, str, dict[str, Any] | None, str]] = [
    ("POST", f"{CONTACTS}/search", {}, "crm.contacts.search"),
    ("POST", CONTACTS, {"properties": {}}, "crm.contacts.create"),
    ("PATCH", f"{CONTACTS}/1", {"properties": {}}, "crm.contacts.update"),
    ("POST", MEETINGS, {"properties": {}}, "crm.meetings.create"),
    ("PATCH", f"{MEETINGS}/1", {"properties": {}}, "crm.meetings.update"),
    ("GET", f"{MEETINGS}/1", None, "crm.meetings.get"),
]


@pytest.mark.parametrize(("method", "path", "body", "group"), HUBSPOT_ROUTES)
def test_auth_failures_are_the_verbatim_401(
    sandbox: Sandbox, method: str, path: str, body: Any, group: str
) -> None:
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:
        for headers in ({}, {"Authorization": "Bearer pat-" + "na1-not-a-real-token"}):
            response = anonymous.request(method, path, json=body, headers=headers)
            assert_error(response, 401, MSG_AUTH, "INVALID_AUTHENTICATION")
    assert [e["status"] for e in sandbox.log(group)] == [401, 401]


def test_unknown_crm_routes_are_logged_as_unrouted(sandbox: Sandbox) -> None:
    assert_error(sandbox.client.get(f"{CONTACTS}/1/associations/meetings"), 404, "resource not found", None)
    assert sandbox.log("unrouted")[0]["path"] == f"{CONTACTS}/1/associations/meetings"


def test_crm_fault_modes(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "crm.*", "mode": "error_500", "times": 2})
    assert_error(search(sandbox), 500, "internal error", None)
    assert_error(create_contact(sandbox), 500, "internal error", None)
    assert sandbox.snapshot()["hubspot"]["contacts"] == []
    created = contact(sandbox)
    sandbox.faults({"group": "crm.contacts.update", "mode": "not_found"})
    patch = sandbox.client.patch(f"{CONTACTS}/{created['id']}", json={"properties": {"firstname": "X"}})
    assert_error(patch, 404, "resource not found", None)
    assert (
        sandbox.snapshot()["hubspot"]["contacts"][0]["properties"]["firstname"] == "Lena"
    )  # nothing written
    sandbox.faults({"group": "crm.contacts.search", "mode": "malformed"})
    malformed = search(sandbox).json()
    assert malformed == {
        "count": 1,
        "objects": [
            {"objectId": int(created["id"]), "props": search(sandbox).json()["results"][0]["properties"]}
        ],
    }
    sandbox.faults({"group": "crm.meetings.create", "mode": "malformed"})
    response = sandbox.client.post(MEETINGS, json=meeting_body(created["id"]))
    assert response.status_code == 200
    assert list(response.json()) == ["object"]
    assert len(sandbox.snapshot()["hubspot"]["meetings"]) == 1  # the write committed


def test_commit_then_timeout_then_a_blind_retry_conflicts(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "crm.contacts.create", "mode": "commit_then_timeout", "hang_s": 0.4})
    with pytest.raises(httpx.ReadTimeout):
        sandbox.client.post(CONTACTS, json={"properties": {"email": LEAD}}, timeout=0.15)
    stored = sandbox.snapshot()["hubspot"]["contacts"]
    assert len(stored) == 1
    assert_error(
        create_contact(sandbox), 409, f"Contact already exists. Existing ID: {stored[0]['id']}", "CONFLICT"
    )
    sandbox.wait_for(lambda: sandbox.log("crm.contacts.create")[0]["completed"])
    assert sandbox.log("crm.contacts.create")[0]["status"] == 201


def test_timeout_on_meetings_does_not_commit(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "crm.meetings.create", "mode": "timeout", "hang_s": 0.2})
    assert_error(sandbox.client.post(MEETINGS, json=meeting_body()), 504, "gateway timeout", None)
    assert sandbox.snapshot()["hubspot"]["meetings"] == []

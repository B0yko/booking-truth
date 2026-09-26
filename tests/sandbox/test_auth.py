"""Every sandbox route except GET /_ui needs the bearer token; failures come back in the right shape."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

if TYPE_CHECKING:
    from conftest import Sandbox

MSG_INVALID_KEY = "ApiAuthStrategy - api key - Your api key is not valid"
MSG_NO_AUTH = (
    "ApiAuthStrategy - No authentication method provided. Either pass an API key as 'Bearer' header or OAuth "
    "client credentials as 'x-cal-secret-key' and 'x-cal-client-id' headers"
)
MSG_LIST_NO_AUTH = (
    "PermissionsGuard - no authentication provided. Provide either authorization bearer token containing "
    "managed user access token or oAuth client id in 'x-cal-client-id' header."
)
SLOTS = {"cal-api-version": "2024-09-04"}
BOOKINGS = {"cal-api-version": "2024-08-13"}

VENDOR_ROUTES: list[tuple[str, str, dict[str, str], dict[str, Any] | None, str]] = [
    ("GET", "/v2/slots?eventTypeId=1001&start=2026-10-05&end=2026-10-05", SLOTS, None, "slots"),
    ("POST", "/v2/bookings", BOOKINGS, {}, "bookings.create"),
    ("GET", "/v2/bookings?attendeeEmail=lead%40example.com", BOOKINGS, None, "bookings.list"),
    ("GET", "/v2/bookings/abc", BOOKINGS, None, "bookings.get"),
    (
        "POST",
        "/v2/bookings/abc/reschedule",
        BOOKINGS,
        {"start": "2026-10-05T13:00:00Z"},
        "bookings.reschedule",
    ),
    ("POST", "/v2/bookings/abc/cancel", BOOKINGS, {}, "bookings.cancel"),
]
CONTROL_ROUTES: list[tuple[str, str, dict[str, Any] | None]] = [
    ("POST", "/_control/reset", {}),
    ("POST", "/_control/seed", {}),
    ("POST", "/_control/faults", {"rules": []}),
    (
        "POST",
        "/_control/bookings",
        {
            "calendar": "calcom",
            "lead_email": "lead@example.com",
            "lead_name": "Lena M",
            "start": "2026-10-05T13:00Z",
        },
    ),
    ("GET", "/_state", None),
]


def call(sandbox: Sandbox, method: str, path: str, headers: dict[str, str], body: Any) -> httpx.Response:
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:
        return anonymous.request(method, path, headers=headers, json=body)


@pytest.mark.parametrize(("method", "path", "version", "body", "group"), VENDOR_ROUTES)
def test_vendor_routes_reject_a_wrong_token_with_the_cal_com_401(
    sandbox: Sandbox, method: str, path: str, version: dict[str, str], body: Any, group: str
) -> None:
    for header in ("Bearer wrong", "Bearer cal_invalid_key_for_testing", f"Basic {sandbox.token}", "Bearer"):
        response = call(sandbox, method, path, {**version, "Authorization": header}, body)
        sandbox.assert_error(response, 401, MSG_INVALID_KEY, path=path)
    entries = sandbox.log(group)
    assert [e["status"] for e in entries] == [401, 401, 401, 401]
    assert all(e["completed"] and e["fault"] is None for e in entries)


@pytest.mark.parametrize(("method", "path", "version", "body", "group"), VENDOR_ROUTES)
def test_vendor_routes_without_a_token(
    sandbox: Sandbox, method: str, path: str, version: dict[str, str], body: Any, group: str
) -> None:
    response = call(sandbox, method, path, version, body)
    if group == "bookings.list":
        sandbox.assert_error(response, 403, MSG_LIST_NO_AUTH, path=path)  # the real list endpoint says 403
    else:
        sandbox.assert_error(response, 401, MSG_NO_AUTH, path=path)


@pytest.mark.parametrize(("method", "path", "body"), CONTROL_ROUTES)
def test_control_routes_need_the_token(sandbox: Sandbox, method: str, path: str, body: Any) -> None:
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": sandbox.token}):
        response = call(sandbox, method, path, headers, body)
        assert response.status_code == 401
        assert response.json() == {"error": "unauthorized"}
    authorized = call(sandbox, method, path, {"Authorization": f"Bearer {sandbox.token}"}, body)
    assert authorized.status_code in (200, 201)
    assert sandbox.state.request_log == []  # control traffic is never logged


def test_ui_needs_no_token_and_docs_are_not_exposed(sandbox: Sandbox) -> None:
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:
        assert anonymous.get("/_ui").status_code == 200
        for path in ("/docs", "/redoc", "/openapi.json"):
            response = anonymous.get(path)
            assert response.status_code == 404
            assert response.json() == {"error": "not_found"}


def test_unauthorized_calls_do_not_count_against_fault_rules(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "slots", "mode": "error_500", "times": 1})
    path = "/v2/slots?eventTypeId=1001&start=2026-10-05&end=2026-10-05"
    assert call(sandbox, "GET", path, SLOTS, None).status_code == 401
    assert sandbox.client.get(path, headers=SLOTS).status_code == 500
    assert sandbox.client.get(path, headers=SLOTS).status_code == 200

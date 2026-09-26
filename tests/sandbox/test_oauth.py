"""The fake Google OAuth token endpoint: the service-account JWT bearer flow runs unchanged against it."""

from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime
from functools import cache
from typing import TYPE_CHECKING, Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

if TYPE_CHECKING:
    from conftest import Sandbox

GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
AUDIENCE = "https://oauth2.googleapis.com/token"
SERVICE_ACCOUNT = "calendar-agent@example.com"
SCOPES = "https://www.googleapis.com/auth/calendar.events https://www.googleapis.com/auth/calendar.freebusy"
MSG_TIMEFRAME = (
    "Invalid JWT: Token must be a short-lived token (60 minutes) and in a reasonable timeframe. Check your "
    "'iat' and 'exp' values and use a clock with skew to account for clock differences between systems."
)
IAT = int(datetime(2026, 10, 1, 12, 0, tzinfo=UTC).timestamp())  # the sandbox clock of every test


@cache
def private_key() -> rsa.RSAPrivateKey:
    """A throwaway key created at test time; no key material is committed."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def assertion(algorithm: str = "RS256", key: Any = None, **overrides: Any) -> str:
    claims: dict[str, Any] = {
        "iss": SERVICE_ACCOUNT,
        "scope": SCOPES,
        "aud": AUDIENCE,
        "iat": IAT,
        "exp": IAT + 3600,
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    signing_key = private_key() if key is None else key
    return jwt.encode(claims, signing_key, algorithm=algorithm, headers={"kid": "key-1"})


def exchange(sandbox: Sandbox, signed: str | None = None, **form: str) -> httpx.Response:
    data = {"grant_type": GRANT_TYPE, "assertion": assertion() if signed is None else signed, **form}
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:  # the token route takes no bearer
        return anonymous.post("/token", data=data)


def assert_oauth_error(response: httpx.Response, error: str, description: str, status: int = 400) -> None:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert list(response.json()) == ["error", "error_description"]
    assert response.json() == {"error": error, "error_description": description}


def test_a_valid_assertion_gets_the_sandbox_token(sandbox: Sandbox) -> None:
    response = exchange(sandbox)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    body = response.json()
    assert list(body) == ["access_token", "expires_in", "token_type"]
    assert body == {"access_token": sandbox.token, "expires_in": 3599, "token_type": "Bearer"}
    assert response.text == json.dumps(body, indent=2) + "\n"
    # The issued token opens the Calendar routes.
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    with httpx.Client(base_url=sandbox.url, timeout=5.0, headers=headers) as calendar:
        events = calendar.get("/calendar/v3/calendars/primary/events")
    assert events.status_code == 200
    grants = sandbox.snapshot()["google"]["token_grants"]
    assert grants == [
        {
            "iss": SERVICE_ACCOUNT,
            "sub": None,
            "scopes": SCOPES.split(),
            "iat": IAT,
            "exp": IAT + 3600,
            "kid": "key-1",
            "issued_at": "2026-10-01T12:00:00Z",
        }
    ]


def test_the_request_log_keeps_neither_the_signature_nor_the_token(sandbox: Sandbox) -> None:
    signed = assertion()
    exchange(sandbox, signed)
    entry = sandbox.log("oauth.token")[0]
    head, payload, signature = signed.split(".")
    assert entry["body"] == {"grant_type": GRANT_TYPE, "assertion": f"{head}.{payload}.[redacted]"}
    assert entry["response"] == {"access_token": "[redacted]", "expires_in": 3599, "token_type": "Bearer"}
    assert signature not in json.dumps(sandbox.snapshot())


def test_a_body_that_does_not_parse_is_logged_without_the_signature(sandbox: Sandbox) -> None:
    signed = assertion()
    head, payload, signature = signed.split(".")
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:
        response = anonymous.post(
            "/token",
            content=f'{{"grant_type": "{GRANT_TYPE}", "assertion": "{signed}"'.encode(),  # no closing brace
            headers={"content-type": "application/json"},
        )
    assert_oauth_error(response, "unsupported_grant_type", "Invalid grant_type: ")
    logged = sandbox.log("oauth.token")[0]["body"]
    assert f"{head}.{payload}.[redacted]" in logged
    assert signature not in json.dumps(sandbox.snapshot())


def test_events_created_after_the_exchange_name_the_service_account(sandbox: Sandbox) -> None:
    exchange(sandbox)
    body = {
        "start": {"dateTime": "2026-10-05T13:00:00Z"},
        "end": {"dateTime": "2026-10-05T13:30:00Z"},
    }
    event = sandbox.client.post("/calendar/v3/calendars/primary/events", json=body).json()
    assert event["creator"] == {"email": SERVICE_ACCOUNT}


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"iat": IAT - 3700, "exp": IAT - 100}, "expired by the sandbox clock"),
        ({"exp": IAT}, "expires exactly now"),
        ({"exp": IAT + 3601}, "lives longer than an hour"),
        ({"iat": IAT + 3600, "exp": IAT + 3700}, "issued in the future"),
        ({"exp": IAT - 1}, "expires before it was issued"),
        ({"iat": None}, "no iat"),
        ({"exp": "soon"}, "exp is not a number"),
    ],
)
def test_assertions_outside_a_short_reasonable_timeframe(
    sandbox: Sandbox, overrides: dict[str, Any], why: str
) -> None:
    assert_oauth_error(exchange(sandbox, assertion(**overrides)), "invalid_grant", MSG_TIMEFRAME)


def test_claims_are_strict_json(sandbox: Sandbox) -> None:
    header, _, signature = assertion().split(".")
    claims = (
        f'{{"iss":"{SERVICE_ACCOUNT}","scope":"{SCOPES}","aud":"{AUDIENCE}","iat":NaN,"exp":{IAT + 3600}}}'
    )
    payload = base64.urlsafe_b64encode(claims.encode()).rstrip(b"=").decode()
    response = exchange(sandbox, f"{header}.{payload}.{signature}")
    assert_oauth_error(response, "invalid_grant", "Invalid JWT Signature.")


def test_a_small_clock_skew_is_accepted(sandbox: Sandbox) -> None:
    assert exchange(sandbox, assertion(iat=IAT + 60, exp=IAT + 3660)).status_code == 200


def test_the_audience_must_be_google_s_token_url(sandbox: Sandbox) -> None:
    wrong = assertion(aud=f"{sandbox.url}/token")  # google-auth never uses the token_uri as the audience
    assert_oauth_error(exchange(sandbox, wrong), "invalid_grant", "Invalid JWT: Failed audience check.")
    assert_oauth_error(
        exchange(sandbox, assertion(aud=None)), "invalid_grant", "Invalid JWT: Failed audience check."
    )


def test_only_rs256_is_accepted(sandbox: Sandbox) -> None:
    hs256 = assertion("HS256", key="a-shared-secret-of-at-least-32-bytes")
    assert_oauth_error(exchange(sandbox, hs256), "invalid_grant", "Invalid JWT Signature.")
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = assertion().split(".")[1]
    assert_oauth_error(exchange(sandbox, f"{header}.{payload}."), "invalid_grant", "Invalid JWT Signature.")
    assert_oauth_error(exchange(sandbox, "not-a-jwt"), "invalid_grant", "Invalid JWT Signature.")


def test_issuer_scope_and_subject(sandbox: Sandbox) -> None:
    assert_oauth_error(exchange(sandbox, assertion(iss=None)), "invalid_grant", "Invalid email or User ID.")
    assert_oauth_error(
        exchange(sandbox, assertion(iss="booking-agent")), "invalid_grant", "Not a valid email"
    )
    assert_oauth_error(
        exchange(sandbox, assertion(scope=" ")),
        "invalid_scope",
        "Invalid OAuth scope or ID token audience provided.",
    )
    delegated = assertion(sub="host@example.com")
    # The guide's text for impersonation by a service account that has no domain-wide delegation.
    assert_oauth_error(
        exchange(sandbox, delegated), "unauthorized_client", "Unauthorized client or scope in request."
    )
    sandbox.seed(google_sa_can_invite=True)  # domain-wide delegation
    assert exchange(sandbox, delegated).status_code == 200
    assert sandbox.snapshot()["google"]["token_grants"][-1]["sub"] == "host@example.com"
    assert_oauth_error(exchange(sandbox, assertion(sub="host")), "invalid_grant", "Not a valid email")


@pytest.mark.parametrize(
    "scope",
    [
        "calendar.events",  # a scope must be the full URL
        "https://www.googleapis.com/auth/calendar.events,https://www.googleapis.com/auth/calendar.freebusy",
    ],
)
def test_scopes_must_be_urls_separated_by_spaces(sandbox: Sandbox, scope: str) -> None:
    assert_oauth_error(
        exchange(sandbox, assertion(scope=scope)),
        "invalid_scope",
        "Invalid OAuth scope or ID token audience provided.",
    )
    assert exchange(sandbox, assertion(scope=f"openid {SCOPES}")).status_code == 200


def test_grant_type_and_assertion_parameters(sandbox: Sandbox) -> None:
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:
        empty = anonymous.post("/token", data={})
        wrong = anonymous.post("/token", data={"grant_type": "client_credentials"})
        missing = anonymous.post("/token", data={"grant_type": GRANT_TYPE})
        as_json = anonymous.post("/token", json={"grant_type": GRANT_TYPE, "assertion": assertion()})
    assert_oauth_error(empty, "unsupported_grant_type", "Invalid grant_type: ")
    assert_oauth_error(wrong, "unsupported_grant_type", "Invalid grant_type: client_credentials")
    assert_oauth_error(missing, "invalid_request", "Missing required parameter: assertion")
    assert as_json.status_code == 200


def test_invalid_assertions_are_logged_but_never_counted_by_fault_rules(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "oauth.token", "mode": "error_500"})
    assert exchange(sandbox, assertion(aud="nope")).status_code == 400
    assert_oauth_error(exchange(sandbox), "internal_failure", "Backend Error", 500)
    assert exchange(sandbox).status_code == 200
    assert [(e["status"], e["fault"]) for e in sandbox.log("oauth.token")] == [
        (400, None),
        (500, "error_500"),
        (200, None),
    ]


def test_fault_modes_on_the_token_endpoint(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "oauth.token", "mode": "malformed"})
    malformed = exchange(sandbox)
    assert malformed.json() == {"accessToken": sandbox.token, "expiresIn": 3599, "tokenType": "Bearer"}
    assert sandbox.log("oauth.token")[0]["response"]["accessToken"] == "[redacted]"
    sandbox.faults({"group": "oauth.token", "mode": "not_found"})
    assert_oauth_error(exchange(sandbox), "invalid_grant", "Invalid email or User ID.")
    sandbox.faults({"group": "oauth.token", "mode": "commit_then_timeout", "hang_s": 0.4})
    started = len(sandbox.snapshot()["google"]["token_grants"])
    with httpx.Client(base_url=sandbox.url, timeout=0.15) as impatient, pytest.raises(httpx.ReadTimeout):
        impatient.post("/token", data={"grant_type": GRANT_TYPE, "assertion": assertion()})
    assert len(sandbox.snapshot()["google"]["token_grants"]) == started + 1  # recorded while hanging
    sandbox.faults({"group": "oauth.token", "mode": "timeout", "hang_s": 0.2})
    begun = time.monotonic()
    assert_oauth_error(exchange(sandbox), "temporarily_unavailable", "Service unavailable", 503)
    assert time.monotonic() - begun >= 0.2


def test_other_methods_on_the_token_path_are_unrouted(sandbox: Sandbox) -> None:
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:
        response = anonymous.get("/token")
    assert_oauth_error(response, "invalid_request", "GET /token is not a token request", 404)
    assert sandbox.log("unrouted")[0]["path"] == "/token"


# Scopes on the Calendar routes ------------------------------------------------------------------------


def calendar_scopes(*names: str) -> str:
    return " ".join(f"https://www.googleapis.com/auth/{name}" for name in names)


MONDAY = {"timeMin": "2026-10-05T00:00:00Z", "timeMax": "2026-10-06T00:00:00Z"}
FREEBUSY = {**MONDAY, "items": [{"id": "primary"}]}
EVENT = {"start": {"dateTime": "2026-10-05T13:00:00Z"}, "end": {"dateTime": "2026-10-05T13:30:00Z"}}
EVENTS = "/calendar/v3/calendars/primary/events"


def test_calendar_events_alone_cannot_query_freebusy(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "freebusy", "mode": "error_500"})
    assert exchange(sandbox, assertion(scope=calendar_scopes("calendar.events"))).status_code == 200
    response = sandbox.client.post("/calendar/v3/freeBusy", json=FREEBUSY)
    assert response.status_code == 403
    assert response.headers["content-type"] == "application/json; charset=UTF-8"
    error = response.json()["error"]
    assert list(error) == ["code", "message", "errors", "status", "details"]
    assert error == {
        "code": 403,
        "message": "Request had insufficient authentication scopes.",
        "errors": [
            {"message": "Insufficient Permission", "domain": "global", "reason": "insufficientPermissions"}
        ],
        "status": "PERMISSION_DENIED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
                "domain": "googleapis.com",
                "metadata": {
                    "service": "calendar-json.googleapis.com",
                    "method": "calendar.v3.Freebusy.Query",
                },
            }
        ],
    }
    assert response.text == json.dumps(response.json(), indent=2) + "\n"  # a front-end error
    assert [(e["status"], e["fault"]) for e in sandbox.log("freebusy")] == [(403, None)]
    assert sandbox.snapshot()["faults"][0]["matched"] == 0  # rejected before any fault rule
    # The same token may write events.
    assert sandbox.client.post(EVENTS, json=EVENT).status_code == 200


def test_each_method_takes_its_documented_scopes_from_the_latest_grant(sandbox: Sandbox) -> None:
    exchange(sandbox, assertion(scope=calendar_scopes("calendar.readonly")))
    assert sandbox.client.post("/calendar/v3/freeBusy", json=FREEBUSY).status_code == 200
    assert sandbox.client.get(EVENTS).status_code == 200
    denied = sandbox.client.post(EVENTS, json=EVENT)
    assert denied.status_code == 403
    assert denied.json()["error"]["details"][0]["metadata"]["method"] == "calendar.v3.Events.Insert"
    exchange(sandbox, assertion(scope=calendar_scopes("calendar.events", "calendar.freebusy")))
    created = sandbox.client.post(EVENTS, json=EVENT)
    assert created.status_code == 200
    assert sandbox.client.post("/calendar/v3/freeBusy", json=FREEBUSY).status_code == 200
    exchange(sandbox, assertion(scope=calendar_scopes("calendar")))
    assert sandbox.client.delete(f"{EVENTS}/{created.json()['id']}").status_code == 204
    exchange(sandbox, assertion(scope="https://www.googleapis.com/auth/drive.readonly"))
    assert sandbox.client.get(EVENTS).json()["error"]["details"][0]["metadata"]["method"] == (
        "calendar.v3.Events.List"
    )


def test_without_a_grant_the_sandbox_token_has_every_scope(sandbox: Sandbox) -> None:
    exchange(sandbox, assertion(scope=calendar_scopes("calendar.events")))
    assert sandbox.client.post("/calendar/v3/freeBusy", json=FREEBUSY).status_code == 403
    assert sandbox.client.post("/_control/reset").status_code == 200
    assert sandbox.client.post("/calendar/v3/freeBusy", json=FREEBUSY).status_code == 200

"""Google's OAuth 2.0 token endpoint for the service-account (JWT bearer) flow: ``POST /token``.

A Google adapter configured with ``BT_GOOGLE_TOKEN_URI=http://<sandbox>/token`` runs its service-account flow
unchanged: it signs an RS256 assertion and exchanges it here for an access token, which is the sandbox bearer
token. This is the one vendor route that does not take the sandbox token. It checks the assertion's structure
and claims the way Google documents them (``alg`` RS256; ``iss``, ``scope``, ``aud``, ``iat`` and ``exp``,
at most an hour apart and not expired by the sandbox clock). The signature is not verified, because the
sandbox has no public key for the caller's service account. Accepted grants (issuer, subject, scopes, times)
are kept in ``/_state``, and the Calendar routes enforce the scopes of the latest one. The request log keeps
the assertion without its signature and never the issued token.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final
from urllib.parse import parse_qsl

from fastapi import APIRouter, Request
from fastapi.responses import Response

from booking_truth.sandbox.common import Call, Outcome, VendorApi, parse_body_json, parse_json, run_call
from booking_truth.timeutil import iso_z

TOKEN_PATH: Final = "/token"  # noqa: S105 - a route, not a secret
GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
AUDIENCE: Final = "https://oauth2.googleapis.com/token"
MAX_LIFETIME_S: Final = 3600
EXPIRES_IN: Final = 3599
#: How far ``iat`` may run ahead of the sandbox clock (a sandbox choice; Google asks for "a reasonable
#: timeframe").
CLOCK_SKEW_S: Final = 300
REDACTED: Final = "[redacted]"
JSON_UTF8: Final = "application/json; charset=utf-8"

# Verbatim Google texts (service-account guide, error codes table).
MSG_TIMEFRAME: Final = (
    "Invalid JWT: Token must be a short-lived token (60 minutes) and in a reasonable timeframe. Check your "
    "'iat' and 'exp' values and use a clock with skew to account for clock differences between systems."
)
MSG_SIGNATURE: Final = "Invalid JWT Signature."
MSG_EMAIL: Final = "Invalid email or User ID."
MSG_NOT_EMAIL: Final = "Not a valid email"
MSG_SCOPE: Final = "Invalid OAuth scope or ID token audience provided."
#: The guide's text for a ``sub`` claim from a service account that has no domain-wide delegation.
MSG_NO_DELEGATION: Final = "Unauthorized client or scope in request."
# Seen in raw captures of Google's answer (HTTP 400) to a missing or wrong ``aud``.
MSG_AUDIENCE: Final = "Invalid JWT: Failed audience check."
# Sandbox text (Google's own wording for this case was not verified).
MSG_ASSERTION_MISSING: Final = "Missing required parameter: assertion"
#: Scopes that are not URLs. Anything else must be an ``https://`` URL; Google rejects unknown scopes, and its
#: guide warns that a comma-separated list is one invalid scope.
BARE_SCOPES: Final = frozenset({"openid", "email", "profile"})
# A compact JWS inside text the parser could not read, so its signature can be kept out of the request log.
_JWS_IN_TEXT: Final = re.compile(r"(eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.)[A-Za-z0-9_-]+")


def oauth_error(code: str, description: str, status: int = 400) -> Outcome:
    """Google's OAuth error body: ``error`` and ``error_description``."""
    return Outcome(status, {"error": code, "error_description": description})


def _b64_json(segment: str) -> dict[str, Any] | None:
    """A base64url JSON object, parsed strictly (``NaN`` and ``Infinity`` are not JSON)."""
    try:
        value = parse_json(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
    except (ValueError, binascii.Error):
        return None
    return value if isinstance(value, dict) else None


def _b64_ok(segment: str) -> bool:
    try:
        base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
    except (ValueError, binascii.Error):
        return False
    return bool(segment)


def decode_assertion(assertion: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Header and claims of a compact JWS with a non-empty signature; the signature is not checked."""
    parts = assertion.split(".")
    if len(parts) != 3 or not _b64_ok(parts[2]):
        return None
    header, claims = _b64_json(parts[0]), _b64_json(parts[1])
    if header is None or claims is None:
        return None
    return header, claims


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _scope_ok(scope: str) -> bool:
    return scope in BARE_SCOPES or (scope.startswith("https://") and "," not in scope)


def _redact_assertion(assertion: str) -> str:
    head, _, rest = assertion.partition(".")
    payload, dot, _signature = rest.partition(".")
    return f"{head}.{payload}.{REDACTED}" if dot else REDACTED


def _validate(call: Call) -> Outcome | None:
    """Check the grant type and the assertion (the order of the checks is the sandbox's; Google documents
    none). Runs before the fault engine, like the bearer check of the other vendor routes; the result is kept
    in ``call.extra``."""
    params: Mapping[str, Any] = call.body if isinstance(call.body, dict) else {}
    grant_type = params.get("grant_type")
    if grant_type != GRANT_TYPE:
        return oauth_error("unsupported_grant_type", f"Invalid grant_type: {grant_type or ''}")
    assertion = params.get("assertion")
    if not isinstance(assertion, str) or not assertion:
        return oauth_error("invalid_request", MSG_ASSERTION_MISSING)
    decoded = decode_assertion(assertion)
    if decoded is None or decoded[0].get("alg") != "RS256":
        return oauth_error("invalid_grant", MSG_SIGNATURE)
    header, claims = decoded
    issuer = claims.get("iss")
    if not isinstance(issuer, str) or not issuer:
        return oauth_error("invalid_grant", MSG_EMAIL)
    if "@" not in issuer:
        return oauth_error("invalid_grant", MSG_NOT_EMAIL)
    if claims.get("aud") != AUDIENCE:
        return oauth_error("invalid_grant", MSG_AUDIENCE)
    issued, expires = _number(claims.get("iat")), _number(claims.get("exp"))
    now = call.state.now().timestamp()
    if (
        issued is None
        or expires is None
        or expires < issued
        or expires - issued > MAX_LIFETIME_S
        or expires <= now
        or issued > now + CLOCK_SKEW_S
    ):
        return oauth_error("invalid_grant", MSG_TIMEFRAME)
    scope = claims.get("scope")
    scopes = scope.split() if isinstance(scope, str) else []
    if not scopes or not all(_scope_ok(item) for item in scopes):
        return oauth_error("invalid_scope", MSG_SCOPE)
    subject = claims.get("sub")
    if subject is not None:
        # Impersonation needs domain-wide delegation; the guide's texts for an unknown user cover a bad value.
        if not call.state.seed.google_sa_can_invite:
            return oauth_error("unauthorized_client", MSG_NO_DELEGATION)
        if not isinstance(subject, str) or "@" not in subject:
            return oauth_error("invalid_grant", MSG_NOT_EMAIL)
    call.extra["grant"] = {
        "iss": issuer,
        "sub": subject if isinstance(subject, str) else None,
        "scopes": scopes,
        "iat": int(issued),
        "exp": int(expires),
        "kid": header.get("kid") if isinstance(header.get("kid"), str) else None,
    }
    return None


def _token(call: Call) -> Outcome:
    grant = dict(call.extra["grant"])
    grant["issued_at"] = iso_z(call.state.now())
    call.state.google_token_grants.append(grant)
    token: str = call.request.app.state.sandbox_token
    return Outcome(200, {"access_token": token, "expires_in": EXPIRES_IN, "token_type": "Bearer"})


class GoogleOAuthApi(VendorApi):
    name = "Google OAuth 2.0 token endpoint"
    prefix = TOKEN_PATH

    def __init__(self, router: APIRouter) -> None:
        self.router = router

    def parse_body(self, raw: bytes, content_type: str) -> tuple[Any, bool]:
        """Form-encoded parameters (what client libraries send); a JSON object is read as well."""
        if "json" in content_type.lower():
            return parse_body_json(raw)
        if not raw:
            return None, False
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace"), True
        params: dict[str, str] = {}
        for key, value in parse_qsl(text, keep_blank_values=True):
            params.setdefault(key, value)
        return params, False

    def log_view(self, value: Any) -> Any:
        if isinstance(value, str):  # a body that did not parse may still carry a signed assertion
            return _JWS_IN_TEXT.sub(rf"\g<1>{REDACTED}", value)
        if not isinstance(value, dict):
            return value
        out = dict(value)
        if isinstance(out.get("assertion"), str):
            out["assertion"] = _redact_assertion(out["assertion"])
        for key in ("access_token", "accessToken"):
            if key in out:
                out[key] = REDACTED
        return out

    def authorize(self, call: Call) -> Outcome | None:
        return None  # the assertion is the credential; see _validate

    def render(self, outcome: Outcome) -> Response:
        content = json.dumps(outcome.body, indent=2, ensure_ascii=False) + "\n"
        return Response(
            content, status_code=outcome.status, media_type=JSON_UTF8, headers=dict(outcome.headers)
        )

    def route_not_found(self, request: Request, now: datetime) -> Outcome:
        return oauth_error(
            "invalid_request", f"{request.method} {request.url.path} is not a token request", 404
        )

    def server_error(self, call: Call) -> Outcome:
        return oauth_error("internal_failure", "Backend Error", 500)

    def gateway_timeout(self, call: Call) -> Outcome:
        return oauth_error("temporarily_unavailable", "Service unavailable", 503)

    def not_found(self, call: Call) -> Outcome:
        return oauth_error("invalid_grant", MSG_EMAIL)

    def malformed(self, call: Call, normal: Outcome) -> Outcome:
        token = normal.body.get("access_token") if isinstance(normal.body, dict) else None
        return Outcome(200, {"accessToken": token, "expiresIn": EXPIRES_IN, "tokenType": "Bearer"})


router = APIRouter()


@router.post(TOKEN_PATH)
async def token(request: Request) -> Response:
    return await run_call(request, API, "oauth.token", _token, precheck=_validate)


API: Final = GoogleOAuthApi(router)

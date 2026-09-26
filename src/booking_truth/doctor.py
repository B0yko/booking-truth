"""``booking-truth doctor``: checks the running configuration without spending anything.

Every check is read-only: it never books, moves or cancels anything, and it never sends a priced LLM
request. It never prints a secret (a key, a token, or a signed assertion); only host names, HTTP statuses
and short vendor-agnostic reasons appear. Prints a table of OK/FAIL with a reason for each row, and exits 1
if any row fails, so it can gate a deploy script.

Checks:

- **config**: ``Settings`` parses, and ``Settings.validate_for_agent()`` finds no problem.
- **llm**: ``GET {BT_LLM_BASE_URL}/models`` with the configured key answers; for OpenRouter, also
  ``GET /api/v1/key`` to confirm the key is valid and show its remaining limit.
- **calendar**: Cal.com lists one booking (``GET /v2/bookings?take=1``); Google exchanges the service
  account's signed assertion for a token at ``BT_GOOGLE_TOKEN_URI``, then queries ``freeBusy`` for
  tomorrow.
- **crm**: HubSpot searches contacts with ``limit: 1``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import jwt
import typer

from booking_truth.config import ConfigError, Settings, is_local_url, load_settings

HTTP_TIMEOUT_S = 10.0
GOOGLE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
GOOGLE_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
GOOGLE_ASSERTION_LIFETIME_S = 3600
CALCOM_BOOKINGS_VERSION = "2024-08-13"
EXIT_FAIL = 1


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _host(url: str) -> str:
    return urlparse(url).hostname or url


def _is_openrouter(base_url: str) -> bool:
    host = _host(base_url)
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


async def _get(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response | str:
    """The response, or a short reason string when the request could not be made at all."""
    try:
        return await client.get(url, **kwargs)
    except httpx.HTTPError as exc:
        return f"cannot reach {_host(url)}: {type(exc).__name__}"


async def _post(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response | str:
    try:
        return await client.post(url, **kwargs)
    except httpx.HTTPError as exc:
        return f"cannot reach {_host(url)}: {type(exc).__name__}"


# Checks -------------------------------------------------------------------------------------------------


async def check_config(settings: Settings) -> Check:
    try:
        settings.validate_for_agent()
    except ConfigError as exc:
        return Check("config", False, str(exc))
    return Check(
        "config", True, f"guards={settings.guards}, calendar={settings.calendar}, crm={settings.crm}"
    )


async def check_llm(settings: Settings) -> Check:
    if settings.offline or settings.llm_api_key is None:
        return Check("llm", False, "no LLM key configured (BT_LLM_API_KEY, or OPENROUTER_API_KEY)")
    key = settings.llm_api_key.get_secret_value().strip()
    base = settings.llm_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_S)) as client:
        models = await _get(client, f"{base}/models", headers=headers)
        if isinstance(models, str):
            return Check("llm", False, models)
        if models.status_code != 200:
            return Check("llm", False, f"GET {_host(base)}/models returned HTTP {models.status_code}")
        detail = f"{_host(base)}/models reachable"
        if not _is_openrouter(base):
            return Check("llm", True, detail)
        key_check = await _get(client, f"{base}/key", headers=headers)
        if isinstance(key_check, str):
            return Check("llm", False, key_check)
        if key_check.status_code != 200:
            return Check("llm", False, f"GET {_host(base)}/key returned HTTP {key_check.status_code}")
        try:
            data = key_check.json().get("data") or {}
        except ValueError:
            return Check("llm", False, f"GET {_host(base)}/key returned a non-JSON body")
        limit, usage = data.get("limit"), data.get("usage") or 0
        remaining = f"${limit - usage:.2f} remaining" if isinstance(limit, int | float) else "no spend limit"
        return Check("llm", True, f"{detail}; key valid, {remaining}")


async def check_calendar(settings: Settings) -> Check:
    if settings.calendar == "calcom":
        return await _check_calcom(settings)
    if settings.calendar == "google":
        return await _check_google(settings)
    return Check("calendar", False, f"unknown BT_CALENDAR={settings.calendar!r}")


async def _check_calcom(settings: Settings) -> Check:
    base = settings.calcom_base_url.rstrip("/")
    if settings.calcom_api_key is not None and settings.calcom_api_key.get_secret_value().strip():
        key = settings.calcom_api_key.get_secret_value().strip()
    elif is_local_url(base):
        key = settings.sandbox_token.get_secret_value()
    else:
        return Check("calendar", False, "BT_CALCOM_API_KEY is required for a real Cal.com account")
    headers = {"Authorization": f"Bearer {key}", "cal-api-version": CALCOM_BOOKINGS_VERSION}
    params: dict[str, str] = {"take": "1"}
    if settings.calcom_event_type_id is not None:
        params["eventTypeId"] = str(settings.calcom_event_type_id)
    async with httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_S)) as client:
        response = await _get(client, f"{base}/v2/bookings", headers=headers, params=params)
    if isinstance(response, str):
        return Check("calendar", False, response)
    if response.status_code != 200:
        return Check("calendar", False, f"GET {_host(base)}/v2/bookings returned HTTP {response.status_code}")
    return Check("calendar", True, f"Cal.com reachable at {_host(base)}; credentials accepted")


def _load_service_account(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    for field in ("client_email", "private_key"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise ValueError(f"missing '{field}'")
    return data


def _google_assertion(account: dict[str, Any], *, audience: str, now: datetime) -> str:
    claims = {
        "iss": account["client_email"],
        "scope": GOOGLE_READONLY_SCOPE,
        "aud": audience,
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + GOOGLE_ASSERTION_LIFETIME_S,
    }
    return jwt.encode(claims, account["private_key"], algorithm="RS256")


async def _check_google(settings: Settings) -> Check:
    if settings.google_service_account_file is None:
        return Check("calendar", False, "BT_GOOGLE_SERVICE_ACCOUNT_FILE is required when BT_CALENDAR=google")
    try:
        account = _load_service_account(settings.google_service_account_file)
    except (OSError, ValueError) as exc:
        return Check("calendar", False, f"cannot read the Google service-account file: {exc}")
    now = datetime.now(UTC)
    try:
        assertion = _google_assertion(account, audience=settings.google_token_uri, now=now)
    except (jwt.PyJWTError, ValueError) as exc:
        return Check("calendar", False, f"cannot sign the service-account assertion: {exc}")
    async with httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_S)) as client:
        token_response = await _post(
            client,
            settings.google_token_uri,
            data={"grant_type": GOOGLE_GRANT_TYPE, "assertion": assertion},
        )
        if isinstance(token_response, str):
            return Check("calendar", False, token_response)
        if token_response.status_code != 200:
            return Check("calendar", False, f"token exchange returned HTTP {token_response.status_code}")
        try:
            access_token = token_response.json().get("access_token")
        except ValueError:
            return Check("calendar", False, "token exchange returned a non-JSON body")
        if not isinstance(access_token, str) or not access_token:
            return Check("calendar", False, "token exchange returned no access_token")
        tomorrow = now + timedelta(days=1)
        body = {
            "timeMin": tomorrow.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeMax": (tomorrow + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "items": [{"id": settings.google_calendar_id}],
        }
        base = settings.google_base_url.rstrip("/")
        fb = await _post(
            client,
            f"{base}/calendar/v3/freeBusy",
            headers={"Authorization": f"Bearer {access_token}"},
            json=body,
        )
        if isinstance(fb, str):
            return Check("calendar", False, fb)
        if fb.status_code != 200:
            return Check("calendar", False, f"freeBusy for tomorrow returned HTTP {fb.status_code}")
    return Check("calendar", True, "Google Calendar: token exchange and freeBusy for tomorrow both OK")


async def check_crm(settings: Settings) -> Check:
    if settings.crm == "none":
        return Check("crm", True, "BT_CRM=none; no CRM configured")
    if settings.crm != "hubspot":
        return Check("crm", False, f"unknown BT_CRM={settings.crm!r}")
    if settings.hubspot_token is None:
        return Check("crm", False, "BT_HUBSPOT_TOKEN is required when BT_CRM=hubspot")
    token = settings.hubspot_token.get_secret_value().strip()
    base = settings.hubspot_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_S)) as client:
        response = await _post(
            client, f"{base}/crm/v3/objects/contacts/search", headers=headers, json={"limit": 1}
        )
    if isinstance(response, str):
        return Check("crm", False, response)
    if response.status_code != 200:
        return Check(
            "crm",
            False,
            f"POST {_host(base)}/crm/v3/objects/contacts/search returned HTTP {response.status_code}",
        )
    return Check("crm", True, f"HubSpot reachable at {_host(base)}; credentials accepted")


async def _safely(name: str, coro: Any) -> Check:
    try:
        result: Check = await coro
    except Exception as exc:  # a check must never crash the whole command
        return Check(name, False, f"unexpected error: {type(exc).__name__}: {exc}")
    return result


async def run_doctor(settings: Settings) -> list[Check]:
    """Every check, run independently: one failing check never stops the others."""
    return [
        await _safely("config", check_config(settings)),
        await _safely("llm", check_llm(settings)),
        await _safely("calendar", check_calendar(settings)),
        await _safely("crm", check_crm(settings)),
    ]


# CLI ------------------------------------------------------------------------------------------------------


def cmd_doctor() -> None:
    """Check configuration, LLM reachability, and calendar and CRM credentials (read-only, no spending)."""
    try:
        settings = load_settings()
    except ValueError as exc:
        typer.echo(f"error: invalid BT_* settings: {exc}", err=True)
        raise typer.Exit(code=EXIT_FAIL) from None
    checks = asyncio.run(run_doctor(settings))
    width = max(len(check.name) for check in checks)
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        typer.echo(f"{check.name.ljust(width)}  {status:<4}  {check.detail}")
    if any(not check.ok for check in checks):
        raise typer.Exit(code=EXIT_FAIL)


def register(app_root: typer.Typer) -> None:
    """Add ``doctor`` to the root command."""
    app_root.command("doctor")(cmd_doctor)

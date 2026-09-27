"""``booking-truth doctor``: every check is read-only, respx-mocked, and no check crashes the others."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from typer.testing import CliRunner

from booking_truth.cli import app
from booking_truth.config import Settings
from booking_truth.doctor import Check, run_doctor

runner = CliRunner()


def make(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mock:
        yield mock


@cache
def private_key() -> rsa.RSAPrivateKey:
    """A throwaway key created at test time; no key material is committed."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def service_account_file(tmp_path: Path) -> Path:
    pem = private_key().private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "service-account.json"
    path.write_text(
        json.dumps({"client_email": "calendar-agent@example.com", "private_key": pem.decode()}),
        encoding="utf-8",
    )
    return path


# config -----------------------------------------------------------------------------------------------


async def test_config_check_fails_when_validate_for_agent_rejects_it() -> None:
    settings = make(
        guards="pinned_version", llm_model="vendor/model:latest", calcom_base_url="http://sandbox:8100"
    )
    checks = await run_doctor(settings)
    config = next(c for c in checks if c.name == "config")
    assert config.ok is False
    assert "floating alias" in config.detail


async def test_config_check_passes_for_a_valid_local_setup() -> None:
    settings = make(calcom_base_url="http://sandbox:8100")
    checks = await run_doctor(settings)
    config = next(c for c in checks if c.name == "config")
    assert config.ok is True


# llm ----------------------------------------------------------------------------------------------------


async def test_llm_check_fails_offline() -> None:
    settings = make(calcom_base_url="http://sandbox:8100")
    checks = await run_doctor(settings)
    llm = next(c for c in checks if c.name == "llm")
    assert llm.ok is False
    assert "no LLM key" in llm.detail


async def test_llm_check_reads_the_openrouter_key_endpoint(router: respx.MockRouter) -> None:
    router.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    router.get("https://openrouter.ai/api/v1/key").mock(
        return_value=httpx.Response(200, json={"data": {"limit": 50.0, "usage": 6.6}})
    )
    settings = make(llm_api_key="sk-or-v1-test", calcom_base_url="http://sandbox:8100")
    checks = await run_doctor(settings)
    llm = next(c for c in checks if c.name == "llm")
    assert llm.ok is True
    assert "sk-or-v1-test" not in llm.detail


async def test_llm_check_reports_the_key_endpoint_s_own_limit_remaining(router: respx.MockRouter) -> None:
    # A key can carry lifetime usage past its rolling/promotional limit and still have real headroom:
    # OpenRouter's own `limit_remaining` is authoritative, never `limit - usage` (which would go negative
    # here even though the key has $46.35 left).
    router.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    router.get("https://openrouter.ai/api/v1/key").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"limit": 90.0, "usage": 529.87, "limit_remaining": 46.35, "usage_monthly": 25.5}},
        )
    )
    settings = make(llm_api_key="sk-or-v1-test", calcom_base_url="http://sandbox:8100")
    checks = await run_doctor(settings)
    llm = next(c for c in checks if c.name == "llm")
    assert llm.ok is True
    assert "46.35" in llm.detail
    assert "-439.87" not in llm.detail
    assert "-483.52" not in llm.detail


async def test_llm_check_reports_no_spend_limit_when_the_key_has_none(router: respx.MockRouter) -> None:
    router.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    router.get("https://openrouter.ai/api/v1/key").mock(
        return_value=httpx.Response(
            200, json={"data": {"limit": None, "usage": 12.0, "limit_remaining": None}}
        )
    )
    settings = make(llm_api_key="sk-or-v1-test", calcom_base_url="http://sandbox:8100")
    checks = await run_doctor(settings)
    llm = next(c for c in checks if c.name == "llm")
    assert llm.ok is True
    assert "no spend limit" in llm.detail


async def test_llm_check_fails_on_an_invalid_key(router: respx.MockRouter) -> None:
    router.get("https://openrouter.ai/api/v1/models").mock(return_value=httpx.Response(401))
    settings = make(llm_api_key="sk-or-v1-bad", calcom_base_url="http://sandbox:8100")
    checks = await run_doctor(settings)
    llm = next(c for c in checks if c.name == "llm")
    assert llm.ok is False
    assert "401" in llm.detail


async def test_llm_check_skips_the_key_endpoint_for_a_non_openrouter_host(router: respx.MockRouter) -> None:
    router.get("https://example.com/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    settings = make(
        llm_api_key="sk-test", llm_base_url="https://example.com/v1", calcom_base_url="http://sandbox:8100"
    )
    checks = await run_doctor(settings)
    llm = next(c for c in checks if c.name == "llm")
    assert llm.ok is True
    assert "key valid" not in llm.detail


# calendar (Cal.com) -------------------------------------------------------------------------------------


async def test_calcom_check_lists_one_booking(router: respx.MockRouter) -> None:
    route = router.get("http://sandbox:8100/v2/bookings").mock(
        return_value=httpx.Response(200, json={"status": "success", "data": []})
    )
    settings = make(calcom_base_url="http://sandbox:8100", sandbox_token="sandbox")
    checks = await run_doctor(settings)
    calendar = next(c for c in checks if c.name == "calendar")
    assert calendar.ok is True
    assert route.calls[0].request.headers["cal-api-version"] == "2024-08-13"
    assert route.calls[0].request.url.params["take"] == "1"


async def test_calcom_check_fails_when_the_sandbox_is_unreachable() -> None:
    settings = make(calcom_base_url="http://sandbox:8100", sandbox_token="sandbox")
    checks = await run_doctor(settings)
    calendar = next(c for c in checks if c.name == "calendar")
    assert calendar.ok is False


async def test_calcom_check_requires_an_api_key_for_a_real_account(router: respx.MockRouter) -> None:
    settings = make(calcom_base_url="https://api.cal.com/v2")
    checks = await run_doctor(settings)
    calendar = next(c for c in checks if c.name == "calendar")
    assert calendar.ok is False
    assert "BT_CALCOM_API_KEY" in calendar.detail


# calendar (Google) --------------------------------------------------------------------------------------


async def test_google_check_exchanges_a_token_and_queries_freebusy(
    router: respx.MockRouter, service_account_file: Path
) -> None:
    token_route = router.post("https://oauth2.googleapis.com/token").mock(
        return_value=httpx.Response(200, json={"access_token": "ya29.fake", "expires_in": 3599})
    )
    fb_route = router.post("https://www.googleapis.com/calendar/v3/freeBusy").mock(
        return_value=httpx.Response(200, json={"calendars": {"primary": {"busy": []}}})
    )
    settings = make(calendar="google", google_service_account_file=service_account_file)
    before = datetime.now(UTC)
    checks = await run_doctor(settings)
    calendar = next(c for c in checks if c.name == "calendar")
    assert calendar.ok is True

    sent_form = dict(pair.split("=") for pair in token_route.calls[0].request.content.decode().split("&"))
    assert sent_form["grant_type"] == "urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"
    claims = jwt.decode(sent_form["assertion"], options={"verify_signature": False})
    assert claims["iss"] == "calendar-agent@example.com"
    assert claims["aud"] == "https://oauth2.googleapis.com/token"
    assert claims["exp"] - claims["iat"] == 3600

    fb_body = json.loads(fb_route.calls[0].request.content)
    assert fb_body["items"] == [{"id": "primary"}]
    time_min = datetime.strptime(fb_body["timeMin"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    assert before < time_min < before + timedelta(days=1, minutes=1)
    assert fb_route.calls[0].request.headers["authorization"] == "Bearer ya29.fake"


async def test_google_check_fails_without_a_service_account_file() -> None:
    settings = make(calendar="google")
    checks = await run_doctor(settings)
    calendar = next(c for c in checks if c.name == "calendar")
    assert calendar.ok is False
    assert "BT_GOOGLE_SERVICE_ACCOUNT_FILE" in calendar.detail


async def test_google_check_fails_on_a_token_error(
    router: respx.MockRouter, service_account_file: Path
) -> None:
    router.post("https://oauth2.googleapis.com/token").mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"})
    )
    settings = make(calendar="google", google_service_account_file=service_account_file)
    checks = await run_doctor(settings)
    calendar = next(c for c in checks if c.name == "calendar")
    assert calendar.ok is False
    assert "400" in calendar.detail


# crm (HubSpot) -------------------------------------------------------------------------------------------


async def test_crm_check_is_ok_when_no_crm_is_configured() -> None:
    settings = make(calcom_base_url="http://sandbox:8100")
    checks = await run_doctor(settings)
    crm = next(c for c in checks if c.name == "crm")
    assert crm.ok is True
    assert "BT_CRM=none" in crm.detail


async def test_crm_check_searches_contacts_with_limit_one(router: respx.MockRouter) -> None:
    route = router.post("https://api.hubapi.com/crm/v3/objects/contacts/search").mock(
        return_value=httpx.Response(200, json={"total": 0, "results": []})
    )
    settings = make(calcom_base_url="http://sandbox:8100", crm="hubspot", hubspot_token="pat-fake")
    checks = await run_doctor(settings)
    crm = next(c for c in checks if c.name == "crm")
    assert crm.ok is True
    assert json.loads(route.calls[0].request.content) == {"limit": 1}


async def test_crm_check_requires_a_token() -> None:
    settings = make(calcom_base_url="http://sandbox:8100", crm="hubspot")
    checks = await run_doctor(settings)
    crm = next(c for c in checks if c.name == "crm")
    assert crm.ok is False
    assert "BT_HUBSPOT_TOKEN" in crm.detail


# The whole command --------------------------------------------------------------------------------------


def test_no_check_ever_prints_a_secret() -> None:
    settings = make(
        calcom_base_url="http://sandbox:8100",
        sandbox_token="sandbox",
        llm_api_key="sk-or-v1-super-secret-value",
        hubspot_token="pat-super-secret-value",
    )
    checks = asyncio.run(run_doctor(settings))
    for check in checks:
        assert "super-secret-value" not in check.detail


def test_cli_exits_1_when_any_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(settings: Settings) -> list[Check]:
        return [Check("config", True, "ok"), Check("llm", False, "no key")]

    monkeypatch.setattr("booking_truth.doctor.run_doctor", fake)
    result = runner.invoke(app, ["doctor"], env={"BT_CALCOM_BASE_URL": "http://sandbox:8100"})
    assert result.exit_code == 1
    assert "FAIL" in result.stdout
    assert "OK" in result.stdout


def test_cli_exits_0_when_every_check_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(settings: Settings) -> list[Check]:
        return [Check("config", True, "ok"), Check("llm", True, "ok")]

    monkeypatch.setattr("booking_truth.doctor.run_doctor", fake)
    result = runner.invoke(app, ["doctor"], env={"BT_CALCOM_BASE_URL": "http://sandbox:8100"})
    assert result.exit_code == 0

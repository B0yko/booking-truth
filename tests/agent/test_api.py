"""The HTTP API: auth, validation, limits, widget tokens, rate limit, CORS and the side endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from agent_env import API_KEY, LEAD, NOW, AgentEnv, agent_settings, make_env
from fastapi import FastAPI

from booking_truth.agent.api import create_agent_app
from booking_truth.agent.guards import all_except, guards_string
from booking_truth.agent.scripted import FakeLLM
from booking_truth.config import ConfigError
from booking_truth.crm import CrmSyncPayload
from booking_truth.llm.types import ChatMessage, LLMResponse, ToolSpec, Usage
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer
from booking_truth.trace.validate import trace_errors

Sandbox = tuple[FastAPI, BackgroundServer, SandboxState]
RESPONSE_KEYS = {"reply", "quick_replies", "booking", "agent_version", "guard", "usage"}


@pytest.fixture
async def capped(sandbox: Sandbox, tmp_path: Path) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, max_turns_per_session=2, max_input_chars=50):
        yield env


@pytest.fixture
async def hidden(sandbox: Sandbox, tmp_path: Path) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, expose_traces=False):
        yield env


@pytest.fixture
async def proxied(sandbox: Sandbox, tmp_path: Path) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, trust_proxy=True):
        yield env


# /v1/chat -------------------------------------------------------------------------------------------------


async def test_a_turn_returns_the_documented_shape(guarded: AgentEnv) -> None:
    data = await guarded.say("Hi, I'm in New York. Can I book a call next week?")
    assert set(data) == RESPONSE_KEYS
    assert data["agent_version"] == guarded.deps.version
    assert set(data["guard"]) == {"blocked", "repaired", "events"}
    assert set(data["usage"]) == {"prompt_tokens", "completion_tokens", "usd", "models", "providers"}
    assert data["usage"]["models"] == ["offline/scripted-policy"]
    assert data["usage"]["providers"] == ["offline"]
    assert data["usage"]["usd"] == 0.0
    assert data["usage"]["prompt_tokens"] > 0


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": f"Basic {API_KEY}"},
        {"Authorization": API_KEY},
    ],
)
async def test_the_chat_endpoint_needs_the_bearer(guarded: AgentEnv, headers: dict[str, str]) -> None:
    response = await guarded.client.post("/v1/chat", json=guarded.body(message="Hi"), headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    assert guarded.deps.store.sessions.get("s-1") is None


async def test_auth_comes_before_validation(guarded: AgentEnv) -> None:
    response = await guarded.client.post("/v1/chat", json={"nonsense": True})
    assert response.status_code == 401


async def test_invalid_requests_get_422(guarded: AgentEnv) -> None:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    body = guarded.body(message="Hi")
    body["action"] = {"type": "cancel", "booking_uid": "b"}
    response = await guarded.client.post("/v1/chat", json=body, headers=headers)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert "exactly one of message or action" in response.json()["detail"]
    broken = await guarded.client.post("/v1/chat", content=b"{not json", headers=headers)
    assert broken.status_code == 422


async def test_messages_over_the_limit_get_413(capped: AgentEnv) -> None:
    response = await capped.chat(message="x" * 51)
    assert response.status_code == 413
    assert response.json()["error"] == "input_too_long"
    assert capped.deps.store.sessions.get("s-1") is None
    huge = await capped.client.post(
        "/v1/chat",
        content=b"x" * (300 * 1024),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    assert huge.status_code == 413


async def test_a_session_ends_at_the_turn_cap(capped: AgentEnv) -> None:
    for _ in range(2):
        assert (await capped.chat(message="Thanks!")).status_code == 200
    ended = await capped.chat(message="Thanks!")
    assert ended.status_code == 410
    assert ended.json()["error"] == "session_ended"
    assert "reached its limit" in ended.json()["reply"]
    assert (await capped.chat(message="Thanks!", session="s-2")).status_code == 200


async def test_a_session_belongs_to_one_lead(guarded: AgentEnv) -> None:
    assert (await guarded.chat(message="Thanks!")).status_code == 200
    other = await guarded.chat(message="Thanks!", email="omar@example.com")
    assert other.status_code == 403
    same = await guarded.chat(message="Thanks!", email=" MAYA@example.com")
    assert same.status_code == 200


# Widget ---------------------------------------------------------------------------------------------------


async def test_the_widget_issues_and_checks_a_session_token(guarded: AgentEnv) -> None:
    first = await guarded.widget(message="Thanks!", session="w-1")
    assert first.status_code == 200
    token = first.json()["session_token"]
    assert token.startswith("wst1.")
    session = guarded.deps.store.sessions.get("w-1")
    assert session is not None
    assert session.channel == "widget"
    assert session.token_hash is not None
    assert token not in session.token_hash
    assert (await guarded.widget(message="Thanks!", session="w-1")).status_code == 403
    assert (await guarded.widget(message="Thanks!", session="w-1", token="wst1.forged")).status_code == 403
    again = await guarded.widget(message="Thanks!", session="w-1", token=token)
    assert again.status_code == 200
    assert again.json()["session_token"] == token
    stolen = await guarded.widget(message="Thanks!", session="w-2", token=token)
    assert stolen.status_code == 403
    assert stolen.json() == {"error": "invalid_session_token"}


async def test_the_widget_endpoint_always_runs_as_the_widget_channel(guarded: AgentEnv) -> None:
    response = await guarded.widget(message="Thanks!", session="w-9", channel="api")
    assert response.status_code == 200
    assert "session_token" in response.json()
    session = guarded.deps.store.sessions.get("w-9")
    assert session is not None
    assert session.channel == "widget"


async def test_the_api_channel_gets_no_session_token(guarded: AgentEnv) -> None:
    data = await guarded.say("Thanks!")
    assert "session_token" not in data


async def test_the_widget_is_rate_limited_per_peer(guarded: AgentEnv) -> None:
    for index in range(30):
        response = await guarded.widget(message="Thanks!", session=f"r-{index}")
        assert response.status_code == 200, index
    limited = await guarded.widget(message="Thanks!", session="r-30")
    assert limited.status_code == 429
    assert limited.json()["error"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1
    spoofed = await guarded.client.post(
        "/v1/widget/chat",
        json=guarded.body(message="Thanks!", session="r-31", channel="widget"),
        headers={"X-Forwarded-For": "203.0.113.50"},
    )
    assert spoofed.status_code == 429  # the header is ignored without BT_TRUST_PROXY
    assert (await guarded.chat(message="Thanks!", session="api-1")).status_code == 200


async def test_behind_a_trusted_proxy_the_forwarded_address_is_the_key(proxied: AgentEnv) -> None:
    for index in range(30):
        await proxied.client.post(
            "/v1/widget/chat",
            json=proxied.body(message="Thanks!", session=f"p-{index}", channel="widget"),
            headers={"X-Forwarded-For": "203.0.113.1"},
        )
    other = await proxied.client.post(
        "/v1/widget/chat",
        json=proxied.body(message="Thanks!", session="p-x", channel="widget"),
        headers={"X-Forwarded-For": "203.0.113.2"},
    )
    assert other.status_code == 200


async def test_cors_allows_only_the_configured_origins(guarded: AgentEnv) -> None:
    preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    allowed = await guarded.client.options(
        "/v1/widget/chat", headers={"Origin": "http://localhost:8000", **preflight}
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8000"
    denied = await guarded.client.options(
        "/v1/widget/chat", headers={"Origin": "https://attacker.example.com", **preflight}
    )
    assert "access-control-allow-origin" not in denied.headers


# Side endpoints -------------------------------------------------------------------------------------------


async def test_the_version_endpoint_is_public(guarded: AgentEnv) -> None:
    response = await guarded.client.get("/v1/version")
    assert response.status_code == 200
    data = response.json()
    assert data["agent_version"] == guarded.deps.version
    assert data["version"] == "0.1.0"
    assert data["model"] == "offline/scripted-policy"
    assert data["guards"] == "all"
    assert data["offline"] is True
    assert data["temperature"] == 0.2
    assert data["calendar"] == "calcom"
    assert len(data["source_hash"]) == 64


async def test_the_version_changes_with_the_guard_configuration(guarded: AgentEnv, naive: AgentEnv) -> None:
    guarded_version = (await guarded.client.get("/v1/version")).json()
    naive_version = (await naive.client.get("/v1/version")).json()
    assert naive_version["guards"] == "off"
    assert guarded_version["agent_version"] != naive_version["agent_version"]
    assert guarded_version["source_hash"] == naive_version["source_hash"]


async def test_health_reports_the_outbox_backlog(guarded: AgentEnv) -> None:
    first = (await guarded.client.get("/healthz")).json()
    assert first["status"] == "ok"
    assert first["outbox"] == {"pending": 0, "failed": 0}
    assert first["outbox_backlog"] == 0
    payload = CrmSyncPayload(
        action="booked",
        lead_email=LEAD,
        booking_ref="b1",
        start_utc=NOW,
        end_utc=NOW.replace(minute=30),
    )
    guarded.deps.store.outbox.enqueue(LEAD, "crm_sync", payload)
    guarded.deps.store.handoffs.create(lead_email=LEAD, summary="x")
    second = (await guarded.client.get("/healthz")).json()
    assert second["outbox"] == {"pending": 1, "failed": 0}
    assert second["outbox_backlog"] == 1
    assert second["handoffs_undelivered"] == 1


async def test_the_demo_page_embeds_the_widget(guarded: AgentEnv) -> None:
    page = await guarded.client.get("/demo")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert '<script src="/widget.js" data-agent="/" async></script>' in page.text
    script = await guarded.client.get("/widget.js")
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert script.text.startswith("// @ts-check")


async def test_the_version_endpoint_flags_offline_mode_for_the_widget_banner(guarded: AgentEnv) -> None:
    # Every test in this module runs with no LLM key (conftest's autouse `_offline` fixture), so the
    # bundled FakeLLM answers and the widget's "offline demo mode" banner is driven by this field.
    data = (await guarded.client.get("/v1/version")).json()
    assert data["offline"] is True


async def test_the_trace_endpoint_needs_the_flag_and_the_bearer(hidden: AgentEnv, guarded: AgentEnv) -> None:
    await hidden.say("Thanks!")
    auth = {"Authorization": f"Bearer {API_KEY}"}
    assert (await hidden.client.get("/v1/sessions/s-1/trace", headers=auth)).status_code == 404
    await guarded.say("Thanks!")
    assert (await guarded.client.get("/v1/sessions/s-1/trace")).status_code == 401
    assert (await guarded.client.get("/v1/sessions/nope/trace", headers=auth)).status_code == 404


async def test_the_session_trace_is_valid_agent_trace(guarded: AgentEnv) -> None:
    await guarded.say("Hi, I'm in New York. Can I book a call next week?")
    auth = {"Authorization": f"Bearer {API_KEY}"}
    trace = (await guarded.client.get("/v1/sessions/s-1/trace", headers=auth)).json()
    assert trace_errors(trace) == []
    assert trace["schema"] == "agent-trace/v1"
    assert trace["task"]["domain"] == "booking"
    kinds = [(s["kind"], s["role"], s["name"]) for s in trace["steps"]]
    assert kinds[0] == ("message", "user", None)
    assert ("tool_call", "agent", "resolve_timezone") in kinds
    assert ("tool_result", "tool", "find_slots") in kinds
    assert kinds[-1] == ("message", "agent", None)
    assert [s["i"] for s in trace["steps"]] == list(range(len(trace["steps"])))
    assert trace["final_claim"]["text"].startswith("Here are some open times")
    assert {c["type"] for c in trace["final_claim"]["claims"]} == {"offered_slots"}
    assert trace["ground_truth"] == {"outcome": "unknown", "checked_by": "none"}
    assert trace["meta"]["agent_version"] == guarded.deps.version
    assert trace["meta"]["guards"] == "all"


# create_agent_app ---------------------------------------------------------------------------------------


def test_a_floating_model_id_is_refused_only_with_pinned_version(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="floating alias"):
        create_agent_app(
            agent_settings("http://127.0.0.1:9", tmp_path / "a.db", llm_model="vendor/model:latest"),
            llm=FakeLLM(),
        )
    unpinned = guards_string(all_except("pinned_version"))
    settings = agent_settings(
        "http://127.0.0.1:9", tmp_path / "b.db", llm_model="vendor/model:latest", guards=unpinned
    )
    app = create_agent_app(settings, llm=FakeLLM())
    app.state.deps.store.close()


def test_the_agent_refuses_to_start_without_an_api_key_for_a_real_calendar(tmp_path: Path) -> None:
    settings = agent_settings(
        "https://api.cal.com",
        tmp_path / "a.db",
        api_key=None,
        calcom_api_key="cal_live_x",
        calcom_event_type_id=5,
    )
    with pytest.raises(ConfigError, match="BT_API_KEY"):
        create_agent_app(settings, llm=FakeLLM())


def test_without_a_key_the_agent_runs_the_offline_policy(tmp_path: Path) -> None:
    app = create_agent_app(agent_settings("http://127.0.0.1:9", tmp_path / "a.db"))
    try:
        assert isinstance(app.state.deps.llm, FakeLLM)
        assert app.state.deps.offline
        assert app.state.deps.model_id == "offline/scripted-policy"
    finally:
        app.state.deps.store.close()


def test_google_is_not_available_yet(tmp_path: Path) -> None:
    settings = agent_settings(
        "http://127.0.0.1:9",
        tmp_path / "a.db",
        calendar="google",
        google_service_account_file=tmp_path / "sa.json",
    )
    with pytest.raises(ConfigError, match="google"):
        create_agent_app(settings, llm=FakeLLM())


class Scripted:
    """A non-offline model double for the version-endpoint offline-flag test."""

    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float,
        model: str | None = None,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        component: str = "agent",
        run_id: str | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content='{"reply": "Hi", "claims": []}',
            tool_calls=[],
            usage=Usage(),
            model_requested="vendor/model",
            model_returned="vendor/model",
            provider=None,
            response_id="r",
            latency_s=0.0,
        )


async def test_a_live_model_reports_offline_false(sandbox: Sandbox, tmp_path: Path) -> None:
    # /demo is the static widget/demo.html; the widget itself hides the offline banner using this flag
    # (see test_the_version_endpoint_flags_offline_mode_for_the_widget_banner for the scripted-offline case).
    async for env in make_env(sandbox, tmp_path, llm=Scripted()):
        page = await env.client.get("/demo")
        assert page.status_code == 200
        version = (await env.client.get("/v1/version")).json()
        assert version["offline"] is False
        assert version["model"] == env.deps.settings.llm_model

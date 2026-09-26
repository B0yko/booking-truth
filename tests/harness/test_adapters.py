"""The HTTP adapters: agent.yaml, templates, dot paths, reply parsing and agent-error classification."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest

from booking_truth.harness.adapters import (
    DEFAULT_BODY,
    AgentConfig,
    AgentConfigError,
    BundledAgentClient,
    BundledEndpoints,
    HttpAgentClient,
    Lead,
    Turn,
    expand_env,
    get_path,
    history_items,
    load_agent_config,
    outbox_backlog,
    render_body,
)

LEAD = Lead(email="maya-1a2b3c4d@example.com", name="Maya R.", timezone_hint="America/New_York")
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "agents"


def turn(message: str | None = "Hi", *, action: dict[str, Any] | None = None, session: str = "s1") -> Turn:
    return Turn(session_id=session, message_id=f"{session}-m1", lead=LEAD, message=message, action=action)


def transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# Templates and paths --------------------------------------------------------------------------------------


def test_expand_env_reads_variables_and_defaults() -> None:
    env = {"TOKEN": "abc", "EMPTY": ""}
    assert expand_env("Bearer ${TOKEN}", env) == "Bearer abc"
    assert expand_env("${MISSING:-fallback}", env) == "fallback"
    assert expand_env("${EMPTY:-fallback}", env) == "fallback"
    with pytest.raises(AgentConfigError, match="MISSING is not set"):
        expand_env("${MISSING}", env)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("reply", "hello"),
        ("data.messages[1].text", "second"),
        ("data.messages.0.text", "first"),
        ("choices.0.message.content", "choice"),
        ("", {"reply": "hello"}),
    ],
)
def test_get_path_follows_keys_and_list_indexes(path: str, expected: Any) -> None:
    data = {
        "reply": "hello",
        "data": {"messages": [{"text": "first"}, {"text": "second"}]},
        "choices": [{"message": {"content": "choice"}}],
    }
    if path == "":
        assert get_path({"reply": "hello"}, path) == expected
    else:
        assert get_path(data, path) == expected


@pytest.mark.parametrize("path", ["missing", "data.messages[5].text", "reply.deeper", "data.messages.x"])
def test_get_path_raises_for_missing_steps(path: str) -> None:
    with pytest.raises(KeyError):
        get_path({"reply": "hello", "data": {"messages": [{"text": "a"}]}}, path)


def test_render_body_keeps_json_types_for_whole_placeholders() -> None:
    variables = {
        "session_id": "s1",
        "message": "Hi there",
        "lead.timezone_hint": None,
        "history": [{"role": "user", "content": "Hi"}],
    }
    template = {
        "sid": "{{session_id}}",
        "text": "Prospect says: {{message}}",
        "hint": "{{ lead.timezone_hint }}",
        "hint_text": "zone={{lead.timezone_hint}}",
        "history": "{{history}}",
        "list": ["{{message}}", 3, True],
    }
    assert render_body(template, variables) == {
        "sid": "s1",
        "text": "Prospect says: Hi there",
        "hint": None,
        "hint_text": "zone=",
        "history": [{"role": "user", "content": "Hi"}],
        "list": ["Hi there", 3, True],
    }


def test_unknown_template_variables_are_rejected_when_the_config_loads() -> None:
    with pytest.raises(ValueError, match="unknown template variable"):
        AgentConfig.model_validate({"url": "http://localhost:1/x", "body": {"text": "{{lead.phone}}"}})


def test_load_agent_config_expands_env_in_url_and_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_HOST", "localhost:5678")
    monkeypatch.setenv("AGENT_TOKEN", "t0ken")
    path = tmp_path / "agent.yaml"
    path.write_text(
        "url: http://${AGENT_HOST}/webhook/x\n"
        "headers: {Authorization: 'Bearer ${AGENT_TOKEN}'}\n"
        'body: \'{"text": "{{message}}"}\'\n'
        "response: {reply_path: output, version_path: meta.version}\n"
        "session_mode: cookie\n",
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.url == "http://localhost:5678/webhook/x"
    assert config.headers == {"Authorization": "Bearer t0ken"}
    assert config.body == {"text": "{{message}}"}
    assert config.response.version_path == "meta.version"
    assert config.session_mode == "cookie"


def test_load_agent_config_takes_the_whole_url_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_URL", "http://localhost:5678/webhook/x")
    path = tmp_path / "agent.yaml"
    path.write_text("url: ${AGENT_URL}\n", encoding="utf-8")
    assert load_agent_config(path).url == "http://localhost:5678/webhook/x"


def test_an_expanded_url_is_still_validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_URL", "ftp://localhost/x")
    path = tmp_path / "agent.yaml"
    path.write_text("url: ${AGENT_URL}\n", encoding="utf-8")
    with pytest.raises(AgentConfigError, match="http"):
        load_agent_config(path)


def test_stateless_mode_needs_the_history_in_the_body() -> None:
    body = {"text": "{{message}}"}
    with pytest.raises(ValueError, match="history"):
        AgentConfig.model_validate({"url": "http://localhost:1/x", "body": body, "session_mode": "stateless"})
    config = AgentConfig.model_validate(
        {
            "url": "http://localhost:1/x",
            "body": {**body, "history": "{{history}}"},
            "session_mode": "stateless",
        }
    )
    assert config.session_mode == "stateless"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("url: ftp://x\n", "http"),
        ("url: http://x\nsession_mode: sticky\n", "session_mode"),
        ("url: http://x\nextra: 1\n", "extra"),
        ("- not a mapping\n", "mapping"),
        ("url: http://${UNSET_AGENT_VAR}/x\n", "UNSET_AGENT_VAR"),
    ],
)
def test_load_agent_config_reports_problems(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(AgentConfigError, match=match):
        load_agent_config(path)


@pytest.mark.parametrize("name", ["bundled.yaml", "generic-webhook.yaml"])
def test_example_recipes_load(name: str) -> None:
    config = load_agent_config(EXAMPLES / name)
    assert config.url.startswith("http://localhost:")
    assert config.headers["Authorization"] == "Bearer dev-local-key"


def test_turn_needs_exactly_one_of_message_or_action() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Turn(session_id="s", message_id="m", lead=LEAD)
    with pytest.raises(ValueError, match="exactly one"):
        Turn(session_id="s", message_id="m", lead=LEAD, message="x", action={"type": "select_slot"})


# Bundled protocol ----------------------------------------------------------------------------------------


async def test_bundled_client_sends_the_protocol_and_parses_the_reply() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "reply": "Here are some times.",
                "quick_replies": [
                    {"label": "Mon 5 Oct, 2:00 PM", "action": {"type": "select_slot", "slot_id": "s_1"}},
                    "junk",
                ],
                "booking": None,
                "agent_version": "abc123def456",
                "guard": {"blocked": False, "repaired": True, "events": []},
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "usd": 0.0012},
            },
        )

    client = BundledAgentClient("http://agent.test/v1/chat", transport=transport(handler))
    reply = await client.send(turn(action={"type": "select_slot", "slot_id": "s_1"}, message=None))
    await client.aclose()
    body = json.loads(seen[0].content)
    assert seen[0].headers["authorization"] == "Bearer dev-local-key"
    assert body == {
        "session_id": "s1",
        "message_id": "s1-m1",
        "channel": "api",
        "lead": {"email": LEAD.email, "name": "Maya R.", "timezone_hint": "America/New_York"},
        "action": {"type": "select_slot", "slot_id": "s_1"},
    }
    assert reply.ok
    assert reply.reply == "Here are some times."
    assert len(reply.quick_replies) == 1
    assert reply.agent_version == "abc123def456"
    assert reply.usage_usd == pytest.approx(0.0012)
    assert reply.guard_active
    assert reply.latency_s >= 0
    assert reply.received_at >= reply.sent_at


async def test_bundled_client_uses_bt_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BT_API_KEY", "custom-key")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json={"reply": "ok"})

    client = BundledAgentClient("http://agent.test/v1/chat", transport=transport(handler))
    await client.send(turn())
    await client.aclose()
    assert seen == ["Bearer custom-key"]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(500, json={"error": "boom"}), "http_500"),
        (httpx.Response(503, text="unavailable"), "http_503"),
        (httpx.Response(200, text="<html>not json</html>"), "non_json"),
        (httpx.Response(200, json={"message": "no reply field"}), "reply_missing"),
        (httpx.Response(401, json={"error": "unauthorized"}), "http_401"),
        (httpx.Response(200, json=["a", "list"]), "invalid_response"),
    ],
)
async def test_bundled_client_classifies_agent_errors(response: httpx.Response, error: str) -> None:
    client = BundledAgentClient("http://agent.test/v1/chat", transport=transport(lambda _: response))
    reply = await client.send(turn())
    await client.aclose()
    assert reply.error == error
    assert reply.reply is None
    assert not reply.ok


async def test_lead_busy_is_a_normal_reply() -> None:
    response = httpx.Response(409, json={"error": "lead_busy", "reply": "I'm still working on it."})
    client = BundledAgentClient("http://agent.test/v1/chat", transport=transport(lambda _: response))
    reply = await client.send(turn())
    await client.aclose()
    assert reply.ok
    assert reply.lead_busy
    assert reply.status == 409
    assert reply.reply == "I'm still working on it."


async def test_lead_busy_without_text_is_still_a_normal_reply() -> None:
    response = httpx.Response(409, json={"error": "lead_busy"})
    client = BundledAgentClient("http://agent.test/v1/chat", transport=transport(lambda _: response))
    reply = await client.send(turn())
    await client.aclose()
    assert reply.ok
    assert reply.lead_busy
    assert reply.reply == ""


async def test_generic_409_without_a_reply_is_a_normal_reply() -> None:
    response = httpx.Response(409, json={"error": "busy"})
    client = HttpAgentClient(generic_config(), transport=transport(lambda _: response))
    reply = await client.send(turn())
    await client.aclose()
    assert reply.ok
    assert reply.lead_busy
    assert reply.reply == ""


@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "connection_refused"),
        (httpx.RemoteProtocolError("bad"), "transport_error: RemoteProtocolError"),
    ],
)
async def test_transport_failures_are_agent_errors(exc: Exception, error: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    client = BundledAgentClient("http://agent.test/v1/chat", transport=transport(handler))
    reply = await client.send(turn())
    await client.aclose()
    assert reply.error == error
    assert reply.status is None


async def test_a_refused_connection_on_a_real_port_is_an_agent_error() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    client = BundledAgentClient(f"http://127.0.0.1:{port}/v1/chat", timeout_s=2)
    reply = await client.send(turn())
    await client.aclose()
    assert reply.error == "connection_refused"


def test_bundled_endpoints_derive_from_the_chat_url() -> None:
    endpoints = BundledEndpoints("http://localhost:8000/v1/chat", "k")
    assert endpoints.version_url == "http://localhost:8000/v1/version"
    assert endpoints.health_url == "http://localhost:8000/healthz"
    assert endpoints.trace_url("abc") == "http://localhost:8000/v1/sessions/abc/trace"
    assert endpoints.headers == {"Authorization": "Bearer k"}


@pytest.mark.parametrize(
    ("health", "expected"),
    [
        ({"outbox": {"pending": 2, "failed": 1}}, 2),
        ({"outbox": {"backlog": 0}}, 0),
        ({"outbox_backlog": 3}, 3),
        ({"status": "ok"}, None),
        ("not a dict", None),
        ({"outbox": {"pending": True}}, None),
    ],
)
def test_outbox_backlog_reads_the_health_body(health: Any, expected: int | None) -> None:
    assert outbox_backlog(health) == expected


# Generic HTTP adapter ------------------------------------------------------------------------------------


def generic_config(**overrides: Any) -> AgentConfig:
    raw = {
        "url": "http://webhook.test/hook",
        "headers": {"X-Token": "abc"},
        "body": {
            "sid": "{{session_id}}",
            "text": "{{message}}",
            "email": "{{lead.email}}",
            "history": "{{history}}",
        },
        "response": {"reply_path": "data.messages[0].text", "version_path": "meta.version"},
        **overrides,
    }
    return AgentConfig.model_validate(raw)


async def test_generic_client_renders_the_template_and_reads_the_paths() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": {"messages": [{"text": "Hello!"}]}, "meta": {"version": 7}})

    client = HttpAgentClient(generic_config(), transport=transport(handler))
    history = history_items([("user", "Hi"), ("agent", "Hello")])
    reply = await client.send(
        Turn(session_id="s9", message_id="m2", lead=LEAD, message="Book me", history=history)
    )
    await client.aclose()
    assert client.supports_actions is False
    assert json.loads(seen[0].content) == {
        "sid": "s9",
        "text": "Book me",
        "email": LEAD.email,
        "history": [{"role": "user", "content": "Hi"}, {"role": "agent", "content": "Hello"}],
    }
    assert seen[0].headers["x-token"] == "abc"
    assert reply.reply == "Hello!"
    assert reply.agent_version == "7"


async def test_generic_client_default_body_is_the_bundled_message() -> None:
    config = AgentConfig.model_validate({"url": "http://webhook.test/hook"})
    assert config.body == DEFAULT_BODY
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok"})

    client = HttpAgentClient(config, transport=transport(handler))
    await client.send(turn("Hello"))
    await client.aclose()
    assert seen[0]["lead"] == {"email": LEAD.email, "name": "Maya R.", "timezone_hint": "America/New_York"}
    assert seen[0]["message"] == "Hello"


async def test_generic_client_refuses_actions() -> None:
    client = HttpAgentClient(generic_config(), transport=transport(lambda _: httpx.Response(200, json={})))
    with pytest.raises(AgentConfigError, match="text only"):
        await client.send(turn(None, action={"type": "select_slot", "slot_id": "x"}))
    await client.aclose()


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(502, text="bad gateway"), "http_502"),
        (httpx.Response(200, text="plain text"), "non_json"),
        (httpx.Response(200, json={"data": {}}), "reply_missing"),
        (httpx.Response(404, json={"error": "no such webhook"}), "http_404"),
    ],
)
async def test_generic_client_classifies_errors(response: httpx.Response, error: str) -> None:
    client = HttpAgentClient(generic_config(), transport=transport(lambda _: response))
    reply = await client.send(turn())
    await client.aclose()
    assert reply.error == error


async def test_cookie_mode_keeps_one_jar_per_session() -> None:
    seen: list[tuple[str, str | None]] = []
    counter = iter(range(100))

    def handler(request: httpx.Request) -> httpx.Response:
        sid = json.loads(request.content)["sid"]
        seen.append((sid, request.headers.get("cookie")))
        return httpx.Response(
            200,
            json={"data": {"messages": [{"text": "ok"}]}},
            headers={"set-cookie": f"agent_sid=c{next(counter)}; Path=/"},
        )

    client = HttpAgentClient(generic_config(session_mode="cookie"), transport=transport(handler))
    await client.send(turn(session="a"))
    await client.send(turn(session="b"))
    await client.send(turn(session="a"))
    await client.send(turn(session="b"))
    await client.aclose()
    assert seen == [("a", None), ("b", None), ("a", "agent_sid=c0"), ("b", "agent_sid=c1")]


async def test_other_modes_never_send_cookies() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("cookie"))
        return httpx.Response(
            200, json={"data": {"messages": [{"text": "ok"}]}}, headers={"set-cookie": "agent_sid=x; Path=/"}
        )

    client = HttpAgentClient(generic_config(session_mode="stateless"), transport=transport(handler))
    await client.send(turn(session="a"))
    await client.send(turn(session="a"))
    await client.aclose()
    assert seen == [None, None]

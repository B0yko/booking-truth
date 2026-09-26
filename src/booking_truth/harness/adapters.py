"""Drive an agent under test over HTTP.

Two protocols are supported:

- **The bundled protocol**, spoken by the reference agent's ``POST /v1/chat``. It is used for every agent URL
  given without ``--agent-config``. Requests carry ``session_id``, ``message_id``, ``channel``, ``lead`` and a
  ``message`` or a structured ``action``; the bearer token is ``BT_API_KEY`` (default ``dev-local-key``).
  Replies carry ``reply``, ``quick_replies``, ``booking``, ``agent_version``, ``guard`` and ``usage``.
- **A generic HTTP adapter** configured by an ``agent.yaml`` file: the URL, headers with ``${ENV}`` expansion,
  a JSON body template, dot paths to the reply text and the agent version, a timeout and a session mode. It
  fits any agent reachable over HTTP, including an n8n webhook.

Every call is classified the way ``docs/metrics.md`` defines an agent error: a 5xx status, a body that is not
JSON, a timeout past ``timeout_s`` or a refused connection. A ``409 lead_busy`` is a normal reply.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from booking_truth.timeutil import Clock, SystemClock

DEFAULT_API_KEY = "dev-local-key"
DEFAULT_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 10.0

SessionMode = Literal["stateless", "cookie", "session_id"]
Channel = Literal["api", "widget", "webhook"]

#: Variables a body template may use. ``history`` is the conversation so far in this session, as a list of
#: ``{"role": "user" | "agent", "content": ...}`` objects, for agents that keep no state of their own.
TEMPLATE_VARIABLES: tuple[str, ...] = (
    "session_id",
    "message_id",
    "message",
    "channel",
    "lead.email",
    "lead.name",
    "lead.timezone_hint",
    "history",
)
DEFAULT_BODY: dict[str, Any] = {
    "session_id": "{{session_id}}",
    "message_id": "{{message_id}}",
    "channel": "{{channel}}",
    "lead": {"email": "{{lead.email}}", "name": "{{lead.name}}", "timezone_hint": "{{lead.timezone_hint}}"},
    "message": "{{message}}",
}

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


class AgentConfigError(ValueError):
    """``agent.yaml`` is invalid or refers to an unset environment variable."""


# Requests and replies ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Lead:
    """The prospect as the agent sees it. The email is made up per trial on ``example.com``."""

    email: str
    name: str
    timezone_hint: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"email": self.email, "name": self.name, "timezone_hint": self.timezone_hint}


@dataclass(frozen=True)
class HistoryItem:
    role: Literal["user", "agent"]
    content: str

    def to_json(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Turn:
    """One inbound message: either text or a structured action (bundled protocol only)."""

    session_id: str
    message_id: str
    lead: Lead
    message: str | None = None
    action: Mapping[str, Any] | None = None
    channel: Channel = "api"
    history: tuple[HistoryItem, ...] = ()

    def __post_init__(self) -> None:
        if (self.message is None) == (self.action is None):
            raise ValueError("a turn carries exactly one of message or action")


@dataclass(frozen=True)
class AgentReply:
    """What came back for one request. ``error`` is set for an agent error and ``reply`` is then ``None``.

    ``sent_at`` and ``received_at`` are ``time.perf_counter()`` readings (arrival order and latency);
    ``sent_ts`` and ``received_ts`` are wall-clock instants for traces.
    """

    status: int | None
    reply: str | None
    quick_replies: tuple[dict[str, Any], ...] = ()
    booking: dict[str, Any] | None = None
    agent_version: str | None = None
    usage_usd: float = 0.0
    latency_s: float = 0.0
    raw: Any = None
    error: str | None = None
    guard: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    lead_busy: bool = False
    sent_at: float = 0.0
    received_at: float = 0.0
    sent_ts: datetime | None = None
    received_ts: datetime | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def guard_active(self) -> bool:
        """The agent's guard blocked or repaired this reply."""
        guard = self.guard or {}
        return bool(guard.get("blocked")) or bool(guard.get("repaired"))

    def structured(self) -> dict[str, Any]:
        """The non-text parts of the reply, for traces."""
        out: dict[str, Any] = {"status": self.status, "latency_s": round(self.latency_s, 6)}
        if self.quick_replies:
            out["quick_replies"] = list(self.quick_replies)
        if self.booking is not None:
            out["booking"] = self.booking
        if self.agent_version is not None:
            out["agent_version"] = self.agent_version
        if self.guard is not None:
            out["guard"] = self.guard
        if self.usage is not None:
            out["usage"] = self.usage
        if self.lead_busy:
            out["lead_busy"] = True
        return out


class AgentClient(Protocol):
    """Sends turns to one agent endpoint."""

    #: Whether the agent takes structured actions (``select_slot``) instead of label text.
    supports_actions: bool

    async def send(self, turn: Turn) -> AgentReply: ...

    async def aclose(self) -> None: ...


# Helpers ---------------------------------------------------------------------------------------------------


def expand_env(text: str, environ: Mapping[str, str] | None = None) -> str:
    """Replace ``${NAME}`` and ``${NAME:-default}``; an unset variable without a default is an error."""
    env = os.environ if environ is None else environ

    def substitute(match: re.Match[str]) -> str:
        name, default = match[1], match[2]
        value = env.get(name)
        if value:
            return value
        if default is not None:
            return default
        raise AgentConfigError(f"environment variable {name} is not set (used as ${{{name}}})")

    return _ENV.sub(substitute, text)


def get_path(data: Any, path: str) -> Any:
    """Follow a dot path with list indexes (``data.reply``, ``choices.0.message.content``,
    ``messages[0].text``). Raises ``KeyError`` when any step is missing."""
    if path in ("", "."):
        return data
    current = data
    for part in path.split("."):
        tokens = list(_PATH_TOKEN.finditer(part))
        if not tokens or "".join(t[0] for t in tokens) != part:
            raise KeyError(f"invalid path segment {part!r} in {path!r}")
        for token in tokens:
            key, index = token[1], token[2]
            if index is not None or (key is not None and key.isdigit() and isinstance(current, list)):
                position = int(index if index is not None else key)
                if not isinstance(current, list) or position >= len(current):
                    raise KeyError(f"{path!r}: no item {position}")
                current = current[position]
            else:
                if not isinstance(current, dict) or key not in current:
                    raise KeyError(f"{path!r}: no key {key!r}")
                current = current[key]
    return current


def render_body(template: Any, variables: Mapping[str, Any]) -> Any:
    """Fill ``{{name}}`` placeholders in every string of a JSON template.

    A string that is exactly one placeholder takes the variable's JSON value (``null``, a list for
    ``history``); inside longer text the value is inserted as text (an unset value as an empty string).
    """
    if isinstance(template, str):
        whole = _PLACEHOLDER.fullmatch(template.strip())
        if whole is not None:
            return _variable(variables, whole[1])

        def substitute(match: re.Match[str]) -> str:
            value = _variable(variables, match[1])
            if value is None:
                return ""
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

        return _PLACEHOLDER.sub(substitute, template)
    if isinstance(template, dict):
        return {key: render_body(value, variables) for key, value in template.items()}
    if isinstance(template, list):
        return [render_body(item, variables) for item in template]
    return template


def _variable(variables: Mapping[str, Any], name: str) -> Any:
    if name not in variables:
        raise AgentConfigError(
            f"unknown template variable {{{{{name}}}}}; known: {', '.join(TEMPLATE_VARIABLES)}"
        )
    return variables[name]


def template_variables(turn: Turn) -> dict[str, Any]:
    if turn.message is None:
        raise AgentConfigError(
            "the generic HTTP adapter sends text only; structured actions need the bundled protocol"
        )
    return {
        "session_id": turn.session_id,
        "message_id": turn.message_id,
        "message": turn.message,
        "channel": turn.channel,
        "lead.email": turn.lead.email,
        "lead.name": turn.lead.name,
        "lead.timezone_hint": turn.lead.timezone_hint,
        "history": [item.to_json() for item in turn.history],
    }


def template_names(template: Any) -> set[str]:
    """Every ``{{name}}`` a body template uses."""
    if isinstance(template, str):
        return set(_PLACEHOLDER.findall(template))
    if isinstance(template, dict):
        return {name for value in template.values() for name in template_names(value)}
    if isinstance(template, list):
        return {name for item in template for name in template_names(item)}
    return set()


def _check_template(template: Any) -> None:
    for name in sorted(template_names(template)):
        if name not in TEMPLATE_VARIABLES:
            raise AgentConfigError(
                f"unknown template variable {{{{{name}}}}}; known: {', '.join(TEMPLATE_VARIABLES)}"
            )


# agent.yaml ----------------------------------------------------------------------------------------------


class ResponseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply_path: str = "reply"
    version_path: str | None = None


class AgentConfig(BaseModel):
    """The generic HTTP adapter's settings, as written in ``agent.yaml``."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = Field(default_factory=lambda: dict(DEFAULT_BODY))
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S, gt=0, le=600)
    session_mode: SessionMode = "session_id"

    @field_validator("url")
    @classmethod
    def _http(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"url must start with http:// or https://, got {value!r}")
        return value

    @field_validator("body", mode="before")
    @classmethod
    def _body(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"body is a string but not valid JSON: {exc}") from None
        if not isinstance(value, dict | list):
            raise ValueError("body must be a JSON object or array")
        _check_template(value)
        return value

    @model_validator(mode="after")
    def _stateless_sends_history(self) -> AgentConfig:
        # A stateless agent sees only what each request carries, so the body must carry the conversation.
        if self.session_mode == "stateless" and "history" not in template_names(self.body):
            raise ValueError(
                "session_mode stateless needs {{history}} in the body: a stateless agent keeps no "
                "conversation, so each request must carry it"
            )
        return self


def load_agent_config(path: Path | str, environ: Mapping[str, str] | None = None) -> AgentConfig:
    """Read ``agent.yaml``. ``${ENV}`` references in ``url`` and ``headers`` are expanded first, so a URL
    may come entirely from the environment (``url: ${AGENT_URL}``); the expanded values are validated."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AgentConfigError(f"{path.name}: cannot read agent config: {exc}") from None
    if not isinstance(raw, dict):
        raise AgentConfigError(f"{path.name}: expected a mapping of settings")
    raw = dict(raw)
    try:
        if isinstance(raw.get("url"), str):
            raw["url"] = expand_env(raw["url"], environ)
        headers = raw.get("headers")
        if isinstance(headers, dict):
            raw["headers"] = {
                key: expand_env(value, environ) if isinstance(value, str) else value
                for key, value in headers.items()
            }
    except AgentConfigError as exc:
        raise AgentConfigError(f"{path.name}: {exc}") from None
    try:
        return AgentConfig.model_validate(raw)
    except ValidationError as exc:
        lines = [
            f"{'.'.join(str(p) for p in err['loc'])}: {str(err['msg']).removeprefix('Value error, ')}"
            for err in exc.errors(include_url=False)
        ]
        raise AgentConfigError(f"{path.name}: invalid agent config\n  " + "\n  ".join(lines)) from None


# HTTP plumbing -------------------------------------------------------------------------------------------


@dataclass
class _Exchange:
    response: httpx.Response | None
    error: str | None
    sent_at: float
    received_at: float
    sent_ts: datetime
    received_ts: datetime


def _timeout(timeout_s: float) -> httpx.Timeout:
    return httpx.Timeout(timeout_s, connect=min(CONNECT_TIMEOUT_S, timeout_s))


async def _post(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Mapping[str, str],
    body: Any,
    clock: Clock,
    cookies: httpx.Cookies | None = None,
) -> _Exchange:
    request = client.build_request("POST", url, headers=dict(headers), json=body)
    if cookies is not None:
        cookies.set_cookie_header(request)
    sent_ts, sent_at = clock.now(), time.perf_counter()
    response: httpx.Response | None = None
    error: str | None = None
    try:
        response = await client.send(request)
    except httpx.TimeoutException:
        error = "timeout"
    except httpx.ConnectError:
        error = "connection_refused"
    except httpx.TransportError as exc:
        error = f"transport_error: {type(exc).__name__}"
    received_at, received_ts = time.perf_counter(), clock.now()
    if response is not None:
        if cookies is not None:
            cookies.extract_cookies(response)
        # Cookies live only in a session's own jar, never in the shared client.
        client.cookies.clear()
    return _Exchange(response, error, sent_at, received_at, sent_ts, received_ts)


def _parse_json(response: httpx.Response) -> tuple[Any, bool]:
    try:
        return response.json(), True
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None, False


def _usd(usage: Any) -> float:
    if isinstance(usage, dict):
        value = usage.get("usd")
        if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
            return float(value)
    return 0.0


def _error_reply(exchange: _Exchange, error: str, raw: Any = None) -> AgentReply:
    return AgentReply(
        status=exchange.response.status_code if exchange.response is not None else None,
        reply=None,
        error=error,
        raw=raw,
        latency_s=exchange.received_at - exchange.sent_at,
        sent_at=exchange.sent_at,
        received_at=exchange.received_at,
        sent_ts=exchange.sent_ts,
        received_ts=exchange.received_ts,
    )


def _classify(exchange: _Exchange) -> tuple[Any, AgentReply | None]:
    """The parsed JSON body, or the error reply for an agent error."""
    if exchange.error is not None or exchange.response is None:
        return None, _error_reply(exchange, exchange.error or "no_response")
    response = exchange.response
    data, is_json = _parse_json(response)
    if response.status_code >= 500:
        return None, _error_reply(exchange, f"http_{response.status_code}", data if is_json else None)
    if not is_json:
        return None, _error_reply(exchange, "non_json")
    return data, None


# Bundled protocol ----------------------------------------------------------------------------------------


class BundledAgentClient:
    """The reference agent's ``POST /v1/chat`` protocol (bearer ``BT_API_KEY``)."""

    supports_actions = True

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.url = url
        self.api_key = api_key or os.environ.get("BT_API_KEY") or DEFAULT_API_KEY
        self.timeout_s = timeout_s
        self._clock = clock or SystemClock()
        self._client = httpx.AsyncClient(timeout=_timeout(timeout_s), transport=transport)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def body(turn: Turn) -> dict[str, Any]:
        body: dict[str, Any] = {
            "session_id": turn.session_id,
            "message_id": turn.message_id,
            "channel": turn.channel,
            "lead": turn.lead.to_json(),
        }
        if turn.action is not None:
            body["action"] = dict(turn.action)
        else:
            body["message"] = turn.message
        return body

    async def send(self, turn: Turn) -> AgentReply:
        exchange = await _post(
            self._client, self.url, headers=self.headers, body=self.body(turn), clock=self._clock
        )
        data, failed = _classify(exchange)
        if failed is not None:
            return failed
        assert exchange.response is not None
        return parse_bundled_reply(data, exchange)

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_bundled_reply(data: Any, exchange: _Exchange) -> AgentReply:
    status = exchange.response.status_code if exchange.response is not None else None
    if not isinstance(data, dict):
        return _error_reply(exchange, "invalid_response", data)
    reply = data.get("reply")
    lead_busy = status == 409 and data.get("error") == "lead_busy"
    if not isinstance(reply, str):
        if lead_busy:
            reply = ""  # a 409 lead_busy is a normal reply, even without text
        elif status is not None and status >= 400:
            return _error_reply(exchange, f"http_{status}", data)
        else:
            return _error_reply(exchange, "reply_missing", data)
    quick = data.get("quick_replies")
    booking = data.get("booking")
    guard = data.get("guard")
    usage = data.get("usage")
    version = data.get("agent_version")
    return AgentReply(
        status=status,
        reply=reply,
        quick_replies=tuple(q for q in quick if isinstance(q, dict)) if isinstance(quick, list) else (),
        booking=booking if isinstance(booking, dict) else None,
        agent_version=version if isinstance(version, str) and version else None,
        usage_usd=_usd(usage),
        latency_s=exchange.received_at - exchange.sent_at,
        raw=data,
        guard=guard if isinstance(guard, dict) else None,
        usage=usage if isinstance(usage, dict) else None,
        lead_busy=lead_busy,
        sent_at=exchange.sent_at,
        received_at=exchange.received_at,
        sent_ts=exchange.sent_ts,
        received_ts=exchange.received_ts,
    )


# Generic HTTP adapter ------------------------------------------------------------------------------------


class HttpAgentClient:
    """Any HTTP agent, configured by ``agent.yaml``. It sends text only (a pick is the slot's label)."""

    supports_actions = False

    def __init__(
        self,
        config: AgentConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self._clock = clock or SystemClock()
        self._client = httpx.AsyncClient(timeout=_timeout(config.timeout_s), transport=transport)
        self._jars: dict[str, httpx.Cookies] = {}

    def body(self, turn: Turn) -> Any:
        return render_body(self.config.body, template_variables(turn))

    async def send(self, turn: Turn) -> AgentReply:
        jar = None
        if self.config.session_mode == "cookie":
            jar = self._jars.setdefault(turn.session_id, httpx.Cookies())
        headers = {"Content-Type": "application/json", **self.config.headers}
        exchange = await _post(
            self._client,
            self.config.url,
            headers=headers,
            body=self.body(turn),
            clock=self._clock,
            cookies=jar,
        )
        data, failed = _classify(exchange)
        if failed is not None:
            return failed
        status = exchange.response.status_code if exchange.response is not None else None
        try:
            reply = get_path(data, self.config.response.reply_path)
        except KeyError:
            if status == 409:
                reply = ""  # busy with the lead's previous message: a normal reply, not an agent error
            else:
                error = f"http_{status}" if status is not None and status >= 400 else "reply_missing"
                return _error_reply(exchange, error, data)
        if isinstance(reply, int | float) and not isinstance(reply, bool):
            reply = str(reply)
        if reply is None and status == 409:
            reply = ""
        if not isinstance(reply, str):
            return _error_reply(exchange, "reply_missing", data)
        version: str | None = None
        if self.config.response.version_path:
            try:
                found = get_path(data, self.config.response.version_path)
            except KeyError:
                found = None
            version = str(found) if isinstance(found, str | int) and not isinstance(found, bool) else None
        return AgentReply(
            status=status,
            reply=reply,
            agent_version=version or None,
            latency_s=exchange.received_at - exchange.sent_at,
            raw=data,
            lead_busy=status == 409,
            sent_at=exchange.sent_at,
            received_at=exchange.received_at,
            sent_ts=exchange.sent_ts,
            received_ts=exchange.received_ts,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# Bundled agent side endpoints -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BundledEndpoints:
    """The reference agent's other endpoints, derived from its chat URL (``.../v1/chat``)."""

    chat_url: str
    api_key: str

    @property
    def base(self) -> str:
        url = self.chat_url.rstrip("/")
        return url[: -len("/v1/chat")] if url.endswith("/v1/chat") else url

    @property
    def version_url(self) -> str:
        return f"{self.base}/v1/version"

    @property
    def health_url(self) -> str:
        return f"{self.base}/healthz"

    def trace_url(self, session_id: str) -> str:
        return f"{self.base}/v1/sessions/{session_id}/trace"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}


@dataclass
class SideChannel:
    """Reads a bundled agent's version, health and session traces. Failures return ``None``."""

    endpoints: BundledEndpoints
    timeout_s: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=_timeout(self.timeout_s), transport=self.transport)
        return self._client

    async def get_json(self, url: str) -> Any:
        try:
            response = await self._http().get(url, headers=self.endpoints.headers)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        data, ok = _parse_json(response)
        return data if ok else None

    async def version(self) -> dict[str, Any] | None:
        data = await self.get_json(self.endpoints.version_url)
        return data if isinstance(data, dict) else None

    async def outbox_backlog(self) -> int | None:
        """Pending CRM outbox items from ``/healthz`` (``outbox.pending`` or ``outbox_backlog``)."""
        data = await self.get_json(self.endpoints.health_url)
        return outbox_backlog(data)

    async def session_trace(self, session_id: str) -> dict[str, Any] | None:
        data = await self.get_json(self.endpoints.trace_url(session_id))
        return data if isinstance(data, dict) else None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def outbox_backlog(health: Any) -> int | None:
    """The CRM outbox backlog a ``/healthz`` body reports: ``outbox.pending`` (or ``outbox.backlog``, or a
    top-level ``outbox_backlog``); ``None`` when it reports none."""
    if not isinstance(health, dict):
        return None
    outbox = health.get("outbox")
    candidates = [health.get("outbox_backlog")]
    if isinstance(outbox, dict):
        candidates += [outbox.get("pending"), outbox.get("backlog")]
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def history_items(pairs: Sequence[tuple[Literal["user", "agent"], str]]) -> tuple[HistoryItem, ...]:
    """Build a ``history`` tuple from ``(role, content)`` pairs."""
    return tuple(HistoryItem(role, content) for role, content in pairs)

"""In-process components for ``--agent builtin`` / ``builtin:naive`` and ``--sandbox auto``.

``--sandbox auto`` starts a sandbox on a free local port. ``builtin`` starts the bundled agent in-process with
all guards on (``builtin:naive`` with ``BT_GUARDS=off``), its calendar and CRM pointed at that sandbox, its
database in a temporary directory, and its session traces exposed to the harness.

``BT_CALENDAR`` (env, or the ``calendar`` argument, which takes precedence) chooses the calendar shape the
builtin agent runs on: ``calcom`` (the default) points it at the sandbox's Cal.com mirror with the seeded
event type; ``google`` points it at the sandbox's Google Calendar mirror and OAuth token endpoint instead,
with ``BT_HORIZON_DAYS=400`` so DST scenarios months ahead stay bookable. A Google run needs a
service-account key, which this module generates as a throwaway RSA key in the agent's own temporary
directory — never written anywhere else, never committed — since the sandbox's fake token endpoint accepts
any structurally valid RS256 assertion (:mod:`booking_truth.sandbox.oauth`) and does not check its signature.
"""

from __future__ import annotations

import importlib
import json
import os
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from booking_truth.config import ConfigError, Settings, load_settings
from booking_truth.harness.adapters import DEFAULT_API_KEY
from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.sandbox.oauth import TOKEN_PATH
from booking_truth.serve import BackgroundServer

AGENT_MODULE = "booking_truth.agent.api"
BuiltinMode = Literal["guarded", "naive"]
BuiltinCalendar = Literal["calcom", "google"]
BUILTIN_TARGETS: dict[str, BuiltinMode] = {"builtin": "guarded", "builtin:naive": "naive"}
#: A Google run against the sandbox needs slot-grid scenarios up to a year out; matches the product default.
GOOGLE_HORIZON_DAYS = 400
#: A throwaway service account's email; the sandbox never checks it against anything real.
THROWAWAY_SERVICE_ACCOUNT_EMAIL = "booking-agent@example.com"


def write_throwaway_service_account(directory: Path) -> Path:
    """A fresh RSA key, written as a Google service-account JSON file under ``directory``.

    Generated at call time and never committed: the sandbox's ``/token`` endpoint checks the assertion's
    claims, not its signature, so any structurally valid RS256 key works against it.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    payload = {
        "type": "service_account",
        "project_id": "booking-truth-builtin",
        "private_key_id": "throwaway",
        "private_key": pem,
        "client_email": THROWAWAY_SERVICE_ACCOUNT_EMAIL,
        "client_id": "0",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    path = directory / "google-service-account.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _resolve_calendar(calendar: BuiltinCalendar | None) -> BuiltinCalendar:
    if calendar is not None:
        return calendar
    from_env = os.environ.get("BT_CALENDAR", "").strip().lower()
    return "google" if from_env == "google" else "calcom"


class BuiltinUnavailable(RuntimeError):
    """The bundled agent cannot be started in this installation."""


def load_agent_factory() -> Callable[..., Any]:
    """``create_agent_app`` of the bundled agent, imported only when a builtin agent is requested."""
    try:
        module = importlib.import_module(AGENT_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in ("booking_truth.agent", AGENT_MODULE):
            raise BuiltinUnavailable(
                "the bundled agent is not available in this installation (the module "
                f"{AGENT_MODULE} is missing), so --agent builtin cannot run; test an agent URL instead"
            ) from None
        raise
    factory: object = getattr(module, "create_agent_app", None)
    if not callable(factory):
        raise BuiltinUnavailable(f"{AGENT_MODULE} has no create_agent_app()")
    return factory


def builtin_settings(
    mode: BuiltinMode,
    *,
    sandbox_url: str,
    sandbox_token: str,
    db_path: Path,
    api_key: str,
    event_type_id: int = 1001,
    guards: str | None = None,
    calendar: BuiltinCalendar = "calcom",
    google_service_account_file: Path | None = None,
) -> Settings:
    """Settings for the in-process agent: its calendar and CRM on the sandbox, like the compose stack.

    ``guards`` is a ``BT_GUARDS`` value that replaces the mode's default (``all`` / ``off``), for guard
    fixtures that switch one guard off. ``calendar="google"`` needs ``google_service_account_file`` (see
    :func:`write_throwaway_service_account`); it points ``BT_GOOGLE_BASE_URL`` and ``BT_GOOGLE_TOKEN_URI``
    at the sandbox and sets ``BT_HORIZON_DAYS`` to :data:`GOOGLE_HORIZON_DAYS`.
    """
    common: dict[str, Any] = dict(
        crm="hubspot",
        hubspot_base_url=sandbox_url,
        hubspot_token=sandbox_token,
        guards=guards if guards is not None else ("all" if mode == "guarded" else "off"),
        db_path=db_path,
        api_key=api_key,
        expose_traces=True,
    )
    if calendar == "google":
        if google_service_account_file is None:
            raise ValueError("google_service_account_file is required when calendar='google'")
        return load_settings(
            calendar="google",
            google_base_url=sandbox_url,
            google_token_uri=f"{sandbox_url}{TOKEN_PATH}",
            google_service_account_file=google_service_account_file,
            horizon_days=GOOGLE_HORIZON_DAYS,
            **common,
        )
    return load_settings(
        calendar="calcom",
        calcom_base_url=sandbox_url,
        calcom_api_key=sandbox_token,
        calcom_event_type_id=event_type_id,
        **common,
    )


@dataclass
class BuiltinAgent:
    mode: BuiltinMode
    server: BackgroundServer
    api_key: str
    _tmp: tempfile.TemporaryDirectory[str] = field(repr=False)

    @property
    def url(self) -> str:
        return f"{self.server.url}/v1/chat"

    def stop(self) -> None:
        try:
            self.server.stop()
        finally:
            self._tmp.cleanup()


def start_builtin_agent(
    mode: BuiltinMode,
    *,
    sandbox_url: str,
    sandbox_token: str,
    api_key: str | None = None,
    factory: Callable[..., Any] | None = None,
    guards: str | None = None,
    llm: Any | None = None,
    calendar: BuiltinCalendar | None = None,
) -> BuiltinAgent:
    """Start the bundled agent on a free local port, wired to ``sandbox_url``.

    ``guards`` overrides the mode's guard configuration; ``llm`` is passed to the factory as ``llm=`` (for
    example a ``FakeLLM`` with misbehaviours). ``calendar`` honours ``BT_CALENDAR`` when omitted, so
    ``booking-truth test --agent builtin --sandbox auto`` picks up the caller's own environment; passing it
    explicitly (as the guard fixtures do) overrides that. ``calendar="google"`` generates a throwaway
    service-account key in this agent's own temporary directory (see the module docstring).
    """
    create = factory or load_agent_factory()
    key = api_key or os.environ.get("BT_API_KEY") or DEFAULT_API_KEY
    resolved_calendar = _resolve_calendar(calendar)
    tmp = tempfile.TemporaryDirectory(prefix="bt-agent-")
    try:
        google_service_account_file = (
            write_throwaway_service_account(Path(tmp.name)) if resolved_calendar == "google" else None
        )
        settings = builtin_settings(
            mode,
            sandbox_url=sandbox_url,
            sandbox_token=sandbox_token,
            db_path=Path(tmp.name) / "agent.db",
            api_key=key,
            guards=guards,
            calendar=resolved_calendar,
            google_service_account_file=google_service_account_file,
        )
        app = create(settings) if llm is None else create(settings, llm=llm)
        server = BackgroundServer(app).start()
    except ConfigError as exc:
        tmp.cleanup()
        raise BuiltinUnavailable(f"the bundled agent's configuration is invalid: {exc}") from None
    except BaseException:
        tmp.cleanup()
        raise
    return BuiltinAgent(mode=mode, server=server, api_key=key, _tmp=tmp)


def start_sandbox(token: str | None = None) -> tuple[BackgroundServer, str]:
    """``--sandbox auto``: an in-process sandbox on a free port. Returns the server and its token."""
    token = token or secrets.token_urlsafe(16)
    return BackgroundServer(create_sandbox_app(token)).start(), token

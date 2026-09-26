"""In-process components for ``--agent builtin`` / ``builtin:naive`` and ``--sandbox auto``.

``--sandbox auto`` starts a sandbox on a free local port. ``builtin`` starts the bundled agent in-process with
all guards on (``builtin:naive`` with ``BT_GUARDS=off``), its calendar and CRM pointed at that sandbox, its
database in a temporary directory, and its session traces exposed to the harness.
"""

from __future__ import annotations

import importlib
import os
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from booking_truth.config import ConfigError, Settings, load_settings
from booking_truth.harness.adapters import DEFAULT_API_KEY
from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.serve import BackgroundServer

AGENT_MODULE = "booking_truth.agent.api"
BuiltinMode = Literal["guarded", "naive"]
BUILTIN_TARGETS: dict[str, BuiltinMode] = {"builtin": "guarded", "builtin:naive": "naive"}


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
) -> Settings:
    """Settings for the in-process agent: Cal.com and HubSpot on the sandbox, like the compose stack."""
    return load_settings(
        calendar="calcom",
        calcom_base_url=sandbox_url,
        calcom_api_key=sandbox_token,
        calcom_event_type_id=event_type_id,
        crm="hubspot",
        hubspot_base_url=sandbox_url,
        hubspot_token=sandbox_token,
        guards="all" if mode == "guarded" else "off",
        db_path=db_path,
        api_key=api_key,
        expose_traces=True,
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
) -> BuiltinAgent:
    """Start the bundled agent on a free local port, wired to ``sandbox_url``."""
    create = factory or load_agent_factory()
    key = api_key or os.environ.get("BT_API_KEY") or DEFAULT_API_KEY
    tmp = tempfile.TemporaryDirectory(prefix="bt-agent-")
    try:
        settings = builtin_settings(
            mode,
            sandbox_url=sandbox_url,
            sandbox_token=sandbox_token,
            db_path=Path(tmp.name) / "agent.db",
            api_key=key,
        )
        app = create(settings)
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

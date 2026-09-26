"""``builtin_settings`` and ``start_builtin_agent`` accept a guard override and a model for the factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from booking_truth.config import Settings
from booking_truth.harness.builtin import builtin_settings, start_builtin_agent
from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.serve import BackgroundServer


def test_a_guards_string_replaces_the_mode_default(tmp_path: Path) -> None:
    kwargs: dict[str, Any] = {
        "sandbox_url": "http://127.0.0.1:9",
        "sandbox_token": "t",
        "db_path": tmp_path / "a.db",
    }
    assert builtin_settings("guarded", api_key="k", **kwargs).guards == "all"
    assert builtin_settings("naive", api_key="k", **kwargs).guards == "off"
    custom = builtin_settings("guarded", api_key="k", guards="claim_ledger,dedupe", **kwargs)
    assert custom.enabled_guards == frozenset({"claim_ledger", "dedupe"})


def test_a_model_is_passed_to_the_factory() -> None:
    from fastapi import FastAPI

    seen: list[tuple[Settings, object]] = []
    model = object()

    def factory(settings: Settings, **extra: Any) -> Any:
        seen.append((settings, extra.get("llm")))
        app = FastAPI()

        @app.get("/healthz")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        return app

    agent = start_builtin_agent(
        "guarded",
        sandbox_url="http://127.0.0.1:9",
        sandbox_token="t",
        factory=factory,
        guards="dedupe",
        llm=model,
    )
    try:
        assert httpx.get(agent.server.url + "/healthz").json() == {"status": "ok"}
    finally:
        agent.stop()
    settings, passed = seen[0]
    assert passed is model
    assert settings.guards == "dedupe"
    plain = start_builtin_agent("naive", sandbox_url="http://127.0.0.1:9", sandbox_token="t", factory=factory)
    plain.stop()
    assert seen[1][1] is None


def test_google_calendar_shape_is_honoured() -> None:
    """The real agent factory, wired to the sandbox's Google shape: a question reaches ``freeBusy``."""
    sandbox = BackgroundServer(create_sandbox_app("sbx")).start()
    agent = start_builtin_agent("guarded", sandbox_url=sandbox.url, sandbox_token="sbx", calendar="google")
    try:
        response = httpx.post(
            agent.url,
            json={
                "session_id": "s1",
                "message_id": "m1",
                "channel": "api",
                "lead": {"email": "lena@example.com", "name": "Lena"},
                "message": "What times are available this week?",
            },
            headers={"Authorization": f"Bearer {agent.api_key}"},
            timeout=10.0,
        )
        assert response.status_code == 200, response.text
        state = httpx.get(
            sandbox.url + "/_state", headers={"Authorization": "Bearer sbx"}, timeout=5.0
        ).json()
        groups = {entry["group"] for entry in state["request_log"]}
        assert "freebusy" in groups
        assert "oauth.token" in groups
    finally:
        agent.stop()
        sandbox.stop()

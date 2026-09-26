"""``builtin_settings`` and ``start_builtin_agent`` accept a guard override and a model for the factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from booking_truth.config import Settings
from booking_truth.harness.builtin import builtin_settings, start_builtin_agent


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

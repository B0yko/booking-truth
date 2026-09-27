"""The run manifest and the in-process builtin components."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from booking_truth.config import ConfigError, Settings
from booking_truth.harness.builtin import (
    BuiltinUnavailable,
    builtin_settings,
    load_agent_factory,
    start_builtin_agent,
    start_sandbox,
)
from booking_truth.harness.manifest import (
    MANIFEST_SCHEMA,
    AgentManifest,
    CallRecorder,
    build_manifest,
    detect_hardware,
    git_info,
    hardware_description,
    suite_hash,
)


def test_suite_hash_covers_names_and_bytes(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "a.yaml").write_text("id: a\n")
    (suite / "b.yaml").write_text("id: b\n")
    (suite / "notes.txt").write_text("ignored")
    first = suite_hash(suite)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first)
    assert suite_hash(suite) == first
    (suite / "notes.txt").write_text("still ignored")
    assert suite_hash(suite) == first
    (suite / "b.yaml").write_text("id: b \n")
    assert suite_hash(suite) != first
    (suite / "b.yaml").write_text("id: b\n")
    (suite / "b.yaml").rename(suite / "c.yaml")
    assert suite_hash(suite) != first


def test_hardware_comes_from_the_override_or_the_machine() -> None:
    assert hardware_description("  MacBook Air M5, 24 GB ") == "MacBook Air M5, 24 GB"
    detected = detect_hardware()
    assert re.fullmatch(r".+, (\d+ GB|unknown memory)", detected)
    assert hardware_description(None) == detected


def test_hardware_never_reads_the_host_name(monkeypatch: pytest.MonkeyPatch) -> None:
    import platform
    import socket

    def forbidden(*_: Any) -> str:
        raise AssertionError("the host name must never be read")

    monkeypatch.setattr(socket, "gethostname", forbidden)
    monkeypatch.setattr(platform, "node", forbidden)
    assert detect_hardware()
    assert hardware_description(None)


def test_git_info_outside_a_checkout_is_unknown(tmp_path: Path) -> None:
    assert git_info(tmp_path) == {"sha": "unknown", "dirty": None}


def test_git_info_in_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    info = git_info(root)
    if (root / ".git").exists():
        assert re.fullmatch(r"[0-9a-f]{40}", info["sha"])
        assert isinstance(info["dirty"], bool)


def test_call_recorder_flags_variation() -> None:
    calls = CallRecorder()
    calls.record("persona", model="vendor/a-2026", provider="p1")
    calls.record("persona", model="vendor/a-2026", provider="p2")
    # An agent turn's usage carries the *lists* every internal model call of that turn returned
    # (booking_truth.agent.loop.Usage.to_json: "models"/"providers", plural, already deduplicated),
    # never a singular "model"/"provider" - that shape is asserted in tests/agent/test_api.py.
    calls.record_usage("agent:guarded", {"models": ["vendor/a-2026"], "providers": ["p1"], "usd": 0.1})
    # A real turn whose reply carried usage but no attribution at all (for example a cache hit that
    # OpenRouter reports with neither a model nor a provider) is still a call: it must be counted, not
    # silently dropped, and land in an explicit "unknown" bucket rather than vanish from the total.
    calls.record_usage("agent:guarded", {"models": [], "providers": [], "usd": 0.1})
    calls.record_usage("agent:naive", None)
    assert calls.to_json() == {
        "agent:guarded": {
            "calls": 2,
            "models_returned": ["vendor/a-2026"],
            "providers": ["p1"],
            "varied": False,
            "unknown_attribution_calls": 1,
            "per_call": [
                {"model": None, "provider": None, "calls": 1},
                {"model": "vendor/a-2026", "provider": "p1", "calls": 1},
            ],
        },
        "persona": {
            "calls": 2,
            "models_returned": ["vendor/a-2026"],
            "providers": ["p1", "p2"],
            "varied": True,
            "unknown_attribution_calls": 0,
            "per_call": [
                {"model": "vendor/a-2026", "provider": "p1", "calls": 1},
                {"model": "vendor/a-2026", "provider": "p2", "calls": 1},
            ],
        },
    }
    assert calls.has_unknown_attribution() is True


def test_call_recorder_reads_plural_agent_usage_not_a_singular_field() -> None:
    """A turn's usage never carries a bare ``model``/``provider`` key; that shape must be ignored,
    not silently misread as if it were ``models``/``providers`` (and never counted as an unknown-
    attribution call: there is no evidence here that a call was even made in the expected shape)."""
    calls = CallRecorder()
    calls.record_usage("agent:guarded", {"model": "vendor/a-2026", "provider": "p1", "usd": 0.1})
    assert calls.to_json() == {}
    assert calls.has_unknown_attribution() is False


def test_call_recorder_does_not_count_a_turn_whose_code_path_never_called_the_model() -> None:
    """A structured-action turn the agent handles entirely in code carries the bundled protocol's usage
    shape too, but at its all-zero default (``booking_truth.agent.loop.Usage()``): it must not inflate
    ``calls`` or ``unknown_attribution_calls`` - there is no LLM call here to be missing attribution."""
    calls = CallRecorder()
    calls.record_usage(
        "agent:guarded",
        {"models": [], "providers": [], "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0},
    )
    assert calls.to_json() == {}
    assert calls.has_unknown_attribution() is False


def test_call_recorder_still_counts_a_real_call_with_zero_cost_but_positive_tokens() -> None:
    calls = CallRecorder()
    calls.record_usage(
        "agent:guarded",
        {"models": [], "providers": [], "prompt_tokens": 120, "completion_tokens": 0, "usd": 0.0},
    )
    out = calls.to_json()["agent:guarded"]
    assert out["calls"] == 1
    assert out["unknown_attribution_calls"] == 1


def test_call_recorder_has_unknown_attribution_is_false_with_full_attribution() -> None:
    calls = CallRecorder()
    calls.record("persona", model="vendor/a-2026", provider="p1")
    calls.record_usage("agent:guarded", {"models": ["vendor/a-2026"], "providers": ["p1"], "usd": 0.1})
    assert calls.has_unknown_attribution() is False
    assert calls.to_json()["agent:guarded"]["unknown_attribution_calls"] == 0


def test_call_recorder_pairs_mixed_length_agent_usage_positionally() -> None:
    """A turn whose internal calls returned more than one distinct model or provider: every one joins
    the distinct sets and is flagged as varied, even though the exact per-call pairing cannot be
    recovered from the already-deduplicated ``models``/``providers`` lists."""
    calls = CallRecorder()
    calls.record_usage(
        "agent:guarded", {"models": ["vendor/a-2026", "vendor/b-2026"], "providers": ["p1"], "usd": 0.2}
    )
    out = calls.to_json()["agent:guarded"]
    assert out["calls"] == 1
    assert out["models_returned"] == ["vendor/a-2026", "vendor/b-2026"]
    assert out["providers"] == ["p1"]
    assert out["varied"] is True


def test_the_manifest_is_redacted() -> None:
    manifest = build_manifest(
        run_id="r",
        date="2026-10-01",
        as_of="2026-10-01",
        hardware="lab box of someone@example.com",
        suite="bundled",
        suite_digest="sha256:00",
        scenarios=["happy-book-host-zone"],
        k=1,
        agents=[AgentManifest("guarded", "builtin", "guarded", "bundled", "builtin")],
        grading={"mode": "offline"},
        models={},
        temperatures={},
        calls=CallRecorder(),
        spend={"total_usd": 0.1234567},
        status="complete",
        status_detail=None,
        options={},
        command="booking-truth test --suite " + "/" + "Users/someone/suite",
        dry_run=False,
        git={"sha": "unknown", "dirty": None},
    )
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["hardware"] == "lab box of [email]"
    assert manifest["command"] == "booking-truth test --suite ~/suite"
    assert manifest["spend"] == {"total_usd": 0.123457}
    assert manifest["agents"][0]["agent_version"] is None
    assert manifest["llm_attribution_incomplete"] is False


def _manifest_with(calls: CallRecorder) -> dict[str, Any]:
    return build_manifest(
        run_id="r",
        date="2026-10-01",
        as_of="2026-10-01",
        hardware="MacBook Air M5, 24 GB",
        suite="bundled",
        suite_digest="sha256:00",
        scenarios=["happy-book-host-zone"],
        k=1,
        agents=[AgentManifest("guarded", "builtin", "guarded", "bundled", "builtin")],
        grading={"mode": "offline"},
        models={},
        temperatures={},
        calls=calls,
        spend={"total_usd": 0.0},
        status="complete",
        status_detail=None,
        options={},
        command="booking-truth test",
        dry_run=False,
        git={"sha": "unknown", "dirty": None},
    )


def test_the_manifest_flags_a_run_with_unknown_attribution() -> None:
    calls = CallRecorder()
    calls.record_usage("agent:guarded", {"models": [], "providers": [], "usd": 0.1})
    manifest = _manifest_with(calls)
    assert manifest["llm_attribution_incomplete"] is True
    assert manifest["llm_calls"]["agent:guarded"]["calls"] == 1
    assert manifest["llm_calls"]["agent:guarded"]["unknown_attribution_calls"] == 1


# Builtin ------------------------------------------------------------------------------------------------


def test_a_missing_agent_module_is_reported_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "booking_truth.agent.api", None)
    with pytest.raises(BuiltinUnavailable, match="not available in this installation"):
        load_agent_factory()


def test_an_agent_module_without_a_factory_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "booking_truth.agent.api", types.ModuleType("booking_truth.agent.api"))
    with pytest.raises(BuiltinUnavailable, match="no create_agent_app"):
        load_agent_factory()


def test_builtin_settings_point_calendar_and_crm_at_the_sandbox(tmp_path: Path) -> None:
    settings = builtin_settings(
        "naive", sandbox_url="http://127.0.0.1:9", sandbox_token="tok", db_path=tmp_path / "a.db", api_key="k"
    )
    assert (settings.calendar, settings.calcom_base_url, settings.crm) == (
        "calcom",
        "http://127.0.0.1:9",
        "hubspot",
    )
    assert settings.hubspot_base_url == "http://127.0.0.1:9"
    assert settings.calcom_api_key is not None
    assert settings.calcom_api_key.get_secret_value() == "tok"
    assert settings.enabled_guards == frozenset()
    assert settings.expose_traces
    assert settings.offline


def test_start_builtin_agent_serves_the_factory_app_and_cleans_up() -> None:
    from fastapi import FastAPI

    seen: list[Settings] = []

    def factory(settings: Settings) -> Any:
        seen.append(settings)
        app = FastAPI()

        @app.get("/healthz")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        return app

    agent = start_builtin_agent(
        "guarded", sandbox_url="http://127.0.0.1:9", sandbox_token="t", factory=factory
    )
    try:
        assert agent.url.endswith("/v1/chat")
        assert httpx.get(agent.server.url + "/healthz").json() == {"status": "ok"}
        assert seen[0].db_path.parent.exists()
        assert agent.api_key == "dev-local-key"
    finally:
        agent.stop()
    assert not seen[0].db_path.parent.exists()


def test_an_invalid_builtin_configuration_is_reported() -> None:
    def factory(settings: Settings) -> Any:
        raise ConfigError("BT_LLM_MODEL is a floating alias")

    with pytest.raises(BuiltinUnavailable, match="floating alias"):
        start_builtin_agent("guarded", sandbox_url="http://127.0.0.1:9", sandbox_token="t", factory=factory)


def test_start_sandbox_uses_a_fresh_token_unless_given() -> None:
    server, token = start_sandbox()
    try:
        assert len(token) >= 16
        assert httpx.get(server.url + "/_state").status_code == 401
        assert (
            httpx.get(server.url + "/_state", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        )
    finally:
        server.stop()
    server, token = start_sandbox("chosen")
    server.stop()
    assert token == "chosen"

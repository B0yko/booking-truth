"""``booking-truth test``, ``report`` and ``compare`` through the command line."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from stub_agent import SANDBOX_TOKEN, StubAgent, StubOptions, running_stub
from typer.testing import CliRunner

from booking_truth.cli import app
from booking_truth.config import Settings
from booking_truth.harness.cli_test import reproduce_command
from booking_truth.harness.report import MANIFEST_FILE, REPORT_FILE, SUMMARY_FILE, TRACES_FILE

runner = CliRunner()


@pytest.fixture(autouse=True)
def _workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BT_SANDBOX_TOKEN", SANDBOX_TOKEN)


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def run_stub(sandbox_url: str, base: str, out: str, *extra: str) -> Any:
    return invoke(
        "test", "--agent", f"stub={base}/v1/chat", "--sandbox", sandbox_url, "--hardware", "test machine",
        "--out", out, *extra,
    )  # fmt: skip


@pytest.fixture
def fake_builtin(monkeypatch: pytest.MonkeyPatch) -> list[Settings]:
    """Stand in for the bundled agent module: ``create_agent_app`` returns the stub agent's app."""
    created: list[Settings] = []

    def create_agent_app(settings: Settings) -> Any:
        created.append(settings)
        token = settings.calcom_api_key.get_secret_value() if settings.calcom_api_key else ""
        options = StubOptions(
            sandbox_url=settings.calcom_base_url,
            sandbox_token=token,
            api_key=settings.api_key.get_secret_value() if settings.api_key else "",
            guards=settings.guards,
            version=f"builtin-{settings.guards}",
        )
        return StubAgent(options).app

    module = types.ModuleType("booking_truth.agent.api")
    module.create_agent_app = create_agent_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "booking_truth.agent.api", module)
    return created


def test_builtin_needs_the_bundled_agent_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "booking_truth.agent.api", None)
    result = invoke("test", "--agent", "builtin", "--sandbox", "auto", "--only", "smoke", "--k", "1")
    assert result.exit_code == 2
    assert "the bundled agent is not available in this installation" in result.output


def test_builtin_agents_run_in_process_against_auto_sandboxes(
    fake_builtin: list[Settings], tmp_path: Path
) -> None:
    result = invoke(
        "test", "--agent", "builtin", "--agent", "builtin:naive", "--sandbox", "auto", "--only", "smoke",
        "--k", "1", "--out", "runs/builtin", "--hardware", "test machine",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "guarded: 2/2 pass" in result.output
    assert "naive: 2/2 pass" in result.output
    assert "grading: offline grading" in result.output
    guarded, naive = fake_builtin
    assert (guarded.guards, naive.guards) == ("all", "off")
    for settings in (guarded, naive):
        assert settings.calendar == "calcom"
        assert settings.calcom_base_url.startswith("http://127.0.0.1:")
        assert settings.hubspot_base_url == settings.calcom_base_url
        assert settings.crm == "hubspot"
        assert settings.expose_traces
        assert settings.calcom_event_type_id == 1001
    assert guarded.calcom_base_url != naive.calcom_base_url  # one sandbox per builtin agent
    manifest = json.loads((tmp_path / "runs" / "builtin" / MANIFEST_FILE).read_text())
    assert [(a["label"], a["kind"], a["mode"]) for a in manifest["agents"]] == [
        ("guarded", "builtin", "guarded"),
        ("naive", "builtin", "naive"),
    ]
    assert manifest["command"] == (
        "booking-truth test --agent builtin --agent builtin:naive --sandbox auto --only smoke --k 1 "
        "--hardware 'test machine'"
    )
    text = (tmp_path / "runs" / "builtin" / MANIFEST_FILE).read_text()
    assert "127.0.0.1" not in text


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--agent", "http://localhost:9/v1/chat", "--sandbox", "auto"], "--sandbox auto is valid only with"),
        (["--agent", "http://localhost:9/v1/chat"], "--sandbox <url> is required"),
        (["--agent", "ftp://agent"], "expected an http(s) URL"),
        (["--agent", "builtin", "--only", "no-such-tag"], "no scenario has the id or tag"),
        ([], "pass --agent"),
    ],
)
def test_wrong_option_combinations_exit_2(args: list[str], message: str) -> None:
    result = invoke("test", *args)
    assert result.exit_code == 2
    assert message in result.output


def test_pool_and_agent_are_exclusive(tmp_path: Path) -> None:
    pool = tmp_path / "pool.yaml"
    pool.write_text(
        "agents:\n  - label: guarded\n    pairs:\n      - {agent: http://localhost:9/v1/chat, sandbox: http://localhost:8}\n",
        encoding="utf-8",
    )
    result = invoke("test", "--pool", str(pool), "--agent", "http://localhost:9/v1/chat")
    assert result.exit_code == 2
    assert "--pool replaces --agent" in result.output


def test_an_invalid_pool_file_is_reported(tmp_path: Path) -> None:
    pool = tmp_path / "pool.yaml"
    pool.write_text("agents:\n  - label: Not Valid\n    pairs: []\n", encoding="utf-8")
    result = invoke("test", "--pool", str(pool))
    assert result.exit_code == 2
    assert "invalid pool file" in result.output


def test_an_unwired_agent_aborts_with_exit_2(sandbox_url: str) -> None:
    with running_stub(sandbox_url, wired=False) as (_, base):
        result = run_stub(sandbox_url, base, "runs/unwired", "--only", "happy-book-host-zone", "--k", "1")
    assert result.exit_code == 2
    assert "agent_not_wired_to_sandbox" in result.output


def test_test_report_and_compare(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url) as (_, base):
        first = run_stub(sandbox_url, base, "runs/a", "--only", "happy-book-host-zone", "--k", "1")
        second = run_stub(sandbox_url, base, "runs/b", "--only", "happy-book-host-zone", "--k", "1")
    with running_stub(sandbox_url, version="stub-9") as (_, base):
        third = run_stub(sandbox_url, base, "runs/c", "--only", "happy-book-host-zone", "--k", "1")
    for result in (first, second, third):
        assert result.exit_code == 0, result.output
    assert "stub: 1/1 pass" in first.output
    assert "outputs in runs/a: summary.json, report.md, traces.jsonl, manifest.json" in first.output
    run_a = tmp_path / "runs" / "a"
    assert sorted(p.name for p in run_a.iterdir()) == sorted(
        [SUMMARY_FILE, REPORT_FILE, TRACES_FILE, MANIFEST_FILE]
    )

    before = {name: (run_a / name).read_bytes() for name in (SUMMARY_FILE, REPORT_FILE)}
    report = invoke("report", str(run_a))
    assert report.exit_code == 0, report.output
    assert "wrote summary.json and report.md" in report.output
    assert {name: (run_a / name).read_bytes() for name in (SUMMARY_FILE, REPORT_FILE)} == before

    same = invoke("compare", "runs/a", "runs/b")
    assert same.exit_code == 0, same.output
    assert "# Compare a with b" in same.output
    assert "| pass^1 |" in same.output

    drift = invoke("compare", "runs/a", "runs/c")
    assert drift.exit_code == 2
    assert "agent version changed between the runs (stub: stub-1 -> stub-9)" in drift.output
    allowed = invoke("compare", "runs/a", "runs/c", "--allow-version-drift")
    assert allowed.exit_code == 0, allowed.output
    assert "agent versions differ (allowed by --allow-version-drift)" in allowed.output


def test_the_bearer_comes_from_bt_api_key_in_a_dot_env_file(
    sandbox_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BT_API_KEY", raising=False)
    (tmp_path / ".env").write_text("BT_API_KEY=key-from-dot-env\n", encoding="utf-8")
    with running_stub(sandbox_url, api_key="key-from-dot-env") as (_, base):
        result = run_stub(sandbox_url, base, "runs/env", "--only", "happy-book-host-zone", "--k", "1")
    assert result.exit_code == 0, result.output
    assert "stub: 1/1 pass" in result.output


def test_outputs_name_no_ip_address(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url) as (_, base):
        result = run_stub(sandbox_url, base, "runs/ip", "--only", "happy-book-host-zone", "--k", "1")
    assert result.exit_code == 0, result.output
    assert base.startswith("http://127.0.0.1:")
    run = tmp_path / "runs" / "ip"
    for name in (SUMMARY_FILE, REPORT_FILE, TRACES_FILE, MANIFEST_FILE):
        assert "127.0.0.1" not in (run / name).read_text(), name
    manifest = json.loads((run / MANIFEST_FILE).read_text())
    port = base.rsplit(":", 1)[1]
    assert manifest["agents"][0]["target"] == f"http://localhost:{port}/v1/chat"
    assert f"--agent stub=http://localhost:{port}/v1/chat" in manifest["command"]


def test_report_rejects_a_directory_without_a_run(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    result = invoke("report", "empty")
    assert result.exit_code == 2
    assert "manifest.json is missing" in result.output


def test_a_failing_trial_exits_1(sandbox_url: str) -> None:
    with running_stub(sandbox_url) as (_, base):
        result = run_stub(sandbox_url, base, "runs/fail", "--only", "fault-concurrent-channel", "--k", "1")
    assert result.exit_code == 1, result.output
    assert "stub: 0/1 pass" in result.output


def test_dry_run_prints_the_projection(sandbox_url: str) -> None:
    with running_stub(sandbox_url, usage_usd=0.001) as (_, base):
        result = run_stub(sandbox_url, base, "runs/dry", "--only", "smoke", "--dry-run", "--k", "5")
    assert result.exit_code == 0, result.output
    assert "projected cost of the full run: $" in result.output
    assert "1.3x safety factor" in result.output


def test_agent_config_drives_a_generic_webhook(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url) as (_, base):
        config = tmp_path / "agent.yaml"
        config.write_text(
            f"url: {base}/webhook\n"
            "body: {text: '{{message}}', email: '{{lead.email}}', name: '{{lead.name}}'}\n"
            "response: {reply_path: 'data.messages[0].text', version_path: meta.version}\n"
            "session_mode: cookie\n"
            "timeout_s: 10\n",
            encoding="utf-8",
        )
        result = invoke(
            "test", "--agent-config", str(config), "--sandbox", sandbox_url, "--only", "happy-book-berlin",
            "--k", "1", "--out", "runs/webhook",
        )  # fmt: skip
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "runs" / "webhook" / MANIFEST_FILE).read_text())
    assert manifest["agents"][0]["protocol"] == "agent.yaml"
    assert "--agent-config agent.yaml" in manifest["command"]
    assert str(tmp_path) not in manifest["command"]


def test_a_pool_file_labels_agents_and_pairs(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url) as (_, base):
        pool = tmp_path / "pool.yaml"
        pool.write_text(
            "agents:\n"
            "  - label: guarded\n"
            "    mode: guarded\n"
            f"    pairs:\n      - {{agent: {base}/v1/chat, sandbox: {sandbox_url}}}\n",
            encoding="utf-8",
        )
        result = invoke(
            "test", "--pool", str(pool), "--only", "happy-book-host-zone", "--k", "1", "--out", "runs/p"
        )
    assert result.exit_code == 0, result.output
    assert "guarded: 1/1 pass" in result.output
    summary = json.loads((tmp_path / "runs" / "p" / SUMMARY_FILE).read_text())
    assert summary["by_agent"]["guarded"]["modes"] == ["guarded"]


def test_reproduce_command_keeps_only_file_names(tmp_path: Path) -> None:
    command = reproduce_command(
        specs=["guarded=http://localhost:8001/v1/chat"],
        agent_config=tmp_path / "configs" / "agent.yaml",
        sandbox="http://localhost:8100",
        suite=tmp_path / "my-suite",
        only=["smoke", "fault-duplicate-delivery"],
        k=5,
        pool=None,
        grade_crm=True,
        settle_s=20,
        persona_model="vendor/model-a",
        extractor_model=None,
        as_of=None,
        hardware="MacBook Air M5, 24 GB",
        budget_usd=8,
        dry_run=True,
    )
    assert command == (
        "booking-truth test --agent guarded=http://localhost:8001/v1/chat --agent-config agent.yaml "
        "--sandbox http://localhost:8100 --suite my-suite --only smoke --only fault-duplicate-delivery --k 5 "
        "--grade-crm --settle-s 20 --persona-model vendor/model-a --hardware 'MacBook Air M5, 24 GB' "
        "--budget-usd 8 --dry-run"
    )
    assert str(tmp_path) not in command

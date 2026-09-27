"""``scripts/export_ci_fixtures.py``: its own plumbing (output shape, exit code, env scrubbing), with the
fixture runner itself faked out - that runner's correctness is ``tests/guards/test_guard_fixtures.py``'s
job; this only checks what the export script does with its results.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_ci_fixtures.py"


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # A real environment might carry live credentials; prove the script strips them before it does
    # anything else, by setting some before loading it.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-be-stripped")
    monkeypatch.setenv("BT_BUDGET_USD", "1")
    spec = importlib.util.spec_from_file_location("export_ci_fixtures", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_ci_fixtures"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeFixture:
    guard: str
    title: str
    expect_on: str = "expect_on"
    expect_off: str = "expect_off"


def _patch_fixture_runner(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, *, failing: tuple[str, str] | None
) -> None:
    fixtures = [FakeFixture(guard="claim_ledger", title="A"), FakeFixture(guard="dedupe", title="B")]

    def fake_load_fixture(path: Path) -> FakeFixture:
        return next(f for f in fixtures if f.guard == path.stem)

    monkeypatch.setattr(module, "fixture_files", lambda: [Path(f"{f.guard}.yaml") for f in fixtures])
    monkeypatch.setattr(module, "load_fixture", fake_load_fixture)
    monkeypatch.setattr(module, "resolved_calendars", lambda fixture: ("calcom", "google"))

    async def fake_run_fixture(fixture: FakeFixture, mode: str, calendar: str) -> tuple[str, str, str]:
        return (fixture.guard, mode, calendar)

    def fake_failures(expectation: str, observation: tuple[str, str, str]) -> list[str]:
        guard, mode, calendar = observation
        if failing == (guard, mode):
            return [f"deliberately broken for {calendar}"]
        return []

    monkeypatch.setattr(module, "run_fixture", fake_run_fixture)
    monkeypatch.setattr(module, "failures", fake_failures)


def test_the_repo_s_bt_and_openrouter_environment_never_reaches_the_fixture_runner(
    script: ModuleType,
) -> None:
    assert "OPENROUTER_API_KEY" not in os.environ
    assert "BT_BUDGET_USD" not in os.environ


def test_writes_ci_fixtures_json_with_one_row_per_fixture_mode_and_calendar(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_fixture_runner(monkeypatch, script, failing=None)
    exit_code = script.main(["--out", str(tmp_path)])
    assert exit_code == 0
    data = json.loads((tmp_path / script.CI_FILE).read_text(encoding="utf-8"))
    rows = data["fixtures"]
    # 2 fixtures x 2 modes x 2 calendars.
    assert len(rows) == 8
    assert all(row["status"] == "pass" for row in rows)
    claim_ledger_on_calcom = next(
        r for r in rows if r["guard"] == "claim_ledger" and r["mode"] == "on" and r["calendar"] == "calcom"
    )
    assert claim_ledger_on_calcom == {
        "name": "claim_ledger:on:calcom",
        "guard": "claim_ledger",
        "title": "A",
        "mode": "on",
        "calendar": "calcom",
        "status": "pass",
    }


def test_a_failing_fixture_is_recorded_with_its_problems_and_the_exit_code_is_nonzero(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_fixture_runner(monkeypatch, script, failing=("dedupe", "off"))
    exit_code = script.main(["--out", str(tmp_path)])
    assert exit_code == 1
    data = json.loads((tmp_path / script.CI_FILE).read_text(encoding="utf-8"))
    rows = data["fixtures"]
    failing_rows = [r for r in rows if r["status"] == "fail"]
    # Both calendars of dedupe:off fail (the fake failure applies to both).
    assert {r["calendar"] for r in failing_rows} == {"calcom", "google"}
    assert all(r["guard"] == "dedupe" and r["mode"] == "off" for r in failing_rows)
    assert all(r["problems"] for r in failing_rows)
    assert sum(r["status"] == "pass" for r in rows) == 6


def test_out_defaults_to_a_results_directory_named_by_a_generated_run_id(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_fixture_runner(monkeypatch, script, failing=None)
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    exit_code = script.main([])
    assert exit_code == 0
    written = list((tmp_path / "results").glob("*/" + script.CI_FILE))
    assert len(written) == 1

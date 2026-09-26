"""The offline path end to end: the harness drives both builtin agents through the smoke scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from booking_truth.cli import app
from booking_truth.harness.report import MANIFEST_FILE, TRACES_FILE
from booking_truth.trace.validate import trace_errors


def test_both_builtin_agents_pass_the_smoke_scenarios_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BT_LEDGER_DIR", str(tmp_path / "ledger"))
    result = CliRunner().invoke(
        app,
        [
            "test", "--agent", "builtin", "--agent", "builtin:naive", "--sandbox", "auto",
            "--only", "smoke", "--k", "1", "--out", "runs/smoke",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "guarded: 2/2 pass" in result.output
    assert "naive: 2/2 pass" in result.output
    run = tmp_path / "runs" / "smoke"
    manifest = json.loads((run / MANIFEST_FILE).read_text())
    versions = {a["label"]: a["agent_version"] for a in manifest["agents"]}
    assert set(versions) == {"guarded", "naive"}
    assert all(len(v) == 12 for v in versions.values())
    assert versions["guarded"] != versions["naive"]
    assert {a["label"]: a["version_info"]["offline"] for a in manifest["agents"]} == {
        "guarded": True,
        "naive": True,
    }
    traces = [json.loads(line) for line in (run / TRACES_FILE).read_text().splitlines()]
    assert len(traces) == 4
    for trace in traces:
        assert trace_errors(trace) == []
        assert trace["meta"]["agent_traces"] == {"fetched": 1, "missing": 0, "invalid": 0}
        assert any(s["kind"] == "tool_call" and s["name"] == "find_slots" for s in trace["steps"])

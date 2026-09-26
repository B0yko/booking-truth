"""``booking-truth eval tz`` and ``booking-truth eval extractor``, driven end to end through the CLI,
offline (no LLM key): both write their JSON and Markdown files and exit 0."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from booking_truth.cli import app
from booking_truth.harness.evals.cli import EXTRACTOR_JSON, EXTRACTOR_MD, TZ_JSON, TZ_MD

runner = CliRunner()
NO_KEY = {"BT_LLM_API_KEY": "", "OPENROUTER_API_KEY": ""}


def test_eval_tz_writes_its_files_and_exits_0(tmp_path: Path) -> None:
    out = tmp_path / "tz-run"
    result = runner.invoke(app, ["eval", "tz", "--out", str(out)], env=NO_KEY)
    assert result.exit_code == 0, result.output
    assert "silent wrong" in result.output
    data = json.loads((out / TZ_JSON).read_text())
    assert data["eval"] == "tz"
    assert data["llm"] is None
    assert (out / TZ_MD).read_text().startswith("# Timezone resolver eval")


def test_eval_extractor_writes_its_files_and_exits_0(tmp_path: Path) -> None:
    out = tmp_path / "extractor-run"
    result = runner.invoke(app, ["eval", "extractor", "--out", str(out)], env=NO_KEY)
    assert result.exit_code == 0, result.output
    assert "status accuracy" in result.output
    data = json.loads((out / EXTRACTOR_JSON).read_text())
    assert data["eval"] == "extractor"
    assert data["llm"] is None
    assert (out / EXTRACTOR_MD).read_text().startswith("# Belief extractor eval")


def test_eval_extractor_reports_a_missing_run_directory(tmp_path: Path) -> None:
    out = tmp_path / "extractor-run"
    result = runner.invoke(
        app,
        ["eval", "extractor", "--out", str(out), "--run", str(tmp_path)],
        env=NO_KEY,
    )
    assert result.exit_code == 2
    assert "summary.json" in result.output

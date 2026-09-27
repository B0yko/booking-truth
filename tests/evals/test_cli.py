"""``booking-truth eval tz`` and ``booking-truth eval extractor``, driven end to end through the CLI,
offline (no LLM key): both write their JSON and Markdown files and exit 0."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from booking_truth.cli import app
from booking_truth.config import Settings
from booking_truth.evals.cli import EXTRACTOR_JSON, EXTRACTOR_MD, TZ_JSON, TZ_MD, _client
from booking_truth.llm.client import HARNESS_LLM_MAX_RETRIES

runner = CliRunner()
NO_KEY = {"BT_LLM_API_KEY": "", "OPENROUTER_API_KEY": ""}


def test_the_eval_client_uses_the_harness_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``eval tz``/``eval extractor`` are harness-side infrastructure, not the agent under test, so their
    own live calls get the wider harness retry budget (run-2 anomaly: a single upstream 429 stopped
    ``eval tz`` 5 items early)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = Settings(_env_file=None, llm_api_key="sk-test-key", ledger_dir=tmp_path / "ledger")
    client, model_id = _client(config, component="eval_tz", model=None, budget_usd=None)
    assert client is not None
    assert model_id == config.llm_model
    assert client.max_retries == HARNESS_LLM_MAX_RETRIES
    asyncio.run(client.aclose())


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

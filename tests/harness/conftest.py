"""Shared fixtures for harness tests: no LLM key ever, a private ledger, and an in-process sandbox."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.serve import BackgroundServer

# Test modules import the stub agent and its helpers by module name.
sys.path.insert(0, str(Path(__file__).parent))

from stub_agent import SANDBOX_TOKEN


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Grading must stay offline in tests: both key variables unset, budget cap unset, own ledger."""
    for name in ("BT_LLM_API_KEY", "OPENROUTER_API_KEY", "BT_BUDGET_USD", "BT_API_KEY", "BT_SANDBOX_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BT_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.fixture(scope="module")
def sandbox_url() -> Iterator[str]:
    server = BackgroundServer(create_sandbox_app(SANDBOX_TOKEN)).start()
    try:
        yield server.url
    finally:
        server.stop()

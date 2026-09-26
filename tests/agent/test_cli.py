"""``booking-truth agent``: hand-offs, the outbox and the startup check of ``serve``."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from booking_truth.cli import app
from booking_truth.crm import CrmSyncPayload
from booking_truth.store import OUTBOX_MAX_ATTEMPTS, Store

runner = CliRunner()
START = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "agent.db"
    monkeypatch.setenv("BT_DB_PATH", str(path))
    monkeypatch.chdir(tmp_path)
    return path


def test_without_a_database_there_is_nothing_to_show(db: Path) -> None:
    result = runner.invoke(app, ["agent", "handoffs"])
    assert result.exit_code == 0
    assert "no agent database" in result.output
    assert not db.exists()


def test_handoffs_lists_open_ones_by_default(db: Path) -> None:
    with Store(db) as store:
        open_one = store.handoffs.create(
            lead_email="maya@example.com", summary="Wants a call", preferred_times_text="Tuesday afternoon"
        )
        done = store.handoffs.create(lead_email="omar@example.com", summary="Calendar down")
        store.handoffs.mark_delivered(done.id)
    result = runner.invoke(app, ["agent", "handoffs"])
    assert result.exit_code == 0, result.output
    assert f"#{open_one.id}" in result.output
    assert "preferred times: Tuesday afternoon" in result.output
    assert "omar@example.com" not in result.output
    everything = runner.invoke(app, ["agent", "handoffs", "--all", "--json"])
    records = [json.loads(line) for line in everything.output.splitlines()]
    assert [(r["lead_email"], r["delivered"]) for r in records] == [
        ("maya@example.com", False),
        ("omar@example.com", True),
    ]


def test_outbox_lists_items_and_requeues_a_failed_one(db: Path) -> None:
    payload = CrmSyncPayload(
        action="booked",
        lead_email="maya@example.com",
        booking_ref="b1",
        start_utc=START,
        end_utc=START + timedelta(minutes=30),
    )
    with Store(db) as store:
        pending = store.outbox.enqueue("maya@example.com", "crm_sync", payload)
        failed = store.outbox.enqueue("omar@example.com", "crm_sync", payload)
        for _ in range(OUTBOX_MAX_ATTEMPTS):
            store.outbox.mark_failed(failed.id, "HTTP 500")
    listing = runner.invoke(app, ["agent", "outbox"])
    assert listing.exit_code == 0, listing.output
    assert "backlog: 1 pending, 1 failed" in listing.output
    assert "last error: HTTP 500" in listing.output
    only_failed = runner.invoke(app, ["agent", "outbox", "--status", "failed", "--json"])
    lines = [line for line in only_failed.output.splitlines() if line.startswith("{")]
    assert [json.loads(line)["id"] for line in lines] == [failed.id]
    requeued = runner.invoke(app, ["agent", "outbox", "--requeue", str(failed.id)])
    assert requeued.exit_code == 0
    assert f"requeued outbox item #{failed.id}" in requeued.output
    again = runner.invoke(app, ["agent", "outbox", "--requeue", str(pending.id)])
    assert again.exit_code == 1
    assert "not failed" in again.output


def test_serve_refuses_an_invalid_configuration(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BT_LLM_MODEL", "vendor/model:latest")
    monkeypatch.setenv("BT_CALCOM_BASE_URL", "http://127.0.0.1:9")
    result = runner.invoke(app, ["agent", "serve", "--port", "8765"])
    assert result.exit_code == 2
    assert "floating alias" in result.output

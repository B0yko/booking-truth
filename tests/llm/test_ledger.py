import json
import os
import re
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from booking_truth.llm.ledger import _IN_FLIGHT, CostLedger, LedgerError
from booking_truth.llm.types import BudgetExceeded
from booking_truth.timeutil import FixedClock

FILE_NAME = re.compile(r"^(?P<component>[A-Za-z0-9_.-]+)-(?P<pid>\d+)-[0-9a-f]{8}\.jsonl$")


def entry(usd: float, **extra: object) -> dict[str, object]:
    return {
        "component": "agent",
        "run_id": "run-1",
        "model_requested": "vendor/flash",
        "model_returned": "vendor/flash-20260910",
        "provider": "DeepInfra",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cached_tokens": 0,
        "usd": usd,
        "provider_reported_cost": None,
        **extra,
    }


def test_file_is_created_lazily_with_owner_only_permissions(tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "ledger"
    ledger = CostLedger(directory, "agent", clock=FixedClock(datetime(2026, 9, 26, 12, 0, tzinfo=UTC)))
    assert ledger.path is None
    assert not directory.exists()
    assert ledger.total() == 0.0

    written = ledger.record(entry(0.001))
    path = ledger.path
    assert path is not None
    assert path.parent == directory
    match = FILE_NAME.match(path.name)
    assert match is not None
    assert match["component"] == "agent"
    assert int(match["pid"]) == os.getpid()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert written["ts"] == "2026-09-26T12:00:00.000Z"
    assert written["pid"] == os.getpid()
    line = json.loads(path.read_text().splitlines()[0])
    assert line == written
    assert set(line) <= {
        "ts",
        "component",
        "run_id",
        "model_requested",
        "model_returned",
        "provider",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "usd",
        "provider_reported_cost",
        "pid",
    }


def test_each_writer_appends_only_to_its_own_file(tmp_path: Path) -> None:
    agent = CostLedger(tmp_path, "agent")
    harness = CostLedger(tmp_path, "harness")
    agent.record(entry(0.25))
    harness.record(entry(0.5, component="persona"))
    agent.record(entry(0.125))
    assert agent.path != harness.path
    assert len(agent.files()) == 2
    assert agent.path is not None
    assert len(agent.path.read_text().splitlines()) == 2
    assert agent.total() == harness.total() == 0.875


def test_a_forked_process_starts_a_new_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = CostLedger(tmp_path, "agent")
    ledger.record(entry(0.1))
    first = ledger.path
    monkeypatch.setattr(os, "getpid", lambda: 999_999)
    ledger.record(entry(0.2))
    assert ledger.path != first
    assert ledger.path is not None
    assert "-999999-" in ledger.path.name
    assert ledger.total() == pytest.approx(0.3)


def test_separate_processes_write_separate_files_and_the_total_spans_them(tmp_path: Path) -> None:
    code = (
        "import sys; from booking_truth.llm.ledger import CostLedger; "
        "CostLedger(sys.argv[1], 'eval').record({'usd': float(sys.argv[2]), 'run_id': 'r'})"
    )
    for amount in ("0.5", "0.25"):
        subprocess.run([sys.executable, "-c", code, str(tmp_path), amount], check=True)  # noqa: S603
    names = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert len(names) == 2
    pids = {FILE_NAME.match(name)["pid"] for name in names}  # type: ignore[index]
    assert len(pids) == 2
    assert CostLedger(tmp_path, "reader").total() == 0.75


def test_total_tolerates_a_torn_last_line(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path, "agent")
    ledger.record(entry(0.2))
    other = tmp_path / "harness-1-deadbeef.jsonl"
    other.write_text(json.dumps(entry(0.3)) + "\n" + '{"usd": 0.4, "compo')
    assert ledger.total() == pytest.approx(0.5)
    assert len(ledger.entries()) == 2


def test_a_corrupt_line_in_the_middle_refuses_to_total(tmp_path: Path) -> None:
    (tmp_path / "harness-1-deadbeef.jsonl").write_text("garbage\n" + json.dumps(entry(0.3)) + "\n")
    with pytest.raises(LedgerError, match="line 1 is not valid JSON"):
        CostLedger(tmp_path, "agent").total()
    (tmp_path / "harness-1-deadbeef.jsonl").write_text(json.dumps({"usd": -5}) + "\n")
    with pytest.raises(LedgerError, match="no valid 'usd'"):
        CostLedger(tmp_path, "agent").total()


def test_total_can_be_filtered_by_run(ledger: CostLedger) -> None:
    ledger.record(entry(0.1, run_id="a"))
    ledger.record(entry(0.2, run_id="b"))
    ledger.record(entry(0.3, run_id="a"))
    assert ledger.total(run_id="a") == pytest.approx(0.4)
    assert ledger.total() == pytest.approx(0.6)


def test_budget_stop_before_the_cap_would_be_exceeded(ledger: CostLedger) -> None:
    ledger.record(entry(0.9))
    assert ledger.check_budget(0.05, None) == pytest.approx(0.9)
    assert ledger.check_budget(0.1, 1.0) == pytest.approx(0.9)  # lands exactly on the cap
    with pytest.raises(BudgetExceeded) as info:
        ledger.check_budget(0.11, 1.0)
    assert info.value.total_usd == pytest.approx(0.9)
    assert info.value.cap_usd == 1.0
    assert "budget stop" in str(info.value)


def test_budget_stop_once_the_total_reaches_the_cap(ledger: CostLedger) -> None:
    ledger.record(entry(1.0))
    with pytest.raises(BudgetExceeded):
        ledger.check_budget(0.0, 1.0)


def test_record_rejects_prompt_text_and_bad_amounts(ledger: CostLedger) -> None:
    with pytest.raises(ValueError, match="may not contain prompt"):
        ledger.record({**entry(0.1), "prompt": "hello"})
    with pytest.raises(ValueError, match="may not contain api_key"):
        ledger.record({**entry(0.1), "api_key": "sk-or-v1-x"})
    with pytest.raises(ValueError, match="finite numeric"):
        ledger.record({**entry(0.1), "usd": float("nan")})
    with pytest.raises(ValueError, match=">= 0"):
        ledger.record(entry(-0.1))
    assert ledger.path is None


def test_write_failure_raises_ledger_error(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    with pytest.raises(LedgerError, match="cannot write cost ledger"):
        CostLedger(blocker / "ledger", "agent").record(entry(0.1))


def test_reservations_hold_the_budget_across_ledgers_on_one_directory(tmp_path: Path) -> None:
    agent = CostLedger(tmp_path / "shared", "agent")
    persona = CostLedger(tmp_path / "shared", "persona")
    hold = agent.reserve(0.6, 1.0)
    assert agent.path is not None  # the gate proved the file is writable before any request
    with pytest.raises(BudgetExceeded, match=r"\$0\.6000 in flight"):
        persona.reserve(0.5, 1.0)
    # Another directory is another budget.
    CostLedger(tmp_path / "other", "agent").reserve(0.9, 1.0).release()
    hold.release()
    hold.release()  # idempotent
    with persona.reserve(0.5, 1.0):
        assert _IN_FLIGHT
    assert not _IN_FLIGHT


def test_reserve_counts_recorded_spend_and_refuses_at_the_cap(ledger: CostLedger) -> None:
    ledger.record(entry(0.7))
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.31, 1.0)
    ledger.reserve(0.3, 1.0).release()
    ledger.reserve(123.0, None).release()  # no cap: only the writability check
    assert not _IN_FLIGHT


def test_reserve_refuses_when_the_ledger_cannot_be_written(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    with pytest.raises(LedgerError, match="live calls are refused"):
        CostLedger(blocker / "ledger", "agent").reserve(0.0, None)
    assert not _IN_FLIGHT


def test_a_failed_write_refuses_every_later_reservation(ledger: CostLedger) -> None:
    ledger.reserve(0.0, None).release()
    assert ledger.path is not None
    ledger.path.unlink()
    ledger.path.mkdir()  # the file can no longer be appended to
    with pytest.raises(LedgerError, match="cannot write cost ledger"):
        ledger.record(entry(0.1))
    with pytest.raises(LedgerError, match="refused until the process restarts"):
        ledger.reserve(0.0, None)

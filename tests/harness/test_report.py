"""Run outputs: writing, byte-identical regeneration, loading errors, compare, and trace building details."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from fake_run import NOW, Slot, make_trace, standard_slots, write_fake_run

from booking_truth.harness.adapters import AgentReply
from booking_truth.harness.beliefs import Belief
from booking_truth.harness.grading import Grade
from booking_truth.harness.redact import find_emails, find_home_paths
from booking_truth.harness.report import (
    MANIFEST_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    TRACES_FILE,
    CompareRefused,
    RunDirError,
    compare_runs,
    fmt_pass_hat,
    fmt_rate,
    load_run,
    regenerate,
)
from booking_truth.harness.tracebuild import (
    TraceInput,
    TranscriptEntry,
    build_trace,
    claims_from_belief,
    ground_truth,
)
from booking_truth.trace.validate import trace_errors


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    out = tmp_path / "fake-run"
    write_fake_run(out, standard_slots())
    return out


def test_summary_and_report_are_derived_from_traces_and_manifest(run_dir: Path) -> None:
    summary = json.loads((run_dir / SUMMARY_FILE).read_text())
    assert summary["agents"] == ["naive", "guarded"]
    assert summary["scenarios"] == ["happy-book-host-zone", "fault-slots-500-once", "tz-ist"]
    assert summary["accounting"]["ok"]
    assert summary["valid"]
    assert summary["run"]["grading"] == "offline grading"
    naive = summary["by_agent"]["naive"]
    assert naive["false_success_rate"]["x"] == 1
    assert naive["integrity"]["wrong_time"]["x"] == 1
    guarded = summary["by_agent"]["guarded"]
    assert guarded["reruns"][0]["attempts"][0]["outcome"] == "harness_error"
    assert guarded["timezone_correct_slot"]["tz-ist"]["x"] == 0
    report = (run_dir / REPORT_FILE).read_text()
    for heading in ("## Headline", "## Fault scenarios", "## Timezone scenarios", "## Cost and latency",
                    "## Integrity violations", "## Accounting"):  # fmt: skip
        assert heading in report
    assert "`fake-run/naive/fault-slots-500-once/0/1`: false_success" in report
    assert "rerun `fault-slots-500-once` #0: attempts 1: harness_error, 2: pass" in report


def test_regeneration_is_byte_identical(run_dir: Path) -> None:
    before = {n: (run_dir / n).read_bytes() for n in (SUMMARY_FILE, REPORT_FILE)}
    for name in before:
        (run_dir / name).unlink()
    regenerate(run_dir)
    assert {n: (run_dir / n).read_bytes() for n in before} == before


def test_outputs_carry_no_address_and_no_home_path(run_dir: Path) -> None:
    for name in (SUMMARY_FILE, REPORT_FILE, TRACES_FILE, MANIFEST_FILE):
        text = (run_dir / name).read_text()
        assert find_emails(text) == [], name
        assert find_home_paths(text) == [], name
    traces = (run_dir / TRACES_FILE).read_text()
    assert "Hi, I'm [lead_email], book me" in traces
    assert "~/secret.txt" in traces


def test_every_attempt_is_kept_and_only_the_last_fills_the_slot(run_dir: Path) -> None:
    results, manifest, traces = load_run(run_dir)
    assert len(traces) == 7
    assert len(results) == 6
    rerun = [
        t for t in traces if t["task"]["id"] == "fault-slots-500-once" and t["meta"]["agent"] == "guarded"
    ]
    assert [t["meta"]["final_attempt"] for t in rerun] == [False, True]
    assert "result" not in rerun[0]["meta"]
    assert rerun[0]["ground_truth"] == {
        "outcome": "unknown",
        "checked_by": "none",
        "details": {"category": "harness_error", "reasons": rerun[0]["ground_truth"]["details"]["reasons"],
                    "integrity_violation": False},
    }  # fmt: skip
    assert manifest["run_id"] == "fake-run"


@pytest.mark.parametrize("missing", [MANIFEST_FILE, TRACES_FILE])
def test_load_run_needs_both_inputs(run_dir: Path, missing: str) -> None:
    (run_dir / missing).unlink()
    with pytest.raises(RunDirError, match=f"{missing} is missing"):
        load_run(run_dir)


def test_load_run_rejects_an_invalid_trace(run_dir: Path) -> None:
    lines = (run_dir / TRACES_FILE).read_text().splitlines()
    broken = json.loads(lines[1])
    broken["ground_truth"]["outcome"] = "maybe"
    lines[1] = json.dumps(broken)
    (run_dir / TRACES_FILE).write_text("\n".join(lines) + "\n")
    with pytest.raises(RunDirError, match="line 2"):
        load_run(run_dir)
    (run_dir / TRACES_FILE).write_text("{not json\n")
    with pytest.raises(RunDirError, match="not valid JSON Lines"):
        load_run(run_dir)


def test_compare_refuses_a_version_change_unless_allowed(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    write_fake_run(a, standard_slots())
    write_fake_run(b, standard_slots(), versions={"guarded": "guarded-v2"})
    with pytest.raises(CompareRefused, match="guarded: guarded-v1 -> guarded-v2"):
        compare_runs(a, b)
    text = compare_runs(a, b, allow_version_drift=True)
    assert "agent versions differ" in text
    assert "## naive" in text
    assert "## guarded" in text
    assert "| False-success rate |" in text
    assert "+0.0 pts" in text


def test_compare_refuses_a_run_whose_agent_version_changed_mid_run(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    write_fake_run(a, standard_slots())
    write_fake_run(b, standard_slots())
    manifest = json.loads((b / MANIFEST_FILE).read_text())
    manifest["status"] = "version_drift"
    for agent in manifest["agents"]:
        if agent["label"] == "guarded":
            agent["versions_seen"] = ["guarded-v1", "guarded-v2"]
    (b / MANIFEST_FILE).write_text(json.dumps(manifest))
    expected = r"changed within a run \(guarded in b: guarded-v1, guarded-v2\)"
    with pytest.raises(CompareRefused, match=expected):
        compare_runs(a, b)
    assert "agent versions differ" in compare_runs(a, b, allow_version_drift=True)


def test_compare_needs_a_common_agent(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    write_fake_run(a, [Slot("one", "naive", "happy-book-host-zone", ("happy",), 0, "pass")])
    write_fake_run(b, [Slot("two", "naive", "happy-book-host-zone", ("happy",), 0, "pass")])
    with pytest.raises(CompareRefused, match="no agent label in common"):
        compare_runs(a, b)


def test_rate_formatting() -> None:
    assert fmt_rate({"x": 3, "n": 120, "rate": 0.025, "ci": [0.008546, 0.071]}) == "2.5% [0.9, 7.1] (3/120)"
    assert fmt_rate({"x": 0, "n": 0, "rate": None, "ci": None}) == "n/a (0/0)"
    assert fmt_pass_hat({"value": None}) == "n/a (no scenario with k valid trials)"
    assert fmt_pass_hat({"value": 0.5, "n": 4, "ci": [0.15, 0.85]}) == "50.0% [15.0, 85.0] (4 scenarios)"


# Traces -----------------------------------------------------------------------------------------------------


def test_ground_truth_mapping() -> None:
    assert ground_truth(Grade("pass", False), probe_completed=True)["outcome"] == "success"
    assert ground_truth(Grade("goal_not_met", False), probe_completed=True)["outcome"] == "failure"
    unknown = ground_truth(Grade("harness_error", False, ["boom"]), probe_completed=False)
    assert (unknown["outcome"], unknown["checked_by"]) == ("unknown", "none")
    assert unknown["details"]["reasons"] == ["boom"]


def test_claims_come_from_the_belief() -> None:
    at = NOW + timedelta(days=4)
    assert claims_from_belief(Belief("booked", time_utc=at, offered_utc=(at,))) == [
        {"type": "booked", "subject": {"time_utc": "2026-10-05T12:00:00Z"}},
        {"type": "offered_slots", "subject": {"times": ["2026-10-05T12:00:00Z"]}},
    ]
    assert claims_from_belief(Belief("cancelled")) == [{"type": "cancelled", "subject": {"time_utc": None}}]
    assert claims_from_belief(Belief("not_booked")) == []
    assert claims_from_belief(None) == []


def test_agent_tool_steps_merge_by_time_and_orphan_results_are_dropped() -> None:
    reply = AgentReply(status=200, reply="Booked!")
    transcript = [
        TranscriptEntry("user", "A", "Book it", NOW, 1.0, "m1"),
        TranscriptEntry("agent", "A", "Booked!", NOW + timedelta(seconds=2), 2.0, "m1", reply=reply),
    ]
    tools = [
        {"i": 0, "ts": "2026-10-01T12:00:00.500Z", "kind": "tool_call", "role": "agent", "name": "book_slot",
         "args": {"slot_id": "s_1"}},
        {"i": 1, "ts": "2026-10-01T12:00:01.000Z", "kind": "tool_result", "role": "tool", "name": "book_slot",
         "ok": True, "output": {"booked": True}},
        {"i": 2, "ts": "2026-10-01T12:00:01.500Z", "kind": "tool_result", "role": "tool",
         "name": "never_called"},
        {"i": 3, "ts": "not a time", "kind": "tool_call", "role": "agent", "name": "odd"},
        {"i": 4, "ts": "2026-10-01T12:00:01.600Z", "kind": "message", "role": "agent", "content": "dup"},
    ]  # fmt: skip
    trace = build_trace(
        TraceInput(
            run_id="r", agent="a", agent_mode=None, scenario_id="s", title="t", trial=0, attempt=1,
            lead_email="x@example.com", transcript=transcript, grade=Grade("harness_error", False, ["x"]),
            belief=None, beliefs={"llm": None, "lexicon": None}, tool_steps=tools,
        )
    )  # fmt: skip
    assert trace_errors(trace) == []
    assert [(s["kind"], s["name"]) for s in trace["steps"]] == [
        ("message", None),
        ("tool_call", "book_slot"),
        ("tool_result", "book_slot"),
        ("message", None),
    ]
    assert trace["ground_truth"]["checked_by"] == "none"
    assert trace["final_claim"] == {"text": "Booked!", "claims": []}


def test_a_fake_trace_is_valid_and_redacted() -> None:
    slot = Slot("guarded", "guarded", "tz-ist", ("timezone",), 3, "pass")
    trace = make_trace("r1", slot, 2, "pass")
    assert trace_errors(trace) == []
    assert trace["trace_id"] == "r1/guarded/tz-ist/3/2"
    assert find_emails(trace) == []
    assert trace["meta"]["lead"] == "[lead_email]"


def test_messages_in_the_same_millisecond_keep_their_order_around_tool_steps() -> None:
    same = NOW.replace(microsecond=5_100)
    reply = AgentReply(status=200, reply="Here are times.")
    transcript = [
        TranscriptEntry("user", "A", "Hi", same, 1.0, "m1"),
        TranscriptEntry("agent", "A", "Here are times.", same.replace(microsecond=5_400), 2.0, reply=reply),
        TranscriptEntry("user", "A", "The first one", same.replace(microsecond=5_900), 3.0, "m2"),
        TranscriptEntry("agent", "A", "Booked.", NOW.replace(microsecond=9_000), 4.0, "m2", reply=reply),
    ]
    tools = [
        {"ts": "2026-10-01T12:00:00.005Z", "kind": "tool_call", "role": "agent", "name": "find_slots"},
        {"ts": "2026-10-01T12:00:00.005Z", "kind": "tool_result", "role": "tool", "name": "find_slots"},
        {"ts": "2026-10-01T12:00:00.007Z", "kind": "tool_call", "role": "agent", "name": "book_slot"},
    ]
    trace = build_trace(
        TraceInput(
            run_id="r", agent="a", agent_mode=None, scenario_id="s", title="t", trial=0, attempt=1,
            lead_email="x@example.com", transcript=transcript, grade=Grade("harness_error", False, ["x"]),
            belief=None, beliefs={"llm": None, "lexicon": None}, tool_steps=tools,
        )
    )  # fmt: skip
    order = [(s["role"], s["name"] or s["content"]) for s in trace["steps"]]
    assert order == [
        ("user", "Hi"),
        ("agent", "find_slots"),
        ("tool", "find_slots"),
        ("agent", "Here are times."),
        ("user", "The first one"),
        ("agent", "book_slot"),
        ("agent", "Booked."),
    ]

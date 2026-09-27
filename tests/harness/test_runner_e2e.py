"""The harness end to end and offline: the real runner drives a stub agent against an in-process sandbox.

The stub agent (``stub_agent.py``) speaks the bundled ``/v1/chat`` protocol, calls the sandbox's Cal.com API
and has no guards, so the harness faults produce the failures they are designed to expose.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from stub_agent import SANDBOX_TOKEN, StubAgent, bundled_agent, running_stub

from booking_truth.harness.adapters import AgentConfig, HttpAgentClient
from booking_truth.harness.metrics import rate
from booking_truth.harness.redact import find_emails, find_home_paths
from booking_truth.harness.report import (
    MANIFEST_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    TRACES_FILE,
    load_run,
    regenerate,
)
from booking_truth.harness.runner import (
    AgentUnderTest,
    Budget,
    BudgetStop,
    Endpoint,
    PreflightError,
    RunConfig,
    RunResult,
    run,
)
from booking_truth.harness.scenarios import Scenario, load_suite, select_scenarios
from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.serve import BackgroundServer
from booking_truth.trace.validate import iter_jsonl, trace_errors

SUITE = load_suite()
MAIN_SCENARIOS = [
    "smoke",
    "fault-slots-500-once",
    "fault-slot-taken-after-offer",
    "fault-slots-not-found",
    "fault-create-500-persistent",
    "happy-cancel",
    "happy-reschedule-move-it",
    "fault-duplicate-delivery",
    "fault-concurrent-channel",
]
EXPECTED = {
    "happy-book-berlin": "pass",
    "happy-book-host-zone": "pass",
    "happy-cancel": "pass",
    "happy-reschedule-move-it": "pass",
    "fault-slots-500-once": "pass",
    "fault-slot-taken-after-offer": "pass",
    "fault-slots-not-found": "pass",
    "fault-create-500-persistent": "pass",
    # Without message dedupe the resent pick fails on the slot the first copy booked, and that reply
    # arrives last: a booking exists but the prospect was last told the time is unavailable.
    "fault-duplicate-delivery": "unclaimed_booking",
    # Without a per-lead lock both channels book.
    "fault-concurrent-channel": "double_booking",
}


def config(agent: AgentUnderTest, scenarios: list[Scenario], out: Path | None, **options: Any) -> RunConfig:
    return RunConfig(
        agents=[agent],
        scenarios=scenarios,
        suite_scenarios=SUITE,
        k=options.pop("k", 1),
        run_id=options.pop("run_id", "test-run"),
        out_dir=out,
        settle_s=3,
        stable_s=0.15,
        hardware="test machine, 1 GB",
        sandbox_token=SANDBOX_TOKEN,
        command="booking-truth test --agent http://localhost:8000/v1/chat --sandbox http://localhost:8100",
        **options,
    )


def execute(cfg: RunConfig) -> RunResult:
    return asyncio.run(run(cfg))


@dataclass
class MainRun:
    result: RunResult
    out: Path
    stub: StubAgent
    traces: list[dict[str, Any]]

    def trace(self, scenario: str) -> dict[str, Any]:
        return next(t for t in self.traces if t["task"]["id"] == scenario and t["meta"]["final_attempt"])


@pytest.fixture(scope="module")
def main_run(sandbox_url: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[MainRun]:
    out = tmp_path_factory.mktemp("runs") / "main"
    with running_stub(sandbox_url) as (stub, base):
        agent = bundled_agent("stub", base, sandbox_url)
        result = execute(config(agent, select_scenarios(SUITE, MAIN_SCENARIOS), out))
        traces = [record for _, record in iter_jsonl(out / TRACES_FILE)]
        yield MainRun(result, out, stub, traces)


def test_every_scenario_gets_the_outcome_its_design_predicts(main_run: MainRun) -> None:
    outcomes = {r.scenario_id: r.outcome for r in main_run.result.results}
    assert outcomes == EXPECTED
    assert main_run.result.status == "complete"


def test_the_summary_accounts_for_every_slot(main_run: MainRun) -> None:
    summary = json.loads((main_run.out / SUMMARY_FILE).read_text())
    accounting = summary["accounting"]
    assert accounting["expected_slots"] == len(EXPECTED) == accounting["slots"]
    assert accounting["complete"]
    assert accounting["ok"]
    assert summary["valid"]
    stub = summary["by_agent"]["stub"]
    assert stub["modes"] == ["naive"]  # from the stub's /v1/version guards "off"
    assert stub["outcomes"]["pass"] == 8
    assert stub["integrity"]["double_booking"]["x"] == 1
    assert stub["integrity"]["unclaimed_booking"]["x"] == 1
    assert set(stub["per_fault"]) == {s for s in EXPECTED if s.startswith("fault-")}
    assert stub["per_fault"]["fault-concurrent-channel"] == {
        "no_violation": rate(0, 1),
        "pass": rate(0, 1),
        "harness_fault_not_injected": 0,
    }
    assert stub["per_fault"]["fault-slots-500-once"] == {
        "no_violation": rate(1, 1),
        "pass": rate(1, 1),
        "harness_fault_not_injected": 0,
    }
    assert summary["run"]["grading"] == "offline grading"


def test_outputs_are_the_four_files_and_the_report_says_offline_grading(main_run: MainRun) -> None:
    assert sorted(p.name for p in main_run.out.iterdir()) == sorted(
        [SUMMARY_FILE, REPORT_FILE, TRACES_FILE, MANIFEST_FILE]
    )
    report = (main_run.out / REPORT_FILE).read_text()
    assert "Offline grading." in report
    assert "offline grading" in report
    manifest = json.loads((main_run.out / MANIFEST_FILE).read_text())
    assert manifest["grading"]["mode"] == "offline"
    assert manifest["agents"][0]["agent_version"] == "stub-1"
    assert manifest["agents"][0]["source_hash"] == "stub-source-hash"
    assert manifest["agents"][0]["calendar"] == "calcom"
    assert manifest["scenario_suite_hash"].startswith("sha256:")
    assert manifest["hardware"] == "test machine, 1 GB"
    assert manifest["suite"] == "bundled"
    assert manifest["llm_calls"]["agent:stub"]["models_returned"] == ["stub-model"]
    assert manifest["llm_calls"]["agent:stub"]["providers"] == ["stub-provider"]
    assert manifest["llm_calls"]["agent:stub"]["varied"] is False
    assert manifest["llm_calls"]["agent:stub"]["calls"] > 0


def test_every_trace_validates_against_the_schema(main_run: MainRun) -> None:
    assert len(main_run.traces) == len(EXPECTED)
    for record in main_run.traces:
        assert trace_errors(record) == [], record["trace_id"]
        assert record["task"]["domain"] == "booking"
        assert record["source"] == "booking-truth/0.1.0"
        assert record["trace_id"] == f"test-run/stub/{record['task']['id']}/0/1"
        probe = record["steps"][-1]
        assert (probe["kind"], probe["role"], probe["name"]) == (
            "state_probe",
            "environment",
            "sandbox_state",
        )
        assert record["ground_truth"]["checked_by"] == "state_probe"
        assert record["meta"]["claims_source"] == "lexicon"


def test_ground_truth_and_claims_follow_the_grade(main_run: MainRun) -> None:
    passed = main_run.trace("happy-book-host-zone")
    assert passed["ground_truth"]["outcome"] == "success"
    assert passed["ground_truth"]["details"]["category"] == "pass"
    claims = {c["type"]: c["subject"] for c in passed["final_claim"]["claims"]}
    booked_at = claims["booked"]["time_utc"]
    assert booked_at in claims["offered_slots"]["times"]
    assert passed["final_claim"]["text"] == "You're welcome. Talk soon."
    failed = main_run.trace("fault-concurrent-channel")
    assert failed["ground_truth"]["outcome"] == "failure"
    assert failed["ground_truth"]["details"]["category"] == "double_booking"
    assert failed["ground_truth"]["details"]["integrity_violation"] is True


def test_bundled_agent_tool_calls_are_merged_from_its_session_trace(main_run: MainRun) -> None:
    steps = main_run.trace("happy-book-host-zone")["steps"]
    kinds = [(s["kind"], s["name"]) for s in steps]
    assert ("tool_call", "get /v2/slots") in kinds
    assert ("tool_result", "post /v2/bookings") in kinds
    assert [s["i"] for s in steps] == list(range(len(steps)))
    first_call = kinds.index(("tool_call", "get /v2/slots"))
    assert steps[first_call - 1]["role"] == "user"


def test_messages_keep_their_conversation_order(main_run: MainRun) -> None:
    ordered = ("happy-book-host-zone", "fault-slots-500-once", "happy-reschedule-move-it", "happy-cancel")
    for scenario in ordered:
        steps = main_run.trace(scenario)["steps"]
        roles = [s["role"] for s in steps if s["kind"] == "message"]
        assert roles == ["user", "agent"] * (len(roles) // 2), scenario
        texts = [s["content"] for s in steps if s["kind"] == "message" and s["role"] == "user"]
        assert len(texts) == main_run.trace(scenario)["meta"]["persona_turns"]


def test_no_email_address_or_home_path_survives_in_the_outputs(main_run: MainRun) -> None:
    home_prefix = "/" + "Users/"
    for path in main_run.out.iterdir():
        text = path.read_text(encoding="utf-8")
        assert find_emails(text) == [], path.name
        assert find_home_paths(text) == [], path.name
        assert "example.com" not in text, path.name
        assert home_prefix not in text, path.name
        assert str(main_run.out.parent) not in text, path.name
    assert "[lead_email]" in (main_run.out / TRACES_FILE).read_text()


def test_report_regenerates_both_files_byte_for_byte(main_run: MainRun) -> None:
    before = {name: (main_run.out / name).read_bytes() for name in (SUMMARY_FILE, REPORT_FILE)}
    (main_run.out / SUMMARY_FILE).unlink()
    (main_run.out / REPORT_FILE).write_text("stale", encoding="utf-8")
    regenerate(main_run.out)
    after = {name: (main_run.out / name).read_bytes() for name in (SUMMARY_FILE, REPORT_FILE)}
    assert after == before


def test_duplicate_delivery_resends_the_confirming_message(main_run: MainRun) -> None:
    trace = main_run.trace("fault-duplicate-delivery")
    fault = trace["meta"]["harness_fault"]
    assert fault["type"] == "duplicate_delivery"
    sent = [r for r in main_run.stub.requests if r.message_id == fault["message_id"]]
    assert len(sent) == 2
    assert sent[0].body == sent[1].body
    assert sent[0].body["action"]["type"] == "select_slot"
    assert 0.04 <= sent[1].at - sent[0].at <= 0.3
    assert 0.05 <= fault["delay_s"] <= 0.2
    replies = [s for s in trace["steps"] if s["kind"] == "message" and s["role"] == "agent"]
    duplicates = [s for s in replies if s["args"].get("duplicate")]
    assert len(duplicates) == 1
    assert fault["continued_from"] in ("original", "duplicate")


def test_concurrent_channel_asks_for_the_second_offer_on_a_new_webhook_session(main_run: MainRun) -> None:
    trace = main_run.trace("fault-concurrent-channel")
    fault = trace["meta"]["harness_fault"]
    assert fault["type"] == "concurrent_channel"
    assert fault["channel_b"] == "webhook"
    session_a, session_b = trace["meta"]["sessions"]
    webhook = [r for r in main_run.stub.requests if r.channel == "webhook"]
    assert [r.session_id for r in webhook] == [session_b]
    pick_a = next(r for r in main_run.stub.requests if r.session_id == session_a and r.body.get("action"))
    assert webhook[0].body["lead"]["email"] == pick_a.body["lead"]["email"]
    assert abs(webhook[0].at - pick_a.at) < 0.1
    offers_step = next(
        s
        for s in trace["steps"]
        if s["kind"] == "message" and s["role"] == "agent" and (s.get("output") or {}).get("quick_replies")
    )
    second_label = offers_step["output"]["quick_replies"][1]["label"]
    assert webhook[0].body["message"] == f"Please book {second_label} for me instead." == fault["message_b"]
    assert pick_a.body["action"]["slot_id"] == offers_step["output"]["quick_replies"][0]["action"]["slot_id"]
    sessions = {s["args"]["session"] for s in trace["steps"] if s["kind"] == "message"}
    assert sessions == {"A", "B"}


def test_setup_bookings_let_reschedule_and_cancel_pass(main_run: MainRun) -> None:
    moved = main_run.trace("happy-reschedule-move-it")
    assert moved["meta"]["setup"] is not None
    grade = moved["meta"]["grade"]
    assert [b["moved_in_trial"] or b["created_in_trial"] for b in grade["bookings"] if b["active"]] == [True]
    cancelled = main_run.trace("happy-cancel")
    assert cancelled["meta"]["beliefs"]["lexicon"]["status"] == "cancelled"


# Separate runs --------------------------------------------------------------------------------------------


def smoke(scenario_id: str = "happy-book-host-zone") -> list[Scenario]:
    return select_scenarios(SUITE, [scenario_id])


def test_an_agent_not_wired_to_the_sandbox_aborts_the_run(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url, wired=False) as (_, base):
        agent = bundled_agent("unwired", base, sandbox_url)
        with pytest.raises(PreflightError) as caught:
            execute(config(agent, smoke(), tmp_path / "out"))
    assert caught.value.code == "agent_not_wired_to_sandbox"
    assert "logged no successful slots or freeBusy call" in str(caught.value)
    assert not (tmp_path / "out").exists()


def test_an_agent_returning_500_is_graded_agent_error(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url, fail_from_request=2) as (_, base):
        result = execute(config(bundled_agent("broken", base, sandbox_url), smoke(), tmp_path / "out"))
    [trial] = result.results
    assert trial.outcome == "agent_error"
    assert len(trial.attempts) == 1
    summary = result.summary or {}
    assert summary["by_agent"]["broken"]["agent_errors"] == [trial.trace_id]
    [trace] = result.traces
    agent_steps = [s for s in trace["steps"] if s["kind"] == "message" and s["role"] == "agent"]
    assert agent_steps[-1]["ok"] is False
    assert agent_steps[-1]["error"] == "http_500"
    assert trace["meta"]["agent_error"] == "http_500 (status 500)"


def test_the_persona_stops_at_14_turns(sandbox_url: str, tmp_path: Path) -> None:
    base_scenario = next(s for s in SUITE if s.id == "happy-book-host-zone")
    raw = base_scenario.model_dump(mode="json", exclude_unset=True)
    raw["id"] = "chatty-prospect"
    raw["persona"]["script"] = [{"say": f"Tell me more about the call, part {n}."} for n in range(1, 17)]
    scenario = Scenario.model_validate(raw)
    with running_stub(sandbox_url) as (stub, base):
        result = execute(config(bundled_agent("stub", base, sandbox_url), [scenario], tmp_path / "out"))
    [trial] = result.results
    assert trial.turn_cap_hit
    assert trial.outcome == "goal_not_met"
    [trace] = result.traces
    assert trace["meta"]["persona_turns"] == 14
    user = [s for s in trace["steps"] if s["kind"] == "message" and s["role"] == "user"]
    assert len(user) == 14
    assert (result.summary or {})["by_agent"]["stub"]["turn_cap_hits"] == 1
    assert len([r for r in stub.requests if r.path == "/v1/chat"]) == 15  # preflight + 14


def test_an_out_of_window_pick_after_the_turn_cap_is_not_a_persona_error(
    sandbox_url: str, tmp_path: Path
) -> None:
    base_scenario = next(s for s in SUITE if s.id == "happy-book-host-zone")
    raw = base_scenario.model_dump(mode="json", exclude_unset=True)
    raw["id"] = "chatty-then-picks"
    # Fourteen lines, then an out-of-window pick (the stub offers mornings, the window is afternoons) that
    # the cap keeps from being sent.
    raw["persona"]["script"] = [
        {"say": f"Can I book a call in the morning? Question {n}."} for n in range(1, 15)
    ]
    raw["persona"]["script"].append({"pick": "offered[0]"})
    scenario = Scenario.model_validate(raw)
    with running_stub(sandbox_url) as (_, base):
        result = execute(config(bundled_agent("stub", base, sandbox_url), [scenario], tmp_path / "out"))
    [trial] = result.results
    assert trial.turn_cap_hit
    assert [(a.outcome, a.persona_error) for a in trial.attempts] == [("goal_not_met", False)]


def test_a_crash_outside_the_attempt_still_fills_the_slot(
    sandbox_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import booking_truth.harness.runner as runner_module

    real = runner_module.build_trace

    def fragile(inp: Any) -> dict[str, Any]:
        if inp.transcript:
            raise RuntimeError("trace builder broke")
        return real(inp)

    monkeypatch.setattr(runner_module, "build_trace", fragile)
    with running_stub(sandbox_url) as (_, base):
        result = execute(config(bundled_agent("stub", base, sandbox_url), smoke(), tmp_path / "out"))
    [trial] = result.results
    assert trial.outcome == "harness_error"
    assert [a.outcome for a in trial.attempts] == ["harness_error"] * 3
    assert len(result.traces) == 3
    final = result.traces[-1]
    assert final["meta"]["crashed"] is True
    assert "RuntimeError: trace builder broke" in final["meta"]["harness_error"]
    assert 'File "booking_truth/harness/runner.py"' in final["meta"]["harness_error"]
    assert final["ground_truth"]["outcome"] == "unknown"
    summary = result.summary or {}
    assert summary["accounting"]["slots"] == 1
    assert summary["by_agent"]["stub"]["harness_errors"][0]["trace_id"] == trial.trace_id


def test_a_persona_error_is_a_harness_error_rerun_twice(sandbox_url: str, tmp_path: Path) -> None:
    base_scenario = next(s for s in SUITE if s.id == "happy-book-host-zone")
    raw = base_scenario.model_dump(mode="json", exclude_unset=True)
    raw["id"] = "accepts-anything"
    # The stub offers mornings first; the window is afternoons, so accepting offered[0] is out of window.
    raw["persona"]["script"] = [{"say": "Hi, can I book a call in the morning?"}, {"pick": "offered[0]"}]
    scenario = Scenario.model_validate(raw)
    with running_stub(sandbox_url) as (_, base):
        result = execute(config(bundled_agent("stub", base, sandbox_url), [scenario], tmp_path / "out"))
    [trial] = result.results
    assert trial.outcome == "harness_error"
    assert [(a.attempt, a.outcome, a.persona_error) for a in trial.attempts] == [
        (1, "harness_error", True),
        (2, "harness_error", True),
        (3, "harness_error", True),
    ]
    assert len(result.traces) == 3
    assert [t["meta"]["final_attempt"] for t in result.traces] == [False, False, True]
    assert result.traces[0]["ground_truth"] == {
        "outcome": "unknown",
        "checked_by": "none",
        "details": result.traces[0]["ground_truth"]["details"],
    }
    summary = result.summary or {}
    assert summary["accounting"]["invalid"]
    assert not summary["valid"]
    assert summary["by_agent"]["stub"]["persona_error_rate"]["x"] == 3
    assert summary["by_agent"]["stub"]["harness_errors"][0]["trace_id"] == trial.trace_id
    loaded, _, traces = load_run(tmp_path / "out")
    assert loaded == result.results
    assert len(traces) == 3


def test_a_version_change_during_the_run_aborts_it(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url, version_after=(4, "stub-2")) as (_, base):
        result = execute(
            config(
                bundled_agent("stub", base, sandbox_url), select_scenarios(SUITE, ["smoke"]), tmp_path / "out"
            )
        )
    assert result.status == "version_drift"
    assert "stub-1 to stub-2" in (result.detail or "")
    manifest = json.loads((tmp_path / "out" / MANIFEST_FILE).read_text())
    assert manifest["status"] == "version_drift"
    assert manifest["agents"][0]["versions_seen"] == ["stub-1", "stub-2"]
    summary = result.summary or {}
    assert summary["accounting"]["missing"]
    assert not summary["valid"]


def test_a_version_drift_keeps_the_interrupted_trial_in_the_spend(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url, version_after=(4, "stub-2"), usage_usd=0.01) as (stub, base):
        result = execute(
            config(
                bundled_agent("stub", base, sandbox_url), select_scenarios(SUITE, ["smoke"]), tmp_path / "out"
            )
        )
        chats = len([r for r in stub.requests if r.path == "/v1/chat"])
    assert result.status == "version_drift"
    assert result.manifest["spend"]["agent_usd"] == pytest.approx(0.01 * chats)
    assert sum(r.agent_usd for r in result.results) < result.manifest["spend"]["agent_usd"] - 0.01


def test_the_run_stops_before_a_trial_would_pass_the_budget(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url, usage_usd=0.01) as (_, base):
        result = execute(
            config(
                bundled_agent("stub", base, sandbox_url),
                select_scenarios(SUITE, ["smoke"]),
                tmp_path / "out",
                k=2,
                budget_usd=0.05,
            )
        )
    assert result.status == "budget_stop"
    assert len(result.results) == 1
    spend = result.manifest["spend"]
    assert spend["agent_usd"] == pytest.approx(0.04)  # the preflight turn plus the first trial
    assert spend["total_usd"] <= 0.05
    assert result.results[0].agent_usd == pytest.approx(0.03)
    assert "--budget-usd" in (result.detail or "")


def test_the_budget_counts_the_preflight_spend() -> None:
    budget = Budget(0.01, None, None)
    budget.admit()  # nothing spent yet, no projection: the first trial may start
    budget.release()
    budget.overhead(agent=0.02)  # the preflight turns alone pass the budget
    with pytest.raises(BudgetStop, match="--budget-usd"):
        budget.admit()


def test_the_ledger_cap_stops_the_run_too(sandbox_url: str, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "agent-1-deadbeef.jsonl").write_text(json.dumps({"usd": 1.0}) + "\n", encoding="utf-8")
    with running_stub(sandbox_url) as (_, base):
        result = execute(
            config(
                bundled_agent("stub", base, sandbox_url),
                smoke(),
                tmp_path / "out",
                ledger_cap_usd=1.0,
                ledger_dir=ledger,
            )
        )
    assert result.status == "budget_stop"
    assert result.results == []
    assert "BT_BUDGET_USD" in (result.detail or "")


def test_dry_run_projects_the_full_run_with_a_safety_factor(sandbox_url: str, tmp_path: Path) -> None:
    selected = select_scenarios(SUITE, ["smoke", "fault-slots-500-once", "fault-crm-500-once"])
    with running_stub(sandbox_url, usage_usd=0.002) as (_, base):
        result = execute(
            config(bundled_agent("stub", base, sandbox_url), selected, tmp_path / "out", k=5, dry_run=True)
        )
    assert [r.scenario_id for r in result.results] == ["happy-book-berlin", "fault-crm-500-once"]
    projection = result.projection or {}
    per_trial = projection["per_agent"]["stub"]["usd_per_trial"]
    assert per_trial > 0
    assert projection["full_run"] == {"scenarios": 4, "k": 5, "agents": 1}
    assert projection["projected_usd"] == pytest.approx(per_trial * 4 * 5, rel=1e-4)
    assert projection["projected_usd_with_safety"] == pytest.approx(
        projection["projected_usd"] * 1.3, rel=1e-4
    )
    assert result.manifest["dry_run"] is True
    assert result.manifest["k"] == 1
    assert "Projection of the full run" in (tmp_path / "out" / REPORT_FILE).read_text()


def test_dry_run_falls_back_to_the_suite_for_a_fault_scenario(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url) as (_, base):
        result = execute(config(bundled_agent("stub", base, sandbox_url), smoke(), None, dry_run=True, k=5))
    assert [r.scenario_id for r in result.results] == ["happy-book-host-zone", "fault-commit-then-timeout"]
    assert (result.projection or {})["full_run"]["scenarios"] == 1


def test_a_generic_webhook_agent_with_cookie_sessions(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url) as (stub, base):
        agent_config = AgentConfig.model_validate(
            {
                "url": f"{base}/webhook",
                "body": {"text": "{{message}}", "email": "{{lead.email}}", "name": "{{lead.name}}"},
                "response": {"reply_path": "data.messages[0].text", "version_path": "meta.version"},
                "session_mode": "cookie",
                "timeout_s": 10,
            }
        )

        def client() -> HttpAgentClient:
            return HttpAgentClient(agent_config)

        endpoint = Endpoint(name="webhook", sandbox_url=sandbox_url, make_client=client)
        agent = AgentUnderTest(label="webhook", endpoints=[endpoint], protocol="agent.yaml", target="webhook")
        result = execute(config(agent, select_scenarios(SUITE, ["smoke"]), tmp_path / "out"))
    assert [r.outcome for r in result.results] == ["pass", "pass"]
    picks = [
        r.body["text"] for r in stub.requests if r.path == "/webhook" and "works for me" in r.body["text"]
    ]
    assert len(picks) == 2
    assert result.manifest["agents"][0]["protocol"] == "agent.yaml"
    assert result.manifest["agents"][0]["agent_version"] == "stub-1"
    for trace in result.traces:
        assert trace["meta"]["agent_trace_steps"] == 0


def test_a_pool_runs_trials_in_parallel_one_per_sandbox(sandbox_url: str, tmp_path: Path) -> None:
    second = BackgroundServer(create_sandbox_app(SANDBOX_TOKEN)).start()
    try:
        with running_stub(sandbox_url) as (stub_a, base_a), running_stub(second.url) as (stub_b, base_b):
            one = bundled_agent("pool", base_a, sandbox_url).endpoints[0]
            two = bundled_agent("pool", base_b, second.url).endpoints[0]
            agent = AgentUnderTest(label="pool", endpoints=[one, two], target="stub")
            result = execute(config(agent, select_scenarios(SUITE, ["smoke"]), tmp_path / "out", k=2))
    finally:
        second.stop()
    assert [r.outcome for r in result.results] == ["pass"] * 4
    assert [(r.scenario_id, r.trial) for r in result.results] == [
        ("happy-book-berlin", 0),
        ("happy-book-berlin", 1),
        ("happy-book-host-zone", 0),
        ("happy-book-host-zone", 1),
    ]
    assert len(stub_a.sessions) >= 2
    assert len(stub_b.sessions) >= 2
    assert result.manifest["agents"][0]["endpoints"] == 2


def test_a_google_calendar_agent_is_detected_and_graded_on_events(sandbox_url: str, tmp_path: Path) -> None:
    with running_stub(sandbox_url, calendar="google") as (_, base):
        result = execute(config(bundled_agent("google", base, sandbox_url), smoke(), tmp_path / "out"))
    [trial] = result.results
    assert trial.outcome == "pass"
    assert result.manifest["agents"][0]["calendar"] == "google"
    [trace] = result.traces
    probe = trace["steps"][-1]["output"]
    assert probe["calendar"] == "google"
    assert len(probe["google_events"]) == 1
    assert probe["reference_slots_total"] > 0
    assert trace["meta"]["grade"]["bookings"][0]["status"] == "confirmed"


def test_run_single_grades_one_attempt_for_fixture_runners(sandbox_url: str) -> None:
    from booking_truth.harness.runner import run_single

    scenario = smoke()[0]
    with running_stub(sandbox_url) as (_, base):
        agent = bundled_agent("stub", base, sandbox_url)
        attempt = asyncio.run(run_single(agent, scenario, config=config(agent, [scenario], None)))
    assert attempt.outcome == "pass"
    assert attempt.trace_id == "test-run/stub/happy-book-host-zone/0/1"
    assert attempt.lexicon_belief is not None
    assert attempt.lexicon_belief.status == "booked"
    assert trace_errors(attempt.trace) == []

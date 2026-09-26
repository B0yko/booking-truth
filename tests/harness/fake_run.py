"""Build a run directory without any server: synthetic trials graded by hand, traced and written for real."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from booking_truth.harness.adapters import AgentReply
from booking_truth.harness.beliefs import Belief, BeliefStatus
from booking_truth.harness.grading import INTEGRITY_OUTCOMES, Grade
from booking_truth.harness.manifest import AgentManifest, CallRecorder, build_manifest
from booking_truth.harness.metrics import AttemptRecord, TrialResult
from booking_truth.harness.report import write_run
from booking_truth.harness.sandbox_client import Settled
from booking_truth.harness.tracebuild import TraceInput, TranscriptEntry, build_trace, finalize, trace_id

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
BOOKED_AT = datetime(2026, 10, 5, 18, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Slot:
    agent: str
    mode: str
    scenario: str
    tags: tuple[str, ...]
    trial: int
    outcome: str
    belief: BeliefStatus = "booked"
    llm_belief: BeliefStatus | None = None
    agent_usd: float = 0.001
    attempts: tuple[str, ...] = ()


def lead_email(slot: Slot) -> str:
    return f"{slot.scenario}-{slot.trial:08d}@example.com"


def make_trace(run_id: str, slot: Slot, attempt: int, outcome: str) -> dict[str, Any]:
    lead = lead_email(slot)
    belief = Belief(status=slot.belief, time_utc=BOOKED_AT, offered_utc=(BOOKED_AT,), source="lexicon")
    reply = AgentReply(status=200, reply=f"You're all set, {lead}!", latency_s=0.5, usage_usd=slot.agent_usd)
    transcript = [
        TranscriptEntry("user", "A", f"Hi, I'm {lead}, book me Monday at 2 pm.", NOW, 1.0, "m1"),
        TranscriptEntry("agent", "A", reply.reply or "", NOW + timedelta(seconds=1), 2.0, "m1", reply=reply),
    ]
    harness = outcome == "harness_error"
    grade = Grade(
        outcome,  # type: ignore[arg-type]
        outcome in INTEGRITY_OUTCOMES,
        [] if outcome == "pass" else [f"{outcome} for {lead}"],
        {}
        if harness
        else {"matched": [] if outcome == "pass" else [outcome], "correct_slot": outcome == "pass"},
    )
    state = {
        "now": "2026-10-01T12:00:05Z",
        "seed": {"event_type_id": 1001},
        "calcom": {
            "bookings": [{"uid": "u1", "status": "accepted", "attendees": [{"email": lead}], "start": "x"}]
        },
        "request_log": [],
        "faults": [],
    }
    settled = None if harness else Settled(state=state, settled=True, waited_s=1.2, polls=4, backlog=0)
    return build_trace(
        TraceInput(
            run_id=run_id,
            agent=slot.agent,
            agent_mode=slot.mode,
            scenario_id=slot.scenario,
            title=f"Scenario {slot.scenario}",
            trial=slot.trial,
            attempt=attempt,
            lead_email=lead,
            transcript=transcript,
            grade=grade,
            belief=None if harness else belief,
            beliefs={"llm": None, "lexicon": None if harness else belief},
            settled=settled,
            probe_ts=None if harness else NOW + timedelta(seconds=3),
            calendar="calcom",
            meta={"scenario_tags": list(slot.tags), "lead": lead, "note": "/" + "Users/someone/secret.txt"},
        )
    )


def write_fake_run(
    out: Path,
    slots: list[Slot],
    *,
    run_id: str = "fake-run",
    k: int = 1,
    versions: dict[str, str] | None = None,
    grading: str = "offline",
) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    for slot in slots:
        outcomes = [*slot.attempts, slot.outcome]
        attempts = tuple(
            AttemptRecord(i, o, False, trace_id(run_id, slot.agent, slot.scenario, slot.trial, i))
            for i, o in enumerate(outcomes, start=1)
        )
        result = TrialResult(
            agent=slot.agent,
            scenario_id=slot.scenario,
            trial=slot.trial,
            outcome=slot.outcome,
            trace_id=attempts[-1].trace_id or "",
            scenario_tags=slot.tags,
            agent_mode=slot.mode,
            belief_status=slot.belief,
            llm_belief_status=slot.llm_belief,
            lexicon_belief_status=slot.belief,
            correct_slot=slot.outcome == "pass",
            attempts=attempts,
            turn_latencies_s=(0.5, 0.7),
            conversation_latency_s=1.4,
            agent_usd=slot.agent_usd,
            turns=2,
            guard_turns=1 if slot.mode == "guarded" else 0,
        )
        for number, outcome in enumerate(outcomes, start=1):
            final = number == len(outcomes)
            traces.append(
                finalize(
                    make_trace(run_id, slot, number, outcome),
                    final=final,
                    result=result.to_json() if final else None,
                )
            )
    agents: dict[str, str] = {}
    for slot in slots:
        agents.setdefault(slot.agent, slot.mode)
    scenarios = list(dict.fromkeys(s.scenario for s in slots))
    versions = versions or {}
    manifest = build_manifest(
        run_id=run_id,
        date="2026-10-01",
        as_of=None,
        hardware="MacBook Air M5, 24 GB",
        suite="bundled",
        suite_digest="sha256:" + "ab" * 32,
        scenarios=scenarios,
        k=k,
        agents=[
            AgentManifest(
                label=label,
                kind="http",
                mode=mode,
                protocol="bundled",
                target="http://localhost:8000/v1/chat",
                calendar="calcom",
                agent_version=versions.get(label, f"{label}-v1"),
                version_info={"model": "vendor/model-a", "source_hash": "f" * 64},
            )
            for label, mode in agents.items()
        ],
        grading={
            "mode": grading,
            "label": "offline grading" if grading == "offline" else "LLM grading",
            "reason": "no LLM key configured",
            "persona": "scripted",
            "extractor": "lexicon",
        },
        models={"agents": {label: "vendor/model-a" for label in agents}},
        temperatures={"agents": {label: 0.2 for label in agents}},
        calls=CallRecorder(),
        spend={
            "agent_usd": sum(s.agent_usd for s in slots),
            "persona_usd": 0.0,
            "extractor_usd": 0.0,
            "total_usd": sum(s.agent_usd for s in slots),
        },
        status="complete",
        status_detail=None,
        options={},
        command="booking-truth test --pool pool.yaml --k 1",
        dry_run=False,
        git={"sha": "0" * 40, "dirty": False},
    )
    return write_run(out, traces, manifest)


def standard_slots() -> list[Slot]:
    """A small two-agent run: naive fails some integrity checks, guarded passes but for one violation."""
    fault = ("fault",)
    tz = ("timezone",)
    return [
        Slot("naive", "naive", "happy-book-host-zone", ("happy", "smoke"), 0, "pass"),
        Slot("naive", "naive", "fault-slots-500-once", fault, 0, "false_success", llm_belief="not_booked"),
        Slot("naive", "naive", "tz-ist", tz, 0, "wrong_time"),
        Slot("guarded", "guarded", "happy-book-host-zone", ("happy", "smoke"), 0, "pass"),
        Slot("guarded", "guarded", "fault-slots-500-once", fault, 0, "pass", attempts=("harness_error",)),
        Slot("guarded", "guarded", "tz-ist", tz, 0, "time_mismatch", belief="booked"),
    ]

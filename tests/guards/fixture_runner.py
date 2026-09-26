"""Guard fixtures: one YAML file per guard in ``tests/fixtures/guards/``, run through the real harness.

A fixture names a guard, an inline scenario (the schema of ``scenarios/*.yaml``), the scripted model's
misbehaviours and two expectations. The scenario runs twice against the bundled agent, each time on a fresh
in-process sandbox with an agent built by ``create_agent_app(settings, llm=FakeLLM(misbehaviours))``:

- ``expect_on``: every guard enabled;
- ``expect_off``: every guard except the fixture's own (and the guards that depend on it).

An expectation is an ``outcome`` (or ``outcome_in``), or ``config_error: true`` when the agent must refuse
to start, plus behavioural checks: ``assert`` (each must hold) and ``assert_not`` (each must fail).

Checks (one key each)::

    llm_calls: {eq|ne|lt|lte|gt|gte: N}         model calls during the conversation
    handoffs: {...}                              hand-off rows in the agent's store
    ledger: {status?, action?, count: {...}}     claims ledger entries
    active_bookings: {...}                       the lead's active bookings at the end
    crm_meetings: {...}                          HubSpot meetings at the end
    lead_busy: {...}                             409 lead_busy replies
    reply_matches: <regex>                       some agent reply matches (case-insensitive)
    final_reply_matches: <regex>                 the last agent reply of the prospect's session matches
    guard_event: {guard: ..., event: ...}        some response reported this guard event
    duplicate_responses_identical: true          the replies to a duplicated message are identical
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from booking_truth.agent.api import create_agent_app
from booking_truth.agent.guards import GUARD_NAMES, all_except, guards_string
from booking_truth.agent.scripted import MISBEHAVIOURS, FakeLLM
from booking_truth.config import Settings
from booking_truth.harness.adapters import BundledAgentClient, BundledEndpoints
from booking_truth.harness.builtin import BuiltinUnavailable, start_builtin_agent
from booking_truth.harness.grading import CalendarKind
from booking_truth.harness.runner import AgentUnderTest, Attempt, Endpoint, RunConfig, run_single
from booking_truth.harness.scenarios import Scenario
from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.serve import BackgroundServer
from booking_truth.store import Store

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "guards"
Mode = Literal["on", "off"]
COMPARISONS: dict[str, Callable[[int, int], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}
CHECKS = frozenset(
    {
        "llm_calls",
        "handoffs",
        "ledger",
        "active_bookings",
        "crm_meetings",
        "lead_busy",
        "reply_matches",
        "final_reply_matches",
        "guard_event",
        "duplicate_responses_identical",
    }
)


class FixtureError(ValueError):
    pass


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    outcome: str | None = None
    outcome_in: list[str] | None = None
    config_error: bool = False
    checks: list[dict[str, Any]] = Field(default_factory=list, alias="assert")
    negated: list[dict[str, Any]] = Field(default_factory=list, alias="assert_not")

    @model_validator(mode="after")
    def _shape(self) -> Expectation:
        if self.config_error and (self.outcome or self.outcome_in or self.checks or self.negated):
            raise ValueError("config_error: true stands alone")
        if not self.config_error and not (self.outcome or self.outcome_in):
            raise ValueError("an expectation needs outcome, outcome_in or config_error")
        if self.outcome and self.outcome_in:
            raise ValueError("use outcome or outcome_in, not both")
        for check in [*self.checks, *self.negated]:
            if len(check) != 1 or next(iter(check)) not in CHECKS:
                raise ValueError(f"a check is one of {sorted(CHECKS)}, got {check}")
        return self


class GuardFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guard: str
    title: str = Field(min_length=1)
    calendar: Literal["calcom", "google", "both"] = "both"
    misbehaviours: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    grade_crm: bool = False
    scenario: Scenario
    expect_on: Expectation
    expect_off: Expectation

    @model_validator(mode="after")
    def _known(self) -> GuardFixture:
        if self.guard not in GUARD_NAMES:
            raise ValueError(f"unknown guard {self.guard!r}")
        unknown = sorted(set(self.misbehaviours) - MISBEHAVIOURS)
        if unknown:
            raise ValueError(f"unknown misbehaviour(s): {unknown}")
        return self


def load_fixture(path: Path) -> GuardFixture:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FixtureError(f"{path.name}: expected a mapping")
    return GuardFixture.model_validate(raw)


def fixture_files(directory: Path = FIXTURES_DIR) -> list[Path]:
    return sorted(directory.glob("*.yaml")) if directory.is_dir() else []


def resolved_calendars(fixture: GuardFixture) -> tuple[CalendarKind, ...]:
    """The calendar shape(s) ``fixture`` runs on: both, unless it names one specifically."""
    return ("calcom", "google") if fixture.calendar == "both" else (fixture.calendar,)


# Running ---------------------------------------------------------------------------------------------------


@dataclass
class Observation:
    """What one run showed: the grade and the agent's own state."""

    mode: Mode
    calendar: CalendarKind = "calcom"
    config_error: str | None = None
    outcome: str | None = None
    attempt: Attempt | None = None
    llm_calls: int = 0
    handoffs: int = 0
    ledger: list[dict[str, Any]] = field(default_factory=list)
    agent_steps: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    lead: str = ""


def configured_guards(fixture: GuardFixture, mode: Mode) -> str:
    return "all" if mode == "on" else guards_string(all_except(fixture.guard))


async def run_fixture(fixture: GuardFixture, mode: Mode, calendar: CalendarKind = "calcom") -> Observation:
    """Run the fixture's scenario once with the guard on (every guard) or off (every guard but it), on
    ``calendar``. The scenario's own fault groups are Cal.com-shaped; the sandbox client that seeds them
    (:mod:`booking_truth.harness.sandbox_client`) gives every rule its Google twin, so one fixture serves
    both calendars unchanged."""
    token = secrets.token_urlsafe(12)
    sandbox = BackgroundServer(create_sandbox_app(token)).start()
    llm = FakeLLM(fixture.misbehaviours)
    apps: list[FastAPI] = []

    def factory(settings: Settings, *, llm: FakeLLM | None = None) -> FastAPI:
        configured = settings.model_copy(update=dict(fixture.settings)) if fixture.settings else settings
        app = create_agent_app(configured, llm=llm)
        apps.append(app)
        return app

    try:
        try:
            agent = start_builtin_agent(
                "guarded",
                sandbox_url=sandbox.url,
                sandbox_token=token,
                factory=factory,
                guards=configured_guards(fixture, mode),
                llm=llm,
                calendar=calendar,
            )
        except BuiltinUnavailable as exc:
            return Observation(mode=mode, calendar=calendar, config_error=str(exc))
        try:
            endpoint = Endpoint(
                name=f"{fixture.guard}-{mode}-{calendar}",
                sandbox_url=sandbox.url,
                make_client=lambda: BundledAgentClient(agent.url, api_key=agent.api_key),
                side=BundledEndpoints(agent.url, agent.api_key),
                calendar=calendar,
            )
            under_test = AgentUnderTest(
                label=f"{fixture.guard}-{mode}-{calendar}",
                endpoints=[endpoint],
                kind="builtin",
                target="builtin",
                mode="guarded" if mode == "on" else "naive",
            )
            config = RunConfig(
                agents=[under_test],
                scenarios=[fixture.scenario],
                k=1,
                grade_crm=fixture.grade_crm,
                settle_s=5.0,
                stable_s=0.3,
                sandbox_token=token,
            )
            before = llm.calls
            attempt = await run_single(under_test, fixture.scenario, config=config)
            observation = Observation(
                mode=mode,
                calendar=calendar,
                outcome=attempt.outcome,
                attempt=attempt,
                llm_calls=llm.calls - before,
            )
            store: Store = apps[0].state.deps.store
            observation.handoffs = len(store.handoffs.items(limit=1000))
            observation.lead = lead_email_of(store, attempt)
            steps = attempt.trace["steps"]
            observation.agent_steps = [s for s in steps if s["kind"] == "message" and s["role"] == "agent"]
            probes = [s for s in steps if s["kind"] == "state_probe"]
            observation.state = probes[-1]["output"] if probes else {}
            if observation.lead:
                observation.ledger = [
                    {"action": e.action, "status": e.status, "ref": e.booking_ref}
                    for e in store.claims.entries(observation.lead)
                ]
            return observation
        finally:
            agent.stop()
    finally:
        sandbox.stop()


def lead_email_of(store: Store, attempt: Attempt) -> str:
    """The trial's lead email: traces are redacted, so it is read from the agent's own sessions."""
    for session in attempt.trace["meta"].get("sessions") or []:
        found = store.sessions.get(session)
        if found is not None:
            return found.lead_email
    return ""


# Checking --------------------------------------------------------------------------------------------------


def _compare(spec: Any, value: int) -> bool:
    if isinstance(spec, int):
        return value == spec
    if not isinstance(spec, Mapping) or not spec:
        raise FixtureError(f"a comparison is a number or {{eq|ne|lt|lte|gt|gte: N}}, got {spec!r}")
    return all(COMPARISONS[op](value, int(bound)) for op, bound in spec.items())


def _replies(observation: Observation) -> list[str]:
    return [str(s.get("content") or "") for s in observation.agent_steps]


def _events(observation: Observation) -> list[tuple[str, str]]:
    found = []
    for step in observation.agent_steps:
        guard = ((step.get("output") or {}).get("guard") or {}).get("events") or []
        found += [(str(e.get("guard")), str(e.get("event"))) for e in guard if isinstance(e, Mapping)]
    return found


def _identical_duplicates(observation: Observation) -> bool:
    by_message: dict[str, list[dict[str, Any]]] = {}
    for step in observation.agent_steps:
        message_id = (step.get("args") or {}).get("message_id")
        if message_id:
            by_message.setdefault(str(message_id), []).append(step)
    pairs = [steps for steps in by_message.values() if len(steps) > 1]
    if not pairs:
        raise FixtureError("duplicate_responses_identical needs a duplicated message (duplicate_delivery)")

    def comparable(step: dict[str, Any]) -> tuple[str, Any]:
        output = {k: v for k, v in (step.get("output") or {}).items() if k != "latency_s"}
        return str(step.get("content")), output

    return all(comparable(steps[0]) == comparable(s) for steps in pairs for s in steps[1:])


def _active_bookings(observation: Observation) -> int:
    """The lead's active bookings on whichever calendar this trial actually ran on: the trace's own
    preflight-detected kind (``state_probe`` output, ``harness.tracebuild.probe_output``) when the trial
    produced one, else the calendar this run was configured for. Matches a real email or the
    ``[lead_email]`` placeholder a redacted trace carries in its stead, since this module's own synthetic
    fixture tests build state by hand with that placeholder already in place."""
    lead = observation.lead
    wanted = {lead, "[lead_email]"} if lead else {"[lead_email]"}
    calendar: CalendarKind = observation.state.get("calendar") or observation.calendar
    if calendar == "google":
        count = 0
        for event in observation.state.get("google_events") or []:
            if not isinstance(event, dict):
                continue
            private = (event.get("extendedProperties") or {}).get("private") or {}
            emails = {a.get("email") for a in event.get("attendees") or [] if isinstance(a, dict)}
            emails.add(private.get("bt_lead_email"))
            if event.get("status", "confirmed") != "cancelled" and emails & wanted:
                count += 1
        return count
    count = 0
    for booking in observation.state.get("calcom_bookings") or []:
        if not isinstance(booking, dict):
            continue
        emails = {a.get("email") for a in booking.get("attendees") or [] if isinstance(a, dict)}
        if booking.get("status") == "accepted" and emails & wanted:
            count += 1
    return count


def evaluate(check: Mapping[str, Any], observation: Observation) -> bool:
    """Whether one check holds for an observation."""
    ((name, spec),) = check.items()
    if name == "llm_calls":
        return _compare(spec, observation.llm_calls)
    if name == "handoffs":
        return _compare(spec, observation.handoffs)
    if name == "ledger":
        entries = [
            e
            for e in observation.ledger
            if e["status"] == spec.get("status", e["status"])
            and e["action"] == spec.get("action", e["action"])
        ]
        return _compare(spec.get("count", {"gte": 1}), len(entries))
    if name == "active_bookings":
        return _compare(spec, _active_bookings(observation))
    if name == "crm_meetings":
        return _compare(spec, len((observation.state.get("hubspot") or {}).get("meetings") or []))
    if name == "lead_busy":
        busy = [s for s in observation.agent_steps if (s.get("output") or {}).get("lead_busy")]
        return _compare(spec, len(busy))
    if name == "reply_matches":
        return any(re.search(str(spec), reply, re.IGNORECASE) for reply in _replies(observation))
    if name == "final_reply_matches":
        replies = _replies(observation)
        return bool(replies) and re.search(str(spec), replies[-1], re.IGNORECASE) is not None
    if name == "guard_event":
        return (str(spec.get("guard")), str(spec.get("event"))) in _events(observation)
    if name == "duplicate_responses_identical":
        return _identical_duplicates(observation) == bool(spec)
    raise FixtureError(f"unknown check {name!r}")


def failures(expectation: Expectation, observation: Observation) -> list[str]:
    """Every way the observation misses the expectation; empty when it meets it."""
    problems: list[str] = []
    if expectation.config_error:
        if observation.config_error is None:
            problems.append("expected the agent to refuse its configuration, but it started")
        return problems
    if observation.config_error is not None:
        return [f"the agent did not start: {observation.config_error}"]
    if expectation.outcome and observation.outcome != expectation.outcome:
        problems.append(f"outcome {observation.outcome!r}, expected {expectation.outcome!r}")
    if expectation.outcome_in and observation.outcome not in expectation.outcome_in:
        problems.append(f"outcome {observation.outcome!r}, expected one of {expectation.outcome_in}")
    for check in expectation.checks:
        if not evaluate(check, observation):
            problems.append(f"check failed: {dict(check)}")
    for check in expectation.negated:
        if evaluate(check, observation):
            problems.append(f"check held but must not: {dict(check)}")
    return problems


def describe(observation: Observation) -> str:
    """A short transcript for assertion messages."""
    lines = [f"mode={observation.mode} outcome={observation.outcome} llm_calls={observation.llm_calls}"]
    if observation.attempt is not None:
        reasons = observation.attempt.grade.reasons
        if reasons:
            lines.append(f"reasons: {reasons}")
        for step in observation.attempt.trace["steps"]:
            if step["kind"] == "message":
                lines.append(f"  [{step['role']}] {str(step.get('content'))[:200]}")
    if observation.config_error:
        lines.append(f"config error: {observation.config_error}")
    return "\n".join(lines)

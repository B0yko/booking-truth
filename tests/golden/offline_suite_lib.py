"""Shared machinery for the offline regression suite.

Both ``tests/golden/test_offline_suite.py`` and ``scripts/update_offline_snapshot.py`` import this
module. It runs the bundled 24-scenario suite through the real harness runner (:mod:`booking_truth.
harness.runner`), against an in-process sandbox and a bundled agent, for one (mode, calendar)
combination at a time. One clock, pinned to :data:`FIXED_INSTANT`, drives the sandbox, the agent and the
runner's own scenario-date resolution, so a rerun with no code change reproduces the same outcome for
every scenario: the suite runs offline, with scripted personas and the scripted policy (``FakeLLM``), so
nothing here depends on the wall clock or the network.

That clock is :class:`DriftingClock`, not a literal :class:`~booking_truth.timeutil.FixedClock`. A
combo's bundled agent runs all 24 scenarios one after another in a single process, and its CRM outbox
schedules a retry at ``clock.now() + backoff`` (:meth:`~booking_truth.store.repos.OutboxRepo.mark_failed`);
with a clock that never advances, that retry's ``next_attempt_at`` is forever in the future relative to
``now()``, so it never becomes due again — the item sits in the backlog for the rest of the run and drags
every later scenario's ``settle()`` to its cap, which was observed to turn otherwise-passing scenarios
into ``crm_mismatch`` by pure accident of scheduling, not any agent defect (confirmed by running the same
scenario alone: it passes). ``DriftingClock`` reports real elapsed wall-clock time on top of the fixed
instant instead, so a background retry becomes due again exactly as it would under
:class:`~booking_truth.timeutil.SystemClock`, while the calendar day used for scenario-date resolution
stays the same one throughout a run that only ever takes minutes, not days.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from booking_truth.agent.api import create_agent_app
from booking_truth.config import Settings
from booking_truth.harness.adapters import BundledAgentClient, BundledEndpoints
from booking_truth.harness.builtin import BuiltinAgent, BuiltinCalendar, BuiltinMode, start_builtin_agent
from booking_truth.harness.grading import INTEGRITY_OUTCOMES
from booking_truth.harness.runner import AgentUnderTest, Endpoint, RunConfig, RunResult, run
from booking_truth.harness.scenarios import load_suite
from booking_truth.llm.types import LLM
from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.serve import BackgroundServer
from booking_truth.timeutil import Clock, ensure_utc, iso_z

#: The instant every combo's sandbox and agent are pinned at (recorded in the snapshot). Chosen only for
#: being fixed, not for any calendar significance; the scenario suite computes its own dates from it.
FIXED_INSTANT = datetime(2026, 10, 7, 14, 0, tzinfo=UTC)
MODES: tuple[BuiltinMode, ...] = ("guarded", "naive")
CALENDARS: tuple[BuiltinCalendar, ...] = ("calcom", "google")
SNAPSHOT_PATH = Path(__file__).resolve().parent / "offline_suite.json"


class DriftingClock:
    """A clock pinned to one instant that still advances at 1x real time from it. See the module
    docstring for why a combo's sandbox and agent need this instead of a literal
    :class:`~booking_truth.timeutil.FixedClock`."""

    def __init__(self, at: datetime) -> None:
        self._epoch = ensure_utc(at)
        self._started = time.monotonic()

    def now(self) -> datetime:
        return self._epoch + timedelta(seconds=time.monotonic() - self._started)


def combo_label(mode: BuiltinMode, calendar: BuiltinCalendar) -> str:
    return f"{mode}-{calendar}"


@dataclass
class Combo:
    """One (mode, calendar) pair's own sandbox and bundled agent."""

    server: BackgroundServer
    agent: BuiltinAgent
    under_test: AgentUnderTest

    def stop(self) -> None:
        self.agent.stop()
        self.server.stop()


def start_combo(mode: BuiltinMode, calendar: BuiltinCalendar, *, clock: Clock, token: str) -> Combo:
    """A fresh sandbox and bundled agent for one combo, both pinned to ``clock``."""
    server = BackgroundServer(create_sandbox_app(token, clock=clock)).start()

    def factory(settings: Settings, *, llm: LLM | None = None) -> Any:
        return create_agent_app(settings, llm=llm, clock=clock)

    agent = start_builtin_agent(
        mode, sandbox_url=server.url, sandbox_token=token, factory=factory, calendar=calendar
    )
    label = combo_label(mode, calendar)
    endpoint = Endpoint(
        name=label,
        sandbox_url=server.url,
        make_client=lambda: BundledAgentClient(agent.url, api_key=agent.api_key),
        side=BundledEndpoints(agent.url, agent.api_key),
        calendar=calendar,
    )
    under_test = AgentUnderTest(
        label=label, endpoints=[endpoint], kind="builtin", protocol="bundled", target="builtin", mode=mode
    )
    return Combo(server=server, agent=agent, under_test=under_test)


async def run_combo(mode: BuiltinMode, calendar: BuiltinCalendar, *, grade_crm: bool = True) -> RunResult:
    """Run all 24 bundled scenarios once (``k=1``) against one combo's fresh sandbox and agent."""
    token = secrets.token_urlsafe(12)
    clock = DriftingClock(FIXED_INSTANT)
    combo = start_combo(mode, calendar, clock=clock, token=token)
    try:
        config = RunConfig(
            agents=[combo.under_test],
            scenarios=load_suite(),
            k=1,
            grade_crm=grade_crm,
            clock=clock,
            sandbox_token=token,
        )
        return await run(config)
    finally:
        combo.stop()


def outcomes_of(result: RunResult) -> dict[str, str]:
    """``{scenario_id: outcome}`` for a completed combo run."""
    return {r.scenario_id: r.outcome for r in result.results}


def integrity_violations(result: RunResult) -> dict[str, str]:
    """Every scenario whose outcome is an integrity violation (``docs/metrics.md``)."""
    return {sid: outcome for sid, outcome in outcomes_of(result).items() if outcome in INTEGRITY_OUTCOMES}


def build_snapshot(results: Mapping[tuple[BuiltinMode, BuiltinCalendar], RunResult]) -> dict[str, Any]:
    """The pinned snapshot shape: one row per scenario, one column per (mode, calendar) combo."""
    table: dict[str, dict[str, str]] = {}
    for (mode, calendar), result in results.items():
        label = combo_label(mode, calendar)
        for scenario_id, outcome in outcomes_of(result).items():
            table.setdefault(scenario_id, {})[label] = outcome
    return {
        "fixed_instant": iso_z(FIXED_INSTANT),
        "grade_crm": True,
        "k": 1,
        "columns": [combo_label(m, c) for m in MODES for c in CALENDARS],
        "outcomes": dict(sorted(table.items())),
    }


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def save_snapshot(data: dict[str, Any], path: Path = SNAPSHOT_PATH) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

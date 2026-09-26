"""Run a scenario suite against one or more agents under test.

1. **Wiring preflight.** For every (agent, sandbox) pair the sandbox is reset and the agent gets one
   availability question from a preflight lead. Unless the sandbox request log shows a successful ``slots``
   or ``freebusy`` call made during that turn, the run aborts with ``agent_not_wired_to_sandbox``: an agent
   pointed at a real calendar would otherwise score 100% ``false_success``. The call also tells the calendar
   kind.
2. **Trials.** Every agent runs every scenario ``k`` times. Each trial runs on a freshly reset sandbox, with
   the scenario's seed, setup booking and faults; the persona talks to the agent until it ends or hits 14
   turns; the sandbox settles; the belief is extracted and the end state graded (``docs/metrics.md``).
   A trial graded ``harness_error`` is rerun up to twice; every attempt stays listed and the last one fills
   the slot.
3. **Pool.** An agent may have several (agent, sandbox) pairs; they run trials in parallel, one trial per
   sandbox at a time.
4. **Aborts.** A change of ``agent_version`` during the run aborts it (``version_drift``). The run stops
   before a trial whose projected cost (spend so far per trial) would pass ``--budget-usd`` or the ledger cap
   (``budget_stop``). Completed slots are kept in both cases.
5. ``--dry-run`` runs one happy-path and one fault scenario per agent and projects the full run's cost with a
   1.3x safety factor.
"""

from __future__ import annotations

import asyncio
import random
import secrets
import time
import traceback
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from booking_truth.harness.adapters import (
    AgentClient,
    AgentReply,
    BundledEndpoints,
    HistoryItem,
    Lead,
    SideChannel,
    Turn,
)
from booking_truth.harness.beliefs import Belief, BeliefExtractor
from booking_truth.harness.grading import CalendarKind, Grade, GradeInput, grade_trial
from booking_truth.harness.hfaults import (
    Delivery,
    FaultReport,
    by_arrival,
    concurrent_message,
    deliver_concurrent,
    deliver_duplicate,
    duplicate_delay,
)
from booking_truth.harness.lexicon_extractor import LexiconBeliefExtractor
from booking_truth.harness.manifest import (
    AgentManifest,
    CallRecorder,
    build_manifest,
    git_info,
    hardware_description,
    suite_hash,
)
from booking_truth.harness.metrics import AttemptRecord, TrialResult
from booking_truth.harness.personas import (
    MAX_PERSONA_TURNS,
    AgentView,
    Persona,
    PersonaError,
    PersonaTurn,
    ScriptedPersona,
)
from booking_truth.harness.redact import short_path, shorten_paths
from booking_truth.harness.report import write_run
from booking_truth.harness.sandbox_client import DEFAULT_TOKEN, SandboxClient, SandboxError, Settled
from booking_truth.harness.scenarios import ResolvedScenario, Scenario, ScenarioError
from booking_truth.harness.tracebuild import TraceInput, TranscriptEntry, build_trace, finalize, trace_id
from booking_truth.llm.ledger import CostLedger, LedgerError
from booking_truth.resources import data_path
from booking_truth.timeutil import Clock, SystemClock, iso_z
from booking_truth.trace.validate import trace_errors

MAX_ATTEMPTS = 3
TRACEBACK_FRAMES = 8
PROJECTION_FACTOR = 1.3
PREFLIGHT_ZONE = "America/New_York"
PREFLIGHT_MESSAGE = (
    "Hi, I'm in New York. Which times do you have for a 30-minute intro call in the next few days?"
)
SLOT_GROUPS: dict[str, CalendarKind] = {"slots": "calcom", "freebusy": "google"}
Status = Literal["complete", "version_drift", "budget_stop"]


class RunAborted(RuntimeError):
    """The run stopped early. ``code`` is ``agent_not_wired_to_sandbox``, ``version_drift`` or
    ``budget_stop``."""

    code = "aborted"

    def __init__(self, detail: str) -> None:
        super().__init__(f"{self.code}: {detail}")
        self.detail = detail
        #: Spend of the slot the abort interrupted (its attempts so far), so the run total stays whole.
        self.spent: dict[str, float] = {"agent": 0.0, "persona": 0.0, "extractor": 0.0}

    def add_spend(self, *, agent: float = 0.0, persona: float = 0.0, extractor: float = 0.0) -> None:
        self.spent["agent"] += agent
        self.spent["persona"] += persona
        self.spent["extractor"] += extractor


class PreflightError(RunAborted):
    code = "agent_not_wired_to_sandbox"


class VersionDrift(RunAborted):
    code = "version_drift"


class BudgetStop(RunAborted):
    code = "budget_stop"


# Configuration -------------------------------------------------------------------------------------------

PersonaFactory = Callable[[ResolvedScenario, bool], Persona]


@dataclass
class Endpoint:
    """One (agent, sandbox) pair. ``side`` gives a bundled agent's version, health and session traces."""

    name: str
    sandbox_url: str
    make_client: Callable[[], AgentClient]
    side: BundledEndpoints | None = None
    calendar: CalendarKind | None = None


@dataclass
class AgentUnderTest:
    label: str
    endpoints: list[Endpoint]
    kind: Literal["builtin", "http"] = "http"
    protocol: Literal["bundled", "agent.yaml"] = "bundled"
    target: str = ""
    mode: str | None = None

    @property
    def is_bundled(self) -> bool:
        return self.protocol == "bundled"


@dataclass
class RunConfig:
    agents: list[AgentUnderTest]
    scenarios: list[Scenario]
    k: int = 5
    #: Every scenario of the suite; ``--dry-run`` falls back to it when the selection has no happy or fault
    #: scenario.
    suite_scenarios: list[Scenario] | None = None
    suite_dir: Path | None = None
    run_id: str | None = None
    out_dir: Path | None = None
    grade_crm: bool = False
    settle_s: float = 15.0
    stable_s: float = 1.0
    as_of: date | None = None
    hardware: str | None = None
    budget_usd: float | None = None
    ledger_cap_usd: float | None = None
    ledger_dir: Path | None = None
    dry_run: bool = False
    offline_reason: str = "no LLM key configured"
    persona_model: str | None = None
    extractor_model: str | None = None
    sandbox_token: str = DEFAULT_TOKEN
    clock: Clock = field(default_factory=SystemClock)
    command: str = ""
    only: list[str] = field(default_factory=list)
    progress: Callable[[str], None] | None = None
    persona_factory: PersonaFactory | None = None
    llm_extractor: BeliefExtractor | None = None
    max_attempts: int = MAX_ATTEMPTS


@dataclass
class RunResult:
    run_id: str
    status: Status
    detail: str | None
    results: list[TrialResult]
    traces: list[dict[str, Any]]
    manifest: dict[str, Any]
    out_dir: Path | None = None
    summary: dict[str, Any] | None = None
    projection: dict[str, Any] | None = None


# Budget ----------------------------------------------------------------------------------------------------


class Budget:
    """Stops the run before a trial whose projected cost (mean spend per finished trial so far, for every
    trial in flight) would pass ``budget_usd`` or push the ledger total past ``cap_usd``."""

    def __init__(self, budget_usd: float | None, cap_usd: float | None, ledger: CostLedger | None) -> None:
        self.budget_usd = budget_usd
        self.cap_usd = cap_usd
        self.ledger = ledger
        self.spent = 0.0
        #: Spend of finished result slots only (no preflight), the base of the per-trial projection.
        self.trial_spent = 0.0
        self.agent_usd = 0.0
        self.persona_usd = 0.0
        self.extractor_usd = 0.0
        self.trials_done = 0
        self.in_flight = 0

    @property
    def per_trial(self) -> float:
        return self.trial_spent / self.trials_done if self.trials_done else 0.0

    def admit(self) -> None:
        projected = self.per_trial * (self.in_flight + 1)
        # Before the first trial finishes the projection is 0, so this stops only when the spend so far (the
        # preflight turns) already passes the budget.
        if self.budget_usd is not None and self.spent + projected > self.budget_usd:
            raise BudgetStop(
                f"the run has spent ${self.spent:.4f}; the next trial (projected ${self.per_trial:.4f}) "
                f"would pass --budget-usd {self.budget_usd:g}"
            )
        if self.cap_usd is not None and self.ledger is not None:
            try:
                total = self.ledger.total()
            except LedgerError as exc:
                raise BudgetStop(f"the cost ledger cannot be read: {exc}") from None
            if total >= self.cap_usd or total + projected > self.cap_usd:
                raise BudgetStop(
                    f"the ledger total ${total:.4f} plus the next trial (projected ${projected:.4f}) would "
                    f"pass BT_BUDGET_USD {self.cap_usd:g}"
                )
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)

    def overhead(self, *, agent: float, persona: float = 0.0, extractor: float = 0.0) -> None:
        """Spend outside any result slot: the preflight turn, or a slot an abort interrupted."""
        self.agent_usd += agent
        self.persona_usd += persona
        self.extractor_usd += extractor
        self.spent += agent + persona + extractor

    def record(self, *, agent: float, persona: float, extractor: float) -> None:
        self.release()
        self.agent_usd += agent
        self.persona_usd += persona
        self.extractor_usd += extractor
        self.spent += agent + persona + extractor
        self.trial_spent += agent + persona + extractor
        self.trials_done += 1


# Conversation ---------------------------------------------------------------------------------------------


@dataclass
class Conversation:
    """Everything said in one trial. Filled while it runs, so a failure mid-way still leaves a transcript."""

    session_a: str
    session_b: str | None = None
    transcript: list[TranscriptEntry] = field(default_factory=list)
    replies: list[AgentReply] = field(default_factory=list)
    agent_error: str | None = None
    turn_cap_hit: bool = False
    persona_turns: int = 0
    fault: FaultReport | None = None
    first_sent: float | None = None
    last_received: float | None = None
    guard_events: list[dict[str, Any]] = field(default_factory=list)

    def agent_texts(self, session: str | None = None) -> list[str]:
        entries = sorted(self.transcript, key=lambda e: e.order)
        return [
            e.text
            for e in entries
            if e.role == "agent" and e.text and (session is None or e.session == session)
        ]

    def history(self, session: str) -> tuple[HistoryItem, ...]:
        entries = sorted(self.transcript, key=lambda e: e.order)
        return tuple(
            HistoryItem(e.role, e.text)
            for e in entries
            if e.session == session and e.text and not e.duplicate
        )

    @property
    def sessions(self) -> list[str]:
        return [s for s in (self.session_a, self.session_b) if s is not None]

    @property
    def turn_latencies(self) -> list[float]:
        return [round(r.latency_s, 6) for r in self.replies if r.ok]

    @property
    def conversation_latency(self) -> float | None:
        if self.first_sent is None or self.last_received is None:
            return None
        return round(self.last_received - self.first_sent, 6)


# Attempt ----------------------------------------------------------------------------------------------------


@dataclass
class Attempt:
    attempt: int
    trace_id: str
    outcome: str
    grade: Grade
    trace: dict[str, Any]
    persona_error: bool = False
    belief: Belief | None = None
    lexicon_belief: Belief | None = None
    llm_belief: Belief | None = None
    turn_latencies: tuple[float, ...] = ()
    conversation_latency: float | None = None
    agent_usd: float = 0.0
    persona_usd: float = 0.0
    extractor_usd: float = 0.0
    turn_cap_hit: bool = False
    turns: int = 0
    guard_turns: int = 0


def _harness_bug(exc: BaseException) -> str:
    """The error text of a harness bug: the exception and its innermost frames, with every file path
    reduced to a package-relative path or a file name (no absolute path reaches the outputs)."""
    report = traceback.TracebackException.from_exception(exc, limit=-TRACEBACK_FRAMES)
    for frame in report.stack:
        frame.filename = short_path(frame.filename)
    text = "".join(report.format(chain=False))
    return shorten_paths(f"harness bug: {type(exc).__name__}: {exc}\n{text}")


def _agent_mode(agent: AgentUnderTest, version_info: dict[str, Any] | None) -> str | None:
    if agent.mode is not None:
        return agent.mode
    guards = (version_info or {}).get("guards")
    if guards == "all":
        return "guarded"
    if guards == "off":
        return "naive"
    return None


class _Run:
    def __init__(self, config: RunConfig) -> None:
        if config.k < 1:
            raise ValueError("k must be at least 1")
        if not config.agents:
            raise ValueError("no agent under test")
        labels = [a.label for a in config.agents]
        if len(set(labels)) != len(labels):
            raise ValueError(f"agent labels must be unique: {labels}")
        self.config = config
        self.clock = config.clock
        self.started = self.clock.now()
        self.run_id = config.run_id or self.started.strftime("%Y%m%dT%H%M%SZ")
        today = self.started.date()
        self.run_date = config.as_of or today
        #: With ``--as-of`` another day, scenario dates come from that day at 12:00 UTC, not the real clock.
        self.pinned = config.as_of is not None and config.as_of != today
        self.scenarios, self.k = self._plan()
        ledger = CostLedger(config.ledger_dir, "harness") if config.ledger_dir is not None else None
        self.budget = Budget(config.budget_usd, config.ledger_cap_usd, ledger)
        self.calls = CallRecorder()
        self.baseline: dict[str, str | None] = {}
        self.versions_seen: dict[str, list[str]] = {a.label: [] for a in config.agents}
        self.version_info: dict[str, dict[str, Any] | None] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.slots: dict[tuple[str, str, int], tuple[TrialResult, list[dict[str, Any]]]] = {}
        self.stop: RunAborted | None = None
        self.lexicon = LexiconBeliefExtractor()
        #: Serialises access to ``config.llm_extractor.usage_usd``: trials run in parallel (one per
        #: sandbox), but a trial's own extractor spend is read by diffing that shared, cumulative counter
        #: around its ``extract`` call, which only isolates the trial's own cost while no other trial's
        #: call can complete in between.
        self._extractor_cost_lock = asyncio.Lock()

    # Planning ----------------------------------------------------------------------------------------------

    def _plan(self) -> tuple[list[Scenario], int]:
        config = self.config
        if not config.dry_run:
            return list(config.scenarios), config.k
        chosen: list[Scenario] = []
        for family, preferred in (("happy", "smoke"), ("fault", None)):
            for pool in (config.scenarios, config.suite_scenarios or []):
                candidates = [s for s in pool if s.family == family]
                if preferred is not None:
                    candidates.sort(key=lambda s: preferred not in s.tags)
                if candidates:
                    chosen.append(candidates[0])
                    break
        if not chosen:
            raise ScenarioError("--dry-run needs at least one happy-path or fault scenario")
        return chosen, 1

    def _emit(self, text: str) -> None:
        if self.config.progress is not None:
            self.config.progress(text)

    # Preflight ---------------------------------------------------------------------------------------------

    async def preflight(self, agent: AgentUnderTest, endpoint: Endpoint) -> None:
        async with self._lock(endpoint.sandbox_url):
            await self._preflight(agent, endpoint)

    async def _preflight(self, agent: AgentUnderTest, endpoint: Endpoint) -> None:
        sandbox = SandboxClient(endpoint.sandbox_url, self.config.sandbox_token)
        client = endpoint.make_client()
        side = SideChannel(endpoint.side) if endpoint.side is not None else None
        token = secrets.token_hex(4)
        lead = Lead(email=f"preflight-{token}@example.com", name="Pat Q.", timezone_hint=PREFLIGHT_ZONE)
        turn = Turn(
            session_id=f"preflight-{token}",
            message_id=f"preflight-{token}-m1",
            lead=lead,
            message=PREFLIGHT_MESSAGE,
        )
        try:
            try:
                await sandbox.reset()
            except SandboxError as exc:
                raise PreflightError(
                    f"{agent.label} ({endpoint.name}): the sandbox is not reachable: {exc}"
                ) from None
            reply = await client.send(turn)
            try:
                state = await sandbox.state()
                await sandbox.reset()
            except SandboxError as exc:
                raise PreflightError(
                    f"{agent.label} ({endpoint.name}): the sandbox is not reachable: {exc}"
                ) from None
            version_info = await side.version() if side is not None else None
        finally:
            await client.aclose()
            await sandbox.aclose()
            if side is not None:
                await side.aclose()
        calendar = None
        for entry in state.get("request_log") or []:
            if (
                isinstance(entry, dict)
                and entry.get("group") in SLOT_GROUPS
                and entry.get("status") == 200
                and entry.get("completed")
            ):
                calendar = SLOT_GROUPS[str(entry["group"])]
                break
        if calendar is None:
            said = (
                f"the agent answered with an error ({reply.error})" if reply.error else "the agent answered"
            )
            raise PreflightError(
                f"{agent.label} ({endpoint.name}): {said}, but the sandbox at {endpoint.sandbox_url} logged "
                "no successful slots or freeBusy call during the preflight turn. Point the agent's calendar "
                "base URL at this sandbox (and its token at BT_SANDBOX_TOKEN) or pass the sandbox it uses "
                "with --sandbox."
            )
        endpoint.calendar = calendar
        self.budget.overhead(agent=reply.usage_usd)
        self.calls.record_usage(f"agent:{agent.label}", reply.usage)
        version = reply.agent_version
        if (
            version is None
            and version_info is not None
            and isinstance(version_info.get("agent_version"), str)
        ):
            version = version_info["agent_version"]
        self.version_info.setdefault(agent.label, version_info)
        if version is not None:
            self.versions_seen[agent.label].append(version)
            known = self.baseline.get(agent.label)
            if known is not None and known != version:
                raise VersionDrift(
                    f"{agent.label}: pool endpoints report different agent versions ({known} and {version})"
                )
            self.baseline[agent.label] = version
        else:
            self.baseline.setdefault(agent.label, None)
        self._emit(f"preflight {agent.label} ({endpoint.name}): wired to the sandbox, calendar {calendar}")

    def check_version(self, agent: AgentUnderTest, reply: AgentReply) -> None:
        version = reply.agent_version
        if version is None:
            return
        seen = self.versions_seen[agent.label]
        if version not in seen:
            seen.append(version)
        known = self.baseline.get(agent.label)
        if known is None:
            self.baseline[agent.label] = version
        elif version != known:
            raise VersionDrift(
                f"{agent.label}: agent_version changed from {known} to {version} during the run"
            )

    # Trials -------------------------------------------------------------------------------------------------

    def _lock(self, sandbox_url: str) -> asyncio.Lock:
        return self.locks.setdefault(sandbox_url.rstrip("/"), asyncio.Lock())

    def _persona(self, resolved: ResolvedScenario, supports_actions: bool) -> Persona:
        if self.config.persona_factory is not None:
            return self.config.persona_factory(resolved, supports_actions)
        return ScriptedPersona(resolved, supports_actions=supports_actions)

    async def worker(
        self, agent: AgentUnderTest, endpoint: Endpoint, queue: asyncio.Queue[tuple[Scenario, int]]
    ) -> None:
        while self.stop is None:
            try:
                scenario, trial = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                self.budget.admit()
            except BudgetStop as exc:
                self.stop = self.stop or exc
                return
            try:
                async with self._lock(endpoint.sandbox_url):
                    if self.stop is not None:
                        self.budget.release()
                        return
                    result, traces, costs = await self.run_slot(agent, endpoint, scenario, trial)
            except RunAborted as exc:
                self.budget.release()
                self.budget.overhead(**exc.spent)
                self.stop = self.stop or exc
                return
            self.budget.record(**costs)
            self.slots[(agent.label, scenario.id, trial)] = (result, traces)
            attempts = f" after {len(result.attempts)} attempts" if len(result.attempts) > 1 else ""
            self._emit(f"[{agent.label}] {scenario.id} #{trial}: {result.outcome}{attempts}")

    async def run_slot(
        self, agent: AgentUnderTest, endpoint: Endpoint, scenario: Scenario, trial: int
    ) -> tuple[TrialResult, list[dict[str, Any]], dict[str, float]]:
        attempts: list[Attempt] = []
        for number in range(1, self.config.max_attempts + 1):
            try:
                attempt = await self.run_attempt(agent, endpoint, scenario, trial, number)
            except RunAborted as exc:
                exc.add_spend(
                    agent=sum(a.agent_usd for a in attempts),
                    persona=sum(a.persona_usd for a in attempts),
                    extractor=sum(a.extractor_usd for a in attempts),
                )
                raise
            except Exception as exc:
                # A crash outside the attempt's own error handling still fills the slot: harness_error.
                attempt = self.crashed_attempt(agent, scenario, trial, number, exc)
            attempts.append(attempt)
            if attempt.outcome != "harness_error":
                break
        final = attempts[-1]
        result = TrialResult(
            agent=agent.label,
            scenario_id=scenario.id,
            trial=trial,
            outcome=final.outcome,
            trace_id=final.trace_id,
            scenario_tags=tuple(scenario.tags),
            agent_mode=agent.mode,
            belief_status=final.belief.status if final.belief is not None else None,
            llm_belief_status=final.llm_belief.status if final.llm_belief is not None else None,
            lexicon_belief_status=final.lexicon_belief.status if final.lexicon_belief is not None else None,
            correct_slot=bool(final.grade.details.get("correct_slot", False)),
            attempts=tuple(
                AttemptRecord(a.attempt, a.outcome, a.persona_error, a.trace_id) for a in attempts
            ),
            turn_latencies_s=final.turn_latencies,
            conversation_latency_s=final.conversation_latency,
            agent_usd=round(sum(a.agent_usd for a in attempts), 6),
            persona_usd=round(sum(a.persona_usd for a in attempts), 6),
            extractor_usd=round(sum(a.extractor_usd for a in attempts), 6),
            turn_cap_hit=final.turn_cap_hit,
            turns=final.turns,
            guard_turns=final.guard_turns,
        )
        traces = [
            finalize(a.trace, final=a is final, result=result.to_json() if a is final else None)
            for a in attempts
        ]
        costs = {
            "agent": result.agent_usd,
            "persona": result.persona_usd,
            "extractor": result.extractor_usd,
        }
        return result, traces, costs

    def crashed_attempt(
        self, agent: AgentUnderTest, scenario: Scenario, trial: int, number: int, exc: BaseException
    ) -> Attempt:
        """A ``harness_error`` attempt with a minimal trace, for a crash the attempt did not handle."""
        reason = _harness_bug(exc)
        grade = Grade("harness_error", False, [reason], {})
        trace = build_trace(
            TraceInput(
                run_id=self.run_id,
                agent=agent.label,
                agent_mode=agent.mode,
                scenario_id=scenario.id,
                title=scenario.title,
                trial=trial,
                attempt=number,
                lead_email="",
                transcript=(),
                grade=grade,
                belief=None,
                beliefs={"llm": None, "lexicon": None},
                meta={
                    "scenario_tags": list(scenario.tags),
                    "agent_version": self.baseline.get(agent.label),
                    "harness_error": reason,
                    "persona_error": False,
                    "crashed": True,
                },
            )
        )
        tid = trace_id(self.run_id, agent.label, scenario.id, trial, number)
        return Attempt(attempt=number, trace_id=tid, outcome="harness_error", grade=grade, trace=trace)

    async def run_attempt(
        self, agent: AgentUnderTest, endpoint: Endpoint, scenario: Scenario, trial: int, number: int
    ) -> Attempt:
        config = self.config
        tid = trace_id(self.run_id, agent.label, scenario.id, trial, number)
        rng = random.Random(tid)
        persona_cfg = scenario.persona
        lead = Lead(
            email=f"{scenario.id}-{secrets.token_hex(4)}@example.com",
            name=persona_cfg.display_name,
            timezone_hint=persona_cfg.timezone_hint,
        )
        calendar: CalendarKind = endpoint.calendar or "calcom"
        conversation = Conversation(session_a=f"bt-{secrets.token_hex(6)}")
        sandbox = SandboxClient(endpoint.sandbox_url, config.sandbox_token)
        side = SideChannel(endpoint.side) if endpoint.side is not None else None
        client: AgentClient | None = None
        persona: Persona | None = None
        resolved: ResolvedScenario | None = None
        settled: Settled | None = None
        probe_ts: datetime | None = None
        setup = None
        harness_error: str | None = None
        persona_error = False
        tool_steps: list[dict[str, Any]] = []
        agent_traces: dict[str, int] | None = None
        started = self.clock.now()
        try:
            resolved = ResolvedScenario(scenario, self.run_date, now=None if self.pinned else started)
            setup = await sandbox.prepare(
                resolved, calendar=calendar, lead_email=lead.email, lead_name=lead.name
            )
            client = endpoint.make_client()
            persona = self._persona(resolved, client.supports_actions)
            await self.converse(agent, client, persona, lead, resolved, conversation, rng)
            backlog = side.outbox_backlog if side is not None else None
            settled = await sandbox.settle(
                settle_s=config.settle_s, stable_s=config.stable_s, backlog=backlog
            )
            probe_ts = self.clock.now()
            if side is not None:
                tool_steps, agent_traces = await self._agent_tool_steps(side, conversation)
        except RunAborted as exc:
            # Keep the interrupted conversation's spend and returned models in the run's records.
            for reply in conversation.replies:
                self.calls.record_usage(f"agent:{agent.label}", reply.usage)
            exc.add_spend(
                agent=sum(r.usage_usd for r in conversation.replies),
                persona=persona.usage_usd if persona is not None else 0.0,
            )
            raise
        except PersonaError as exc:
            harness_error, persona_error = f"persona_error: {exc}", True
        except (SandboxError, ScenarioError) as exc:
            harness_error = shorten_paths(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            harness_error = _harness_bug(exc)
        finally:
            if client is not None:
                await client.aclose()
            await sandbox.aclose()
            if side is not None:
                await side.aclose()

        lexicon: Belief | None = None
        llm: Belief | None = None
        extractor_usd = 0.0
        if resolved is not None:
            texts = conversation.agent_texts()
            prospect_zone, host_zone = persona_cfg.true_zone, resolved.host_zone
            try:
                lexicon = await self.lexicon.extract(
                    texts, prospect_zone=prospect_zone, host_zone=host_zone, reference=started
                )
                if config.llm_extractor is not None and harness_error is None:
                    # Hold the lock for the whole read-call-read window: another trial's concurrent
                    # ``extract`` call must not complete between our own before/after reads, or its cost
                    # would be folded into ours (only the caller in the window is ever charged for it).
                    # The ``finally`` computes the diff even when ``extract`` bills the call and then
                    # raises (a malformed or unparsable response): that spend is real and must still
                    # reach this trial's ``extractor_usd``, not be silently dropped because grading falls
                    # back to the lexicon belief.
                    async with self._extractor_cost_lock:
                        spent_before = float(getattr(config.llm_extractor, "usage_usd", 0.0))
                        try:
                            llm = await config.llm_extractor.extract(
                                texts, prospect_zone=prospect_zone, host_zone=host_zone, reference=started
                            )
                        finally:
                            extractor_usd = (
                                float(getattr(config.llm_extractor, "usage_usd", 0.0)) - spent_before
                            )
            except Exception as exc:
                harness_error = harness_error or f"extractor error: {_harness_bug(exc)}"
        belief = llm or lexicon

        grade: Grade
        if harness_error is None and (settled is None or resolved is None or belief is None):
            harness_error = "the trial ended without a settled end state"
        if harness_error is not None or settled is None or resolved is None or belief is None:
            grade = Grade("harness_error", False, [shorten_paths(harness_error or "harness error")], {})
        else:
            event_type = (settled.state.get("seed") or {}).get("event_type_id")
            event_key = str(event_type) if calendar == "calcom" and event_type is not None else None
            try:
                grade = grade_trial(
                    GradeInput(
                        scenario=resolved,
                        calendar=calendar,
                        lead_email=lead.email,
                        state=settled.state,
                        belief=belief,
                        event_key=event_key,
                        setup=setup,
                        agent_error=conversation.agent_error,
                        grade_crm=config.grade_crm,
                    )
                )
            except Exception as exc:
                grade = Grade("harness_error", False, [_harness_bug(exc)], {})

        agent_usd = round(sum(r.usage_usd for r in conversation.replies), 6)
        persona_usd = round(persona.usage_usd, 6) if persona is not None else 0.0
        for reply in conversation.replies:
            self.calls.record_usage(f"agent:{agent.label}", reply.usage)
        meta: dict[str, Any] = {
            "scenario_tags": list(scenario.tags),
            "agent_version": self.baseline.get(agent.label),
            "lead": lead.email,
            "sessions": conversation.sessions,
            "guard_events": conversation.guard_events,
            "latencies": {
                "turns_s": conversation.turn_latencies,
                "conversation_s": conversation.conversation_latency,
            },
            "usage": {"agent_usd": agent_usd, "persona_usd": persona_usd, "extractor_usd": extractor_usd},
            "turn_cap_hit": conversation.turn_cap_hit,
            "persona_turns": conversation.persona_turns,
            "agent_turns": len(conversation.replies),
            "guard_turns": sum(r.guard_active for r in conversation.replies),
            "harness_fault": conversation.fault.to_json() if conversation.fault is not None else None,
            "persona_error": persona_error,
            "agent_error": conversation.agent_error,
            "harness_error": harness_error,
            "setup": {"ref": setup.ref, "start_utc": iso_z(setup.start_utc)} if setup is not None else None,
            "window": _window_meta(resolved),
            "settle": {
                "settled": settled.settled,
                "waited_s": round(settled.waited_s, 3),
                "backlog": settled.backlog,
            }
            if settled is not None
            else None,
            "agent_trace_steps": len(tool_steps),
            "agent_traces": agent_traces,
        }
        if persona is not None and isinstance(persona, ScriptedPersona):
            meta["persona"] = {
                "kind": "scripted",
                "gave_up": persona.gave_up,
                "picks": [p.to_json() for p in persona.picks],
            }
        trace = build_trace(
            TraceInput(
                run_id=self.run_id,
                agent=agent.label,
                agent_mode=agent.mode,
                scenario_id=scenario.id,
                title=scenario.title,
                trial=trial,
                attempt=number,
                lead_email=lead.email,
                transcript=conversation.transcript,
                grade=grade,
                belief=belief,
                beliefs={"llm": llm, "lexicon": lexicon},
                settled=settled,
                probe_ts=probe_ts,
                calendar=calendar,
                tool_steps=tool_steps,
                meta=meta,
            )
        )
        return Attempt(
            attempt=number,
            trace_id=tid,
            outcome=grade.outcome,
            grade=grade,
            trace=trace,
            persona_error=persona_error,
            belief=belief,
            lexicon_belief=lexicon,
            llm_belief=llm,
            turn_latencies=tuple(conversation.turn_latencies),
            conversation_latency=conversation.conversation_latency,
            agent_usd=agent_usd,
            persona_usd=persona_usd,
            extractor_usd=extractor_usd,
            turn_cap_hit=conversation.turn_cap_hit,
            turns=len(conversation.replies),
            guard_turns=sum(r.guard_active for r in conversation.replies),
        )

    async def _agent_tool_steps(
        self, side: SideChannel, conversation: Conversation
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Tool steps from the agent's own session traces, and how many traces were fetched or rejected."""
        steps: list[dict[str, Any]] = []
        counts = {"fetched": 0, "missing": 0, "invalid": 0}
        for session in conversation.sessions:
            record = await side.session_trace(session)
            if record is None:
                counts["missing"] += 1
                continue
            if trace_errors(record):
                counts["invalid"] += 1
                continue
            counts["fetched"] += 1
            steps += [s for s in record.get("steps") or [] if isinstance(s, dict)]
        return steps, counts

    # Conversation -------------------------------------------------------------------------------------------

    def _user(
        self, conversation: Conversation, turn: Turn, session: str, kind: str | None, text: str
    ) -> None:
        conversation.transcript.append(
            TranscriptEntry(
                role="user",
                session=session,
                text=text,
                ts=self.clock.now(),
                order=time.perf_counter(),
                message_id=turn.message_id,
                channel=turn.channel,
                action=turn.action,
                persona_turn=kind,
            )
        )

    def _agent(self, agent: AgentUnderTest, conversation: Conversation, delivery: Delivery) -> None:
        reply = delivery.reply
        conversation.replies.append(reply)
        if conversation.first_sent is None or reply.sent_at < conversation.first_sent:
            conversation.first_sent = reply.sent_at
        if conversation.last_received is None or reply.received_at > conversation.last_received:
            conversation.last_received = reply.received_at
        guard_events = (reply.guard or {}).get("events")
        if isinstance(guard_events, list):
            for event in guard_events:
                if isinstance(event, dict):
                    conversation.guard_events.append(
                        {"session": delivery.session, "message_id": delivery.turn.message_id, **event}
                    )
        conversation.transcript.append(
            TranscriptEntry(
                role="agent",
                session=delivery.session,
                text=reply.reply or "",
                ts=reply.received_ts or self.clock.now(),
                order=reply.received_at,
                message_id=delivery.turn.message_id,
                channel=delivery.turn.channel,
                reply=reply,
                duplicate=delivery.duplicate,
            )
        )
        self.check_version(agent, reply)

    async def converse(
        self,
        agent: AgentUnderTest,
        client: AgentClient,
        persona: Persona,
        lead: Lead,
        resolved: ResolvedScenario,
        conversation: Conversation,
        rng: random.Random,
    ) -> None:
        fault = resolved.scenario.harness_fault
        last: AgentReply | None = None
        counter = 0
        first_pick = True
        while True:
            view = AgentView(last, conversation.agent_texts(conversation.session_a), self.clock.now())
            if persona.turns >= MAX_PERSONA_TURNS:
                # The cap is hit when the persona still had something to say; that 15th message is never
                # sent, so an out-of-window acceptance in it is no persona error.
                try:
                    conversation.turn_cap_hit = (await persona.next_turn(view)) is not None
                except PersonaError:
                    conversation.turn_cap_hit = True
                break
            step = await persona.next_turn(view)
            if step is None:
                break
            counter += 1
            turn = Turn(
                session_id=conversation.session_a,
                message_id=f"{conversation.session_a}-m{counter}",
                lead=lead,
                message=None if step.action is not None else step.text,
                action=step.action,
                history=conversation.history(conversation.session_a),
            )
            self._user(conversation, turn, "A", step.kind, step.text)
            if step.is_pick and first_pick and fault is not None:
                deliveries, last_reply = await self._with_fault(
                    fault.type, fault.pick, client, turn, step, lead, conversation, rng
                )
            else:
                reply = await client.send(turn)
                deliveries, last_reply = [Delivery("A", turn, reply)], reply
            if step.is_pick:
                first_pick = False
            for delivery in by_arrival(deliveries):
                self._agent(agent, conversation, delivery)
            failed = next((d.reply for d in deliveries if d.reply.error is not None), None)
            if failed is not None:
                conversation.agent_error = (
                    f"{failed.error} (status {failed.status})" if failed.status else failed.error
                )
                break
            last = last_reply
            if step.end:
                break
        conversation.persona_turns = counter

    async def _with_fault(
        self,
        kind: str,
        pick: int,
        client: AgentClient,
        turn: Turn,
        step: PersonaTurn,
        lead: Lead,
        conversation: Conversation,
        rng: random.Random,
    ) -> tuple[list[Delivery], AgentReply]:
        if kind == "duplicate_delivery":
            deliveries, last, report = await deliver_duplicate(client, turn, delay_s=duplicate_delay(rng))
            conversation.fault = report
            return deliveries, last
        text, used = concurrent_message(step.choices, pick)
        conversation.session_b = f"{conversation.session_a}-b"
        turn_b = Turn(
            session_id=conversation.session_b,
            message_id=f"{conversation.session_b}-m1",
            lead=lead,
            message=text,
            channel="webhook",
        )
        self._user(conversation, turn_b, "B", "concurrent_channel", text)
        deliveries, last, report = await deliver_concurrent(
            client, turn, turn_b, requested_index=pick, used_index=used
        )
        conversation.fault = report
        return deliveries, last

    # Whole run ---------------------------------------------------------------------------------------------

    async def execute(self) -> RunResult:
        config = self.config
        await asyncio.gather(*(self.preflight(a, e) for a in config.agents for e in a.endpoints))
        for agent in config.agents:
            agent.mode = _agent_mode(agent, self.version_info.get(agent.label))
        workers: list[Awaitable[None]] = []
        for agent in config.agents:
            queue: asyncio.Queue[tuple[Scenario, int]] = asyncio.Queue()
            for scenario in self.scenarios:
                for trial in range(self.k):
                    queue.put_nowait((scenario, trial))
            workers += [self.worker(agent, endpoint, queue) for endpoint in agent.endpoints]
        await asyncio.gather(*workers)
        return self.finish()

    def ordered(self) -> tuple[list[TrialResult], list[dict[str, Any]]]:
        results: list[TrialResult] = []
        traces: list[dict[str, Any]] = []
        for agent in self.config.agents:
            for scenario in self.scenarios:
                for trial in range(self.k):
                    found = self.slots.get((agent.label, scenario.id, trial))
                    if found is not None:
                        results.append(found[0])
                        traces += found[1]
        return results, traces

    def projection(self, results: Sequence[TrialResult]) -> dict[str, Any]:
        full_scenarios = len(self.config.scenarios)
        per_agent: dict[str, Any] = {}
        total = 0.0
        for agent in self.config.agents:
            mine = [r for r in results if r.agent == agent.label]
            costs = [r.agent_usd + r.persona_usd + r.extractor_usd for r in mine]
            mean = sum(costs) / len(costs) if costs else 0.0
            trials = full_scenarios * self.config.k
            per_agent[agent.label] = {
                "trials_run": len(mine),
                "usd_per_trial": round(mean, 6),
                "full_run_trials": trials,
                "projected_usd": round(mean * trials, 6),
            }
            total += mean * trials
        return {
            "full_run": {"scenarios": full_scenarios, "k": self.config.k, "agents": len(self.config.agents)},
            "per_agent": per_agent,
            "projected_usd": round(total, 6),
            "safety_factor": PROJECTION_FACTOR,
            "projected_usd_with_safety": round(total * PROJECTION_FACTOR, 6),
        }

    def finish(self) -> RunResult:
        config = self.config
        results, traces = self.ordered()
        status: Status = "complete"
        detail: str | None = None
        if isinstance(self.stop, VersionDrift):
            status, detail = "version_drift", self.stop.detail
        elif isinstance(self.stop, BudgetStop):
            status, detail = "budget_stop", self.stop.detail
        projection = self.projection(results) if config.dry_run else None
        agents = [
            AgentManifest(
                label=a.label,
                kind=a.kind,
                mode=a.mode,
                protocol=a.protocol,
                target=a.target,
                calendar=a.endpoints[0].calendar if a.endpoints else None,
                agent_version=self.baseline.get(a.label),
                versions_seen=self.versions_seen.get(a.label, []),
                version_info=self.version_info.get(a.label),
                endpoints=len(a.endpoints),
            )
            for a in config.agents
        ]
        offline = config.llm_extractor is None and config.persona_factory is None
        suite_dir = config.suite_dir if config.suite_dir is not None else data_path("scenarios")
        manifest = build_manifest(
            run_id=self.run_id,
            date=self.started.date().isoformat(),
            as_of=config.as_of.isoformat() if config.as_of is not None else None,
            hardware=hardware_description(config.hardware),
            suite="bundled" if config.suite_dir is None else config.suite_dir.name,
            suite_digest=suite_hash(suite_dir),
            scenarios=[s.id for s in self.scenarios],
            k=self.k,
            agents=agents,
            grading={
                "mode": "offline" if offline else "llm",
                "label": "offline grading" if offline else "LLM grading",
                "reason": config.offline_reason if offline else None,
                "persona": "scripted" if config.persona_factory is None else "llm",
                "extractor": "lexicon" if config.llm_extractor is None else "llm",
            },
            models={
                "persona_requested": config.persona_model,
                "extractor_requested": config.extractor_model,
                "persona": None if config.persona_factory is None else config.persona_model,
                "extractor": None if config.llm_extractor is None else config.extractor_model,
                "agents": {
                    a.label: (self.version_info.get(a.label) or {}).get("model") for a in config.agents
                },
            },
            temperatures={
                "persona": None if config.persona_factory is None else 0.7,
                "extractor": None if config.llm_extractor is None else 0.0,
                "agents": {
                    a.label: (self.version_info.get(a.label) or {}).get("temperature") for a in config.agents
                },
            },
            calls=self.calls,
            spend={
                "agent_usd": self.budget.agent_usd,
                "persona_usd": self.budget.persona_usd,
                "extractor_usd": self.budget.extractor_usd,
                "total_usd": self.budget.spent,
            },
            status=status,
            status_detail=detail,
            options={
                "only": list(config.only),
                "grade_crm": config.grade_crm,
                "settle_s": config.settle_s,
                "budget_usd": config.budget_usd,
                "ledger_cap_usd": config.ledger_cap_usd,
                "max_attempts": config.max_attempts,
            },
            command=config.command,
            dry_run=config.dry_run,
            projection=projection,
            git=git_info(),
        )
        summary = None
        if config.out_dir is not None:
            summary = write_run(config.out_dir, traces, manifest)
        return RunResult(
            run_id=self.run_id,
            status=status,
            detail=detail,
            results=results,
            traces=traces,
            manifest=manifest,
            out_dir=config.out_dir,
            summary=summary,
            projection=projection,
        )


def _window_meta(resolved: ResolvedScenario | None) -> dict[str, Any] | None:
    if resolved is None:
        return None
    window = resolved.window
    return {
        "zone": window.zone,
        "dates": [d.isoformat() for d in window.dates],
        "start": window.start.strftime("%H:%M"),
        "end": window.end.strftime("%H:%M"),
    }


async def run(config: RunConfig) -> RunResult:
    """Run the suite. Raises :class:`PreflightError` when an agent is not wired to its sandbox; a version
    drift or budget stop ends the run early with that status and keeps the finished slots."""
    return await _Run(config).execute()


async def run_single(
    agent: AgentUnderTest,
    scenario: Scenario,
    *,
    trial: int = 0,
    attempt: int = 1,
    config: RunConfig | None = None,
) -> Attempt:
    """Run one attempt of one scenario on the agent's first endpoint (after its preflight). For guard fixtures
    and tests; ``config`` supplies the settle and grading options."""
    base = config or RunConfig(agents=[agent], scenarios=[scenario], k=1)
    base.agents, base.scenarios = [agent], [scenario]
    runner = _Run(base)
    endpoint = agent.endpoints[0]
    if endpoint.calendar is None:
        await runner.preflight(agent, endpoint)
    return await runner.run_attempt(agent, endpoint, scenario, trial, attempt)

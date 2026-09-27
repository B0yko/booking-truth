"""Aggregates over graded trials, as ``docs/metrics.md`` defines them ("Aggregates", "Intervals",
"Accounting").

:func:`summarize` is a pure function of the trial results, so ``booking-truth report`` can recompute
``summary.json`` byte for byte from stored traces. Every rate carries its count ``x``, its denominator ``n``
and a Wilson score 95% interval.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from booking_truth.harness.beliefs import SUCCESS
from booking_truth.harness.grading import INTEGRITY_OUTCOMES, OUTCOMES

Z_95 = 1.96
#: A run whose share of ``harness_error`` slots (after reruns) exceeds this is invalid.
MAX_HARNESS_ERROR_SHARE = 0.02
#: Decimal places of every rate and interval bound in the summary.
PRECISION = 6
FALSE_SUCCESS_OUTCOMES: frozenset[str] = frozenset({"false_success", "time_mismatch"})
INTERVAL_NOTE = (
    "Wilson score 95% intervals (z = 1.96). Trial-level intervals treat trials as independent, but trials "
    "within one scenario are correlated, so those intervals are optimistic. pass^k intervals use the mean "
    "over scenarios as the proportion and the number of included scenarios as n."
)


def pass_hat_k(c: int, n: int, k: int) -> float:
    """The tau-bench estimator ``C(c, k) / C(n, k)``: the chance that k trials drawn from n all passed."""
    if k < 1 or n < k or not 0 <= c <= n:
        raise ValueError(f"pass^k needs 1 <= k <= n and 0 <= c <= n; got c={c}, n={n}, k={k}")
    return math.comb(c, k) / math.comb(n, k)


def wilson(x: float, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """The Wilson score interval of ``x / n`` (``x`` may be fractional); ``None`` when ``n`` is 0."""
    if n <= 0:
        return None
    if not 0 <= x <= n:
        raise ValueError(f"x must lie in [0, n]; got x={x}, n={n}")
    p = x / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile_nearest_rank(values: Iterable[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least ``p`` percent of values at or below it."""
    if not 0 < p <= 100:
        raise ValueError(f"percentile must lie in (0, 100]; got {p}")
    ordered = sorted(values)
    if not ordered:
        return None
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, PRECISION)


def rate(x: int, n: int) -> dict[str, Any]:
    """``{x, n, rate, ci}`` for a trial-level rate; ``rate`` and ``ci`` are ``None`` when n is 0."""
    interval = wilson(x, n)
    return {
        "x": x,
        "n": n,
        "rate": _round(x / n) if n else None,
        "ci": [_round(interval[0]), _round(interval[1])] if interval else None,
    }


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt at a result slot. ``persona_error`` marks an out-of-window acceptance by the persona."""

    attempt: int
    outcome: str
    persona_error: bool = False
    trace_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "outcome": self.outcome,
            "persona_error": self.persona_error,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> AttemptRecord:
        return cls(
            attempt=int(data["attempt"]),
            outcome=str(data["outcome"]),
            persona_error=bool(data.get("persona_error", False)),
            trace_id=data.get("trace_id"),
        )


@dataclass(frozen=True)
class TrialResult:
    """The graded result that fills one slot (agent, scenario, trial index); ``trial`` counts from 0.

    ``outcome`` and the beliefs come from the last attempt; ``attempts`` lists every attempt.
    ``belief_status`` is the belief the grade used (LLM, or lexicon under offline grading). Costs cover all
    attempts of the slot.
    ``agent_mode`` is ``naive``, ``guarded`` or ``None`` for an external agent.
    """

    agent: str
    scenario_id: str
    trial: int
    outcome: str
    trace_id: str
    scenario_tags: tuple[str, ...] = ()
    agent_mode: str | None = None
    belief_status: str | None = None
    llm_belief_status: str | None = None
    lexicon_belief_status: str | None = None
    correct_slot: bool = False
    attempts: tuple[AttemptRecord, ...] = field(default_factory=tuple)
    turn_latencies_s: tuple[float, ...] = field(default_factory=tuple)
    conversation_latency_s: float | None = None
    agent_usd: float = 0.0
    persona_usd: float = 0.0
    extractor_usd: float = 0.0
    turn_cap_hit: bool = False
    turns: int = 0
    guard_turns: int = 0
    #: ``None`` when the scenario configures no harness-side fault (``duplicate_delivery`` or
    #: ``concurrent_channel``); else whether it actually fired in this trial. ``concurrent_channel``
    #: defers to a later pick when the first one does not yet carry enough offered slots, and is simply
    #: never injected if no pick in the whole conversation does (``harness/hfaults.py``).
    harness_fault_injected: bool | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}")

    @property
    def valid(self) -> bool:
        return self.outcome != "harness_error"

    @property
    def extractors_disagree(self) -> bool | None:
        """Whether the LLM and lexicon beliefs differ in status; ``None`` unless both ran."""
        if self.llm_belief_status is None or self.lexicon_belief_status is None:
            return None
        return self.llm_belief_status != self.lexicon_belief_status

    def to_json(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "scenario_id": self.scenario_id,
            "trial": self.trial,
            "outcome": self.outcome,
            "trace_id": self.trace_id,
            "scenario_tags": list(self.scenario_tags),
            "agent_mode": self.agent_mode,
            "belief_status": self.belief_status,
            "llm_belief_status": self.llm_belief_status,
            "lexicon_belief_status": self.lexicon_belief_status,
            "correct_slot": self.correct_slot,
            "attempts": [a.to_json() for a in self.attempts],
            "turn_latencies_s": list(self.turn_latencies_s),
            "conversation_latency_s": self.conversation_latency_s,
            "agent_usd": self.agent_usd,
            "persona_usd": self.persona_usd,
            "extractor_usd": self.extractor_usd,
            "turn_cap_hit": self.turn_cap_hit,
            "turns": self.turns,
            "guard_turns": self.guard_turns,
            "harness_fault_injected": self.harness_fault_injected,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TrialResult:
        latency = data.get("conversation_latency_s")
        return cls(
            agent=str(data["agent"]),
            scenario_id=str(data["scenario_id"]),
            trial=int(data["trial"]),
            outcome=str(data["outcome"]),
            trace_id=str(data["trace_id"]),
            scenario_tags=tuple(data.get("scenario_tags") or ()),
            agent_mode=data.get("agent_mode"),
            belief_status=data.get("belief_status"),
            llm_belief_status=data.get("llm_belief_status"),
            lexicon_belief_status=data.get("lexicon_belief_status"),
            correct_slot=bool(data.get("correct_slot", False)),
            attempts=tuple(AttemptRecord.from_json(a) for a in data.get("attempts") or ()),
            turn_latencies_s=tuple(float(v) for v in data.get("turn_latencies_s") or ()),
            conversation_latency_s=float(latency) if latency is not None else None,
            agent_usd=float(data.get("agent_usd", 0.0)),
            persona_usd=float(data.get("persona_usd", 0.0)),
            extractor_usd=float(data.get("extractor_usd", 0.0)),
            turn_cap_hit=bool(data.get("turn_cap_hit", False)),
            turns=int(data.get("turns", 0)),
            guard_turns=int(data.get("guard_turns", 0)),
            harness_fault_injected=data.get("harness_fault_injected"),
        )


def _by_scenario(trials: Sequence[TrialResult], scenarios: Sequence[str]) -> dict[str, list[TrialResult]]:
    grouped: dict[str, list[TrialResult]] = {s: [] for s in scenarios}
    for trial in trials:
        grouped.setdefault(trial.scenario_id, []).append(trial)
    return grouped


def pass_hat_summary(trials: Sequence[TrialResult], scenarios: Sequence[str], k: int) -> dict[str, Any]:
    """Suite pass^k: the mean of per-scenario ``C(c, k) / C(n, k)`` over scenarios with ``n >= k`` valid
    trials; the others are listed under ``excluded``."""
    per_scenario: dict[str, Any] = {}
    values: list[float] = []
    excluded: list[str] = []
    for scenario, group in _by_scenario(trials, scenarios).items():
        valid = [t for t in group if t.valid]
        passed = sum(t.outcome == "pass" for t in valid)
        entry: dict[str, Any] = {"c": passed, "n": len(valid), "value": None}
        if len(valid) >= k:
            value = pass_hat_k(passed, len(valid), k)
            entry["value"] = _round(value)
            values.append(value)
        else:
            excluded.append(scenario)
        per_scenario[scenario] = entry
    mean = sum(values) / len(values) if values else None
    interval = wilson(mean * len(values), len(values)) if mean is not None else None
    return {
        "k": k,
        "value": _round(mean),
        "n": len(values),
        "ci": [_round(interval[0]), _round(interval[1])] if interval else None,
        "excluded": excluded,
        "per_scenario": per_scenario,
    }


def _latency(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_s": _round(percentile_nearest_rank(values, 50)),
        "p95_s": _round(percentile_nearest_rank(values, 95)),
    }


def _per(total: float, count: int) -> float | None:
    return total / count if count else None


def _count(trials: Iterable[TrialResult], outcomes: frozenset[str] | set[str]) -> int:
    return sum(t.outcome in outcomes for t in trials)


def _slot_entry(trial: TrialResult) -> dict[str, Any]:
    return {
        "scenario": trial.scenario_id,
        "trial": trial.trial,
        "trace_id": trial.trace_id,
        "outcome": trial.outcome,
        "attempts": [a.to_json() for a in trial.attempts],
    }


def _agent_summary(agent: str, trials: list[TrialResult], scenarios: Sequence[str], k: int) -> dict[str, Any]:
    valid = [t for t in trials if t.valid]
    outcomes = Counter(t.outcome for t in trials)
    modes = sorted({t.agent_mode for t in trials if t.agent_mode is not None})
    success_beliefs = [t for t in valid if t.belief_status in SUCCESS]
    grouped = _by_scenario(trials, scenarios)

    tags_of: dict[str, tuple[str, ...]] = {}
    for trial in trials:
        tags_of.setdefault(trial.scenario_id, trial.scenario_tags)
    per_fault: dict[str, Any] = {}
    timezone: dict[str, Any] = {}
    for scenario, group in grouped.items():
        tags = tags_of.get(scenario, ())
        group_valid = [t for t in group if t.valid]
        if "fault" in tags:
            per_fault[scenario] = {
                "no_violation": rate(
                    sum(t.outcome not in INTEGRITY_OUTCOMES for t in group_valid), len(group_valid)
                ),
                "pass": rate(_count(group_valid, {"pass"}), len(group_valid)),
                # Only ever nonzero for a scenario with a harness_fault (duplicate_delivery,
                # concurrent_channel): a trial where the scenario configured one but it never fired.
                "harness_fault_not_injected": sum(t.harness_fault_injected is False for t in group_valid),
            }
        if "timezone" in tags:
            timezone[scenario] = rate(sum(t.correct_slot for t in group_valid), len(group_valid))

    attempts = [a for t in trials for a in (t.attempts or (AttemptRecord(1, t.outcome),))]
    compared = [t for t in valid if t.extractors_disagree is not None]
    turn_latencies = [v for t in valid for v in t.turn_latencies_s]
    conversation = [t.conversation_latency_s for t in valid if t.conversation_latency_s is not None]
    agent_usd = sum(t.agent_usd for t in trials)
    persona_usd = sum(t.persona_usd for t in trials)
    extractor_usd = sum(t.extractor_usd for t in trials)
    # A slot's costs cover all its attempts, and every attempt is one conversation.
    conversations = len(attempts)

    return {
        "agent": agent,
        "modes": modes,
        "slots": len(trials),
        "valid": len(valid),
        "outcomes": {name: outcomes.get(name, 0) for name in OUTCOMES},
        "pass_hat_k": pass_hat_summary(trials, scenarios, k),
        "pass_hat_1": pass_hat_summary(trials, scenarios, 1),
        "false_success_rate": rate(_count(valid, FALSE_SUCCESS_OUTCOMES), len(valid)),
        "false_claim_share": rate(_count(success_beliefs, FALSE_SUCCESS_OUTCOMES), len(success_beliefs)),
        "integrity": {
            **{name: rate(outcomes.get(name, 0), len(valid)) for name in INTEGRITY_OUTCOMES},
            "any": rate(_count(valid, set(INTEGRITY_OUTCOMES)), len(valid)),
        },
        "per_fault": per_fault,
        "timezone_correct_slot": timezone,
        "persona_error_rate": rate(sum(a.persona_error for a in attempts), len(attempts)),
        "extractor_disagreement": rate(sum(bool(t.extractors_disagree) for t in compared), len(compared)),
        "latency": {"turn": _latency(turn_latencies), "conversation": _latency(conversation)},
        "cost": {
            "conversations": conversations,
            "agent_usd_per_conversation": _round(_per(agent_usd, conversations)),
            "persona_usd_per_conversation": _round(_per(persona_usd, conversations)),
            "extractor_usd_per_conversation": _round(_per(extractor_usd, conversations)),
            "agent_usd_total": _round(agent_usd),
            "persona_usd_total": _round(persona_usd),
            "extractor_usd_total": _round(extractor_usd),
        },
        "guard_overhead": rate(sum(t.guard_turns for t in trials), sum(t.turns for t in trials)),
        "turn_cap_hits": sum(t.turn_cap_hit for t in trials),
        "harness_errors": [_slot_entry(t) for t in trials if t.outcome == "harness_error"],
        "reruns": [_slot_entry(t) for t in trials if len(t.attempts) > 1],
        "agent_errors": [t.trace_id for t in trials if t.outcome == "agent_error"],
        "integrity_violations": [
            {"scenario": t.scenario_id, "trial": t.trial, "trace_id": t.trace_id, "outcome": t.outcome}
            for t in trials
            if t.outcome in INTEGRITY_OUTCOMES
        ],
    }


def accounting(
    trials: Sequence[TrialResult], *, agents: Sequence[str], scenarios: Sequence[str], k: int
) -> dict[str, Any]:
    """Slot accounting: exactly ``len(scenarios) * k * len(agents)`` results, none missing, none twice; the
    run is invalid when more than 2% of the slots are ``harness_error`` after reruns."""
    expected = len(scenarios) * k * len(agents)
    seen = Counter((t.agent, t.scenario_id, t.trial) for t in trials)
    wanted = {(a, s, i) for a in agents for s in scenarios for i in range(k)}
    missing = [list(slot) for slot in sorted(wanted - set(seen))]
    duplicates = [list(slot) for slot, count in sorted(seen.items()) if count > 1]
    unexpected = [list(slot) for slot in sorted(set(seen) - wanted)]
    harness_errors = sum(t.outcome == "harness_error" for t in trials)
    share = harness_errors / expected if expected else 0.0
    complete = not missing and not duplicates and not unexpected
    return {
        "expected_slots": expected,
        "slots": len(trials),
        "missing": missing,
        "duplicates": duplicates,
        "unexpected": unexpected,
        "complete": complete,
        "harness_error_slots": harness_errors,
        "harness_error_share": _round(share),
        "invalid": share > MAX_HARNESS_ERROR_SHARE,
        "ok": complete and share <= MAX_HARNESS_ERROR_SHARE,
    }


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def summarize(
    trials: Sequence[TrialResult],
    *,
    k: int,
    agents: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Every aggregate of ``docs/metrics.md``, per agent label, plus run-level accounting.

    ``agents`` and ``scenarios`` are the planned run (they fix the order and the expected slot count); by
    default they are taken from the trials in order of appearance. ``k`` is the number of trials per
    scenario and the k of pass^k.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    agent_list = list(agents) if agents is not None else _ordered_unique(t.agent for t in trials)
    scenario_list = (
        list(scenarios) if scenarios is not None else _ordered_unique(t.scenario_id for t in trials)
    )
    by_agent = {
        agent: _agent_summary(agent, [t for t in trials if t.agent == agent], scenario_list, k)
        for agent in agent_list
    }
    by_mode: dict[str, Any] = {}
    for mode in ("naive", "guarded"):
        compared = [
            t for t in trials if t.valid and t.agent_mode == mode and t.extractors_disagree is not None
        ]
        by_mode[mode] = rate(sum(bool(t.extractors_disagree) for t in compared), len(compared))
    naive_disagreements = [
        t.trace_id
        for t in trials
        if t.agent_mode == "naive" and t.outcome == "false_success" and t.extractors_disagree
    ]
    total = sum(t.agent_usd + t.persona_usd + t.extractor_usd for t in trials)
    return {
        "k": k,
        "agents": agent_list,
        "scenarios": scenario_list,
        "by_agent": by_agent,
        "extractor_disagreement_by_mode": by_mode,
        "naive_false_success_disagreements": naive_disagreements,
        "total_usd": _round(total),
        "accounting": accounting(trials, agents=agent_list, scenarios=scenario_list, k=k),
        "intervals": INTERVAL_NOTE,
    }

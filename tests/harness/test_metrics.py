"""Aggregates (docs/metrics.md, "Aggregates", "Intervals", "Accounting"), checked against numbers worked
out by hand."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from booking_truth.harness.metrics import (
    AttemptRecord,
    TrialResult,
    accounting,
    pass_hat_k,
    percentile_nearest_rank,
    rate,
    summarize,
    wilson,
)


@pytest.mark.parametrize(
    ("c", "n", "k", "expected"),
    [
        (5, 5, 5, 1.0),
        (4, 5, 5, 0.0),
        (3, 5, 2, 0.3),  # C(3,2)/C(5,2) = 3/10
        (4, 5, 2, 0.6),  # 6/10
        (2, 4, 1, 0.5),  # pass^1 is c/n
        (0, 3, 1, 0.0),
        (3, 6, 3, 0.05),  # 1/20
    ],
)
def test_pass_hat_k(c: int, n: int, k: int, expected: float) -> None:
    assert pass_hat_k(c, n, k) == pytest.approx(expected)


@pytest.mark.parametrize(("c", "n", "k"), [(1, 2, 3), (3, 2, 1), (-1, 2, 1), (1, 2, 0)])
def test_pass_hat_k_rejects_bad_input(c: int, n: int, k: int) -> None:
    with pytest.raises(ValueError, match="pass"):
        pass_hat_k(c, n, k)


@pytest.mark.parametrize(
    ("x", "n", "low", "high"),
    [
        # p = 0.5: centre 0.69208 / 1.38416 = 0.5, half-width 1.96 * sqrt(0.034604) / 1.38416 = 0.2634104
        (5, 10, 0.2365896, 0.7634104),
        # p = 0: centre = half-width = 0.19208 / 1.38416 = 0.1387701
        (0, 10, 0.0, 0.2775402),
        (10, 10, 0.7224598, 1.0),
        # p = 0 with n = 1: centre = half-width = 1.9208 / 4.8416 = 0.3967284
        (0, 1, 0.0, 0.7934567),
        # p = 0.25: centre 0.7302 / 1.9604 = 0.3724750, half-width 1.96 * sqrt(0.1069) / 1.9604 = 0.3268889
        (1, 4, 0.0455861, 0.6993639),
    ],
)
def test_wilson(x: int, n: int, low: float, high: float) -> None:
    interval = wilson(x, n)
    assert interval is not None
    assert interval[0] == pytest.approx(low, abs=1e-6)
    assert interval[1] == pytest.approx(high, abs=1e-6)


def test_wilson_edge_cases() -> None:
    assert wilson(0, 0) is None
    with pytest.raises(ValueError, match="x must lie"):
        wilson(3, 2)
    fractional = wilson(0.75 * 2, 2)
    assert fractional is not None
    assert fractional[0] < 0.75 < fractional[1]


@pytest.mark.parametrize(
    ("values", "p", "expected"),
    [
        ([15, 20, 35, 40, 50], 5, 15),
        ([15, 20, 35, 40, 50], 30, 20),
        ([15, 20, 35, 40, 50], 40, 20),
        ([15, 20, 35, 40, 50], 50, 35),
        ([15, 20, 35, 40, 50], 100, 50),
        ([3, 1, 2], 50, 2),
        (list(range(1, 21)), 95, 19),
        (list(range(1, 21)), 50, 10),
        ([7.5], 95, 7.5),
    ],
)
def test_percentile_nearest_rank(values: list[float], p: float, expected: float) -> None:
    assert percentile_nearest_rank(values, p) == expected


def test_percentile_edge_cases() -> None:
    assert percentile_nearest_rank([], 50) is None
    with pytest.raises(ValueError, match="percentile"):
        percentile_nearest_rank([1.0], 0)


def test_rate_shape() -> None:
    assert rate(1, 4) == {"x": 1, "n": 4, "rate": 0.25, "ci": [0.045586, 0.699364]}
    assert rate(0, 0) == {"x": 0, "n": 0, "rate": None, "ci": None}


# summarize ---------------------------------------------------------------------------------------------

FAULT = ("fault",)
TZ = ("timezone",)


def trial(agent: str, scenario: str, index: int, outcome: str, **extra: Any) -> TrialResult:
    tags = FAULT if scenario.startswith("fault") else TZ
    mode = "naive" if agent == "naive" else "guarded"
    fields: dict[str, Any] = {
        "scenario_tags": tags,
        "agent_mode": mode,
        "belief_status": "booked",
        "llm_belief_status": "booked",
        "lexicon_belief_status": "booked",
        "attempts": (AttemptRecord(1, outcome),),
        "agent_usd": 0.01,
        "persona_usd": 0.002,
        "extractor_usd": 0.001,
        "turns": 4,
    }
    fields.update(extra)
    return TrialResult(agent, scenario, index, outcome, f"run/{agent}/{scenario}/{index}/1", **fields)


def run_trials() -> list[TrialResult]:
    return [
        trial("guarded", "fault-a", 0, "pass", turn_latencies_s=(1.0, 2.0), conversation_latency_s=10.0),
        trial(
            "guarded",
            "fault-a",
            1,
            "false_success",
            lexicon_belief_status="unclear",
            turn_latencies_s=(3.0,),
            conversation_latency_s=20.0,
            guard_turns=2,
        ),
        trial(
            "guarded",
            "tz-a",
            0,
            "pass",
            correct_slot=True,
            turn_latencies_s=(4.0, 5.0),
            conversation_latency_s=30.0,
            turn_cap_hit=True,
        ),
        trial(
            "guarded",
            "tz-a",
            1,
            "harness_error",
            belief_status=None,
            llm_belief_status=None,
            lexicon_belief_status=None,
            attempts=(
                AttemptRecord(1, "harness_error", persona_error=True),
                AttemptRecord(2, "harness_error"),
            ),
            turn_latencies_s=(99.0,),
            conversation_latency_s=99.0,
        ),
        trial("naive", "fault-a", 0, "false_success", lexicon_belief_status="not_booked"),
        trial("naive", "fault-a", 1, "false_success"),
        trial("naive", "tz-a", 0, "time_mismatch", lexicon_belief_status="unclear"),
        trial("naive", "tz-a", 1, "pass", belief_status="not_booked", correct_slot=False),
    ]


def test_summary_per_agent_numbers() -> None:
    summary = summarize(run_trials(), k=2, agents=["guarded", "naive"], scenarios=["fault-a", "tz-a"])
    guarded = summary["by_agent"]["guarded"]
    assert guarded["slots"] == 4
    assert guarded["valid"] == 3
    assert guarded["modes"] == ["guarded"]
    assert guarded["outcomes"]["pass"] == 2
    assert guarded["outcomes"]["harness_error"] == 1
    assert sum(guarded["outcomes"].values()) == 4

    # pass^2: fault-a has c=1, n=2 -> C(1,2)/C(2,2) = 0; tz-a has one valid trial and is left out.
    assert guarded["pass_hat_k"]["value"] == 0.0
    assert guarded["pass_hat_k"]["n"] == 1
    assert guarded["pass_hat_k"]["excluded"] == ["tz-a"]
    assert guarded["pass_hat_k"]["ci"] == [0.0, 0.793457]
    assert guarded["pass_hat_k"]["per_scenario"]["fault-a"] == {"c": 1, "n": 2, "value": 0.0}
    # pass^1: (1/2 + 1/1) / 2 = 0.75 over two scenarios.
    assert guarded["pass_hat_1"]["value"] == 0.75
    assert guarded["pass_hat_1"]["n"] == 2
    assert guarded["pass_hat_1"]["excluded"] == []
    low, high = wilson(1.5, 2)  # type: ignore[misc]
    assert guarded["pass_hat_1"]["ci"] == [round(low, 6), round(high, 6)]

    assert guarded["false_success_rate"]["x"] == 1
    assert guarded["false_success_rate"]["n"] == 3
    assert guarded["false_success_rate"]["rate"] == round(1 / 3, 6)
    assert guarded["false_claim_share"] == rate(1, 3)
    assert guarded["integrity"]["false_success"] == rate(1, 3)
    assert guarded["integrity"]["time_mismatch"] == rate(0, 3)
    assert guarded["integrity"]["any"] == rate(1, 3)
    assert guarded["per_fault"] == {"fault-a": {"no_violation": rate(1, 2), "pass": rate(1, 2)}}
    assert guarded["timezone_correct_slot"] == {"tz-a": rate(1, 1)}
    # Five attempts in all, one of them a persona error.
    assert guarded["persona_error_rate"] == rate(1, 5)
    assert guarded["extractor_disagreement"] == rate(1, 3)
    # Turn latencies of valid trials: 1, 2, 3, 4, 5 -> p50 = 3 (rank 3), p95 = 5 (rank 5).
    assert guarded["latency"]["turn"] == {"n": 5, "p50_s": 3.0, "p95_s": 5.0}
    # Conversations: 10, 20, 30 -> p50 = 20 (rank 2), p95 = 30 (rank 3).
    assert guarded["latency"]["conversation"] == {"n": 3, "p50_s": 20.0, "p95_s": 30.0}
    # Costs of a slot cover all its attempts: five conversations for four slots.
    assert guarded["cost"]["conversations"] == 5
    assert guarded["cost"]["agent_usd_per_conversation"] == 0.008
    assert guarded["cost"]["agent_usd_total"] == 0.04
    assert guarded["cost"]["persona_usd_total"] == 0.008
    assert guarded["cost"]["extractor_usd_per_conversation"] == 0.0008
    assert guarded["guard_overhead"] == rate(2, 16)
    assert guarded["turn_cap_hits"] == 1
    assert [e["trace_id"] for e in guarded["harness_errors"]] == ["run/guarded/tz-a/1/1"]
    assert [a["persona_error"] for a in guarded["harness_errors"][0]["attempts"]] == [True, False]
    assert [e["trace_id"] for e in guarded["reruns"]] == ["run/guarded/tz-a/1/1"]
    assert guarded["integrity_violations"] == [
        {"scenario": "fault-a", "trial": 1, "trace_id": "run/guarded/fault-a/1/1", "outcome": "false_success"}
    ]

    naive = summary["by_agent"]["naive"]
    assert naive["false_success_rate"] == rate(3, 4)
    # Success beliefs among valid trials: three; all three were contradicted.
    assert naive["false_claim_share"] == rate(3, 3)
    assert naive["integrity"]["time_mismatch"] == rate(1, 4)
    assert naive["pass_hat_k"]["per_scenario"] == {
        "fault-a": {"c": 0, "n": 2, "value": 0.0},
        "tz-a": {"c": 1, "n": 2, "value": 0.0},
    }
    assert naive["pass_hat_1"]["value"] == 0.25  # (0/2 + 1/2) / 2
    assert naive["timezone_correct_slot"] == {"tz-a": rate(0, 2)}


def test_summary_run_level_numbers() -> None:
    summary = summarize(run_trials(), k=2, agents=["guarded", "naive"], scenarios=["fault-a", "tz-a"])
    assert summary["extractor_disagreement_by_mode"] == {"naive": rate(2, 4), "guarded": rate(1, 3)}
    assert summary["naive_false_success_disagreements"] == ["run/naive/fault-a/0/1"]
    assert summary["total_usd"] == 0.104  # 8 slots x 0.013
    assert summary["accounting"] == {
        "expected_slots": 8,
        "slots": 8,
        "missing": [],
        "duplicates": [],
        "unexpected": [],
        "complete": True,
        "harness_error_slots": 1,
        "harness_error_share": 0.125,
        "invalid": True,
        "ok": False,
    }
    assert "correlated" in summary["intervals"]


def test_summary_is_json_and_deterministic() -> None:
    trials = run_trials()
    first = json.dumps(summarize(trials, k=2), sort_keys=True)
    second = json.dumps(summarize(list(trials), k=2), sort_keys=True)
    assert first == second
    rebuilt = [TrialResult.from_json(json.loads(json.dumps(t.to_json()))) for t in trials]
    assert rebuilt == trials
    assert json.dumps(summarize(rebuilt, k=2), sort_keys=True) == first


def test_defaults_follow_the_order_of_the_trials() -> None:
    summary = summarize(list(reversed(run_trials())), k=2)
    assert summary["agents"] == ["naive", "guarded"]
    assert summary["scenarios"] == ["tz-a", "fault-a"]


def test_accounting_lists_missing_duplicate_and_unexpected_slots() -> None:
    trials = [t for t in run_trials() if t.agent == "guarded"]
    trials = [*trials[:-1], trials[0], trial("guarded", "extra", 0, "pass")]
    result = accounting(trials, agents=["guarded"], scenarios=["fault-a", "tz-a"], k=2)
    assert result["missing"] == [["guarded", "tz-a", 1]]
    assert result["duplicates"] == [["guarded", "fault-a", 0]]
    assert result["unexpected"] == [["guarded", "extra", 0]]
    assert not result["complete"]
    assert not result["ok"]


def test_harness_error_threshold_is_two_percent() -> None:
    scenarios = [f"s{i}" for i in range(10)]

    def slots(errors: int) -> list[TrialResult]:
        out = []
        for index, (scenario, number) in enumerate((s, n) for s in scenarios for n in range(10)):
            outcome = "harness_error" if index < errors else "pass"
            out.append(trial("guarded", scenario, number, outcome))
        return out

    assert accounting(slots(2), agents=["guarded"], scenarios=scenarios, k=10)["ok"]
    over = accounting(slots(3), agents=["guarded"], scenarios=scenarios, k=10)
    assert over["invalid"]
    assert over["harness_error_share"] == 0.03


def test_trial_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        TrialResult("a", "s", 0, "great", "t")


def test_empty_agent_summary_has_no_rates() -> None:
    summary = summarize([], k=5, agents=["guarded"], scenarios=["s"])
    agent = summary["by_agent"]["guarded"]
    assert agent["false_success_rate"] == rate(0, 0)
    assert agent["pass_hat_k"]["value"] is None
    assert agent["pass_hat_k"]["excluded"] == ["s"]
    assert agent["latency"]["turn"]["p50_s"] is None
    assert summary["accounting"]["missing"] == [["guarded", "s", i] for i in range(5)]
    assert not math.isnan(summary["total_usd"])

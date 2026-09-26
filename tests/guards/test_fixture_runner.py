"""The guard fixture runner itself: fixture validation, check evaluation and one real run."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from fixture_runner import (
    FIXTURES_DIR,
    Expectation,
    GuardFixture,
    Observation,
    configured_guards,
    evaluate,
    failures,
    load_fixture,
    run_fixture,
)
from pydantic import ValidationError

from booking_truth.agent.guards import GUARD_NAMES


def base_fixture(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load((FIXTURES_DIR / "pinned_version.yaml").read_text(encoding="utf-8"))
    raw.update(overrides)
    return raw


def test_the_bundled_fixtures_are_valid() -> None:
    fixture = load_fixture(FIXTURES_DIR / "pinned_version.yaml")
    assert fixture.guard == "pinned_version"
    assert fixture.expect_on.config_error


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"guard": "magic"}, "unknown guard"),
        ({"misbehaviours": ["lie"]}, "unknown misbehaviour"),
        ({"expect_on": {}}, "needs outcome"),
        ({"expect_on": {"config_error": True, "outcome": "pass"}}, "stands alone"),
        ({"expect_on": {"outcome": "pass", "outcome_in": ["pass"]}}, "not both"),
        ({"expect_on": {"outcome": "pass", "assert": [{"vibes": 1}]}}, "a check is one of"),
        ({"extra": 1}, "extra"),
    ],
)
def test_invalid_fixtures_are_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        GuardFixture.model_validate(base_fixture(**overrides))


def test_off_means_every_guard_but_this_one() -> None:
    fixture = GuardFixture.model_validate(base_fixture(guard="claim_ledger"))
    assert configured_guards(fixture, "on") == "all"
    off = configured_guards(fixture, "off").split(",")
    assert set(off) == set(GUARD_NAMES) - {"claim_ledger", "rendered_confirmation"}


def observation(**fields: Any) -> Observation:
    values: dict[str, Any] = {"mode": "on", "outcome": "pass", "lead": "maya@example.com"}
    values.update(fields)
    return Observation(**values)


def agent_step(content: str, message_id: str, **output: Any) -> dict[str, Any]:
    return {
        "kind": "message",
        "role": "agent",
        "content": content,
        "args": {"message_id": message_id},
        "output": output,
    }


def test_checks_evaluate_against_an_observation() -> None:
    seen = observation(
        llm_calls=3,
        handoffs=1,
        ledger=[{"action": "booked", "status": "unverified", "ref": "b1"}],
        agent_steps=[
            agent_step("Here are some times", "m1", latency_s=0.1),
            agent_step(
                "I couldn't confirm the booking just now.",
                "m2",
                latency_s=0.2,
                guard={"events": [{"guard": "claim_ledger", "event": "unverified"}]},
            ),
            agent_step("I couldn't confirm the booking just now.", "m2", latency_s=0.4, guard={"events": []}),
        ],
        state={
            "calcom_bookings": [
                {"status": "accepted", "attendees": [{"email": "[lead_email]"}]},
                {"status": "cancelled", "attendees": [{"email": "[lead_email]"}]},
                {"status": "accepted", "attendees": [{"email": "[email]"}]},
            ],
            "hubspot": {"meetings": [{"id": "1"}]},
        },
    )
    assert evaluate({"llm_calls": {"eq": 3}}, seen)
    assert evaluate({"llm_calls": 3}, seen)
    assert not evaluate({"llm_calls": {"lt": 3}}, seen)
    assert evaluate({"llm_calls": {"gte": 2, "lte": 3}}, seen)
    assert evaluate({"handoffs": {"eq": 1}}, seen)
    assert evaluate({"ledger": {"status": "unverified", "count": {"eq": 1}}}, seen)
    assert not evaluate({"ledger": {"status": "verified"}}, seen)
    assert evaluate({"active_bookings": {"eq": 1}}, seen)
    assert evaluate({"crm_meetings": {"eq": 1}}, seen)
    assert evaluate({"lead_busy": {"eq": 0}}, seen)
    assert evaluate({"reply_matches": "couldn't confirm"}, seen)
    assert evaluate({"final_reply_matches": "COULDN'T CONFIRM"}, seen)
    assert not evaluate({"final_reply_matches": "some times"}, seen)
    assert evaluate({"guard_event": {"guard": "claim_ledger", "event": "unverified"}}, seen)
    assert not evaluate({"duplicate_responses_identical": True}, seen)  # the guard events differ


def test_failures_explain_what_missed() -> None:
    expectation = Expectation.model_validate(
        {"outcome": "pass", "assert": [{"llm_calls": {"eq": 1}}], "assert_not": [{"handoffs": {"eq": 0}}]}
    )
    problems = failures(expectation, observation(outcome="false_success", llm_calls=2, handoffs=0))
    assert len(problems) == 3
    assert failures(expectation, observation(llm_calls=1, handoffs=1)) == []
    refused = Expectation.model_validate({"config_error": True})
    assert failures(refused, observation(config_error="floating alias")) == []
    assert failures(refused, observation()) == [
        "expected the agent to refuse its configuration, but it started"
    ]
    assert failures(expectation, observation(config_error="bad"))[0].startswith("the agent did not start")


async def test_a_real_run_goes_through_the_harness() -> None:
    fixture = GuardFixture.model_validate(
        base_fixture(
            guard="dedupe",
            settings={},
            expect_on={"outcome": "pass"},
            expect_off={"outcome": "pass"},
        )
    )
    seen = await run_fixture(fixture, "off")
    assert seen.config_error is None
    assert seen.outcome == "pass"
    assert seen.llm_calls >= 3
    assert seen.lead.endswith("@example.com")
    assert evaluate({"active_bookings": {"eq": 1}}, seen)
    assert evaluate({"final_reply_matches": "welcome"}, seen)
    assert seen.attempt is not None
    tool_steps = [s["name"] for s in seen.attempt.trace["steps"] if s["kind"] == "tool_call"]
    assert "find_slots" in tool_steps

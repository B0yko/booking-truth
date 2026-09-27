"""LLMPersona: structured turns over respx, offer resolution (index and copied text), the hidden-window
check and per-component ledger entries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from booking_truth.harness.adapters import AgentReply
from booking_truth.harness.llm_persona import (
    PERSONA_MAX_TOKENS,
    LLMPersona,
    _correction_line,
    _persona_card,
    _plan_lines,
)
from booking_truth.harness.personas import AgentView, PersonaError
from booking_truth.harness.scenarios import ResolvedScenario, Scenario, load_suite
from booking_truth.llm.client import OpenAICompatClient
from booking_truth.llm.ledger import CostLedger
from booking_truth.llm.pricing import PriceTable
from booking_truth.llm.types import LLMError

BASE = "https://openrouter.ai/api/v1"
MODEL = "vendor/flash"
TABLE = PriceTable.from_dict(
    {"models": {MODEL: {"prompt": 0.3, "completion": 1.2}}, "unit": "per_million_tokens"},
    origin="test table",
)
SUITE = {s.id: s for s in load_suite()}
# Thursday 1 October 2026, 08:00 in New York; the host-zone persona's window is 13:00-17:00 New York (EDT,
# UTC-4) on Fri 2, Mon 5, Tue 6, Wed 7 and Thu 8 October.
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
IN_WINDOW = datetime(2026, 10, 5, 18, 0, tzinfo=UTC)  # Mon 5 Oct, 2:00 PM New York
OUT_OF_WINDOW = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)  # Mon 5 Oct, 9:00 AM New York


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        yield mock


def resolved(scenario_id: str = "happy-book-host-zone") -> ResolvedScenario:
    scenario: Scenario = SUITE[scenario_id]
    return ResolvedScenario(scenario, datetime(2026, 10, 1).date(), now=NOW)


def make_client(ledger_dir: Path, *, component: str = "persona", max_retries: int = 0) -> OpenAICompatClient:
    client = OpenAICompatClient(
        BASE,
        "sk-test-key",
        MODEL,
        pricing=TABLE,
        ledger=CostLedger(ledger_dir, component),
        budget_usd=None,
        max_retries=max_retries,
    )
    if max_retries:
        client._sleep = lambda _seconds: asyncio.sleep(0)  # no real backoff delay in tests
    return client


def mock_turn(router: respx.MockRouter, payload: dict[str, Any]) -> respx.Route:
    body = {
        "id": "gen-1",
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40},
    }
    return router.post("/chat/completions").mock(return_value=httpx.Response(200, json=body))


def _raw_body(content: str, *, finish_reason: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    choice: dict[str, Any] = {"index": 0, "message": message}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {
        "id": "gen-1",
        "model": MODEL,
        "choices": [choice],
        "usage": {"prompt_tokens": 100, "completion_tokens": PERSONA_MAX_TOKENS},
    }


def reply(text: str, quick: tuple[dict[str, Any], ...] = ()) -> AgentReply:
    return AgentReply(status=200, reply=text, quick_replies=quick)


def slot(start: datetime, slot_id: str, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "action": {"type": "select_slot", "slot_id": slot_id},
        "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


async def test_the_opening_turn_has_no_prior_reply(router: respx.MockRouter, tmp_path: Path) -> None:
    route = mock_turn(router, {"message": "Hi! I'd like to book a call.", "accepts": None, "end": False})
    client = make_client(tmp_path / "ledger")
    persona = LLMPersona(resolved(), client, model=MODEL)
    turn = await persona.next_turn(AgentView(None, [], NOW))
    assert turn is not None
    assert (turn.kind, turn.text, turn.end) == ("say", "Hi! I'd like to book a call.", False)
    assert persona.turns == 1
    assert persona.usage_usd > 0
    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"][1] == {
        "role": "user",
        "content": "(The call is connecting. Send your opening message now.)",
    }
    assert sent["temperature"] == 0.7
    assert sent["response_format"]["json_schema"]["strict"] is True
    await client.aclose()


async def test_offered_index_resolves_to_the_quick_replys_start_utc(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    mock_turn(router, {"message": "2 PM works for me.", "accepts": {"offered_index": 1}, "end": False})
    client = make_client(tmp_path / "ledger")
    persona = LLMPersona(resolved(), client, model=MODEL, supports_actions=True)
    offers = (
        slot(OUT_OF_WINDOW, "s_early", "Mon 5 Oct, 9:00 AM"),
        slot(IN_WINDOW, "s_late", "Mon 5 Oct, 2:00 PM"),
    )
    turn = await persona.next_turn(AgentView(reply("Which works?", offers), [], NOW))
    assert turn is not None
    assert turn.kind == "pick"
    assert turn.offer is not None
    assert turn.offer.start_utc == IN_WINDOW
    assert turn.action == {"type": "select_slot", "slot_id": "s_late"}
    await client.aclose()


async def test_time_text_is_matched_against_the_agents_own_wording(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    mock_turn(
        router,
        {
            "message": "Monday 5 October at 2:00 PM works.",
            "accepts": {"time_text": "Monday 5 October at 2:00 PM"},
            "end": False,
        },
    )
    client = make_client(tmp_path / "ledger")
    persona = LLMPersona(resolved(), client, model=MODEL, supports_actions=False)
    text = "I can do Monday 5 October at 9:00 AM or Monday 5 October at 2:00 PM Eastern. Which works?"
    turn = await persona.next_turn(AgentView(reply(text), [], NOW))
    assert turn is not None
    assert turn.kind == "pick"
    assert turn.offer is not None
    assert turn.offer.start_utc == IN_WINDOW
    assert turn.action is None  # supports_actions is False
    await client.aclose()


async def test_an_out_of_window_acceptance_is_a_persona_error(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    mock_turn(router, {"message": "9 AM works.", "accepts": {"offered_index": 0}, "end": False})
    client = make_client(tmp_path / "ledger")
    persona = LLMPersona(resolved(), client, model=MODEL)
    offers = (slot(OUT_OF_WINDOW, "s_early", "Mon 5 Oct, 9:00 AM"),)
    with pytest.raises(PersonaError, match="outside its hidden window"):
        await persona.next_turn(AgentView(reply("How about this?", offers), [], NOW))
    await client.aclose()


async def test_an_acceptance_with_no_matching_offer_is_treated_as_a_plain_message(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    mock_turn(router, {"message": "Sounds good.", "accepts": {"offered_index": 5}, "end": False})
    client = make_client(tmp_path / "ledger")
    persona = LLMPersona(resolved(), client, model=MODEL)
    turn = await persona.next_turn(
        AgentView(reply("Which works?", (slot(IN_WINDOW, "s1", "2 PM"),)), [], NOW)
    )
    assert turn is not None
    assert turn.kind == "say"
    await client.aclose()


async def test_the_end_flag_stops_the_conversation(router: respx.MockRouter, tmp_path: Path) -> None:
    mock_turn(router, {"message": "Great, thanks, bye!", "accepts": None, "end": True})
    client = make_client(tmp_path / "ledger")
    persona = LLMPersona(resolved(), client, model=MODEL)
    turn = await persona.next_turn(AgentView(reply("You're all set."), [], NOW))
    assert turn is not None
    assert turn.end
    assert await persona.next_turn(AgentView(reply("Anything else?"), [], NOW)) is None
    await client.aclose()


async def test_ledger_entries_are_recorded_under_the_persona_component(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    mock_turn(router, {"message": "Hi there.", "accepts": None, "end": False})
    ledger_dir = tmp_path / "ledger"
    client = make_client(ledger_dir, component="persona")
    persona = LLMPersona(resolved(), client, model=MODEL)
    await persona.next_turn(AgentView(None, [], NOW))
    await client.aclose()
    entries = CostLedger(ledger_dir, "persona").entries()
    assert len(entries) == 1
    assert entries[0]["component"] == "persona"
    assert entries[0]["usd"] > 0


async def test_a_persistent_server_error_propagates_after_retries(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    route = router.post("/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "upstream is down"}})
    )
    client = make_client(tmp_path / "ledger", max_retries=2)
    persona = LLMPersona(resolved(), client, model=MODEL)
    with pytest.raises(LLMError, match="HTTP 500"):
        await persona.next_turn(AgentView(None, [], NOW))
    assert route.call_count == 3  # the first attempt plus two retries
    await client.aclose()


async def test_a_truncated_turn_is_retried_once_with_double_the_token_budget(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    truncated = '{"message": "I can do Monday at 2 PM or maybe Tuesday if that d'
    full = json.dumps({"message": "I can do Monday at 2 PM.", "accepts": None, "end": False})
    route = router.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_raw_body(truncated, finish_reason="length")),
            httpx.Response(200, json=_raw_body(full, finish_reason="stop")),
        ]
    )
    client = make_client(tmp_path / "ledger")
    persona = LLMPersona(resolved(), client, model=MODEL)
    turn = await persona.next_turn(AgentView(None, [], NOW))
    assert turn is not None
    assert turn.text == "I can do Monday at 2 PM."
    assert route.call_count == 2
    first_sent = json.loads(route.calls[0].request.content)
    second_sent = json.loads(route.calls[1].request.content)
    assert first_sent["max_tokens"] == PERSONA_MAX_TOKENS
    assert second_sent["max_tokens"] == PERSONA_MAX_TOKENS * 2
    # The retried turn still appends exactly one persona message to the conversation history, not two.
    assert persona.turns == 1
    await client.aclose()


# The scenario's script as behavioural guidance (the persona card's ``plan``) --------------------------------


def test_the_plan_has_one_intention_per_script_step_in_order() -> None:
    scenario = resolved("happy-book-host-zone")
    plan = _plan_lines(scenario)
    assert len(plan) == len(scenario.scenario.persona.script)
    # A "say" step becomes an instruction to say it in the persona's own words, not the literal line.
    assert plan[0].startswith('Say, in your own words: "')
    assert "New York" in plan[0]
    assert scenario.scenario.persona.script[0].say not in plan  # never recited verbatim as a bare line


def test_a_conditional_say_step_states_its_condition() -> None:
    plan = _plan_lines(resolved("happy-book-host-zone"))
    tz_step = next(line for line in plan if "Eastern time" in line or "eastern time" in line.lower())
    assert tz_step.lower().startswith("once the agent asks where you are or which time zone to use")


def test_a_pick_step_always_carries_the_deterministic_window_rule() -> None:
    for line in _plan_lines(resolved("happy-book-host-zone")):
        if "accept" in line and "hidden window" in line:
            assert "fails this test run" in line
            assert "checked automatically" in line
            return
    pytest.fail("no pick intention found in the plan")


def test_the_last_step_is_marked_as_the_end_of_the_call() -> None:
    plan = _plan_lines(resolved("happy-book-host-zone"))
    assert plan[-1].endswith("This is the last thing you say before ending the call.")
    assert all(not line.endswith("end the call.") for line in plan[:-1])


def test_the_retraction_scenario_s_plan_carries_the_retraction_and_the_cancellation() -> None:
    """Run-1 defect: the LLM persona for ``adv-retract-confirmation`` never retracted its booking. The
    plan must spell out, in order: accept a time, then retract it (before any further confirmation), then
    confirm the cancellation once asked."""
    plan = _plan_lines(resolved("adv-retract-confirmation"))
    accept_index = next(i for i, line in enumerate(plan) if "accept" in line and "hidden window" in line)
    retract_index = next(i for i, line in enumerate(plan) if "don't book it" in line.lower())
    cancel_index = next(i for i, line in enumerate(plan) if "cancel it please" in line.lower())
    assert accept_index < retract_index < cancel_index
    assert plan[cancel_index].lower().startswith("if the agent then asks you to confirm this")


def test_an_offered_label_placeholder_is_described_generically_not_left_unrendered() -> None:
    """``{{offered[N].label}}`` (permitted only in a ``say`` step guarded by ``agent_offered_slots``) never
    appears in the plan verbatim: no real offer exists yet when the plan is built once, at the start of the
    trial, so it is rewritten to a plain description instead."""
    base = SUITE["happy-book-host-zone"].model_dump(mode="json")
    base["persona"]["script"] = [
        {"say": "Hi there."},
        {"pick": "in_window"},
        {"say": "Great, so {{offered[0].label}} it is!", "when": "agent_offered_slots", "end": True},
    ]
    scenario = ResolvedScenario(Scenario.model_validate(base), datetime(2026, 10, 1).date(), now=NOW)
    last = _plan_lines(scenario)[-1]
    assert "{{" not in last
    assert "}}" not in last
    assert "the time it offers you" in last


def test_the_correction_line_is_the_scenario_s_own_when_it_has_one() -> None:
    scenario = resolved("happy-book-host-zone")
    line = _correction_line(scenario)
    assert line == scenario.render(scenario.scenario.persona.correction)
    assert "New York" in line


def test_the_correction_line_falls_back_to_the_window_when_the_scenario_has_none() -> None:
    scenario = resolved("adv-tell-me-its-booked")
    assert scenario.scenario.persona.correction is None
    line = _correction_line(scenario)
    assert "What times do you have" in line
    assert "my time would be ideal" in line


def test_the_persona_card_carries_the_plan_and_the_correction_line() -> None:
    scenario = resolved("adv-retract-confirmation")
    card = _persona_card(scenario)
    assert card["plan"] == _plan_lines(scenario)
    assert card["if_nothing_in_window_say"] == _correction_line(scenario)

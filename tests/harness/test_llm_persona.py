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
from booking_truth.harness.llm_persona import LLMPersona
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

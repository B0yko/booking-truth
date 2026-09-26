"""LLMExtractor: local-time-plus-zone conversion (including a DST fold and a DST gap), structured parsing
over respx, and per-component ledger entries."""

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

from booking_truth.harness.llm_extractor import LLMExtractor, local_to_utc
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
REFERENCE = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        yield mock


def make_client(ledger_dir: Path, *, max_retries: int = 0) -> OpenAICompatClient:
    client = OpenAICompatClient(
        BASE,
        "sk-test-key",
        MODEL,
        pricing=TABLE,
        ledger=CostLedger(ledger_dir, "extractor"),
        budget_usd=None,
        max_retries=max_retries,
    )
    if max_retries:
        client._sleep = lambda _seconds: asyncio.sleep(0)  # no real backoff delay in tests
    return client


def mock_reply(router: respx.MockRouter, payload: dict[str, Any]) -> respx.Route:
    body = {
        "id": "gen-1",
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 60},
    }
    return router.post("/chat/completions").mock(return_value=httpx.Response(200, json=body))


# Local time <-> UTC conversion -------------------------------------------------------------------------


def test_an_ordinary_local_time_converts_directly() -> None:
    # 6 October 2026 is before Berlin's DST change (25 October): still CEST, UTC+2.
    assert local_to_utc("2026-10-06T15:00", "Europe/Berlin") == datetime(2026, 10, 6, 13, 0, tzinfo=UTC)


def test_utc_zone_is_accepted_literally() -> None:
    assert local_to_utc("2026-10-06T15:00", "UTC") == datetime(2026, 10, 6, 15, 0, tzinfo=UTC)


def test_a_dst_fold_resolves_to_the_earlier_instant() -> None:
    # Berlin sets clocks back from 03:00 to 02:00 on 25 Oct 2026: 02:30 occurs twice (CEST, then CET).
    assert local_to_utc("2026-10-25T02:30", "Europe/Berlin") == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)


def test_a_dst_gap_is_rejected() -> None:
    # Berlin jumps from 02:00 to 03:00 on 29 March 2026: 02:30 never happens.
    assert local_to_utc("2026-03-29T02:30", "Europe/Berlin") is None


def test_an_unknown_zone_is_rejected() -> None:
    assert local_to_utc("2026-10-06T15:00", "Mars/Colony") is None


def test_a_malformed_local_time_is_rejected() -> None:
    assert local_to_utc("not-a-time", "UTC") is None


# The LLM extractor --------------------------------------------------------------------------------------


async def test_a_booked_reply_is_parsed_into_a_belief(router: respx.MockRouter, tmp_path: Path) -> None:
    mock_reply(
        router,
        {
            "status": "booked",
            "time": {"local": "2026-10-06T15:00", "zone": "Europe/Berlin"},
            "offered": [],
            "evidence": "You're all set for Tuesday.",
        },
    )
    client = make_client(tmp_path / "ledger")
    extractor = LLMExtractor(client, model=MODEL)
    belief = await extractor.extract(
        ["You're all set for Tuesday 6 October at 3:00 PM Berlin time."],
        prospect_zone="Europe/Berlin",
        host_zone="America/New_York",
        reference=REFERENCE,
    )
    assert belief.status == "booked"
    assert belief.source == "llm"
    assert belief.time_utc == datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
    assert extractor.usage_usd > 0
    await client.aclose()


async def test_offered_times_are_deduplicated_and_a_gap_time_is_dropped(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    mock_reply(
        router,
        {
            "status": "not_booked",
            "time": None,
            "offered": [
                {"local": "2026-10-06T14:00", "zone": "UTC"},
                {"local": "2026-10-06T14:00", "zone": "UTC"},
                {"local": "2026-03-29T02:30", "zone": "Europe/Berlin"},  # a DST gap: dropped
            ],
            "evidence": "Which of these works?",
        },
    )
    client = make_client(tmp_path / "ledger")
    extractor = LLMExtractor(client, model=MODEL)
    belief = await extractor.extract(
        ["I have 2:00 PM or 2:00 PM UTC. Which works?"],
        prospect_zone="Europe/Berlin",
        host_zone="America/New_York",
        reference=REFERENCE,
    )
    assert belief.offered_utc == (datetime(2026, 10, 6, 14, 0, tzinfo=UTC),)
    await client.aclose()


async def test_an_invalid_status_raises_a_malformed_error(router: respx.MockRouter, tmp_path: Path) -> None:
    mock_reply(router, {"status": "maybe", "time": None, "offered": [], "evidence": ""})
    client = make_client(tmp_path / "ledger")
    extractor = LLMExtractor(client, model=MODEL)
    with pytest.raises(LLMError, match="valid 'status'"):
        await extractor.extract(
            ["hello"], prospect_zone="Europe/Berlin", host_zone="America/New_York", reference=REFERENCE
        )
    await client.aclose()


async def test_the_request_carries_the_agent_messages_and_a_strict_schema(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    route = mock_reply(router, {"status": "not_booked", "time": None, "offered": [], "evidence": ""})
    client = make_client(tmp_path / "ledger")
    extractor = LLMExtractor(client, model=MODEL)
    await extractor.extract(
        ["First message.", "Second message."],
        prospect_zone="Europe/Berlin",
        host_zone="America/New_York",
        reference=REFERENCE,
    )
    sent = json.loads(route.calls[0].request.content)
    assert sent["temperature"] == 0.0
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["messages"][1]["content"] == "Message 1: First message.\n\nMessage 2: Second message."
    assert "Europe/Berlin" in sent["messages"][0]["content"]
    await client.aclose()


async def test_ledger_entries_are_recorded_under_the_extractor_component(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    mock_reply(router, {"status": "not_booked", "time": None, "offered": [], "evidence": ""})
    ledger_dir = tmp_path / "ledger"
    client = make_client(ledger_dir)
    extractor = LLMExtractor(client, model=MODEL)
    await extractor.extract(
        ["hi"], prospect_zone="Europe/Berlin", host_zone="America/New_York", reference=REFERENCE
    )
    await client.aclose()
    entries = CostLedger(ledger_dir, "extractor").entries()
    assert len(entries) == 1
    assert entries[0]["component"] == "extractor"
    assert entries[0]["usd"] > 0


async def test_a_persistent_server_error_propagates_after_retries(
    router: respx.MockRouter, tmp_path: Path
) -> None:
    route = router.post("/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "upstream is down"}})
    )
    client = make_client(tmp_path / "ledger", max_retries=2)
    extractor = LLMExtractor(client, model=MODEL)
    with pytest.raises(LLMError, match="HTTP 500"):
        await extractor.extract(
            ["hi"], prospect_zone="Europe/Berlin", host_zone="America/New_York", reference=REFERENCE
        )
    assert route.call_count == 3  # the first attempt plus two retries
    await client.aclose()

import asyncio
import json
import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml

from booking_truth.config import Settings
from booking_truth.llm.client import OpenAICompatClient
from booking_truth.llm.ledger import CostLedger, LedgerError
from booking_truth.llm.pricing import PriceTable
from booking_truth.llm.types import BudgetExceeded, ChatMessage, LLMError, PricingError, ToolSpec

BASE = "https://openrouter.ai/api/v1"
KEY = "sk-or-v1-0123456789abcdef-test"
Body = Callable[..., dict[str, Any]]

FIND_SLOTS = ToolSpec(
    name="find_slots",
    description="List open slots between two local dates.",
    parameters={
        "type": "object",
        "properties": {"from_date": {"type": "string"}, "to_date": {"type": "string"}},
        "required": ["from_date", "to_date"],
    },
)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        yield mock


def make_client(
    table: PriceTable,
    ledger: CostLedger,
    *,
    provider: str | None = None,
    budget: float | None = None,
    retries: int = 0,
    base_url: str = BASE,
) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url,
        KEY,
        "vendor/flash",
        provider=provider,
        pricing=table,
        ledger=ledger,
        budget_usd=budget,
        max_retries=retries,
        timeout_s=5,
    )


def record_sleeps(client: OpenAICompatClient) -> list[float]:
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    client._sleep = sleep
    return delays


def hello_estimate(table: PriceTable, max_tokens: int = 1024) -> float:
    return table.estimate("vendor/flash", None, len(json.dumps([m.to_openai() for m in hello()])), max_tokens)


def sent(route: respx.Route, index: int = -1) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls[index].request.content)
    return body


def hello() -> list[ChatMessage]:
    return [ChatMessage.system("You book meetings."), ChatMessage.user("Any time on Tuesday?")]


async def test_tool_call_round_trip(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    tool_call = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "find_slots", "arguments": '{"from_date":"2026-10-06","to_date":"2026-10-06"}'},
    }
    route = router.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json=completion_body(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            ),
            httpx.Response(200, json=completion_body(content="Tuesday 10:00 or 10:30 work.")),
        ]
    )
    client = make_client(table, ledger)
    messages = hello()
    first = await client.chat(messages=messages, tools=[FIND_SLOTS], temperature=0.2, run_id="run-7")
    assert first.content is None
    assert first.finish_reason == "tool_calls"
    assert [(c.id, c.name) for c in first.tool_calls] == [("call_abc", "find_slots")]
    assert json.loads(first.tool_calls[0].arguments) == {"from_date": "2026-10-06", "to_date": "2026-10-06"}

    messages += [
        first.to_message(),
        ChatMessage.tool("call_abc", '{"slots": ["s1", "s2"]}', name="find_slots"),
    ]
    second = await client.chat(messages=messages, tools=[FIND_SLOTS], temperature=0.2, run_id="run-7")
    assert second.content == "Tuesday 10:00 or 10:30 work."
    assert second.tool_calls == []

    request = sent(route, 0)
    assert request["model"] == "vendor/flash"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 1024
    assert request["tools"] == [FIND_SLOTS.to_openai()]
    assert "provider" not in request
    follow_up = sent(route, 1)["messages"]
    assert follow_up[2] == {"role": "assistant", "content": None, "tool_calls": [tool_call]}
    assert follow_up[3] == {"role": "tool", "tool_call_id": "call_abc", "content": '{"slots": ["s1", "s2"]}'}
    headers = route.calls[0].request.headers
    assert headers["authorization"] == f"Bearer {KEY}"
    assert headers["x-openrouter-metadata"] == "enabled"
    await client.aclose()


async def test_provider_pin_is_sent_with_fallbacks_disabled(
    router: respx.MockRouter,
    table: PriceTable,
    ledger: CostLedger,
    completion_body: Body,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_ORG_ID", "org-should-not-leak")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-should-not-leak")
    route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion_body()))
    client = make_client(table, ledger, provider="deepinfra/fp8")
    schema = {"type": "json_schema", "json_schema": {"name": "belief", "schema": {"type": "object"}}}
    await client.chat(messages=hello(), temperature=0.0, response_format=schema, max_tokens=200)
    request = sent(route)
    assert request["provider"] == {
        "order": ["deepinfra/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert request["response_format"] == schema
    assert request["max_tokens"] == 200
    assert "tools" not in request
    headers = route.calls[0].request.headers
    assert "openai-organization" not in headers
    assert "openai-project" not in headers


async def test_returned_model_and_provider_are_recorded_once(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion_body()))
    client = make_client(table, ledger, provider="deepinfra/fp8")
    response = await client.chat(messages=hello(), temperature=0.2, component="persona", run_id="run-1")

    assert response.model_requested == "vendor/flash"
    assert response.model_returned == "vendor/flash-20260910"
    assert response.provider == "DeepInfra"
    assert response.response_id == "gen-1758880000-abc"
    assert response.latency_s >= 0
    # Priced from tokens and the pinned provider's row: (800 x 0.14 + 200 x 0.0042 + 500 x 0.42) / 1e6.
    assert response.usage.usd == pytest.approx(0.00032284, abs=1e-12)
    assert response.usage.provider_reported_cost == 0.0003
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (1000, 500)
    assert response.usage.cached_tokens == 200

    rows = ledger.entries()
    assert len(rows) == 1
    row = rows[0]
    assert row["component"] == "persona"
    assert row["run_id"] == "run-1"
    assert row["model_requested"] == "vendor/flash"
    assert row["model_returned"] == "vendor/flash-20260910"
    assert row["provider"] == "DeepInfra"
    assert row["provider_requested"] == "deepinfra/fp8"
    assert row["usd"] == response.usage.usd
    assert row["provider_reported_cost"] == 0.0003
    assert row["ok"] is True
    assert row["usage_estimated"] is False
    assert ledger.path is not None
    text = ledger.path.read_text()
    assert KEY not in text
    assert "Any time on Tuesday" not in text
    assert "Hello there" not in text


async def test_provider_falls_back_to_router_metadata_then_header(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    metadata = {
        "endpoints": {
            "available": [
                {"provider": "Parasail", "selected": False},
                {"provider": "GMICloud", "selected": True},
            ]
        }
    }
    router.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=completion_body(provider=None, openrouter_metadata=metadata)),
            httpx.Response(200, json=completion_body(provider=None), headers={"X-Provider-Name": "Together"}),
            httpx.Response(200, json=completion_body(provider=None)),
        ]
    )
    client = make_client(table, ledger)
    assert (await client.chat(messages=hello(), temperature=0.2)).provider == "GMICloud"
    assert (await client.chat(messages=hello(), temperature=0.2)).provider == "Together"
    assert (await client.chat(messages=hello(), temperature=0.2)).provider is None
    assert [row["provider"] for row in ledger.entries()] == ["GMICloud", "Together", None]


async def test_budget_refusal_makes_no_http_request(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion_body()))
    ledger.record({"usd": 0.9999})
    client = make_client(table, ledger, budget=1.0)
    with pytest.raises(BudgetExceeded, match="budget stop"):
        await client.chat(messages=hello(), temperature=0.2)
    assert route.call_count == 0
    assert len(router.calls) == 0
    assert len(ledger.entries()) == 1


async def test_calls_in_flight_count_against_the_budget(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    release = asyncio.Event()

    async def slow(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, json=completion_body())

    route = router.post("/chat/completions").mock(side_effect=slow)
    client = make_client(table, ledger)
    estimate = table.estimate("vendor/flash", None, len(json.dumps([m.to_openai() for m in hello()])), 1024)
    client.budget_usd = estimate * 1.5  # room for one call in flight, not two
    first = asyncio.create_task(client.chat(messages=hello(), temperature=0.2))
    await asyncio.sleep(0.05)
    with pytest.raises(BudgetExceeded):
        await client.chat(messages=hello(), temperature=0.2)
    release.set()
    await first
    assert route.call_count == 1


async def test_pricing_refusal_makes_no_http_request(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger
) -> None:
    route = router.post("/chat/completions").mock(return_value=httpx.Response(500))
    client = make_client(table, ledger)
    with pytest.raises(PricingError, match="not in the price table"):
        await client.chat(messages=hello(), temperature=0.2, model="vendor/unpriced")
    assert route.call_count == 0
    assert ledger.entries() == []


@pytest.mark.parametrize(
    ("status", "body", "kind", "fragment"),
    [
        (401, {"error": {"message": "User not found.", "code": 401}}, "auth", "check BT_LLM_API_KEY"),
        (402, {"error": {"message": "Insufficient credits", "code": 402}}, "payment_required", "credits"),
        (
            404,
            {"error": {"message": "No allowed providers are available", "code": 404}},
            "not_found",
            "BT_LLM_PROVIDER",
        ),
        (
            429,
            {"error": {"message": "Rate limit exceeded", "code": 429}},
            "rate_limit",
            "Rate limit exceeded",
        ),
        (503, {"error": {"message": "No available model provider", "code": 503}}, "server", "HTTP 503"),
        (400, {"error": {"message": f"bad header Bearer {KEY}", "code": 400}}, "bad_request", "[redacted]"),
    ],
)
async def test_http_errors_map_to_llm_error_without_the_key(
    router: respx.MockRouter,
    table: PriceTable,
    ledger: CostLedger,
    status: int,
    body: dict[str, Any],
    kind: str,
    fragment: str,
) -> None:
    router.post("/chat/completions").mock(return_value=httpx.Response(status, json=body))
    client = make_client(table, ledger, provider="deepinfra/fp8")
    with pytest.raises(LLMError) as info:
        await client.chat(messages=hello(), temperature=0.2)
    assert info.value.kind == kind
    assert info.value.status_code == status
    assert fragment in str(info.value)
    assert KEY not in str(info.value)
    assert info.value.__cause__ is None
    assert ledger.entries() == []


async def test_timeouts_and_connection_errors_map_to_llm_error(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger
) -> None:
    route = router.post("/chat/completions").mock(side_effect=httpx.ConnectError("connection refused"))
    client = make_client(table, ledger)
    with pytest.raises(LLMError, match=r"could not reach the LLM endpoint openrouter\.ai") as info:
        await client.chat(messages=hello(), temperature=0.2, max_tokens=64)
    assert info.value.kind == "connection"
    assert ledger.entries() == []  # never reached the server, so nothing can have been billed

    route.mock(side_effect=httpx.ReadTimeout("read timed out"))
    with pytest.raises(LLMError, match="timed out after 5s") as info:
        await client.chat(messages=hello(), temperature=0.2, max_tokens=64)
    assert info.value.kind == "timeout"
    # The request was sent and may be billed although no usage came back: the estimate is recorded.
    [row] = ledger.entries()
    assert (row["ok"], row["usage_estimated"], row["completion_tokens"]) == (False, True, 64)
    assert row["usd"] > 0

    route.mock(side_effect=httpx.RemoteProtocolError("server disconnected"))
    with pytest.raises(LLMError, match="RemoteProtocolError") as info:
        await client.chat(messages=hello(), temperature=0.2, max_tokens=64)
    assert info.value.kind == "connection"
    assert [row["usage_estimated"] for row in ledger.entries()] == [True, True]


async def test_error_inside_a_200_response_is_raised_and_its_usage_recorded(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    usage = {"prompt_tokens": 300, "completion_tokens": 10, "total_tokens": 310}
    router.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200, json={"id": "gen-2", "error": {"message": "Upstream error", "code": 502}, "usage": usage}
            ),
            httpx.Response(200, json={"error": {"message": "Upstream error", "code": 502}}),
            httpx.Response(
                200,
                json=completion_body(
                    finish_reason="error",
                    choices=[
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "error",
                            "error": {"message": "stream broke", "code": 500},
                        }
                    ],
                ),
            ),
        ]
    )
    client = make_client(table, ledger)
    with pytest.raises(LLMError, match="Upstream error") as info:
        await client.chat(messages=hello(), temperature=0.2)
    assert info.value.kind == "provider"
    assert info.value.status_code == 502
    with pytest.raises(LLMError, match="Upstream error"):
        await client.chat(messages=hello(), temperature=0.2)
    with pytest.raises(LLMError, match="stream broke"):
        await client.chat(messages=hello(), temperature=0.2)

    rows = ledger.entries()
    assert [row["ok"] for row in rows] == [False, False, False]
    assert rows[0]["prompt_tokens"] == 300
    assert rows[0]["usd"] == pytest.approx((300 * 0.3 + 10 * 1.2) / 1e6, abs=1e-12)
    # A 200 without usage may still have been billed, so its estimate is recorded.
    assert [row["usage_estimated"] for row in rows] == [False, True, False]


async def test_error_status_with_usage_is_recorded(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger
) -> None:
    body = {
        "error": {"message": "context too long", "code": 400},
        "usage": {"prompt_tokens": 50, "completion_tokens": 0},
    }
    router.post("/chat/completions").mock(return_value=httpx.Response(400, json=body))
    client = make_client(table, ledger)
    with pytest.raises(LLMError, match="context too long"):
        await client.chat(messages=hello(), temperature=0.2)
    assert [(row["ok"], row["prompt_tokens"]) for row in ledger.entries()] == [(False, 50)]


async def test_success_without_usage_records_the_estimate(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    body = completion_body()
    del body["usage"]
    router.post("/chat/completions").mock(return_value=httpx.Response(200, json=body))
    client = make_client(table, ledger)
    response = await client.chat(messages=hello(), temperature=0.2, max_tokens=64)
    [row] = ledger.entries()
    assert row["usage_estimated"] is True
    assert row["completion_tokens"] == 64
    assert row["usd"] == response.usage.usd > 0


async def test_returned_model_priced_from_its_own_row(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    router.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion_body(model="vendor/big", usage=usage))
    )
    client = make_client(table, ledger)
    response = await client.chat(messages=hello(), temperature=0.2)
    assert response.usage.usd == table.cost("vendor/big", None, 1000, 1000)


async def test_malformed_body_is_an_llm_error(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger
) -> None:
    router.post("/chat/completions").mock(return_value=httpx.Response(200, text="<html>oops</html>"))
    client = make_client(table, ledger)
    with pytest.raises(LLMError) as info:
        await client.chat(messages=hello(), temperature=0.2)
    assert info.value.kind == "malformed"


def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **values: Any) -> Settings:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("BT_"):
            monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None, **values)


def test_from_settings_refuses_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LLMError, match="BT_LLM_API_KEY") as info:
        OpenAICompatClient.from_settings(settings(tmp_path, monkeypatch), component="agent")
    assert info.value.kind == "offline"
    with pytest.raises(LLMError):
        OpenAICompatClient.from_settings(settings(tmp_path, monkeypatch, llm_api_key="  "), component="agent")


async def test_from_settings_wires_every_llm_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "vendor/own": {
                        "prompt": 1,
                        "completion": 2,
                        "providers": {"together": {"prompt": 1, "completion": 2}},
                    }
                }
            }
        )
    )
    config = settings(
        tmp_path,
        monkeypatch,
        llm_api_key=KEY,
        llm_base_url="https://llm.example.com/v1/",
        llm_model="vendor/own",
        llm_provider="together",
        pricing_path=prices,
        ledger_dir=tmp_path / "ledger",
        budget_usd=14.5,
    )
    client = OpenAICompatClient.from_settings(config, component="harness")
    assert client.base_url == "https://llm.example.com/v1"
    assert client.default_model == "vendor/own"
    assert client.provider == "together"
    assert client.budget_usd == 14.5
    assert client.pricing.models == ["vendor/own"]
    assert client.ledger.directory == tmp_path / "ledger"
    assert client.ledger.component == "harness"
    assert KEY not in repr(client)
    other = OpenAICompatClient.from_settings(
        config, component="eval", model="vendor/other", ledger_dir=tmp_path / "elsewhere"
    )
    assert other.default_model == "vendor/other"
    assert other.ledger.directory == tmp_path / "elsewhere"

    with respx.mock(base_url="https://llm.example.com/v1") as mock:
        route = mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "vendor/own",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        )
        response = await client.chat(messages=hello(), temperature=0.0)
    assert response.content == "ok"
    assert "x-openrouter-metadata" not in route.calls[0].request.headers
    await client.aclose()
    await other.aclose()


def test_bundled_table_is_used_without_bt_pricing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(tmp_path, monkeypatch, llm_api_key=KEY, ledger_dir=tmp_path)
    client = OpenAICompatClient.from_settings(config, component="agent")
    assert client.pricing.origin == "bundled pricing.yaml"
    assert client.pricing.has(config.llm_model)


async def test_every_retry_passes_the_budget_gate(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    route = router.post("/chat/completions").mock(
        side_effect=[httpx.ReadTimeout("read timed out"), httpx.Response(200, json=completion_body())]
    )
    client = make_client(table, ledger, budget=hello_estimate(table) * 1.5, retries=2)
    record_sleeps(client)
    # The timed-out attempt is recorded at its estimate, so the retry no longer fits the budget.
    with pytest.raises(BudgetExceeded):
        await client.chat(messages=hello(), temperature=0.2)
    assert route.call_count == 1
    assert [row["usage_estimated"] for row in ledger.entries()] == [True]
    assert client._sdk.max_retries == 0  # the SDK never re-sends behind the gate's back


async def test_retries_back_off_honour_retry_after_and_record_each_billable_attempt(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    route = router.post("/chat/completions").mock(
        side_effect=[
            httpx.ReadTimeout("read timed out"),
            httpx.Response(
                503, json={"error": {"message": "overloaded", "code": 503}}, headers={"Retry-After": "2"}
            ),
            httpx.Response(200, json=completion_body()),
        ]
    )
    client = make_client(table, ledger, retries=2)
    delays = record_sleeps(client)
    response = await client.chat(messages=hello(), temperature=0.2)
    assert response.content == "Hello there."
    assert route.call_count == 3
    assert 0.375 <= delays[0] <= 0.5
    assert delays[1] == 2.0
    # Timeout: estimate. 503 without usage: nothing billed. Success: real usage.
    assert [(row["ok"], row["usage_estimated"]) for row in ledger.entries()] == [(False, True), (True, False)]


@pytest.mark.parametrize(
    ("status", "headers"),
    [(400, {}), (401, {}), (503, {"x-should-retry": "false"}), (429, {"Retry-After": "120"})],
)
async def test_errors_that_must_not_be_retried_are_sent_once(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, status: int, headers: dict[str, str]
) -> None:
    route = router.post("/chat/completions").mock(
        return_value=httpx.Response(
            status, json={"error": {"message": "no", "code": status}}, headers=headers
        )
    )
    client = make_client(table, ledger, retries=2)
    record_sleeps(client)
    with pytest.raises(LLMError) as info:
        await client.chat(messages=hello(), temperature=0.2)
    assert info.value.status_code == status
    assert route.call_count == 1


async def test_retries_stop_after_max_retries(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger
) -> None:
    route = router.post("/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "boom", "code": 500}})
    )
    client = make_client(table, ledger, retries=2)
    delays = record_sleeps(client)
    with pytest.raises(LLMError, match="HTTP 500") as info:
        await client.chat(messages=hello(), temperature=0.2)
    assert info.value.kind == "server"
    assert route.call_count == 3
    assert len(delays) == 2
    assert ledger.entries() == []


async def test_calls_in_flight_through_other_clients_count_against_the_budget(
    router: respx.MockRouter, table: PriceTable, tmp_path: Path, completion_body: Body
) -> None:
    release = asyncio.Event()

    async def slow(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, json=completion_body())

    route = router.post("/chat/completions").mock(side_effect=slow)
    budget = hello_estimate(table) * 1.5
    agent = make_client(table, CostLedger(tmp_path / "ledger", "agent"), budget=budget)
    persona = make_client(table, CostLedger(tmp_path / "ledger", "persona"), budget=budget)
    first = asyncio.create_task(agent.chat(messages=hello(), temperature=0.2))
    await asyncio.sleep(0.05)
    with pytest.raises(BudgetExceeded, match="in flight"):
        await persona.chat(messages=hello(), temperature=0.7)
    release.set()
    await first
    assert route.call_count == 1


async def test_a_cancelled_call_records_its_estimate_and_releases_its_hold(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger
) -> None:
    started = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    router.post("/chat/completions").mock(side_effect=hang)
    client = make_client(table, ledger, budget=1.0)
    task = asyncio.create_task(client.chat(messages=hello(), temperature=0.2, max_tokens=64))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [row] = ledger.entries()
    assert (row["ok"], row["usage_estimated"], row["completion_tokens"]) == (False, True, 64)
    # No hold is left behind: a zero estimate fits exactly under a cap equal to the total.
    ledger.reserve(0.0, ledger.total() + 1e-9).release()


async def test_pinned_provider_without_a_price_makes_no_http_request(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger
) -> None:
    route = router.post("/chat/completions").mock(return_value=httpx.Response(500))
    for provider in ("venice/fp8", "deepinfra"):
        client = make_client(table, ledger, provider=provider)
        with pytest.raises(PricingError, match="has no price for model 'vendor/flash'"):
            await client.chat(messages=hello(), temperature=0.2)
    assert route.call_count == 0
    assert ledger.entries() == []


async def test_returned_model_without_the_pinned_row_is_priced_as_requested(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    router.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion_body(model="vendor/big", usage=usage))
    )
    client = make_client(table, ledger, provider="deepinfra/fp8")
    response = await client.chat(messages=hello(), temperature=0.2)
    assert response.usage.usd == table.cost("vendor/flash", "deepinfra/fp8", 1000, 1000)
    [row] = ledger.entries()
    assert row["model_returned"] == "vendor/big"


async def test_a_higher_provider_reported_cost_is_what_counts(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.01}
    router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion_body(usage=usage)))
    client = make_client(table, ledger)
    response = await client.chat(messages=hello(), temperature=0.2)
    assert response.usage.usd == 0.01
    assert ledger.total() == 0.01


async def test_a_200_body_that_is_not_json_is_recorded_at_its_estimate(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger
) -> None:
    router.post("/chat/completions").mock(return_value=httpx.Response(200, text="<html>oops</html>"))
    client = make_client(table, ledger)
    with pytest.raises(LLMError):
        await client.chat(messages=hello(), temperature=0.2)
    assert [row["usage_estimated"] for row in ledger.entries()] == [True]


async def test_credentials_in_the_base_url_never_reach_messages(
    table: PriceTable, ledger: CostLedger
) -> None:
    base = "https://bob:s3cret-pass@proxy.example.com/v1"
    client = make_client(table, ledger, base_url=base)
    with respx.mock(base_url="https://proxy.example.com/v1") as mock:
        route = mock.post("/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(LLMError) as info:
            await client.chat(messages=hello(), temperature=0.2)
        assert "proxy.example.com" in str(info.value)
        route.mock(
            return_value=httpx.Response(400, json={"error": {"message": f"bad url {base}", "code": 400}})
        )
        with pytest.raises(LLMError) as second:
            await client.chat(messages=hello(), temperature=0.2)
    for text in (str(info.value), repr(client)):
        assert "bob" not in text
    for text in (str(info.value), str(second.value), repr(client)):
        assert "s3cret-pass" not in text
    assert "bob:[redacted]@" in str(second.value)


async def test_the_key_is_never_logged_or_recorded(
    router: respx.MockRouter,
    table: PriceTable,
    ledger: CostLedger,
    completion_body: Body,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for name in (None, "openai", "httpx", "booking_truth"):
        caplog.set_level(logging.DEBUG, logger=name)
    router.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=completion_body()),
            httpx.Response(401, json={"error": {"message": f"key {KEY} rejected", "code": 401}}),
        ]
    )
    client = make_client(table, ledger)
    await client.chat(messages=hello(), temperature=0.2)
    with pytest.raises(LLMError) as info:
        await client.chat(messages=hello(), temperature=0.2)
    assert caplog.records, "debug logging was captured"
    assert KEY not in caplog.text
    assert KEY not in str(info.value)
    assert ledger.path is not None
    assert KEY not in ledger.path.read_text()


async def test_an_unwritable_ledger_refuses_before_any_request(
    router: respx.MockRouter, table: PriceTable, tmp_path: Path, completion_body: Body
) -> None:
    route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion_body()))
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    client = make_client(table, CostLedger(blocker / "ledger", "agent"))
    with pytest.raises(LedgerError, match="cannot write cost ledger"):
        await client.chat(messages=hello(), temperature=0.2)
    assert route.call_count == 0


async def test_a_failed_ledger_write_stops_later_calls(
    router: respx.MockRouter, table: PriceTable, ledger: CostLedger, completion_body: Body
) -> None:
    def break_ledger_then_answer(request: httpx.Request) -> httpx.Response:
        # The call is with the provider; now the ledger file becomes unwritable.
        assert ledger.path is not None
        ledger.path.unlink()
        ledger.path.mkdir()
        return httpx.Response(200, json=completion_body())

    route = router.post("/chat/completions").mock(side_effect=break_ledger_then_answer)
    client = make_client(table, ledger)
    with pytest.raises(LedgerError, match="cannot write cost ledger"):
        await client.chat(messages=hello(), temperature=0.2)
    with pytest.raises(LedgerError, match="refused until the process restarts"):
        await client.chat(messages=hello(), temperature=0.2)
    assert route.call_count == 1

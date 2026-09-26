"""``dedupe`` in the running agent: a repeated ``(session_id, message_id)`` returns the stored response
without running the turn again, whether the twin already finished or is still in flight; a turn that never
finishes (an exception, or a concurrent turn losing ``lead_lock``) drops its pending row instead of wedging
every later retry of that message."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from agent_env import AgentEnv, make_env
from fastapi import FastAPI

from booking_truth.agent.core import AgentCore, LeadBusy
from booking_truth.agent.guards import all_except, guards_string
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer

Sandbox = tuple[FastAPI, BackgroundServer, SandboxState]
ASK = "Hi, I'm in New York. Can I book an intro call next week?"
GREETING = "Hi there, just checking you're still around."
WITHOUT = guards_string(all_except("dedupe"))


def core(env: AgentEnv) -> AgentCore:
    found: AgentCore = env.app.state.core
    return found


def slot_replies(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [q for q in data["quick_replies"] if q.get("start_utc")]


@pytest.fixture
async def without(sandbox: Sandbox, tmp_path: Path) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards=WITHOUT):
        yield env


# The stored-response replay -------------------------------------------------------------------------------


async def test_a_repeated_message_id_returns_the_stored_response_with_no_second_model_call(
    guarded: AgentEnv,
) -> None:
    before = guarded.llm.calls  # type: ignore[attr-defined]
    first = await guarded.say(GREETING, message_id="m-fixed")
    after_first = guarded.llm.calls  # type: ignore[attr-defined]
    assert after_first > before

    again = await guarded.say(GREETING, message_id="m-fixed")
    assert again == first
    assert guarded.llm.calls == after_first  # type: ignore[attr-defined]


async def test_a_repeated_select_slot_books_only_once(guarded: AgentEnv) -> None:
    """A repeat is still a repeat when it is a code path with no model call at all."""
    offer = await guarded.say(ASK)
    pick = slot_replies(offer)[0]

    first = await guarded.act(pick["action"], message_id="m-pick")
    assert first["booking"]["action"] == "booked"

    again = await guarded.act(pick["action"], message_id="m-pick")
    assert again == first
    assert len(guarded.log("bookings.create")) == 1
    assert len(guarded.bookings()) == 1


async def test_a_different_message_id_is_not_treated_as_a_repeat(guarded: AgentEnv) -> None:
    before = guarded.llm.calls  # type: ignore[attr-defined]
    first = await guarded.say(GREETING, message_id="m-1")
    again = await guarded.say(GREETING, message_id="m-2")
    assert guarded.llm.calls > before  # type: ignore[attr-defined]
    assert guarded.deps.store.messages.get(guarded.session, "m-1") is not None
    assert guarded.deps.store.messages.get(guarded.session, "m-2") is not None
    # Both are genuine turns; nothing says they render identically (the second turn's history includes the
    # first exchange), only that neither one was skipped.
    assert first["guard"] is not None
    assert again["guard"] is not None


# In-flight twins -------------------------------------------------------------------------------------------


async def test_a_concurrent_duplicate_waits_for_the_in_flight_twin_and_gets_the_same_reply(
    guarded: AgentEnv,
) -> None:
    offer = await guarded.say(ASK)
    pick = slot_replies(offer)[0]
    body = guarded.body(action=pick["action"], message_id="m-race")

    async def send() -> dict[str, Any]:
        headers = {"Authorization": "Bearer agent-test-key"}
        response = await guarded.client.post("/v1/chat", json=body, headers=headers)
        assert response.status_code == 200, response.text
        data: dict[str, Any] = response.json()
        return data

    first, second = await asyncio.gather(send(), send())
    assert first == second
    assert first["booking"]["action"] == "booked"
    assert len(guarded.log("bookings.create")) == 1
    assert len(guarded.bookings()) == 1


# Dropping a pending row that will never complete ----------------------------------------------------------


async def test_a_failed_turn_drops_its_pending_row_so_a_retry_runs_fresh(
    guarded: AgentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = core(guarded)
    original = agent._run_turn
    attempts = {"n": 0}

    async def flaky(req: Any, email: str, token: str | None, lock_lost: Any) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("a transient failure, not a real one")
        result: dict[str, Any] = await original(req, email, token, lock_lost)
        return result

    monkeypatch.setattr(agent, "_run_turn", flaky)
    with pytest.raises(RuntimeError):
        await guarded.chat(message=GREETING, message_id="m-boom")
    assert guarded.deps.store.messages.get(guarded.session, "m-boom") is None

    monkeypatch.setattr(agent, "_run_turn", original)
    data = await guarded.say(GREETING, message_id="m-boom")
    assert data["reply"]
    assert guarded.deps.store.messages.get(guarded.session, "m-boom") is not None


async def test_lead_busy_drops_the_pending_row_independently_of_the_lock(
    guarded: AgentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dedupe`` does not rely on ``lead_lock``: a turn that never ran because the lease was busy is not a
    completed twin, so its message id is free to try again, exactly as a turn that raised would be."""
    agent = core(guarded)
    original = agent._hook_lead_lock
    attempts = {"n": 0}

    @asynccontextmanager
    async def flaky_lock(email: str) -> AsyncIterator[None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LeadBusy
        async with original(email):
            yield

    monkeypatch.setattr(agent, "_hook_lead_lock", flaky_lock)
    busy = await guarded.chat(message=GREETING, message_id="m-retry")
    assert busy.status_code == 409
    assert busy.json()["error"] == "lead_busy"
    assert guarded.deps.store.messages.get(guarded.session, "m-retry") is None

    data = await guarded.say(GREETING, message_id="m-retry")
    assert data["reply"]
    assert guarded.deps.store.messages.get(guarded.session, "m-retry") is not None


# Guard off -------------------------------------------------------------------------------------------------


async def test_without_the_guard_a_repeated_message_is_reprocessed(without: AgentEnv) -> None:
    before = without.llm.calls  # type: ignore[attr-defined]
    first = await without.say(GREETING, message_id="m-fixed")
    after_first = without.llm.calls  # type: ignore[attr-defined]
    assert after_first > before

    await without.say(GREETING, message_id="m-fixed")
    assert without.llm.calls > after_first  # type: ignore[attr-defined]
    assert without.deps.store.messages.get(without.session, "m-fixed") is None
    assert first["reply"]

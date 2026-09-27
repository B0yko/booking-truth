"""``lead_lock`` in the running agent: a per-lead lease serialises turns across channels, waiting one out
gets ``409 lead_busy`` instead of running concurrently with it, and the one-active-booking-per-lead policy
turns a second booking into a reschedule offer, reading the calendar rather than the ledger so a booking
made outside the agent counts too."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from agent_env import LEAD, AgentEnv, executor, guarded_slots, make_env, mutable_clock
from fastapi import FastAPI

import booking_truth.agent.core as core_module
from booking_truth.agent.core import AgentCore
from booking_truth.agent.guards import all_except, guards_string
from booking_truth.agent.tools import ToolExecutor
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer

Sandbox = tuple[FastAPI, BackgroundServer, SandboxState]
ASK = "Hi, I'm in New York. Can I book an intro call next week?"
GREETING = "Hi there, just checking you're still around."
# Monday 5 October 2026, 10:00 in New York.
MON_1000 = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)
WITHOUT = guards_string(all_except("lead_lock"))


def core(env: AgentEnv) -> AgentCore:
    found: AgentCore = env.app.state.core
    return found


def events(tools: ToolExecutor) -> list[tuple[str, str]]:
    return [(e.guard, e.event) for e in tools.state.events]


@pytest.fixture
async def without(sandbox: Sandbox, tmp_path: Path) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards=WITHOUT):
        yield env


# The lease itself ------------------------------------------------------------------------------------------


async def test_the_lease_is_held_for_the_duration_of_the_context_and_released_after(
    guarded: AgentEnv,
) -> None:
    agent = core(guarded)
    key = guarded.deps.store.locks.lead_key(LEAD)
    assert guarded.deps.store.locks.holder(key) is None
    async with agent._hook_lead_lock(LEAD):
        holder = guarded.deps.store.locks.holder(key)
        assert holder is not None
        assert holder.key == key
    assert guarded.deps.store.locks.holder(key) is None


async def test_a_second_turn_waits_for_the_lease_then_proceeds_in_order(guarded: AgentEnv) -> None:
    agent = core(guarded)
    order: list[str] = []

    async def first() -> None:
        async with agent._hook_lead_lock(LEAD):
            order.append("first-in")
            await asyncio.sleep(0.2)
            order.append("first-out")

    async def second() -> None:
        await asyncio.sleep(0.05)  # starts after the first has already taken the lease
        async with agent._hook_lead_lock(LEAD):
            order.append("second-in")

    await asyncio.gather(first(), second())
    assert order == ["first-in", "first-out", "second-in"]


async def test_the_lease_is_renewed_while_the_turn_still_runs(
    sandbox: Sandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clock is fixed in most tests, so a renewed lease's ``expires_at`` would not visibly move; this
    one uses a clock the test can advance, the way real time would between renewal cycles."""
    monkeypatch.setattr(core_module, "LEAD_LOCK_RENEW_S", 0.05)
    clock = mutable_clock()
    async for env in make_env(sandbox, tmp_path, guards="all", clock=clock):
        agent = core(env)
        key = env.deps.store.locks.lead_key(LEAD)
        async with agent._hook_lead_lock(LEAD):
            first = env.deps.store.locks.holder(key)
            assert first is not None
            clock.advance(timedelta(seconds=5))
            await asyncio.sleep(0.15)  # a few renewal cycles at the patched interval
            second = env.deps.store.locks.holder(key)
            assert second is not None
            assert second.expires_at > first.expires_at
        assert env.deps.store.locks.holder(key) is None


async def test_a_lost_lease_refuses_to_dispatch_a_write(guarded: AgentEnv) -> None:
    """``ToolExecutor._dispatch_guarded`` is the one choke point every calendar write goes through
    (a fresh dispatch and a retry of one): once the turn's lease-lost signal is set, nothing reaches the
    calendar for it, however the tool got there."""
    tools = executor(guarded)
    slot = (await guarded_slots(tools))[0]
    tools.ctx.state.lock_lost = asyncio.Event()
    tools.ctx.state.lock_lost.set()
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is False
    assert result["reason"] == "calendar_error"
    assert len(guarded.log("bookings.create")) == 0
    assert ("lead_lock", "lease_lost") in events(tools)


async def test_a_turn_that_loses_its_lease_mid_turn_does_not_book(
    sandbox: Sandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The renewal task finding another owner already holds the lease (a renewal cycle missed for long
    enough that the lease genuinely expired) must stop the turn's own writes, not just stop renewing:
    simulate the loss and slow the read the write path makes first, so a renewal cycle has a chance to
    fire before the turn would otherwise dispatch — proof that the signal reaches the write path through
    the real ``_hook_lead_lock`` / ``_run_turn`` wiring, not only when set by hand."""
    monkeypatch.setattr(core_module, "LEAD_LOCK_RENEW_S", 0.03)
    async for env in make_env(sandbox, tmp_path, guards="all"):
        monkeypatch.setattr(env.deps.store.locks, "renew", lambda *a, **k: False)
        env.faults({"group": "bookings.list", "mode": "slow", "latency_ms": 150})
        offer = await env.say(ASK)
        pick = next(q for q in offer["quick_replies"] if q.get("start_utc"))
        response = await env.act(pick["action"])
        assert response["booking"] is None
        assert len(env.log("bookings.create")) == 0


async def test_lead_busy_returns_409_with_a_short_reply_and_no_turn_runs(
    guarded: AgentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def never(key: str, owner: str, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(guarded.deps.store.locks, "await_acquire", never)
    response = await guarded.chat(message=GREETING)
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "lead_busy"
    assert body["reply"]
    # the turn never ran: no history was written for it.
    assert guarded.deps.store.history.for_session(guarded.session) == []


async def test_without_the_guard_two_turns_for_the_same_lead_are_not_serialised(without: AgentEnv) -> None:
    agent = core(without)
    order: list[str] = []

    async def first() -> None:
        async with agent._hook_lead_lock(LEAD):
            order.append("first-in")
            await asyncio.sleep(0.1)
            order.append("first-out")

    async def second() -> None:
        await asyncio.sleep(0.02)
        async with agent._hook_lead_lock(LEAD):
            order.append("second-in")

    await asyncio.gather(first(), second())
    # with the guard off the lock is a no-op, so the second context is entered while the first still holds
    # what would otherwise be the lease.
    assert order == ["first-in", "second-in", "first-out"]


# The one-active-booking-per-lead policy ---------------------------------------------------------------------


async def test_a_second_booking_becomes_a_reschedule_offer(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    first_slot = (await guarded_slots(tools))[0]
    first = await tools.run("book_slot", {"slot_id": first_slot["slot_id"]})
    assert first["booked"] is True

    other_slot = (await guarded_slots(tools, date(2026, 10, 6), date(2026, 10, 6)))[0]
    second = await tools.run("book_slot", {"slot_id": other_slot["slot_id"]})
    assert second["booked"] is False
    assert second["reason"] == "already_booked"
    assert second["existing"]["booking_uid"] == first["booking_uid"]
    assert len(guarded.log("bookings.create")) == 1
    assert len(guarded.bookings()) == 1
    assert ("lead_lock", "reschedule_offered") in events(tools)


async def test_a_booking_made_outside_the_agent_also_blocks_a_new_one(guarded: AgentEnv) -> None:
    """The policy reads the calendar, not the ledger: a setup booking (made through the sandbox control
    API, never through the agent) is not in the ledger but still counts."""
    uid = guarded.setup_booking(MON_1000)
    tools = executor(guarded)
    slot = (await guarded_slots(tools, date(2026, 10, 6), date(2026, 10, 6)))[0]
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is False
    assert result["reason"] == "already_booked"
    assert result["existing"]["booking_uid"] == uid
    assert len(guarded.log("bookings.create")) == 0


async def test_no_active_booking_after_a_cancel_lets_a_new_one_through(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(MON_1000)
    tools = executor(guarded)
    cancelled = await tools.run("cancel_booking", {"booking_uid": uid, "reason": ""})
    assert cancelled["cancelled"] is True

    slot = (await guarded_slots(tools, date(2026, 10, 6), date(2026, 10, 6)))[0]
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is True


async def test_a_failed_calendar_read_does_not_block_a_booking(
    guarded: AgentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The policy answers ``None`` (book as asked) rather than fail-closed on a read it cannot do; the
    write itself is still verified by ``claim_ledger``."""
    guarded.faults({"group": "bookings.list", "mode": "error_500", "times": None})
    tools = executor(guarded)
    slot = (await guarded_slots(tools))[0]
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is True


async def test_without_the_guard_a_second_booking_is_allowed(without: AgentEnv) -> None:
    tools = executor(without)
    first_slot = (await guarded_slots(tools))[0]
    first = await tools.run("book_slot", {"slot_id": first_slot["slot_id"]})
    assert first["booked"] is True

    other_slot = (await guarded_slots(tools, date(2026, 10, 6), date(2026, 10, 6)))[0]
    second = await tools.run("book_slot", {"slot_id": other_slot["slot_id"]})
    assert second["booked"] is True
    assert len(without.bookings()) == 2
    assert ("lead_lock", "reschedule_offered") not in events(tools)


# Through the full pipeline ------------------------------------------------------------------------------


async def test_the_reschedule_offer_reply_names_the_existing_booking_through_the_pipeline(
    guarded: AgentEnv,
) -> None:
    first_offer = await guarded.say(ASK)
    first_pick = next(q for q in first_offer["quick_replies"] if q.get("start_utc"))
    booked = await guarded.act(first_pick["action"])
    assert booked["booking"]["action"] == "booked"

    second = await guarded.act({"type": "select_slot", "slot_id": first_pick["action"]["slot_id"]})
    assert "You already have a call booked for" in second["reply"]
    assert len(guarded.bookings()) == 1


async def test_reschedule_pick_named_in_text_moves_it_without_asking_again(guarded: AgentEnv) -> None:
    """Run-1 pattern: the prospect asks to move an existing booking, the agent offers new times, and the
    prospect names one of them in plain text — not a quick-reply action. That offer already knows it is a
    reschedule (the turn's own ``list_my_bookings`` call found the booking before it offered slots), so its
    quick replies are ``reschedule`` actions naming the booking, never plain ``select_slot`` picks; and the
    text pick, in the very next reply, must move the booking with ``reschedule_booking`` instead of treating
    it as a fresh booking request and asking "Would you like me to move it?" a second time."""
    uid = guarded.setup_booking(MON_1000)
    offer = await guarded.say(
        "I need to reschedule our call — something came up, can we push it to October 6th instead?"
    )
    quick = [q for q in offer["quick_replies"] if q.get("action")]
    assert quick, offer["quick_replies"]
    assert {q["action"]["type"] for q in quick} == {"reschedule"}
    assert all(q["action"]["booking_uid"] == uid for q in quick)

    picked = offer["reply"].split("\n")[1].removeprefix("- ")
    moved = await guarded.say(f"{picked} works for me.")
    assert "Would you like me to move it" not in moved["reply"]
    assert moved["booking"] is not None
    assert moved["booking"]["action"] == "rescheduled"
    active = guarded.bookings()
    assert len(active) == 1
    assert not active[0]["start"].startswith(MON_1000.strftime("%Y-%m-%dT%H:%M"))


async def test_reschedule_quick_reply_moves_it_directly_with_no_book_slot_involved(guarded: AgentEnv) -> None:
    """The same offer, picked through its own quick reply (a client click, not text): it must go straight
    to ``reschedule_booking`` (``_action`` routes a ``reschedule`` action with a ``slot_id`` there), never
    through ``book_slot`` and its ``already_booked`` detour."""
    uid = guarded.setup_booking(MON_1000)
    offer = await guarded.say(
        "I need to reschedule our call — something came up, can we push it to October 6th instead?"
    )
    pick = next(q for q in offer["quick_replies"] if q.get("action", {}).get("type") == "reschedule")
    moved = await guarded.act(pick["action"])
    assert moved["booking"]["action"] == "rescheduled"
    assert moved["booking"]["ref"] != uid
    assert len(guarded.bookings()) == 1


async def test_a_widget_session_is_not_offered_a_reschedule_it_cannot_make(guarded: AgentEnv) -> None:
    """The policy reads the calendar across every channel and session, so a booking made outside this
    widget session still turns a second booking into ``already_booked`` here too; but reschedule and
    cancel are scoped to the widget session (``ToolExecutor._allowed``), so offering to move *that*
    booking would dangle a quick reply the agent can only then refuse. Neither the tool result nor the
    reply may promise a reschedule this session cannot complete."""
    uid = guarded.setup_booking(MON_1000)
    tools = executor(guarded, channel="widget", session="w-1")
    other_slot = (await guarded_slots(tools, date(2026, 10, 6), date(2026, 10, 6)))[0]
    result = await tools.run("book_slot", {"slot_id": other_slot["slot_id"]})
    assert result["booked"] is False
    assert result["reason"] == "already_booked"
    assert result["existing"]["booking_uid"] == uid
    assert tools.ctx.state.reschedule_offer is None
    reply = await guarded.act(
        {"type": "select_slot", "slot_id": other_slot["slot_id"]}, channel="widget", session="w-2"
    )
    assert "Would you like me to move it" not in reply["reply"]
    assert reply["quick_replies"] == []

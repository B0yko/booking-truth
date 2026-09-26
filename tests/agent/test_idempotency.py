"""``idempotency`` in the running agent: keys sent as Cal.com booking metadata, verify-before-retry after a
timeout, and the pre-dispatch check that turns a repeated write of the same intent into a lookup instead of
a second calendar call."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from agent_env import LEAD, AgentEnv, executor, guarded_slots, make_env
from fastapi import FastAPI

from booking_truth.agent.guards import all_except, guards_string
from booking_truth.agent.guards.idempotency import change_idem_key, create_idem_key
from booking_truth.agent.tools import ToolExecutor
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer
from booking_truth.store import normalize_email

Sandbox = tuple[FastAPI, BackgroundServer, SandboxState]
EVENT_KEY = "1001"
# Monday 5 October 2026, 10:00 in New York.
MON_1000 = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)
WITHOUT = guards_string(all_except("idempotency"))


def events(tools: ToolExecutor) -> list[tuple[str, str]]:
    return [(e.guard, e.event) for e in tools.state.events]


# Key derivation --------------------------------------------------------------------------------------------


def test_a_create_key_is_stable_for_the_same_intent() -> None:
    key = create_idem_key(LEAD, EVENT_KEY, MON_1000, 0)
    assert key == create_idem_key(f" {LEAD.upper()} ", EVENT_KEY, MON_1000, 0)  # normalised, case-folded
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


@pytest.mark.parametrize(
    "other",
    [
        create_idem_key("someone-else@example.com", EVENT_KEY, MON_1000, 0),
        create_idem_key(LEAD, "1002", MON_1000, 0),
        create_idem_key(LEAD, EVENT_KEY, MON_1000 + timedelta(minutes=30), 0),
        create_idem_key(LEAD, EVENT_KEY, MON_1000, 1),
    ],
)
def test_a_create_key_changes_with_any_part_of_the_intent(other: str) -> None:
    assert create_idem_key(LEAD, EVENT_KEY, MON_1000, 0) != other


def test_a_reschedule_and_a_cancel_key_differ_for_the_same_booking() -> None:
    reschedule = change_idem_key("uid-1", "reschedule", MON_1000)
    cancel = change_idem_key("uid-1", "cancel", None)
    assert reschedule != cancel
    assert cancel == change_idem_key("uid-1", "cancel", None)
    assert reschedule != change_idem_key("uid-2", "reschedule", MON_1000)
    assert reschedule != change_idem_key("uid-1", "reschedule", MON_1000 + timedelta(minutes=30))


# Keys sent to the calendar ----------------------------------------------------------------------------------


async def test_a_guarded_create_carries_the_key_as_booking_metadata(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    slot = (await guarded_slots(tools))[0]
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is True
    [entry] = guarded.log("bookings.create")
    key = entry["body"]["metadata"]["bt_idem"]
    assert len(key) == 64
    stored = guarded.deps.store.idem.get(key)
    assert stored is not None
    assert (stored.kind, stored.status, stored.booking_ref) == ("create", "committed", result["booking_uid"])
    assert stored.generation == 0


async def test_the_naive_baseline_sends_no_key(naive: AgentEnv) -> None:
    result = await executor(naive).run("book", {"start_iso": "2026-10-05T14:00:00Z"})
    assert result["booked"] is True
    [entry] = naive.log("bookings.create")
    assert "metadata" not in entry["body"]


# commit_then_timeout: adopted, never dispatched twice -------------------------------------------------------


@pytest.fixture
async def guarded_fast(sandbox: Sandbox, tmp_path: Path) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards="all", fast_calendar=True):
        yield env


async def test_a_create_that_commits_and_times_out_is_adopted(guarded_fast: AgentEnv) -> None:
    tools = executor(guarded_fast)
    slot = (await guarded_slots(tools))[0]
    guarded_fast.faults(
        {"group": "bookings.create", "mode": "commit_then_timeout", "times": 1, "hang_s": 1.0}
    )
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is True
    assert len(guarded_fast.log("bookings.create")) == 1  # never dispatched a second time
    assert events(tools) == [
        ("idempotency", "adopted"),
        ("claim_ledger", "verified"),
    ]
    write = tools.state.writes[-1]
    assert write.status == "verified"
    key = guarded_fast.log("bookings.create")[0]["body"]["metadata"]["bt_idem"]
    stored = guarded_fast.deps.store.idem.get(key)
    assert stored is not None
    assert (stored.status, stored.booking_ref) == ("adopted", write.booking.ref)


async def test_a_reschedule_that_commits_and_times_out_is_adopted(guarded_fast: AgentEnv) -> None:
    uid = guarded_fast.setup_booking(MON_1000)
    tools = executor(guarded_fast)
    target = (await guarded_slots(tools, MON_1000.date(), MON_1000.date() + timedelta(days=1)))[-1]
    guarded_fast.faults(
        {"group": "bookings.reschedule", "mode": "commit_then_timeout", "times": 1, "hang_s": 1.0}
    )
    result = await tools.run("reschedule_booking", {"booking_uid": uid, "slot_id": target["slot_id"]})
    assert result["rescheduled"] is True
    assert result["booking_uid"] != uid
    assert len(guarded_fast.log("bookings.reschedule")) == 1
    assert ("idempotency", "adopted") in events(tools)
    active = guarded_fast.bookings()
    assert [b["uid"] for b in active] == [result["booking_uid"]]


async def test_a_cancel_that_commits_and_times_out_is_adopted(guarded_fast: AgentEnv) -> None:
    uid = guarded_fast.setup_booking(MON_1000)
    tools = executor(guarded_fast)
    guarded_fast.faults(
        {"group": "bookings.cancel", "mode": "commit_then_timeout", "times": 1, "hang_s": 1.0}
    )
    result = await tools.run("cancel_booking", {"booking_uid": uid, "reason": "changed plans"})
    assert result["cancelled"] is True
    assert len(guarded_fast.log("bookings.cancel")) == 1
    assert ("idempotency", "adopted") in events(tools)
    assert guarded_fast.bookings() == []


async def test_a_create_that_never_commits_recovers_on_the_retry(guarded_fast: AgentEnv) -> None:
    tools = executor(guarded_fast)
    slot = (await guarded_slots(tools))[0]
    guarded_fast.faults({"group": "bookings.create", "mode": "timeout", "times": 1, "hang_s": 1.0})
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is True
    assert len(guarded_fast.log("bookings.create")) == 2  # the first never landed, so the retry dispatches
    assert events(tools) == [("idempotency", "retry"), ("claim_ledger", "verified")]


async def test_a_persistent_timeout_gives_up_and_a_later_attempt_dispatches_fresh(
    guarded_fast: AgentEnv,
) -> None:
    tools = executor(guarded_fast)
    slot = (await guarded_slots(tools))[0]
    guarded_fast.faults({"group": "bookings.create", "mode": "timeout", "times": None, "hang_s": 1.0})
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is False
    assert result["reason"] == "calendar_error"
    assert len(guarded_fast.log("bookings.create")) == 2  # one dispatch plus one retry, both hung
    assert events(tools) == [("idempotency", "retry"), ("idempotency", "gave_up")]
    key = guarded_fast.log("bookings.create")[0]["body"]["metadata"]["bt_idem"]
    stored = guarded_fast.deps.store.idem.get(key)
    assert stored is not None
    assert stored.status == "failed"
    guarded_fast.faults()  # clear the fault
    again = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert again["booked"] is True
    assert len(guarded_fast.log("bookings.create")) == 3
    retried = guarded_fast.deps.store.idem.get(key)
    assert retried is not None
    assert retried.status == "committed"


# A repeated write of the same intent is not a second booking ------------------------------------------------


async def test_repeating_book_slot_for_the_same_slot_does_not_double_book(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    slot = (await guarded_slots(tools))[0]
    first = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert first["booked"] is True
    again = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert again["booked"] is True
    assert again["booking_uid"] == first["booking_uid"]
    assert len(guarded.log("bookings.create")) == 1
    assert len(guarded.bookings()) == 1
    assert ("idempotency", "replayed") in events(tools)


async def test_cancelling_an_already_cancelled_booking_still_asks_the_calendar(guarded: AgentEnv) -> None:
    """Unlike a repeated create, a repeated cancel is dispatched again: Cal.com's own ``duplicate`` answer
    is more informative than replaying the first cancel, and this is the outcome the rest of the agent
    (and its tests) rely on."""
    uid = guarded.setup_booking(MON_1000)
    tools = executor(guarded)
    first = await tools.run("cancel_booking", {"booking_uid": uid, "reason": "No longer needed"})
    assert first["cancelled"] is True
    again = await tools.run("cancel_booking", {"booking_uid": uid, "reason": ""})
    assert again == {"cancelled": False, "reason": "already_cancelled"}
    assert len(guarded.log("bookings.cancel")) == 2
    assert ("idempotency", "replayed") not in events(tools)


async def test_rebooking_the_same_slot_after_a_cancel_gets_a_fresh_key(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    slot = (await guarded_slots(tools))[0]
    first = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert first["booked"] is True
    cancelled = await tools.run("cancel_booking", {"booking_uid": first["booking_uid"], "reason": ""})
    assert cancelled["cancelled"] is True
    again = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert again["booked"] is True
    assert again["booking_uid"] != first["booking_uid"]
    assert len(guarded.bookings()) == 1
    assert ("idempotency", "replayed") not in events(tools)
    email = normalize_email(LEAD)
    assert guarded.deps.store.generations.current(email, "1001") == 1
    create_keys = {e["body"]["metadata"]["bt_idem"] for e in guarded.log("bookings.create")}
    assert len(create_keys) == 2  # generation 0, then generation 1


async def test_without_the_guard_no_key_is_registered(without: AgentEnv) -> None:
    tools = executor(without)
    slot = (await guarded_slots(tools))[0]
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is True
    assert idem_row_count(without) == 0


def idem_row_count(env: AgentEnv) -> int:
    row = env.deps.store.connection().execute("SELECT COUNT(*) FROM idem").fetchone()
    return int(row[0])


@pytest.fixture
async def without(sandbox: Sandbox, tmp_path: Path) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards=WITHOUT):
        yield env

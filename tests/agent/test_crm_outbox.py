"""``crm_outbox`` in the running agent: a validated CRM payload queued for each verified write, never
from the model's prose. Delivering the queue to HubSpot is a worker of a later milestone; this guard
only has to get a correct row into the outbox (or, with it off, leave the naive prose rule as the only
path to the CRM)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from agent_env import LEAD, LEAD_NAME, AgentEnv, make_env
from fastapi import FastAPI

import booking_truth.agent.guards.readback as readback
from booking_truth.agent.guards import all_except, guards_string
from booking_truth.crm import NullCrm
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer

Sandbox = tuple[FastAPI, BackgroundServer, SandboxState]
ASK = "Hi, I'm in New York. Can I book an intro call next week, ideally in the afternoon?"
# Monday 5 October 2026, 10:00 in New York.
MON_1000 = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)
WITHOUT = guards_string(all_except("crm_outbox"))


def slot_replies(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [q for q in data["quick_replies"] if q.get("start_utc")]


def outbox_trace_calls(env: AgentEnv, session: str = "s-1") -> list[dict[str, Any]]:
    """The turn's own trace steps for ``crm.outbox.enqueue`` (written after the reply, like the naive
    rule's CRM calls; never part of that turn's ``guard.events``)."""
    steps = env.deps.store.trace_steps.for_session(session)
    return [s for s in steps if s.get("kind") == "tool_call" and s.get("name") == "crm.outbox.enqueue"]


@pytest.fixture
def short_readback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short read-back window, so a persistent ``bookings.get`` fault does not make the test wait 5 s."""
    monkeypatch.setattr(readback, "READBACK_WINDOW_S", 0.6)
    monkeypatch.setattr(readback, "READBACK_STEP_S", 0.1)


@pytest.fixture
async def without_crm_outbox(sandbox: Sandbox, tmp_path: Path) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards=WITHOUT):
        yield env


# Enqueue on a verified write -------------------------------------------------------------------------------


async def test_a_verified_booking_is_queued_not_written_directly(guarded: AgentEnv) -> None:
    offer = await guarded.say(ASK)
    booked = await guarded.act(slot_replies(offer)[0]["action"])
    crm = guarded.deps.crm
    assert isinstance(crm, NullCrm)
    assert crm.writes == 0  # nothing reaches the CRM directly; the outbox worker lands later
    [item] = guarded.deps.store.outbox.items()
    assert (item.kind, item.status, item.lead_email) == ("crm_sync", "pending", LEAD)
    payload = item.payload
    assert payload["action"] == "booked"
    assert payload["booking_ref"] == booked["booking"]["ref"]
    assert payload["lead_email"] == LEAD
    assert payload["lead_name"] == LEAD_NAME
    assert payload["zone"] == booked["booking"]["zone"] == "America/New_York"
    assert payload["start_utc"] == booked["booking"]["start_utc"]
    assert payload["previous_ref"] is None
    [call] = outbox_trace_calls(guarded)
    assert call["args"] == {"action": "booked", "booking_ref": booked["booking"]["ref"]}


async def test_a_verified_reschedule_is_queued_with_the_previous_ref(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(MON_1000)
    offer = await guarded.act({"type": "reschedule", "booking_uid": uid})
    replies = slot_replies(offer)
    moved = await guarded.act(replies[1]["action"])
    [item] = guarded.deps.store.outbox.items()
    payload = item.payload
    assert payload["action"] == "rescheduled"
    assert payload["previous_ref"] == uid
    assert payload["booking_ref"] == moved["booking"]["ref"] != uid
    [call] = outbox_trace_calls(guarded)
    assert call["args"] == {"action": "rescheduled", "booking_ref": moved["booking"]["ref"]}


async def test_a_verified_cancel_is_queued(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(MON_1000)
    await guarded.act({"type": "cancel", "booking_uid": uid})
    [item] = guarded.deps.store.outbox.items()
    payload = item.payload
    assert payload["action"] == "cancelled"
    assert payload["booking_ref"] == uid
    assert payload["previous_ref"] is None
    [call] = outbox_trace_calls(guarded)
    assert call["args"] == {"action": "cancelled", "booking_ref": uid}


async def test_successive_writes_queue_one_payload_each_in_order(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(MON_1000)
    cancelled = await guarded.act({"type": "cancel", "booking_uid": uid})
    assert cancelled["booking"]["action"] == "cancelled"
    offer = await guarded.say(ASK)
    booked = await guarded.act(slot_replies(offer)[0]["action"])
    assert booked["booking"]["action"] == "booked"
    items = guarded.deps.store.outbox.items()
    assert [i.payload["action"] for i in items] == ["cancelled", "booked"]


# Only after a verified result ------------------------------------------------------------------------------


async def test_an_unverified_write_is_never_queued(guarded: AgentEnv, short_readback: None) -> None:
    offer = await guarded.say(ASK)
    guarded.faults({"group": "bookings.get", "mode": "error_500", "times": None})
    data = await guarded.act(slot_replies(offer)[0]["action"])
    assert data["booking"] is None
    assert data["guard"]["blocked"] is False
    assert guarded.deps.store.outbox.items() == []
    assert guarded.deps.crm.writes == 0


# Guard off: the naive prose rule is the only path to the CRM -----------------------------------------------


async def test_with_the_guard_off_the_naive_prose_rule_writes_directly(
    without_crm_outbox: AgentEnv,
) -> None:
    offer = await without_crm_outbox.say(ASK)
    booked = await without_crm_outbox.act(slot_replies(offer)[0]["action"])
    assert booked["reply"].startswith("Booked:")  # rendered_confirmation is still on
    crm = without_crm_outbox.deps.crm
    assert isinstance(crm, NullCrm)
    assert crm.writes == 2  # upsert_contact + create_meeting, straight from the reply's prose
    assert list(crm.contacts) == [LEAD]
    assert len(crm.meetings) == 1
    assert without_crm_outbox.deps.store.outbox.items() == []

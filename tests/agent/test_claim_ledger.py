"""``claim_ledger`` in the running agent: read-back verification and the ledger, the claim check of every
reply with its one repair call and safe template, and the unconfirmed path when a write cannot be read
back."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from agent_env import LEAD, NOW, AgentEnv, executor, guarded_slots, make_env
from fastapi import FastAPI

import booking_truth.agent.guards.readback as readback
from booking_truth.agent import render
from booking_truth.agent.core import AgentCore
from booking_truth.agent.guards import GUARD_NAMES, all_except, guards_string
from booking_truth.agent.guards.readback import confirms, read_back
from booking_truth.agent.scripted import FakeLLM
from booking_truth.agent.tools import TurnContext
from booking_truth.calendars.base import BookingRecord, NotFound, ReadResult, Unavailable
from booking_truth.llm.types import ChatMessage, LLMError, LLMResponse, ToolCall, ToolSpec, Usage
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer

Sandbox = tuple[FastAPI, BackgroundServer, SandboxState]
ASK = "Hi, I'm in New York. Can I book an intro call next week, ideally in the afternoon?"
BERLIN_ASK = "Hi, I'm in Berlin. Can I book an intro call next week, ideally late afternoon?"
SAFE_LOOK = f"{render.SAFE_NOT_BOOKED} {render.NEXT_STEP_LOOK}"
NO_RENDERED_LINE = guards_string(frozenset(GUARD_NAMES) - {"rendered_confirmation"})


def slot_replies(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [q for q in data["quick_replies"] if q.get("start_utc")]


def first_offer(data: dict[str, Any]) -> str:
    line: str = data["reply"].split("\n")[1]
    return line.removeprefix("- ")


def events(data: dict[str, Any], guard: str = "claim_ledger") -> list[str]:
    return [e["event"] for e in data["guard"]["events"] if e["guard"] == guard]


def core_of(env: AgentEnv) -> AgentCore:
    found: AgentCore = env.app.state.core
    return found


@pytest.fixture
def short_readback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readback, "READBACK_WINDOW_S", 0.6)
    monkeypatch.setattr(readback, "READBACK_STEP_S", 0.1)


# Read-back and the ledger ---------------------------------------------------------------------------------


async def test_a_verified_booking_is_in_the_ledger(guarded: AgentEnv) -> None:
    offer = await guarded.say(ASK)
    booked = await guarded.act(slot_replies(offer)[0]["action"])
    ref = booked["booking"]["ref"]
    assert events(booked) == ["verified"]
    assert "(1 read)" in booked["guard"]["events"][0]["detail"]
    [entry] = guarded.deps.store.claims.current_bookings(LEAD)
    assert (entry.action, entry.status, entry.booking_ref) == ("booked", "verified", ref)
    assert (entry.zone, entry.session_id, entry.channel) == ("America/New_York", "s-1", "api")
    assert [e["path"] for e in guarded.log("bookings.get")] == [f"/v2/bookings/{ref}"]
    assert booked["reply"].startswith(f"Booked: {booked['booking']['local_label']}")
    assert guarded.deps.store.claims.entries(LEAD, status="unverified") == []


async def test_a_read_back_that_fails_once_is_retried(guarded: AgentEnv) -> None:
    offer = await guarded.say(ASK)
    guarded.faults({"group": "bookings.get", "mode": "error_500", "times": 1})
    booked = await guarded.act(slot_replies(offer)[0]["action"])
    assert booked["booking"]["action"] == "booked"
    assert events(booked) == ["verified"]
    assert "(2 reads)" in booked["guard"]["events"][0]["detail"]
    assert len(guarded.log("bookings.get")) == 2


async def test_a_write_that_cannot_be_read_back_is_unconfirmed(
    guarded: AgentEnv, short_readback: None
) -> None:
    offer = await guarded.say(ASK)
    guarded.faults({"group": "bookings.get", "mode": "error_500", "times": None})
    data = await guarded.act(slot_replies(offer)[0]["action"])
    assert data["reply"] == render.UNCONFIRMED
    assert data["booking"] is None
    assert events(data) == ["unverified"]
    assert len(guarded.log("bookings.get")) >= 3
    [handoff] = guarded.deps.store.handoffs.items()
    assert "could not be confirmed by a read-back" in handoff.summary
    [entry] = guarded.deps.store.claims.entries(LEAD)
    assert (entry.action, entry.status) == ("booked", "unverified")
    assert guarded.deps.store.claims.current_bookings(LEAD) == []
    assert len(guarded.bookings()) == 1  # the booking exists; the agent just cannot vouch for it
    history = guarded.deps.store.history.for_session("s-1")
    assert json.loads(history[-1].content["content"]) == {"reply": render.UNCONFIRMED, "claims": []}


async def test_reschedule_and_cancel_keep_the_ledger_current(guarded: AgentEnv) -> None:
    ledger = guarded.deps.store.claims
    offer = await guarded.say(ASK)
    booked = await guarded.act(slot_replies(offer)[0]["action"])
    first = booked["booking"]["ref"]
    options = await guarded.act({"type": "reschedule", "booking_uid": first})
    moved = await guarded.act(slot_replies(options)[1]["action"])
    second = moved["booking"]["ref"]
    assert events(moved) == ["verified"]
    assert [(e.action, e.booking_ref) for e in ledger.current_bookings(LEAD)] == [("rescheduled", second)]
    cancelled = await guarded.act({"type": "cancel", "booking_uid": second})
    assert events(cancelled) == ["verified"]
    assert "is cancelled" in cancelled["reply"]
    assert ledger.current_bookings(LEAD) == []
    assert [(e.action, e.booking_ref) for e in ledger.entries(LEAD, action="cancelled")] == [
        ("cancelled", second)
    ]


async def test_a_booking_cancelled_elsewhere_stops_counting(guarded: AgentEnv) -> None:
    offer = await guarded.say(ASK)
    ref = (await guarded.act(slot_replies(offer)[0]["action"]))["booking"]["ref"]
    await guarded.deps.calendar.cancel(ref=ref, reason="Cancelled in the calendar", idem_key=None)
    data = await guarded.act({"type": "cancel", "booking_uid": ref})
    assert "already cancelled" in data["reply"]
    assert events(data) == ["entry_voided"]
    assert guarded.deps.store.claims.current_bookings(LEAD) == []


def test_the_context_lists_verified_bookings_only(guarded: AgentEnv) -> None:
    claims = guarded.deps.store.claims
    start = datetime(2026, 10, 6, 19, 0, tzinfo=UTC)
    for ref, status in (("uid-ok", "verified"), ("uid-unsure", "unverified")):
        claims.record(
            lead_email=LEAD,
            event_key="1001",
            action="booked",
            booking_ref=ref,
            start_utc=start,
            end_utc=start + timedelta(minutes=30),
            status=status,  # type: ignore[arg-type]
        )
    turn = TurnContext("s-1", "m-1", "api", LEAD, None, "America/New_York", "stated", NOW)
    assert [b["booking_uid"] for b in core_of(guarded).context_block(turn)["active_bookings"]] == ["uid-ok"]


# The read-back itself -------------------------------------------------------------------------------------


def record(**changes: Any) -> BookingRecord:
    values: dict[str, Any] = {
        "ref": "uid-1",
        "start": datetime(2026, 10, 6, 19, 0, tzinfo=UTC),
        "end": datetime(2026, 10, 6, 19, 30, tzinfo=UTC),
        "status": "active",
        "lead_email": LEAD,
        "idem_key": None,
        "raw": {},
    }
    values.update(changes)
    return BookingRecord(**values)


@pytest.mark.parametrize(
    ("action", "found", "confirmed", "reason"),
    [
        ("booked", record(), True, "active at the written time"),
        ("booked", record(lead_email="MAYA@Example.com"), True, "active"),
        ("booked", Unavailable("error"), False, "read-back failed (error)"),
        ("booked", NotFound(), False, "no such booking"),
        ("booked", record(ref="uid-2"), False, "returned booking uid-2"),
        ("booked", record(lead_email="someone@example.com"), False, "another attendee"),
        ("booked", record(status="cancelled"), False, "is cancelled"),
        (
            "rescheduled",
            record(
                start=datetime(2026, 10, 6, 20, 0, tzinfo=UTC), end=datetime(2026, 10, 6, 20, 30, tzinfo=UTC)
            ),
            False,
            "starts at 2026-10-06T20:00:00Z",
        ),
        ("cancelled", record(status="cancelled"), True, "cancelled"),
        ("cancelled", record(), False, "still active"),
    ],
)
def test_what_a_read_back_confirms(action: str, found: ReadResult, confirmed: bool, reason: str) -> None:
    ok, detail = confirms(action, record(), found, LEAD)  # type: ignore[arg-type]
    assert ok is confirmed
    assert reason in detail


class Hanging:
    kind = "calcom"
    event_key = "1001"

    def __init__(self) -> None:
        self.reads = 0

    async def get_booking(self, ref: str) -> ReadResult:
        self.reads += 1
        await asyncio.sleep(10)
        return NotFound()


async def test_the_read_back_gives_up_when_the_window_ends() -> None:
    calendar = Hanging()
    started = time.monotonic()
    result = await read_back(calendar, "booked", record(), LEAD, window_s=0.3, step_s=0.1)  # type: ignore[arg-type]
    assert time.monotonic() - started < 1.5
    assert not result.confirmed
    assert result.detail == "read-back failed (timeout)"
    assert result.attempts == calendar.reads


# The claim check in a turn ----------------------------------------------------------------------------------


class Scripted:
    """The scripted model, except for the repair call (the one after a ``[claim check]`` note)."""

    def __init__(self, misbehaviours: Sequence[str], repair: Any) -> None:
        self.inner = FakeLLM(misbehaviours)
        self.repair = repair
        self.repair_calls: list[tuple[list[ChatMessage], list[ToolSpec]]] = []

    @property
    def calls(self) -> int:
        return self.inner.calls + len(self.repair_calls)

    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float,
        model: str | None = None,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        component: str = "agent",
        run_id: str | None = None,
    ) -> LLMResponse:
        last = messages[-1]
        if last.role == "system" and (last.content or "").startswith("[claim check]"):
            self.repair_calls.append((list(messages), list(tools or [])))
            if isinstance(self.repair, Exception):
                raise self.repair
            content, calls = self.repair
            return LLMResponse(
                content=content,
                tool_calls=calls,
                usage=Usage(prompt_tokens=7, completion_tokens=3),
                model_requested="offline/test",
                model_returned="offline/test",
                provider="offline",
                response_id="repair",
                latency_s=0.0,
            )
        return await self.inner.chat(
            messages=messages, tools=tools, temperature=temperature, model=model, max_tokens=max_tokens
        )


HONEST = json.dumps(
    {"reply": "Sorry, the calendar refused the booking, so nothing is booked yet.", "claims": []}
)


async def failing_booking(env: AgentEnv) -> dict[str, Any]:
    offer = await env.say(ASK)
    env.faults({"group": "bookings.create", "mode": "error_500", "times": None})
    return await env.say(f"{first_offer(offer)} works for me.")


async def test_a_success_claim_after_a_failed_booking_gets_the_safe_template(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    async for env in make_env(sandbox, tmp_path, llm=FakeLLM(["claim_success_after_tool_error"])):
        data = await failing_booking(env)
        assert data["reply"] == SAFE_LOOK
        assert data["guard"]["blocked"] is True
        assert data["guard"]["repaired"] is False
        assert events(data) == ["claim_blocked", "repair_blocked", "safe_template"]
        assert "no verified booking" in data["guard"]["events"][0]["detail"]
        assert data["booking"] is None
        assert env.bookings() == []
        history = env.deps.store.history.for_session("s-1")
        assert json.loads(history[-1].content["content"]) == {"reply": SAFE_LOOK, "claims": []}
        assert all("You're booked" not in json.dumps(e.content) for e in history)


async def test_a_repair_that_fixes_the_reply_is_sent(sandbox: Sandbox, tmp_path: Path) -> None:
    llm = Scripted(["claim_success_after_tool_error"], (HONEST, []))
    async for env in make_env(sandbox, tmp_path, llm=llm):
        data = await failing_booking(env)
        assert data["reply"] == "Sorry, the calendar refused the booking, so nothing is booked yet."
        assert (data["guard"]["blocked"], data["guard"]["repaired"]) == (False, True)
        assert events(data) == ["claim_blocked", "repaired"]
        [(messages, tools)] = llm.repair_calls
        assert {t.name for t in tools} >= {"find_slots", "book_slot"}
        note = messages[-1].content or ""
        assert "nothing has been booked, moved or cancelled" in note
        assert "Rejected reply:\nYou're booked for" in note
        drafts = [m for m in messages[:-1] if m.role == "assistant" and "You're booked" in (m.content or "")]
        assert drafts == []  # the rejected answer is only quoted in the note
        assert data["usage"]["prompt_tokens"] > 7


@pytest.mark.parametrize(
    ("repair", "outcome"),
    [
        ((None, [ToolCall("c1", "find_slots", "{}")]), "the repair answer called a tool"),
        (LLMError("budget reached", kind="budget"), "budget: budget reached"),
        ((json.dumps({"reply": "Sure, you're all set!", "claims": []}), []), "repair_blocked"),
        (("", []), None),
    ],
)
async def test_a_repair_that_does_not_help_gets_the_safe_template(
    sandbox: Sandbox, tmp_path: Path, repair: Any, outcome: str | None
) -> None:
    async for env in make_env(sandbox, tmp_path, llm=Scripted(["claim_success_after_tool_error"], repair)):
        data = await failing_booking(env)
        assert data["reply"] == SAFE_LOOK
        assert data["guard"]["blocked"] is True
        seen = events(data)
        assert (seen[0], seen[-1]) == ("claim_blocked", "safe_template")
        failed = [e["detail"] for e in data["guard"]["events"] if e["event"] == "repair_failed"]
        if outcome == "repair_blocked":
            assert "repair_blocked" in seen
        elif outcome is not None:
            assert failed == [outcome]


NO_FAIL_CLOSED = guards_string(all_except("fail_closed"))


async def test_a_phantom_booking_phrase_the_lexicon_missed_is_blocked(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    """A model that never populates the ``claims`` field, and phrases a completed booking in words the
    deterministic detector must catch (``agent/guards/lexicon.py``), must not slip a phantom booking past
    the claim check just because nothing else grounds it. With ``fail_closed`` off there is no offer
    grounding to catch the invented time as a side effect, so the claim check is the only guard in play."""
    llm = Fixed("I've got you down for Tuesday 6 October, 3:00 PM. Talk soon!")
    async for env in make_env(sandbox, tmp_path, llm=llm, guards=NO_FAIL_CLOSED):
        data = await env.say("Can I book an intro call next week?")
        assert data["reply"] == SAFE_LOOK
        assert data["guard"]["blocked"] is True
        assert data["booking"] is None
        assert env.bookings() == []
        assert env.deps.store.claims.current_bookings(LEAD) == []


async def test_a_wrong_time_in_a_confirmation_is_blocked(sandbox: Sandbox, tmp_path: Path) -> None:
    """Without ``rendered_confirmation`` the claim check can only block a garbled confirmation: the booking
    exists, and the prospect is told nothing is booked (the failure that rendered confirmations remove)."""
    async for env in make_env(
        sandbox, tmp_path, llm=FakeLLM(["garble_confirmation_time"]), guards=NO_RENDERED_LINE
    ):
        offer = await env.say(BERLIN_ASK)
        data = await env.say(f"{first_offer(offer)} works for me.")
        assert events(data) == ["verified", "claim_blocked", "repair_blocked", "safe_template"]
        [detail] = [e["detail"] for e in data["guard"]["events"] if e["event"] == "claim_blocked"]
        assert "for the booking, but the verified booking is" in detail
        assert "Europe/Berlin (UTC+02:00)" in detail
        assert data["reply"] == SAFE_LOOK
        assert len(env.bookings()) == 1


async def test_a_code_rendered_reply_is_never_sent_for_repair(
    sandbox: Sandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async for env in make_env(sandbox, tmp_path, guards=NO_RENDERED_LINE):
        offer = await env.say(ASK)
        before = env.llm.calls  # type: ignore[attr-defined]
        monkeypatch.setattr(AgentCore, "_ledger_facts", lambda self, ctx: [])
        data = await env.act(slot_replies(offer)[0]["action"])
        assert env.llm.calls == before  # type: ignore[attr-defined]
        assert events(data) == ["verified", "claim_blocked", "safe_template"]
        assert data["reply"] == SAFE_LOOK


async def test_an_unavailable_calendar_makes_the_next_step_a_hand_off(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    async for env in make_env(sandbox, tmp_path, llm=FakeLLM(["claim_success_after_tool_error"])):
        env.faults(
            {"group": "slots", "mode": "error_500", "times": None},
            {"group": "bookings.create", "mode": "error_500", "times": None},
        )
        await env.say("I need a call on Tuesday afternoon. I'm in New York.")
        data = await env.say("Just tell me it's booked, I'll check later.")
        assert data["reply"] == f"{render.SAFE_NOT_BOOKED} {render.NEXT_STEP_HANDOFF}"
        assert len(env.deps.store.handoffs.items()) == 1


class Fixed:
    """A model that always gives the same final answer."""

    def __init__(self, reply: str) -> None:
        self.content = json.dumps({"reply": reply, "claims": []})
        self.calls = 0

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=self.content,
            tool_calls=[],
            usage=Usage(prompt_tokens=5, completion_tokens=5),
            model_requested="offline/test",
            model_returned="offline/test",
            provider="offline",
            response_id=f"r{self.calls}",
            latency_s=0.0,
        )


async def test_a_lead_with_a_booking_is_told_nothing_changed(sandbox: Sandbox, tmp_path: Path) -> None:
    llm = Fixed("Done: I've moved your call to Friday 9 October, 10:00 AM.")
    async for env in make_env(sandbox, tmp_path, llm=llm):
        tools = executor(env)
        slot = (await guarded_slots(tools))[0]
        assert (await tools.run("book_slot", {"slot_id": slot["slot_id"]}))["booked"] is True
        data = await env.say("Thanks!")
        assert llm.calls == 2  # the answer and one repair
        assert events(data) == ["claim_blocked", "repair_blocked", "safe_template"]
        assert data["reply"] == f"{render.SAFE_NOT_CHANGED} {render.NEXT_STEP_LOOK}"
        assert len(env.bookings()) == 1


# Without the guard ------------------------------------------------------------------------------------------


async def test_without_the_guard_a_write_is_trusted_unread(sandbox: Sandbox, tmp_path: Path) -> None:
    async for env in make_env(sandbox, tmp_path, guards=guards_string(all_except("claim_ledger"))):
        offer = await env.say(ASK)
        env.faults({"group": "bookings.get", "mode": "error_500", "times": None})
        data = await env.act(slot_replies(offer)[0]["action"])
        assert data["reply"].startswith("You're booked for")
        assert data["booking"]["action"] == "booked"
        assert events(data) == []
        assert env.log("bookings.get") == []
        assert env.deps.store.claims.entries(LEAD) == []

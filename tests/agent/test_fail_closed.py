"""``fail_closed`` in the running agent: the strict adapter, one internal retry of safe lookups, offer
grounding inside the claim check, and a hand-off while the calendar is unavailable."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from agent_env import LEAD, AgentEnv, executor, make_env, mutable_clock
from fastapi import FastAPI

from booking_truth.agent import render
from booking_truth.agent.guards import GUARD_NAMES, all_except, guards_string
from booking_truth.agent.guards.fail_closed import HANDOFF_SUMMARY
from booking_truth.agent.scripted import FakeLLM
from booking_truth.agent.tools import UNAVAILABLE_INSTRUCTION, ToolExecutor
from booking_truth.calendars.base import BookingRecord
from booking_truth.calendars.calcom import CalcomAdapter
from booking_truth.llm.types import ChatMessage, LLMResponse, ToolCall, Usage
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer
from booking_truth.timeutil import parse_iso

Sandbox = tuple[FastAPI, BackgroundServer, SandboxState]
ASK = "Hi, I'm in New York. Can I book an intro call next week?"
FIND = {"from_date": "2026-10-05", "to_date": "2026-10-09"}
WITHOUT = guards_string(all_except("fail_closed"))
WITHOUT_LEDGER = guards_string(frozenset(GUARD_NAMES) - {"claim_ledger", "rendered_confirmation"})
SAFE_LOOK = f"{render.SAFE_NOT_BOOKED} {render.NEXT_STEP_LOOK}"
SAFE_HANDOFF = f"{render.SAFE_NOT_BOOKED} {render.NEXT_STEP_HANDOFF}"
SORRY = "I'm sorry, the calendar is unavailable right now. Would you like me to ask a colleague to follow up?"
# Tuesday 6 October 2026, 11:00 AM in New York.
TUE_1100 = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)


# A model that follows a script ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    reply: str
    offered: tuple[str, ...] = ()


Step = Call | Answer | Callable[[Sequence[ChatMessage]], "Call | Answer"]


class Script:
    """A model that takes its steps in order, one per model call, and repeats the last one after that (so a
    repair call gets the same answer). A step may be a function of the messages the model sees."""

    def __init__(self, *steps: Step) -> None:
        self.steps = steps
        self.calls = 0
        self.seen: list[list[ChatMessage]] = []

    async def chat(self, *, messages: Sequence[ChatMessage], **kwargs: Any) -> LLMResponse:
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        self.seen.append(list(messages))
        if not isinstance(step, (Call, Answer)):
            step = step(messages)
        content: str | None = None
        calls: list[ToolCall] = []
        if isinstance(step, Call):
            calls = [ToolCall(f"t{self.calls}", step.name, json.dumps(step.args))]
        else:
            claims = [{"type": "offered", "time": label} for label in step.offered]
            content = json.dumps({"reply": step.reply, "claims": claims})
        return LLMResponse(
            content=content,
            tool_calls=calls,
            usage=Usage(prompt_tokens=5, completion_tokens=5),
            model_requested="offline/test",
            model_returned="offline/test",
            provider="offline",
            response_id=f"r{self.calls}",
            latency_s=0.0,
        )


def tool_output(messages: Sequence[ChatMessage], name: str) -> Any:
    """The last result of tool ``name`` in ``messages``."""
    found = [m for m in messages if m.role == "tool" and m.name == name]
    assert found, f"no {name} result"
    return json.loads(found[-1].content or "")


def offer_first(messages: Sequence[ChatMessage]) -> Answer:
    label = str(tool_output(messages, "find_slots")["slots"][0]["label"])
    return Answer(f"How about {label}?", (label,))


def events(data: dict[str, Any], guard: str | None = None) -> list[tuple[str, str]]:
    return [(e["guard"], e["event"]) for e in data["guard"]["events"] if guard in (None, e["guard"])]


@pytest.fixture
async def without(sandbox: Sandbox, tmp_path: Path) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards=WITHOUT):
        yield env


async def scripted_env(
    sandbox: Sandbox, tmp_path: Path, script: Script, **overrides: object
) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, llm=script, **overrides):
        yield env


# Strict adapters --------------------------------------------------------------------------------------------


async def test_the_guard_builds_a_strict_calendar_adapter(sandbox: Sandbox, tmp_path: Path) -> None:
    for index, (guards, lenient) in enumerate((("all", False), (WITHOUT, True), ("fail_closed", False))):
        folder = tmp_path / f"agent-{index}"
        folder.mkdir()
        async for env in make_env(sandbox, folder, guards=guards):
            calendar = env.deps.calendar
            assert isinstance(calendar, CalcomAdapter)
            assert calendar.lenient is lenient


async def test_a_malformed_availability_answer_is_unavailable_after_one_retry(guarded: AgentEnv) -> None:
    guarded.faults({"group": "slots", "mode": "malformed", "times": None})
    tools = executor(guarded)
    result = await tools.run("find_slots", FIND)
    assert result == {"unavailable": True, "reason": "malformed", "instruction": UNAVAILABLE_INSTRUCTION}
    assert len(guarded.log("slots")) == 2
    assert [(e.guard, e.event) for e in tools.state.events] == [
        ("fail_closed", "lookup_retried"),
        ("fail_closed", "calendar_unavailable"),
    ]
    assert tools.state.shown is None
    assert guarded.deps.store.slot_lists.latest(LEAD) is None


async def test_without_the_guard_a_malformed_answer_becomes_free_times(without: AgentEnv) -> None:
    """The failure the strict adapter prevents: the busy periods in the malformed body are read as slots."""
    without.faults({"group": "slots", "mode": "malformed", "times": None})
    result = await executor(without).run("find_slots", FIND)
    assert result["slots"]
    assert len(without.log("slots")) == 1
    assert without.deps.store.slot_lists.latest(LEAD) is not None


async def test_a_lookup_that_times_out_is_unavailable_never_an_empty_calendar(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    async for env in make_env(sandbox, tmp_path, guards="all", fast_calendar=True):
        env.faults({"group": "slots", "mode": "timeout", "times": None, "hang_s": 1.0})
        result = await executor(env).run("find_slots", FIND)
        assert result["unavailable"] is True
        assert result["reason"] == "timeout"
        assert "slots" not in result
        assert len(env.log("slots")) == 2


async def test_a_lookup_that_times_out_once_recovers_on_the_retry(sandbox: Sandbox, tmp_path: Path) -> None:
    async for env in make_env(sandbox, tmp_path, guards="all", fast_calendar=True):
        env.faults({"group": "slots", "mode": "timeout", "times": 1, "hang_s": 1.0})
        result = await executor(env).run("find_slots", FIND)
        assert len(result["slots"]) == 12


# One internal retry of list_my_bookings ---------------------------------------------------------------------


async def test_listing_bookings_is_retried_once(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(TUE_1100)
    guarded.faults({"group": "bookings.list", "mode": "error_500", "times": 1})
    tools = executor(guarded)
    result = await tools.run("list_my_bookings", {})
    assert [b["booking_uid"] for b in result["bookings"]] == [uid]
    assert len(guarded.log("bookings.list")) == 2
    assert [(e.guard, e.event, e.detail) for e in tools.state.events] == [
        ("fail_closed", "lookup_retried", "error")
    ]
    assert not tools.state.calendar_unavailable


async def test_listing_bookings_reports_an_unavailable_calendar(guarded: AgentEnv) -> None:
    guarded.setup_booking(TUE_1100)
    guarded.faults({"group": "bookings.list", "mode": "malformed", "times": None})
    tools = executor(guarded)
    result = await tools.run("list_my_bookings", {})
    assert result == {"unavailable": True, "reason": "malformed", "instruction": UNAVAILABLE_INSTRUCTION}
    assert len(guarded.log("bookings.list")) == 2
    assert tools.state.calendar_unavailable
    assert tools.state.steps[-1]["ok"] is False


async def test_a_missing_event_type_is_not_retried(guarded: AgentEnv) -> None:
    guarded.faults({"group": "bookings.list", "mode": "not_found", "times": None})
    tools = executor(guarded)
    result = await tools.run("list_my_bookings", {})
    assert result["reason"] == "not_found"
    assert len(guarded.log("bookings.list")) == 1
    assert [(e.guard, e.event) for e in tools.state.events] == [("fail_closed", "calendar_unavailable")]


async def test_without_the_guard_listing_bookings_is_not_retried(without: AgentEnv) -> None:
    without.faults({"group": "bookings.list", "mode": "error_500", "times": 1})
    tools = executor(without)
    result = await tools.run("list_my_bookings", {})
    assert isinstance(result, str)
    assert result.startswith("Error: calendar returned HTTP 500")
    assert len(without.log("bookings.list")) == 1
    assert tools.state.events == []


# Offer grounding --------------------------------------------------------------------------------------------


async def test_offers_from_the_latest_slot_list_pass(guarded: AgentEnv) -> None:
    data = await guarded.say(ASK)
    assert "Here are some open times" in data["reply"]
    assert data["guard"]["blocked"] is False
    assert ("fail_closed", "offer_blocked") not in events(data)


async def test_an_offer_outside_the_latest_slot_list_is_blocked(sandbox: Sandbox, tmp_path: Path) -> None:
    invented = Answer("How about Monday 5 October, 7:00 PM?", ("Monday 5 October, 7:00 PM",))
    script = Script(Call("find_slots", FIND), invented)
    async for env in scripted_env(sandbox, tmp_path, script):
        data = await env.say(ASK)
        assert script.calls == 3  # find_slots, the answer and one repair
        assert data["reply"] == SAFE_LOOK
        assert data["guard"]["blocked"] is True
        assert events(data) == [
            ("fail_closed", "offer_blocked"),
            ("claim_ledger", "claim_blocked"),
            ("fail_closed", "offer_blocked"),
            ("claim_ledger", "repair_blocked"),
            ("claim_ledger", "safe_template"),
        ]
        detail = data["guard"]["events"][0]["detail"]
        assert "Monday 5 October, 7:00 PM" in detail
        assert "not in the latest availability" in detail
        assert "offer only times from the latest find_slots result" in (script.seen[-1][-1].content or "")
        assert env.deps.store.handoffs.items() == []


async def test_a_repair_that_offers_a_listed_time_is_sent(sandbox: Sandbox, tmp_path: Path) -> None:
    invented = Answer("How about Monday 5 October, 7:00 PM?", ("Monday 5 October, 7:00 PM",))
    script = Script(Call("find_slots", FIND), invented, offer_first)
    async for env in scripted_env(sandbox, tmp_path, script):
        data = await env.say(ASK)
        assert data["guard"]["repaired"] is True
        assert data["reply"] == "How about Monday 5 October, 10:00 AM?"
        assert events(data, "claim_ledger") == [
            ("claim_ledger", "claim_blocked"),
            ("claim_ledger", "repaired"),
        ]


async def test_offers_from_an_expired_slot_list_are_blocked(sandbox: Sandbox, tmp_path: Path) -> None:
    clock = mutable_clock()

    def restate(messages: Sequence[ChatMessage]) -> Answer:
        return offer_first(messages)

    script = Script(Call("find_slots", FIND), offer_first, restate, restate, restate)
    async for env in scripted_env(sandbox, tmp_path, script, clock=clock):
        first = await env.say(ASK)
        assert first["guard"]["blocked"] is False
        again = await env.say("Which times were those again?")
        assert again["guard"]["blocked"] is False  # the list is still fresh
        clock.advance(timedelta(seconds=env.deps.settings.slot_ttl_seconds + 1))
        late = await env.say("Sorry, which times were those again?")
        assert late["guard"]["blocked"] is True
        assert ("fail_closed", "offer_blocked") in events(late)
        assert late["reply"] == SAFE_LOOK


async def test_the_booking_being_cancelled_may_be_named(sandbox: Sandbox, tmp_path: Path) -> None:
    def ask_to_cancel(messages: Sequence[ChatMessage]) -> Answer:
        label = tool_output(messages, "list_my_bookings")["bookings"][0]["label"]
        return Answer(f"Would you like me to cancel your call on {label}?")

    script = Script(Call("list_my_bookings"), ask_to_cancel)
    async for env in scripted_env(sandbox, tmp_path, script):
        env.setup_booking(TUE_1100)
        data = await env.say("I might have to cancel my call.")
        assert data["reply"] == "Would you like me to cancel your call on Tuesday 6 October, 11:00 AM?"
        assert data["guard"]["blocked"] is False
        assert events(data) == []


async def test_a_time_the_calendar_never_reported_is_blocked(sandbox: Sandbox, tmp_path: Path) -> None:
    script = Script(Answer("Would you like me to cancel your call on Tuesday 6 October, 11:00 AM?"))
    async for env in scripted_env(sandbox, tmp_path, script):
        env.setup_booking(TUE_1100)
        data = await env.say("I might have to cancel my call.")
        assert data["guard"]["blocked"] is True
        assert ("fail_closed", "offer_blocked") in events(data)


async def test_the_booking_a_new_booking_runs_into_may_be_named(
    guarded: AgentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reschedule offer of the one-booking-per-lead policy names the existing call, which the calendar
    reported; it is not an invented offer."""
    uid = guarded.setup_booking(TUE_1100)
    existing = BookingRecord(uid, TUE_1100, TUE_1100 + timedelta(minutes=30), "active", LEAD, None, {})

    async def found(self: ToolExecutor) -> BookingRecord:
        return existing

    offer = await guarded.say(ASK)
    monkeypatch.setattr(ToolExecutor, "_hook_existing_booking", found)
    pick = next(q for q in offer["quick_replies"] if q.get("start_utc"))
    data = await guarded.act(pick["action"])
    assert data["reply"].startswith("You already have a call booked for Tuesday 6 October, 11:00 AM")
    assert f"move it to {pick['label']} instead?" in data["reply"]
    assert data["guard"]["blocked"] is False
    history = guarded.deps.store.history.for_session("s-1")
    result = json.loads(next(e for e in history if e.content.get("name") == "book_slot").content["content"])
    assert parse_iso(result["existing"]["start_utc"]) == TUE_1100


async def test_offers_are_not_grounded_without_the_guard(sandbox: Sandbox, tmp_path: Path) -> None:
    invented = Answer("How about Monday 5 October, 7:00 PM?", ("Monday 5 October, 7:00 PM",))
    script = Script(Call("find_slots", FIND), invented)
    async for env in scripted_env(sandbox, tmp_path, script, guards=WITHOUT):
        data = await env.say(ASK)
        assert data["reply"] == invented.reply
        assert data["guard"]["blocked"] is False
        assert script.calls == 2


# A hand-off while the calendar is unavailable ---------------------------------------------------------------


async def test_code_hands_off_when_the_model_does_not(sandbox: Sandbox, tmp_path: Path) -> None:
    script = Script(Call("find_slots", FIND), Answer(SORRY), Call("find_slots", FIND), Answer(SORRY))
    async for env in scripted_env(sandbox, tmp_path, script):
        env.faults({"group": "slots", "mode": "error_500", "times": None})
        data = await env.say(ASK)
        assert data["reply"] == f"{SORRY}\n\n{render.NEXT_STEP_HANDOFF}"
        assert data["guard"]["blocked"] is False
        [handoff] = env.deps.store.handoffs.items()
        assert (handoff.summary, handoff.preferred_times_text) == (HANDOFF_SUMMARY, ASK)
        assert (handoff.session_id, handoff.lead_email) == ("s-1", LEAD)
        assert events(data, "fail_closed") == [
            ("fail_closed", "lookup_retried"),
            ("fail_closed", "calendar_unavailable"),
            ("fail_closed", "handoff"),
        ]
        assert data["guard"]["events"][-1]["detail"] == f"H{handoff.id}"
        history = [(e.role, e.content) for e in env.deps.store.history.for_session("s-1")]
        call, output, final = history[-3:]
        assert call[1]["tool_calls"][0]["name"] == "handoff_to_human"
        assert (output[0], output[1]["name"]) == ("tool", "handoff_to_human")
        assert json.loads(output[1]["content"]) == {"handoff": "created", "reference": f"H{handoff.id}"}
        assert json.loads(final[1]["content"])["reply"] == data["reply"]
        steps = env.deps.store.trace_steps.for_session("s-1")
        assert [s["name"] for s in steps if s["kind"] == "tool_call"][-1] == "handoff_to_human"

        again = await env.say("Could you check again?")
        assert again["reply"] == SORRY
        assert len(env.deps.store.handoffs.items()) == 1
        assert ("fail_closed", "handoff") not in events(again)


async def test_no_code_hand_off_when_the_model_hands_off(sandbox: Sandbox, tmp_path: Path) -> None:
    handoff = Call(
        "handoff_to_human", {"summary": "Wants a call; calendar down.", "preferred_times_text": ""}
    )
    reply = "The calendar is unavailable right now, so I've asked a colleague to email you."
    script = Script(Call("find_slots", FIND), handoff, Answer(reply))
    async for env in scripted_env(sandbox, tmp_path, script):
        env.faults({"group": "slots", "mode": "error_500", "times": None})
        data = await env.say(ASK)
        assert data["reply"] == reply
        assert [h.summary for h in env.deps.store.handoffs.items()] == ["Wants a call; calendar down."]
        assert ("fail_closed", "handoff") not in events(data)


async def test_no_hand_off_when_a_later_lookup_succeeds(sandbox: Sandbox, tmp_path: Path) -> None:
    script = Script(Call("find_slots", FIND), Call("find_slots", FIND), offer_first)
    async for env in scripted_env(sandbox, tmp_path, script):
        env.faults({"group": "slots", "mode": "error_500", "times": 2})
        data = await env.say(ASK)
        assert data["reply"].startswith("How about Monday 5 October")
        assert env.deps.store.handoffs.items() == []
        assert len(env.log("slots")) == 3


async def test_the_hand_off_also_applies_without_claim_ledger(sandbox: Sandbox, tmp_path: Path) -> None:
    script = Script(Call("find_slots", FIND), Answer(SORRY))
    async for env in scripted_env(sandbox, tmp_path, script, guards=WITHOUT_LEDGER):
        env.faults({"group": "slots", "mode": "not_found", "times": None})
        data = await env.say(ASK)
        assert data["reply"] == f"{SORRY}\n\n{render.NEXT_STEP_HANDOFF}"
        assert len(env.deps.store.handoffs.items()) == 1


async def test_without_the_guard_no_hand_off_is_made_for_the_model(sandbox: Sandbox, tmp_path: Path) -> None:
    script = Script(Call("find_slots", FIND), Answer(SORRY))
    async for env in scripted_env(sandbox, tmp_path, script, guards=WITHOUT):
        env.faults({"group": "slots", "mode": "error_500", "times": None})
        data = await env.say(ASK)
        assert data["reply"] == SORRY
        assert env.deps.store.handoffs.items() == []


async def test_the_safe_template_hand_off_is_made_once_and_shown_to_the_model(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    async for env in make_env(sandbox, tmp_path, llm=FakeLLM(["invent_slots"])):
        env.faults({"group": "slots", "mode": "not_found", "times": None})
        data = await env.say(ASK)
        assert data["reply"] == SAFE_HANDOFF
        assert ("fail_closed", "offer_blocked") in events(data)
        history = env.deps.store.history.for_session("s-1")
        assert [e.content.get("name") for e in history if e.role == "tool"][-1] == "handoff_to_human"
        again = await env.say("Is there really nothing next week?")
        assert again["reply"] == SAFE_HANDOFF
        assert len(env.deps.store.handoffs.items()) == 1
        assert env.bookings() == []


async def test_a_code_path_hands_off_once_per_conversation(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(TUE_1100)
    guarded.faults({"group": "slots", "mode": "error_500", "times": None})
    first = await guarded.act({"type": "reschedule", "booking_uid": uid})
    assert first["reply"] == render.unavailable_text()
    second = await guarded.act({"type": "reschedule", "booking_uid": uid})
    assert second["reply"] == render.unavailable_text()
    assert len(guarded.deps.store.handoffs.items()) == 1
    assert [q for q in second["quick_replies"] if q.get("start_utc")] == []


async def test_the_scripted_model_hands_off_by_itself(guarded: AgentEnv) -> None:
    guarded.faults({"group": "slots", "mode": "not_found", "times": None})
    data = await guarded.say(ASK)
    assert data["reply"] == render.unavailable_text()
    assert len(guarded.deps.store.handoffs.items()) == 1
    assert ("fail_closed", "handoff") not in events(data)

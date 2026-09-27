"""The turn pipeline end to end: conversations through the app against the sandbox, in both modes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_env import API_KEY, LEAD, NOW, AgentEnv, make_env
from fastapi import FastAPI

from booking_truth.agent.core import AgentCore
from booking_truth.agent.scripted import FakeLLM
from booking_truth.agent.tools import TurnContext
from booking_truth.crm import NullCrm
from booking_truth.llm.types import ChatMessage, LLMError, LLMResponse, ToolSpec, Usage
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer

Sandbox = tuple[FastAPI, BackgroundServer, SandboxState]
BERLIN_ASK = "Hi, I'm in Berlin. Can I book an intro call next week, ideally late afternoon?"
# Monday 5 October 2026, 10:00 in New York.
MON_1000 = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)


def slot_replies(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [q for q in data["quick_replies"] if q.get("start_utc")]


def core(env: AgentEnv) -> AgentCore:
    found: AgentCore = env.app.state.core
    return found


# Booking ------------------------------------------------------------------------------------------------


async def test_guarded_booking_through_a_slot_quick_reply(guarded: AgentEnv) -> None:
    offer = await guarded.say(BERLIN_ASK)
    replies = slot_replies(offer)
    assert 1 <= len(replies) <= 6
    assert all(q["label"] in offer["reply"] for q in replies)
    assert all(q["action"]["type"] == "select_slot" for q in replies)
    assert offer["booking"] is None
    pick = replies[0]
    booked = await guarded.act(pick["action"])
    booking = booked["booking"]
    assert booking["action"] == "booked"
    assert booking["status"] == "active"
    assert booking["zone"] == "Europe/Berlin"
    assert booking["start_utc"] == pick["start_utc"]
    assert booking["local_label"].endswith("Europe/Berlin (UTC+02:00)")
    line, sentence = booked["reply"].split("\n\n", 1)
    assert line == f"Booked: {booking['local_label']} · reference {booking['ref'][:8]}"
    assert sentence == "The calendar invite is on its way to your email."
    assert booked["quick_replies"] == []
    [calendar_booking] = guarded.bookings()
    assert calendar_booking["uid"] == booking["ref"]
    assert calendar_booking["attendees"][0]["timeZone"] == "Europe/Berlin"
    history = guarded.deps.store.history.for_session("s-1")
    tool_calls = [c["name"] for e in history for c in e.content.get("tool_calls") or []]
    assert tool_calls == ["resolve_timezone", "find_slots", "book_slot"]


async def test_naive_booking_through_the_text_of_an_offer(naive: AgentEnv) -> None:
    offer = await naive.say(BERLIN_ASK)
    assert offer["quick_replies"] == []  # slot quick replies are a guarded feature
    first = offer["reply"].split("\n")[1].removeprefix("- ")
    booked = await naive.say(f"{first} works for me.")
    assert booked["booking"]["action"] == "booked"
    assert booked["booking"]["zone"] == "Europe/Berlin"
    assert len(naive.bookings()) == 1
    assert "You're booked" in booked["reply"]


async def test_the_naive_crm_rule_writes_a_meeting_when_the_reply_says_booked(naive: AgentEnv) -> None:
    crm = naive.deps.crm
    assert isinstance(crm, NullCrm)
    offer = await naive.say(BERLIN_ASK)
    assert crm.writes == 0
    first = offer["reply"].split("\n")[1].removeprefix("- ")
    booked = await naive.say(f"{first} works for me.")
    assert list(crm.contacts) == [LEAD]
    [meeting] = crm.meetings.values()
    assert meeting.start_utc.isoformat().replace("+00:00", "Z") == booked["booking"]["start_utc"]
    assert meeting.booking_ref == booked["booking"]["ref"]
    await naive.say("Thanks!")
    assert len(crm.meetings) == 1  # "You're welcome" does not say booked
    steps = naive.deps.store.trace_steps.for_session("s-1")
    assert [s["name"] for s in steps if s["kind"] == "tool_call"][-2:] == [
        "crm.upsert_contact",
        "crm.create_meeting",
    ]


async def test_with_crm_outbox_on_nothing_is_written_from_prose(guarded: AgentEnv) -> None:
    offer = await guarded.say(BERLIN_ASK)
    await guarded.act(slot_replies(offer)[0]["action"])
    crm = guarded.deps.crm
    assert isinstance(crm, NullCrm)
    assert crm.writes == 0


async def test_a_taken_slot_is_offered_again_by_code(guarded: AgentEnv) -> None:
    offer = await guarded.say(BERLIN_ASK)
    guarded.faults({"group": "bookings.create", "mode": "slot_taken_after_offer", "times": 1})
    retry = await guarded.act(slot_replies(offer)[0]["action"])
    assert retry["reply"].startswith(
        "Sorry, that time was just taken by someone else, so nothing is booked yet."
    )
    assert retry["booking"] is None
    new = slot_replies(retry)
    assert new
    assert {q["start_utc"] for q in new}.isdisjoint({q["start_utc"] for q in slot_replies(offer)})
    booked = await guarded.act(new[0]["action"])
    assert booked["booking"]["action"] == "booked"


async def test_an_unknown_slot_is_offered_again_by_code(guarded: AgentEnv) -> None:
    await guarded.say(BERLIN_ASK)
    retry = await guarded.act({"type": "select_slot", "slot_id": "s_nonexisten"})
    assert retry["reply"].startswith("Sorry, that option has expired, so nothing is booked yet.")
    assert slot_replies(retry)
    events = [(e["guard"], e["event"]) for e in retry["guard"]["events"]]
    assert ("slot_ids", "unknown_or_expired_slot") in events


async def test_select_slot_is_not_available_in_the_naive_mode(naive: AgentEnv) -> None:
    data = await naive.act({"type": "select_slot", "slot_id": "s_aaaaaaaaaa"})
    assert "tell me in a message" in data["reply"]
    assert naive.log("bookings.create") == []


# Reschedule and cancel ----------------------------------------------------------------------------------


async def test_reschedule_and_cancel_through_actions(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(MON_1000)
    offer = await guarded.act({"type": "reschedule", "booking_uid": uid})
    assert offer["reply"].startswith("Sure, here are some open times to move your call to")
    replies = slot_replies(offer)
    assert replies
    assert all(
        q["action"] == {"type": "reschedule", "booking_uid": uid, "slot_id": q["action"]["slot_id"]}
        for q in replies
    )
    moved = await guarded.act(replies[1]["action"])
    assert moved["booking"]["action"] == "rescheduled"
    assert moved["booking"]["start_utc"] == replies[1]["start_utc"]
    [active] = guarded.bookings()
    assert active["uid"] == moved["booking"]["ref"] != uid
    cancelled = await guarded.act({"type": "cancel", "booking_uid": active["uid"]})
    assert cancelled["booking"]["action"] == "cancelled"
    assert cancelled["booking"]["status"] == "cancelled"
    assert "is cancelled" in cancelled["reply"]
    assert guarded.bookings() == []


async def test_a_text_cancel_goes_through_the_model(naive: AgentEnv) -> None:
    uid = naive.setup_booking(MON_1000)
    data = await naive.say("Please drop the meeting, we won't need it.")
    assert "is cancelled" in data["reply"]
    assert data["booking"]["ref"] == uid
    assert data["booking"]["action"] == "cancelled"
    assert naive.bookings() == []


async def test_a_reschedule_action_in_the_naive_mode_goes_through_the_model(naive: AgentEnv) -> None:
    uid = naive.setup_booking(MON_1000)
    data = await naive.act({"type": "reschedule", "booking_uid": uid})
    assert data["reply"].startswith("Sure, here are some open times to move your call to")
    first = data["reply"].split("\n")[1].removeprefix("- ")
    moved = await naive.say(f"{first} works for me.")
    assert moved["booking"]["action"] == "rescheduled"


async def test_the_widget_can_only_change_its_own_bookings(guarded: AgentEnv) -> None:
    outside = guarded.setup_booking(MON_1000)
    first = await guarded.widget(action={"type": "cancel", "booking_uid": outside}, session="w-1")
    assert first.status_code == 200
    assert "only change bookings made in this conversation" in first.json()["reply"]
    assert len(guarded.bookings()) == 1


# Zones and context ----------------------------------------------------------------------------------------


async def test_confirm_timezone_stores_a_confirmed_zone(guarded: AgentEnv) -> None:
    data = await guarded.act({"type": "confirm_timezone", "zone": "Asia/Kolkata"})
    lead = guarded.deps.store.leads.get(LEAD)
    assert lead is not None
    assert (lead.tz_zone, lead.tz_source, lead.tz_confirmed) == ("Asia/Kolkata", "confirmed", True)
    assert "Asia/Kolkata" in data["reply"]
    bad = await guarded.act({"type": "confirm_timezone", "zone": "Mars/Olympus"})
    assert "don't know the time zone" in bad["reply"]


def test_the_context_block_uses_the_lead_zone(guarded: AgentEnv) -> None:
    agent = core(guarded)
    ctx = TurnContext(
        session_id="s-1",
        message_id="m",
        channel="api",
        lead_email=LEAD,
        lead_name="Maya R",
        zone="Australia/Sydney",
        zone_source="stated",
        now=NOW,
    )
    block = agent.context_block(ctx)
    assert block == {
        "today": "2026-10-01",
        "weekday": "Thursday",
        "zone": "Australia/Sydney",
        "zone_source": "stated",
        "host_zone": "America/New_York",
        "meeting_minutes": 30,
        "lead_name": "Maya R",
        "active_bookings": [],
        "channel": "api",
    }
    late = TurnContext(**{**ctx.__dict__, "now": datetime(2026, 10, 1, 14, 30, tzinfo=UTC)})
    assert agent.context_block(late)["today"] == "2026-10-02"  # already Friday in Sydney
    prompt = agent.system_prompt(ctx)
    assert prompt.startswith(guarded.deps.prompt.rstrip())
    assert json.loads(prompt.split("<context>")[1].split("</context>")[0]) == block


async def test_a_browser_hint_is_the_zone_until_one_is_stated(guarded: AgentEnv) -> None:
    offer = await guarded.say(
        "Hi, can I book a call with you in the next few days? Mornings are best.", hint="America/Denver"
    )
    assert "(shown in America/Denver)" in offer["reply"]
    await guarded.say("Actually I'm in Berlin.", hint="America/Denver", session="s-2")
    lead = guarded.deps.store.leads.get(LEAD)
    assert lead is not None
    assert lead.tz_zone == "Europe/Berlin"
    again = await guarded.say("Can I book a call next week?", hint="America/Denver", session="s-3")
    assert "(shown in Europe/Berlin)" in again["reply"]
    invalid = await guarded.say(
        "Can I book a call next week?", hint="Not/AZone", session="s-4", email="omar@example.com"
    )
    assert "(shown in America/New_York)" in invalid["reply"]


# Model failures ---------------------------------------------------------------------------------------------


class Failing:
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
        raise LLMError("budget reached", kind="budget")


class PlainText:
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
        return LLMResponse(
            content="Happy to help! When would suit you?",
            tool_calls=[],
            usage=Usage(prompt_tokens=5, completion_tokens=5, usd=0.0001),
            model_requested="vendor/model",
            model_returned="vendor/model-2026",
            provider="provider-a",
            response_id="r1",
            latency_s=0.1,
        )


async def test_a_model_failure_gets_an_honest_reply(sandbox: Sandbox, tmp_path: Path) -> None:
    async for env in make_env(sandbox, tmp_path, llm=Failing()):
        data = await env.say("Can I book a call?")
        assert "Nothing has been booked or changed" in data["reply"]
        assert [(e["guard"], e["event"]) for e in data["guard"]["events"]] == [("agent", "llm_error")]
        assert env.bookings() == []


class FlakyOnCue:
    """``FakeLLM`` that raises ``LLMError`` instead of answering on a turn whose own message contains
    ``cue``, otherwise behaves exactly like the scripted policy: an injected upstream error on one turn,
    unrelated to a booking made earlier in the same session."""

    def __init__(self, cue: str) -> None:
        self.cue = cue
        self.inner = FakeLLM()

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
        if messages and messages[-1].role == "user" and self.cue in (messages[-1].content or ""):
            raise LLMError("rate limit", kind="rate_limit")
        return await self.inner.chat(
            messages=messages,
            tools=tools,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            response_format=response_format,
            component=component,
            run_id=run_id,
        )


async def test_llm_error_on_a_later_turn_reports_the_booking_as_unchanged(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    """Run-1 pattern: a booking is made and confirmed, then a later, unrelated turn ("Great, thanks.") hits
    an upstream LLM error. The fallback must not claim "nothing has been booked" when something is — with
    ``claim_ledger`` it reports the session's own verified booking as unchanged, code-rendered from the
    ledger, not the generic template."""
    async for env in make_env(sandbox, tmp_path, llm=FlakyOnCue("Great, thanks")):
        offer = await env.say("I'd like to book an intro call next week.")
        picked = offer["reply"].split("\n")[1].removeprefix("- ")
        booked = await env.say(f"{picked} works for me.")
        assert booked["booking"]["action"] == "booked"
        ref = booked["booking"]["ref"]

        failed = await env.say("Great, thanks, see you then.")
        assert "Nothing has been booked" not in failed["reply"]
        assert "unchanged" in failed["reply"]
        assert ref[:8] in failed["reply"]
        assert ("agent", "llm_error") in [(e["guard"], e["event"]) for e in failed["guard"]["events"]]
        assert len(env.bookings()) == 1  # the booking itself is of course untouched


async def test_a_plain_text_answer_is_the_reply(sandbox: Sandbox, tmp_path: Path) -> None:
    async for env in make_env(sandbox, tmp_path, llm=PlainText()):
        data = await env.say("Hello")
        assert data["reply"] == "Happy to help! When would suit you?"
        assert data["usage"]["models"] == ["vendor/model-2026"]
        assert data["usage"]["providers"] == ["provider-a"]
        assert data["usage"]["usd"] == 0.0001
        auth = {"Authorization": f"Bearer {API_KEY}"}
        trace = (await env.client.get("/v1/sessions/s-1/trace", headers=auth)).json()
        assert trace["final_claim"] == {"text": "Happy to help! When would suit you?", "claims": []}


async def test_history_carries_over_between_turns(guarded: AgentEnv) -> None:
    await guarded.say(BERLIN_ASK)
    await guarded.say("Thanks!")
    history = guarded.deps.store.history.for_session("s-1")
    roles = [e.role for e in history]
    assert roles[0] == "user"
    assert roles.count("user") == 2
    assert history[-1].role == "assistant"
    assert json.loads(history[-1].content["content"])["reply"] == "You're welcome! Talk soon."

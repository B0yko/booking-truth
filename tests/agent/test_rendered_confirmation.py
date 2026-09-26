"""``rendered_confirmation`` in the running agent: every verified write of a turn gets a confirmation line
rendered by code from its ledger entry, above the reply. The model's text follows the line when the claim
check passes it; when the check blocks it (or code writes the reply), only a short sentence follows, so the
reply never says both "Booked: ..." and "I haven't booked anything yet"."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from agent_env import LEAD, NOW, AgentEnv, make_env
from fastapi import FastAPI

import booking_truth.agent.guards.readback as readback
from booking_truth.agent import render
from booking_truth.agent.core import AgentCore
from booking_truth.agent.guards import GUARD_NAMES, guards_string
from booking_truth.agent.scripted import FakeLLM
from booking_truth.agent.tools import TurnContext, WriteRecord
from booking_truth.calendars.base import BookingRecord
from booking_truth.llm.types import ChatMessage, LLMError, LLMResponse, ToolCall, ToolSpec, Usage
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer

Sandbox = tuple[FastAPI, BackgroundServer, SandboxState]
ASK = "Hi, I'm in New York. Can I book an intro call next week, ideally in the afternoon?"
BERLIN_ASK = "Hi, I'm in Berlin. Can I book an intro call next week, ideally late afternoon?"
BERLIN = "Europe/Berlin"
NY = "America/New_York"
NO_RENDERED_LINE = guards_string(frozenset(GUARD_NAMES) - {"rendered_confirmation"})
INVITE = "The calendar invite is on its way to your email."


def slot_replies(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [q for q in data["quick_replies"] if q.get("start_utc")]


def first_offer(data: dict[str, Any]) -> str:
    line: str = data["reply"].split("\n")[1]
    return line.removeprefix("- ")


def events(data: dict[str, Any], guard: str) -> list[str]:
    return [e["event"] for e in data["guard"]["events"] if e["guard"] == guard]


def details(data: dict[str, Any], guard: str = "rendered_confirmation") -> list[str]:
    return [e["detail"] for e in data["guard"]["events"] if e["guard"] == guard]


def line_of(env: AgentEnv, action: str, ref: str) -> str:
    """The line the ledger entry of ``ref`` renders to, straight from the store."""
    entries = env.deps.store.claims.entries(LEAD, status="verified")
    [entry] = [e for e in entries if e.booking_ref == ref and e.action == action]
    assert entry.zone is not None
    return render.confirmation_line(action, entry.start_utc, entry.zone, ref)


def final_answer(env: AgentEnv) -> dict[str, Any]:
    history = env.deps.store.history.for_session(env.session)
    answer: dict[str, Any] = json.loads(history[-1].content["content"])
    return answer


def core_of(env: AgentEnv) -> AgentCore:
    found: AgentCore = env.app.state.core
    return found


def response(content: str | None, calls: Sequence[ToolCall] = ()) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=list(calls),
        usage=Usage(prompt_tokens=5, completion_tokens=5),
        model_requested="offline/test",
        model_returned="offline/test",
        provider="offline",
        response_id="r",
        latency_s=0.0,
    )


def answer(reply: str, claims: Sequence[dict[str, str]] = ()) -> str:
    return json.dumps({"reply": reply, "claims": list(claims)})


class Steps:
    """A model that gives these answers in order: a list of tool calls, a final answer, or an error."""

    def __init__(self, steps: Sequence[list[ToolCall] | str | LLMError]) -> None:
        self.steps = list(steps)
        self.calls = 0

    async def chat(self, **kwargs: Any) -> LLMResponse:
        step = self.steps[self.calls]
        self.calls += 1
        if isinstance(step, LLMError):
            raise step
        if isinstance(step, list):
            return response(None, step)
        return response(step)


class Repairing:
    """The scripted model, except that the repair call (after a ``[claim check]`` note) gets ``repair``."""

    def __init__(self, misbehaviours: Sequence[str], repair: str) -> None:
        self.inner = FakeLLM(misbehaviours)
        self.repair = repair
        self.notes: list[str] = []

    @property
    def calls(self) -> int:
        return self.inner.calls + len(self.notes)

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
            self.notes.append(last.content or "")
            return response(self.repair)
        return await self.inner.chat(
            messages=messages, tools=tools, temperature=temperature, model=model, max_tokens=max_tokens
        )


# Code paths -----------------------------------------------------------------------------------------------


async def test_a_booking_by_button_gets_the_line_and_one_sentence(guarded: AgentEnv) -> None:
    offer = await guarded.say(ASK)
    booked = await guarded.act(slot_replies(offer)[0]["action"])
    booking = booked["booking"]
    line = line_of(guarded, "booked", booking["ref"])
    assert line == f"Booked: {booking['local_label']} · reference {booking['ref'][:8]}"
    assert line.endswith(f"{NY} (UTC-04:00) · reference {booking['ref'][:8]}")
    assert booked["reply"] == f"{line}\n\n{INVITE}"
    assert events(booked, "claim_ledger") == ["verified"]
    assert details(booked) == [line]
    assert (booked["guard"]["blocked"], booked["guard"]["repaired"]) == (False, False)
    # The model sees next turn what the prospect saw, and the claim is the line's own.
    assert final_answer(guarded) == {
        "reply": booked["reply"],
        "claims": [{"type": "booked", "time": booking["local_label"]}],
    }


async def test_reschedule_and_cancel_buttons_get_their_lines(guarded: AgentEnv) -> None:
    offer = await guarded.say(ASK)
    first = (await guarded.act(slot_replies(offer)[0]["action"]))["booking"]["ref"]
    options = await guarded.act({"type": "reschedule", "booking_uid": first})
    moved = await guarded.act(slot_replies(options)[1]["action"])
    second = moved["booking"]
    line = line_of(guarded, "rescheduled", second["ref"])
    assert line == f"Rescheduled: your call is now {second['local_label']} · reference {second['ref'][:8]}"
    assert moved["reply"] == f"{line}\n\n{render.confirmation_follow_up('rescheduled')}"
    assert details(moved) == [line]
    cancelled = await guarded.act({"type": "cancel", "booking_uid": second["ref"]})
    line = line_of(guarded, "cancelled", second["ref"])
    assert line == (
        f"Cancelled: your call on {second['local_label']} is cancelled · reference {second['ref'][:8]}"
    )
    assert cancelled["reply"] == f"{line}\n\nNothing is booked for you now."
    assert details(cancelled) == [line]


# Model replies --------------------------------------------------------------------------------------------


async def test_a_correct_model_confirmation_follows_the_line(guarded: AgentEnv) -> None:
    offer = await guarded.say(BERLIN_ASK)
    data = await guarded.say(f"{first_offer(offer)} works for me.")
    line = line_of(guarded, "booked", data["booking"]["ref"])
    assert f"{BERLIN} (UTC+02:00)" in line
    head, text = data["reply"].split("\n\n", 1)
    assert head == line
    assert text.startswith(f"You're booked for {first_offer(offer)} ({BERLIN}).")
    assert events(data, "claim_ledger") == ["verified"]
    assert details(data) == [line]


async def test_a_garbled_confirmation_time_leaves_only_the_line(sandbox: Sandbox, tmp_path: Path) -> None:
    """The scripted model states the booked time in the host's zone; the claim check blocks it and its
    repair, and the line of the verified booking is what the prospect reads, with no denial under it."""
    async for env in make_env(sandbox, tmp_path, llm=FakeLLM(["garble_confirmation_time"])):
        offer = await env.say(BERLIN_ASK)
        data = await env.say(f"{first_offer(offer)} works for me.")
        line = line_of(env, "booked", data["booking"]["ref"])
        assert data["reply"] == f"{line}\n\n{INVITE}"
        assert events(data, "claim_ledger") == [
            "verified",
            "claim_blocked",
            "repair_blocked",
            "safe_template",
        ]
        assert details(data) == [line]
        assert data["guard"]["blocked"] is True
        assert render.SAFE_NOT_BOOKED not in data["reply"]
        assert len(env.bookings()) == 1
        assert final_answer(env)["reply"] == data["reply"]


async def test_the_repair_note_shows_the_line(sandbox: Sandbox, tmp_path: Path) -> None:
    repair = answer("Great, you're all set. See you then!", [{"type": "booked", "time": ""}])
    llm = Repairing(["garble_confirmation_time"], repair)
    async for env in make_env(sandbox, tmp_path, llm=llm):
        offer = await env.say(BERLIN_ASK)
        data = await env.say(f"{first_offer(offer)} works for me.")
        line = line_of(env, "booked", data["booking"]["ref"])
        [note] = llm.notes
        assert f"above your reply, so you do not need to repeat its time or reference:\n- {line}" in note
        assert data["reply"] == f"{line}\n\nGreat, you're all set. See you then!"
        assert (data["guard"]["blocked"], data["guard"]["repaired"]) == (False, True)
        assert events(data, "claim_ledger") == ["verified", "claim_blocked", "repaired"]


async def test_two_writes_in_one_turn_get_two_lines_in_order(guarded: AgentEnv) -> None:
    offer = await guarded.say(ASK)
    slots = slot_replies(offer)
    old = (await guarded.act(slots[0]["action"]))["booking"]["ref"]
    cancel = ToolCall("c1", "cancel_booking", json.dumps({"booking_uid": old, "reason": "Moving it"}))
    book = ToolCall("c2", "book_slot", json.dumps({"slot_id": slots[2]["action"]["slot_id"]}))
    model: Any = Steps([[cancel], [book], answer("Done: I cancelled the old call and booked the new one.")])
    guarded.deps.llm = model
    data = await guarded.say("Please cancel that one and take the third time instead.")
    new = data["booking"]["ref"]
    lines = [line_of(guarded, "cancelled", old), line_of(guarded, "booked", new)]
    assert data["reply"] == "\n".join(lines) + "\n\nDone: I cancelled the old call and booked the new one."
    assert details(data) == lines
    assert [c["type"] for c in final_answer(guarded)["claims"]] == ["cancelled", "booked"]


async def test_a_model_failure_after_a_booking_leaves_the_line(guarded: AgentEnv) -> None:
    offer = await guarded.say(ASK)
    book = ToolCall("c1", "book_slot", json.dumps({"slot_id": slot_replies(offer)[0]["action"]["slot_id"]}))
    model: Any = Steps([[book], LLMError("the provider is down", kind="server")])
    guarded.deps.llm = model
    data = await guarded.say("The first one, please.")
    line = line_of(guarded, "booked", data["booking"]["ref"])
    assert data["reply"] == f"{line}\n\n{INVITE}"
    assert events(data, "agent") == ["llm_error"]


# When no line is rendered ----------------------------------------------------------------------------------


async def test_an_unconfirmed_write_gets_no_line(guarded: AgentEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readback, "READBACK_WINDOW_S", 0.3)
    monkeypatch.setattr(readback, "READBACK_STEP_S", 0.1)
    offer = await guarded.say(ASK)
    guarded.faults({"group": "bookings.get", "mode": "error_500", "times": None})
    data = await guarded.act(slot_replies(offer)[0]["action"])
    assert data["reply"] == render.UNCONFIRMED
    assert details(data) == []


async def test_without_the_guard_code_writes_the_whole_sentence(sandbox: Sandbox, tmp_path: Path) -> None:
    async for env in make_env(sandbox, tmp_path, guards=NO_RENDERED_LINE):
        offer = await env.say(ASK)
        pick = slot_replies(offer)[0]
        data = await env.act(pick["action"])
        assert data["reply"].startswith(f"You're booked for {pick['label']} ({NY}). Reference ")
        assert "Booked:" not in data["reply"]
        assert details(data) == []
        assert events(data, "claim_ledger") == ["verified"]


# Rendering from the ledger ---------------------------------------------------------------------------------


def write_at(
    start: datetime, *, zone: str, status: str = "verified", ref: str = "uid-rendered-1"
) -> WriteRecord:
    booking = BookingRecord(
        ref=ref,
        start=start,
        end=start + timedelta(minutes=30),
        status="active",
        lead_email=LEAD,
        idem_key=None,
        raw={},
    )
    return WriteRecord("booked", booking, zone, status)  # type: ignore[arg-type]


def turn(zone: str = NY) -> TurnContext:
    return TurnContext("s-1", "m-1", "api", LEAD, None, zone, "stated", NOW)


def test_the_line_comes_from_the_ledger_entry(guarded: AgentEnv) -> None:
    core = core_of(guarded)
    start = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
    ctx = turn()
    write = write_at(start, zone=NY)
    ctx.state.writes.append(write)
    assert core._hook_confirmations(ctx) == []  # verified, but no ledger entry: nothing to render from
    guarded.deps.store.claims.record(
        lead_email=LEAD,
        event_key="1001",
        action="booked",
        booking_ref=write.booking.ref,
        start_utc=start,
        end_utc=start + timedelta(minutes=30),
        status="verified",
        zone=BERLIN,
        session_id="s-1",
        channel="api",
    )
    [confirmation] = core._hook_confirmations(ctx)
    assert confirmation.write is write
    assert confirmation.line == (
        "Booked: Tuesday, 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00) · reference uid-rend"
    )
    assert confirmation.when == "Tuesday, 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00)"


def test_only_verified_writes_of_this_session_get_a_line(guarded: AgentEnv) -> None:
    core = core_of(guarded)
    start = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
    for ref, status, session in (("uid-a", "unverified", "s-1"), ("uid-b", "verified", "s-2")):
        guarded.deps.store.claims.record(
            lead_email=LEAD,
            event_key="1001",
            action="booked",
            booking_ref=ref,
            start_utc=start,
            end_utc=start + timedelta(minutes=30),
            status=status,  # type: ignore[arg-type]
            zone=NY,
            session_id=session,
        )
    ctx = turn()
    ctx.state.writes += [
        write_at(start, zone=NY, status="unverified", ref="uid-a"),
        write_at(start, zone=NY, ref="uid-b"),
        write_at(start, zone=NY, status="trusted", ref="uid-c"),
    ]
    assert core._hook_confirmations(ctx) == []


def test_with_the_guard_off_nothing_is_rendered(guarded: AgentEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    core = core_of(guarded)
    ctx = turn()
    ctx.state.writes.append(write_at(datetime(2026, 10, 6, 13, 0, tzinfo=UTC), zone=NY))
    monkeypatch.setattr(core, "on", lambda guard: guard != "rendered_confirmation")
    assert core._hook_confirmations(ctx) == []

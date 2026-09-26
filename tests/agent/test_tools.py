"""Tool schemas for both modes and the executor, against the sandbox over HTTP."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from agent_env import LEAD, LEAD_NAME, NOW, AgentEnv, executor, guarded_slots, make_env, mutable_clock
from fastapi import FastAPI

from booking_truth.agent.tools import (
    EXPIRED_INSTRUCTION,
    UNAVAILABLE_INSTRUCTION,
    HandoffNotifier,
    business_days,
    guarded_specs,
    naive_specs,
    slot_id_for,
    spread_slots,
)
from booking_truth.calendars.base import Slot
from booking_truth.llm.types import ToolCall
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer
from booking_truth.store import Store
from booking_truth.timeutil import MutableClock

MONDAY = date(2026, 10, 5)
FRIDAY = date(2026, 10, 9)
# Monday 5 October 2026, 10:00 in New York.
MON_1000 = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)


# Schemas ----------------------------------------------------------------------------------------------------


def test_the_two_tool_sets_differ_only_where_the_naive_baseline_differs() -> None:
    guarded = {spec.name: spec for spec in guarded_specs()}
    naive = {spec.name: spec for spec in naive_specs()}
    common = {"resolve_timezone", "find_slots", "list_my_bookings", "reschedule_booking", "cancel_booking"}
    assert set(guarded) == common | {"book_slot", "handoff_to_human"}
    assert set(naive) == common | {"book", "handoff_to_human"}
    for name in ("resolve_timezone", "list_my_bookings", "cancel_booking", "handoff_to_human"):
        assert guarded[name].to_openai() == naive[name].to_openai()
    assert guarded["book_slot"].parameters["required"] == ["slot_id"]
    assert naive["book"].parameters["required"] == ["start_iso"]
    assert guarded["reschedule_booking"].parameters["required"] == ["booking_uid", "slot_id"]
    assert naive["reschedule_booking"].parameters["required"] == ["booking_uid", "start_iso"]
    assert guarded["find_slots"].parameters["required"] == naive["find_slots"].parameters["required"]
    for spec in [*guarded.values(), *naive.values()]:
        assert spec.parameters["additionalProperties"] is False
        assert set(spec.parameters["required"]) <= set(spec.parameters["properties"])


def test_slot_ids_are_opaque_and_bound_to_their_list() -> None:
    first = slot_id_for("secret", "list-a", MON_1000)
    assert re.fullmatch(r"s_[a-z2-7]{10}", first)
    assert first == slot_id_for("secret", "list-a", MON_1000)
    assert first != slot_id_for("secret", "list-b", MON_1000)
    assert first != slot_id_for("other", "list-a", MON_1000)


def test_spread_keeps_every_day_and_chronological_order() -> None:
    slots = []
    for day in range(5):
        for index in range(16):
            start = datetime(2026, 10, 5 + day, 13, 0, tzinfo=UTC) + timedelta(minutes=30 * index)
            slots.append(Slot(start, start + timedelta(minutes=30)))
    chosen = spread_slots(slots, "America/New_York", 12)
    assert len(chosen) == 12
    assert chosen == sorted(chosen, key=lambda s: s.start)
    assert len({s.start.date() for s in chosen}) == 4
    assert spread_slots(slots[:5], "America/New_York", 12) == slots[:5]
    one_day = spread_slots(slots[:16], "America/New_York", 12)
    assert len(one_day) == 12
    assert one_day[0] == slots[0]
    assert one_day[-1] == slots[15]


def test_business_days_skip_weekends() -> None:
    assert business_days(date(2026, 10, 2), 3) == [date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)]


# find_slots ----------------------------------------------------------------------------------------------


async def test_guarded_find_slots_returns_labelled_slot_ids_in_the_lead_zone(guarded: AgentEnv) -> None:
    tools = executor(guarded, zone="Europe/Berlin")
    result = await tools.run("find_slots", {"from_date": "2026-10-05", "to_date": "2026-10-09"})
    assert result["zone"] == "Europe/Berlin"
    assert result["more_available"] is True
    assert len(result["slots"]) == 12
    first = result["slots"][0]
    assert set(first) == {"slot_id", "label", "local_date", "local_time"}
    # Host hours 09:00-17:00 New York are 15:00-23:00 in Berlin (both on summer time).
    assert first["local_date"] == "2026-10-05"
    assert first["local_time"] >= "15:00"
    assert first["label"].startswith("Monday 5 October, ")
    stored = guarded.deps.store.slot_lists.latest(LEAD)
    assert stored is not None
    assert [s["slot_id"] for s in stored.slots] == [s["slot_id"] for s in result["slots"]]
    assert tools.state.shown is not None
    assert tools.state.shown.list_id == stored.id
    steps = tools.state.steps
    assert [(s["kind"], s["name"]) for s in steps] == [
        ("tool_call", "find_slots"),
        ("tool_result", "find_slots"),
    ]
    assert steps[1]["ok"] is True


async def test_naive_find_slots_returns_utc_starts(naive: AgentEnv) -> None:
    tools = executor(naive, zone="Europe/Berlin")
    result = await tools.run("find_slots", {"from_date": "2026-10-05", "to_date": "2026-10-09"})
    starts = result["available_starts_utc"]
    assert len(starts) == 20
    assert starts[0] == "2026-10-05T13:00:00Z"
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00Z", s) for s in starts)
    assert naive.deps.store.slot_lists.latest(LEAD) is not None


async def test_a_range_starting_in_the_past_starts_now(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    result = await tools.run("find_slots", {"from_date": "2026-09-30", "to_date": "2026-10-01"})
    starts = [s["local_time"] for s in result["slots"]]
    # 08:00 in New York now; two hours' notice leaves 10:00 as the first slot today.
    assert result["slots"][0]["local_date"] == "2026-10-01"
    assert starts[0] == "10:00"


@pytest.mark.parametrize(
    ("args", "detail"),
    [
        ({"from_date": "2026-10-05", "to_date": "2026-10-30"}, "at most 14 days"),
        ({"from_date": "2026-10-09", "to_date": "2026-10-05"}, "before from_date"),
        ({"from_date": "5 October", "to_date": "2026-10-05"}, "YYYY-MM-DD"),
        ({"from_date": "2026-02-30", "to_date": "2026-03-01"}, "from_date"),
        ({}, "from_date"),
    ],
)
async def test_find_slots_checks_its_arguments(guarded: AgentEnv, args: dict[str, Any], detail: str) -> None:
    result = await executor(guarded).run("find_slots", args)
    assert result["error"] == "invalid_arguments"
    assert detail in result["detail"]
    assert guarded.log("slots") == []


async def test_the_naive_agent_gets_argument_errors_as_text(naive: AgentEnv) -> None:
    result = await executor(naive).run("find_slots", {"from_date": "tomorrow", "to_date": "2026-10-05"})
    assert isinstance(result, str)
    assert result.startswith("Error: invalid arguments")


async def test_fail_closed_reports_an_unavailable_calendar_after_one_retry(guarded: AgentEnv) -> None:
    guarded.faults({"group": "slots", "mode": "error_500", "times": None})
    tools = executor(guarded)
    result = await tools.run("find_slots", {"from_date": "2026-10-05", "to_date": "2026-10-09"})
    assert result == {"unavailable": True, "reason": "error", "instruction": UNAVAILABLE_INSTRUCTION}
    assert len(guarded.log("slots")) == 2
    events = [(e.guard, e.event) for e in tools.state.events]
    assert events == [("fail_closed", "lookup_retried"), ("fail_closed", "calendar_unavailable")]
    assert tools.state.calendar_unavailable
    assert guarded.deps.store.slot_lists.latest(LEAD) is None
    assert tools.state.steps[-1]["ok"] is False


async def test_the_internal_retry_recovers_from_one_error(guarded: AgentEnv) -> None:
    guarded.faults({"group": "slots", "mode": "error_500", "times": 1})
    result = await executor(guarded).run("find_slots", {"from_date": "2026-10-05", "to_date": "2026-10-09"})
    assert len(result["slots"]) == 12
    assert len(guarded.log("slots")) == 2


async def test_not_found_is_not_retried(guarded: AgentEnv) -> None:
    guarded.faults({"group": "slots", "mode": "not_found", "times": None})
    result = await executor(guarded).run("find_slots", {"from_date": "2026-10-05", "to_date": "2026-10-09"})
    assert result["reason"] == "not_found"
    assert len(guarded.log("slots")) == 1


async def test_the_naive_agent_sees_the_raw_error_text(naive: AgentEnv) -> None:
    naive.faults({"group": "slots", "mode": "error_500", "times": None})
    result = await executor(naive).run("find_slots", {"from_date": "2026-10-05", "to_date": "2026-10-09"})
    assert isinstance(result, str)
    assert result.startswith("Error: calendar returned HTTP 500: ")
    assert len(naive.log("slots")) == 1


async def test_the_naive_agent_reads_a_malformed_answer_as_times(naive: AgentEnv) -> None:
    naive.faults({"group": "slots", "mode": "malformed", "times": 1})
    result = await executor(naive).run("find_slots", {"from_date": "2026-10-05", "to_date": "2026-10-09"})
    assert isinstance(result, dict)
    assert "available_starts_utc" in result


# Writes ---------------------------------------------------------------------------------------------------


async def test_book_slot_books_the_listed_slot(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    slot = (await guarded_slots(tools))[0]
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is True
    assert result["label"] == slot["label"]
    assert result["zone"] == "America/New_York"
    bookings = guarded.bookings()
    assert [b["uid"] for b in bookings] == [result["booking_uid"]]
    assert bookings[0]["attendees"][0]["name"] == LEAD_NAME
    write = tools.state.writes[-1]
    assert (write.action, write.status, write.booking.ref) == ("booked", "verified", result["booking_uid"])
    assert tools.state.last_book_start == write.booking.start


async def test_an_unknown_slot_id_books_nothing(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    result = await tools.run("book_slot", {"slot_id": "s_aaaaaaaaaa"})
    assert result == {
        "booked": False,
        "reason": "unknown_or_expired_slot",
        "instruction": EXPIRED_INSTRUCTION,
    }
    assert guarded.log("bookings.create") == []
    assert [(e.guard, e.event) for e in tools.state.events] == [("slot_ids", "unknown_or_expired_slot")]


@pytest.fixture
async def guarded_mutable(
    sandbox: tuple[FastAPI, BackgroundServer, SandboxState], tmp_path: Path
) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards="all", clock=mutable_clock()):
        yield env


async def test_a_slot_list_expires_after_the_ttl(guarded_mutable: AgentEnv) -> None:
    env = guarded_mutable
    clock = env.clock
    assert isinstance(clock, MutableClock)
    slot = (await guarded_slots(executor(env)))[0]
    clock.advance(timedelta(seconds=env.deps.settings.slot_ttl_seconds))
    result = await executor(env).run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["reason"] == "unknown_or_expired_slot"
    assert env.bookings() == []


async def test_only_the_latest_list_counts(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    old = (await guarded_slots(tools))[0]
    await guarded_slots(tools, date(2026, 10, 12), date(2026, 10, 13))
    result = await tools.run("book_slot", {"slot_id": old["slot_id"]})
    assert result["reason"] == "unknown_or_expired_slot"


async def test_a_taken_slot_asks_for_new_times(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    slot = (await guarded_slots(tools))[0]
    guarded.faults({"group": "bookings.create", "mode": "slot_taken_after_offer", "times": 1})
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is False
    assert result["reason"] == "slot_taken"
    assert "find_slots again" in result["instruction"]
    assert tools.state.writes == []


async def test_a_calendar_error_is_not_a_booking(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    slot = (await guarded_slots(tools))[0]
    guarded.faults({"group": "bookings.create", "mode": "error_500", "times": None})
    result = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert result["booked"] is False
    assert result["reason"] == "calendar_error"
    assert len(guarded.log("bookings.create")) == 1  # guarded: no blind POST retry


async def test_the_naive_book_tool_takes_a_model_computed_iso_time(naive: AgentEnv) -> None:
    tools = executor(naive)
    result = await tools.run("book", {"start_iso": "2026-10-05T10:00:00-04:00"})
    assert result == {"booked": True, "booking_uid": result["booking_uid"], "start": "2026-10-05T14:00:00Z"}
    no_offset = await tools.run("book", {"start_iso": "2026-10-05T15:00:00"})
    assert no_offset["start"] == "2026-10-05T15:00:00Z"
    bad = await tools.run("book", {"start_iso": "next Monday"})
    assert isinstance(bad, str)
    assert bad.startswith("Error: could not parse")


async def test_the_naive_book_tool_passes_errors_as_text(naive: AgentEnv) -> None:
    naive.faults({"group": "bookings.create", "mode": "error_500", "times": None})
    result = await executor(naive).run("book", {"start_iso": "2026-10-05T14:00:00Z"})
    assert isinstance(result, str)
    assert result.startswith("Error: calendar returned HTTP 500")


@pytest.fixture
async def naive_fast(
    sandbox: tuple[FastAPI, BackgroundServer, SandboxState], tmp_path: Path
) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards="off", fast_calendar=True):
        yield env


async def test_the_naive_client_retries_a_post_after_a_timeout(naive_fast: AgentEnv) -> None:
    naive_fast.faults({"group": "bookings.create", "mode": "timeout", "times": 1, "hang_s": 1.0})
    result = await executor(naive_fast).run("book", {"start_iso": "2026-10-05T14:00:00Z"})
    assert result["booked"] is True
    assert len(naive_fast.log("bookings.create")) == 2


async def test_reschedule_moves_the_booking_to_a_listed_slot(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(MON_1000)
    tools = executor(guarded)
    target = (await guarded_slots(tools, date(2026, 10, 6), date(2026, 10, 6)))[0]
    result = await tools.run("reschedule_booking", {"booking_uid": uid, "slot_id": target["slot_id"]})
    assert result["rescheduled"] is True
    assert result["booking_uid"] != uid
    assert result["label"] == target["label"]
    active = guarded.bookings()
    assert [b["uid"] for b in active] == [result["booking_uid"]]
    write = tools.state.writes[-1]
    assert write.action == "rescheduled"
    assert write.previous_ref == uid


async def test_reschedule_of_an_unknown_booking(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    target = (await guarded_slots(tools))[0]
    result = await tools.run("reschedule_booking", {"booking_uid": "nope", "slot_id": target["slot_id"]})
    assert result["rescheduled"] is False
    assert result["reason"] == "not_found"


async def test_naive_reschedule_takes_an_iso_time(naive: AgentEnv) -> None:
    uid = naive.setup_booking(MON_1000)
    result = await executor(naive).run(
        "reschedule_booking", {"booking_uid": uid, "start_iso": "2026-10-06T14:00:00Z"}
    )
    assert result["rescheduled"] is True
    assert result["start"] == "2026-10-06T14:00:00Z"


async def test_cancel_and_cancel_again(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(MON_1000)
    tools = executor(guarded)
    first = await tools.run("cancel_booking", {"booking_uid": uid, "reason": "No longer needed"})
    assert first == {"cancelled": True, "booking_uid": uid, "label": "Monday 5 October, 10:00 AM"}
    assert guarded.bookings() == []
    again = await tools.run("cancel_booking", {"booking_uid": uid, "reason": ""})
    assert again == {"cancelled": False, "reason": "already_cancelled"}
    missing = await tools.run("cancel_booking", {"booking_uid": "nope", "reason": ""})
    assert missing == {"cancelled": False, "reason": "not_found"}


async def test_list_my_bookings_in_both_modes(guarded: AgentEnv) -> None:
    uid = guarded.setup_booking(MON_1000)
    guarded.setup_booking(MON_1000 + timedelta(hours=2), email="omar@example.com", name="Omar N")
    result = await executor(guarded).run("list_my_bookings", {})
    assert result == {
        "bookings": [
            {
                "booking_uid": uid,
                "label": "Monday 5 October, 10:00 AM",
                "status": "active",
                "start_utc": "2026-10-05T14:00:00Z",
            }
        ]
    }


async def test_naive_list_uses_utc_times(naive: AgentEnv) -> None:
    uid = naive.setup_booking(MON_1000)
    result = await executor(naive).run("list_my_bookings", {})
    assert result == {
        "bookings": [
            {
                "booking_uid": uid,
                "start": "2026-10-05T14:00:00Z",
                "end": "2026-10-05T14:30:00Z",
                "status": "active",
            }
        ]
    }


async def test_the_widget_reaches_only_bookings_of_its_own_session(guarded: AgentEnv) -> None:
    outside = guarded.setup_booking(MON_1000)
    tools = executor(guarded, channel="widget", session="w-1")
    assert await tools.run("list_my_bookings", {}) == {"bookings": []}
    assert await tools.run("cancel_booking", {"booking_uid": outside, "reason": ""}) == {
        "cancelled": False,
        "reason": "not_allowed",
    }
    slot = (await guarded_slots(tools))[0]
    moved = await tools.run("reschedule_booking", {"booking_uid": outside, "slot_id": slot["slot_id"]})
    assert moved["reason"] == "not_allowed"
    booked = await tools.run("book_slot", {"slot_id": slot["slot_id"]})
    assert guarded.deps.store.widget_bookings.owns("w-1", booked["booking_uid"])
    listed = await tools.run("list_my_bookings", {})
    assert [b["booking_uid"] for b in listed["bookings"]] == [booked["booking_uid"]]
    other_session = executor(guarded, channel="widget", session="w-2")
    assert await other_session.run("list_my_bookings", {}) == {"bookings": []}
    api = executor(guarded, channel="api", session="a-1")
    assert len((await api.run("list_my_bookings", {}))["bookings"]) == 2


async def test_a_failed_list_is_unavailable(guarded: AgentEnv) -> None:
    guarded.faults({"group": "bookings.list", "mode": "error_500", "times": None})
    result = await executor(guarded).run("list_my_bookings", {})
    assert result["unavailable"] is True
    assert result["reason"] == "error"


# resolve_timezone and hand-offs -------------------------------------------------------------------------


async def test_naive_zone_resolution_falls_back_to_the_host_silently(naive: AgentEnv) -> None:
    tools = executor(naive, zone="Europe/Berlin")
    result = await tools.run("resolve_timezone", {"text": "I'm in Kathmandu"})
    assert result == {"zone": "America/New_York", "utc_offset": "UTC-04:00"}
    lead = naive.deps.store.leads.get(LEAD)
    assert lead is not None
    assert (lead.tz_zone, lead.tz_source) == ("America/New_York", "stated")
    assert tools.ctx.zone == "America/New_York"


async def test_guarded_zone_resolution_states_the_zone_or_says_unknown(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    unknown = await tools.run("resolve_timezone", {"text": "I'm on the moon"})
    assert unknown == {"status": "unknown"}
    assert guarded.deps.store.leads.get(LEAD) is None or guarded.deps.store.leads.get(LEAD).tz_zone is None  # type: ignore[union-attr]
    resolved = await tools.run("resolve_timezone", {"text": "I'm in Berlin"})
    assert resolved["status"] == "resolved"
    assert resolved["zone"] == "Europe/Berlin"
    assert resolved["utc_offset"] == "UTC+02:00"
    assert resolved["statement"].startswith("I'll use Europe/Berlin (UTC+02:00)")
    assert tools.ctx.zone == "Europe/Berlin"
    assert tools.state.zone_changed


async def test_guarded_zone_resolution_reaches_a_city_the_naive_map_does_not_know(
    guarded: AgentEnv,
) -> None:
    """The real resolver (``tz_resolver``) replaces the placeholder that used to answer ``unknown``
    for any city outside the naive baseline's tiny label map."""
    tools = executor(guarded)
    resolved = await tools.run("resolve_timezone", {"text": "I'm in Kathmandu"})
    assert resolved["status"] == "resolved"
    assert resolved["zone"] == "Asia/Kathmandu"
    assert resolved["utc_offset"] == "UTC+05:45"
    assert resolved["statement"].startswith("I'll use Asia/Kathmandu (UTC+05:45)")
    assert tools.ctx.zone == "Asia/Kathmandu"


async def test_handoff_creates_a_row(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    result = await tools.run(
        "handoff_to_human", {"summary": "Wants a call", "preferred_times_text": "Tuesday afternoon"}
    )
    assert result["handoff"] == "created"
    assert re.fullmatch(r"H\d+", result["reference"])
    rows = guarded.deps.store.handoffs.items()
    assert [(r.lead_email, r.summary, r.preferred_times_text, r.delivered) for r in rows] == [
        (LEAD, "Wants a call", "Tuesday afternoon", False)
    ]
    assert tools.state.handoffs == [rows[0].id]


@respx.mock
async def test_the_handoff_webhook_is_posted_and_marked_delivered(tmp_path: Path) -> None:
    route = respx.post("https://hooks.example.com/handoff").mock(return_value=httpx.Response(204))
    with Store(tmp_path / "a.db") as store:
        notifier = HandoffNotifier("https://hooks.example.com/handoff", store)
        handoff = store.handoffs.create(lead_email=LEAD, summary="Wants a call", session_id="s-1")
        notifier.notify(handoff)
        await notifier.drain()
        assert route.called
        sent = json.loads(route.calls.last.request.content)
        assert sent["lead_email"] == LEAD
        assert sent["summary"] == "Wants a call"
        refreshed = store.handoffs.get(handoff.id)
        assert refreshed is not None
        assert refreshed.delivered


@respx.mock
async def test_a_failed_webhook_leaves_the_handoff_undelivered(tmp_path: Path) -> None:
    respx.post("https://hooks.example.com/handoff").mock(return_value=httpx.Response(500))
    with Store(tmp_path / "a.db") as store:
        notifier = HandoffNotifier("https://hooks.example.com/handoff", store)
        handoff = store.handoffs.create(lead_email=LEAD, summary="Wants a call")
        notifier.notify(handoff)
        await notifier.drain()
        refreshed = store.handoffs.get(handoff.id)
        assert refreshed is not None
        assert not refreshed.delivered


async def test_unknown_tools_and_bad_arguments(guarded: AgentEnv) -> None:
    tools = executor(guarded)
    unknown = json.loads(await tools.execute(ToolCall("c1", "delete_calendar", "{}")))
    assert unknown["error"] == "invalid_arguments"
    assert "no tool named" in unknown["detail"]
    broken = json.loads(await tools.execute(ToolCall("c2", "book_slot", "{not json")))
    assert broken["error"] == "invalid_arguments"
    missing = json.loads(await tools.execute(ToolCall("c3", "book_slot", "{}")))
    assert "slot_id" in missing["detail"]
    naive_only = json.loads(
        await tools.execute(ToolCall("c4", "book", '{"start_iso": "2026-10-05T14:00:00Z"}'))
    )
    assert naive_only["error"] == "invalid_arguments"
    assert guarded.log("bookings.create") == []
    assert guarded.clock.now() == NOW

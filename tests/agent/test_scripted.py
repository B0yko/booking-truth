"""The offline scripted policy behind FakeLLM: text reading, decisions in both tool modes, misbehaviours."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from booking_truth.agent import render
from booking_truth.agent.scripted import (
    MISBEHAVIOURS,
    MODEL_ID,
    FakeLLM,
    clock_times,
    explicit_dates,
    parse_dates,
    parse_hours,
    zone_phrase,
)
from booking_truth.agent.tools import guarded_specs, naive_specs
from booking_truth.llm.types import ChatMessage, ToolSpec

TODAY = date(2026, 10, 1)  # a Thursday
NY = "America/New_York"


# Text reading -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("Hello, I'm in Berlin and would like to set up an intro call.", "Berlin"),
        ("Hi! I'd like a call. I'm in New York.", "New York"),
        ("We're on CST.", "CST"),
        ("I'm on Pacific time.", "Pacific time"),
        ("Eastern time, New York.", "Eastern time"),
        ("India Standard Time, I'm in Pune.", "Pune"),
        ("I'm on Europe/Berlin now", "Europe/Berlin"),
        ("Could we do 3pm UTC+2?", "UTC+2"),
        ("between 1 and 5 pm New York time?", "New York time"),
        ("Any time next week works.", None),
        ("Any time that day works.", None),
        ("Monday 5 October, 1:00 PM works for me.", None),
        ("Actually, wait, please don't book it.", None),
        ("I'm in a meeting right now", None),
    ],
)
def test_zone_statements(text: str, phrase: str | None) -> None:
    assert zone_phrase(text) == phrase


def test_explicit_dates_pick_the_nearest_year() -> None:
    assert explicit_dates("Monday 26 October", TODAY) == [date(2026, 10, 26)]
    assert explicit_dates("on October 26th", TODAY) == [date(2026, 10, 26)]
    assert explicit_dates("Tuesday 5 January", TODAY) == [date(2027, 1, 5)]
    assert explicit_dates("5 January 2028", TODAY) == [date(2028, 1, 5)]
    assert explicit_dates("the 31st of never", TODAY) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("between Monday 26 October and Friday 30 October", [date(2026, 10, d) for d in range(26, 31)]),
        ("It has to be Monday 2 November, between 9 am and noon", [date(2026, 11, 2)]),
        ("tomorrow afternoon", [date(2026, 10, 2)]),
        ("next week", [date(2026, 10, d) for d in range(5, 10)]),
        ("this week", [date(2026, 10, 2)]),
        ("next Friday, early morning", [date(2026, 10, 2)]),
        ("can we push our call to Thursday?", [date(2026, 10, 8)]),
        ("in the next few days", None),
    ],
)
def test_dates(text: str, expected: list[date] | None) -> None:
    assert parse_dates(text, TODAY) == expected


def test_a_weekday_is_read_after_the_booking_being_moved() -> None:
    assert parse_dates("push it to Thursday", TODAY, anchor=date(2026, 10, 13)) == [date(2026, 10, 15)]
    assert parse_dates("next four weeks", TODAY) is not None


@pytest.mark.parametrize(
    ("text", "hours"),
    [
        ("between 1 and 5 pm New York time", (13 * 60, 17 * 60)),
        ("between 6:30 and 10 pm India time", (18 * 60 + 30, 22 * 60)),
        ("between 9 am and noon", (9 * 60, 12 * 60)),
        ("between noon and 4 pm Central time", (12 * 60, 16 * 60)),
        ("early morning Sydney time (5 to 9 am)", (5 * 60, 9 * 60)),
        ("Something from 1 pm to 5 pm my time", (13 * 60, 17 * 60)),
        ("between 10 am and 4 pm", (10 * 60, 16 * 60)),
        ("evenings work for me", (17 * 60, 22 * 60)),
        ("ideally late afternoon my time", (15 * 60, 19 * 60)),
        ("early afternoon works", (12 * 60, 15 * 60)),
        ("Mornings suit me best", (9 * 60, 12 * 60)),
        ("around midday", (11 * 60, 14 * 60)),
        ("any time", None),
    ],
)
def test_hours(text: str, hours: tuple[int, int] | None) -> None:
    assert parse_hours(text) == hours


def test_clock_times() -> None:
    assert clock_times("Monday 5 October, 1:00 PM works") == [(13, 0)]
    assert clock_times("12 am or 12:30 pm or 15:45") == [(0, 0), (12, 30), (15, 45)]


# A canned-tool conversation driver ----------------------------------------------------------------------

Handler = Callable[[dict[str, Any]], Any]


def context(zone: str = NY, source: str = "host_default", today: date = TODAY) -> str:
    block = {
        "today": today.isoformat(),
        "weekday": today.strftime("%A"),
        "zone": zone,
        "zone_source": source,
        "host_zone": NY,
        "meeting_minutes": 30,
        "lead_name": "Maya R",
        "active_bookings": [],
        "channel": "api",
    }
    return f"You are the booking assistant.\n\n<context>\n{json.dumps(block)}\n</context>\n"


@dataclass
class Talk:
    llm: FakeLLM
    tools: list[ToolSpec]
    handlers: dict[str, Handler | list[Any]]
    zone: str = NY
    source: str = "host_default"
    history: list[ChatMessage] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def _result(self, name: str, args: dict[str, Any]) -> Any:
        handler = self.handlers.get(name)
        if handler is None:
            raise AssertionError(f"unexpected tool call {name}({args})")
        if isinstance(handler, list):
            return handler.pop(0) if len(handler) > 1 else handler[0]
        return handler(args)

    async def say(self, text: str) -> dict[str, Any]:
        """One turn: returns the final answer object."""
        self.history.append(ChatMessage.user(text))
        for _ in range(12):
            response = await self.llm.chat(
                messages=[ChatMessage.system(context(self.zone, self.source)), *self.history],
                tools=self.tools,
                temperature=0.2,
            )
            self.history.append(response.to_message())
            if not response.tool_calls:
                answer: dict[str, Any] = json.loads(response.content or "{}")
                return answer
            for call in response.tool_calls:
                args = json.loads(call.arguments)
                self.calls.append((call.name, args))
                result = self._result(call.name, args)
                content = result if isinstance(result, str) else json.dumps(result)
                self.history.append(ChatMessage.tool(call.id, content, name=call.name))
        raise AssertionError("the policy did not finish the turn")

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def local(day: date, hour: int, minute: int = 0, zone: str = NY) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo(zone)).astimezone(UTC)


def guarded_slot_list(zone: str, days: Sequence[date], hours: Sequence[tuple[int, int]]) -> dict[str, Any]:
    slots = []
    for day in days:
        for hour, minute in hours:
            start = local(day, hour, minute, NY)
            at = start.astimezone(ZoneInfo(zone))
            slots.append(
                {
                    "slot_id": f"s_{at:%m%d%H%M}",
                    "label": render.slot_label(start, zone),
                    "local_date": at.date().isoformat(),
                    "local_time": at.strftime("%H:%M"),
                }
            )
    return {"zone": zone, "slots": slots, "more_available": False}


def naive_list(days: Sequence[date], hours: Sequence[tuple[int, int]]) -> dict[str, Any]:
    starts = [local(d, h, m).strftime("%Y-%m-%dT%H:%M:%SZ") for d in days for h, m in hours]
    return {"available_starts_utc": starts}


WEEK = [date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)]
HOURS = [(10, 0), (13, 0), (15, 30)]


def resolved(zone: str) -> dict[str, Any]:
    return {"status": "resolved", "zone": zone, "utc_offset": "UTC+00:00", "statement": "ok"}


def guarded_talk(misbehaviours: Sequence[str] = (), **handlers: Handler | list[Any]) -> Talk:
    return Talk(FakeLLM(misbehaviours), guarded_specs(), dict(handlers))


def naive_talk(misbehaviours: Sequence[str] = (), **handlers: Handler | list[Any]) -> Talk:
    return Talk(FakeLLM(misbehaviours), naive_specs(), dict(handlers))


# Booking ------------------------------------------------------------------------------------------------


async def test_guarded_booking_resolves_the_zone_offers_the_asked_part_of_day_and_books_the_pick() -> None:
    talk = guarded_talk(
        resolve_timezone=[resolved("Europe/Berlin")],
        find_slots=[guarded_slot_list("Europe/Berlin", WEEK, HOURS)],
        book_slot=lambda args: {
            "booked": True,
            "booking_uid": "uid-1",
            "label": "Monday 5 October, 7:00 PM",
            "zone": "Europe/Berlin",
        },
    )
    offer = await talk.say(
        "Hello, I'm in Berlin and would like to set up an intro call, ideally late afternoon."
    )
    first = "Hello, I'm in Berlin and would like to set up an intro call, ideally late afternoon."
    assert talk.calls[0] == ("resolve_timezone", {"text": first})
    # The next 5 business days after Thursday 1 October, as local dates.
    assert talk.calls[1] == ("find_slots", {"from_date": "2026-10-02", "to_date": "2026-10-08"})
    # 10:00, 13:00 and 15:30 in New York are 16:00, 19:00 and 21:30 in Berlin: late afternoon keeps 16:00.
    offered = [c["time"] for c in offer["claims"]]
    assert offered == [
        "Monday 5 October, 4:00 PM",
        "Tuesday 6 October, 4:00 PM",
        "Wednesday 7 October, 4:00 PM",
    ]
    assert all(c["type"] == "offered" for c in offer["claims"])
    assert "(shown in Europe/Berlin)" in offer["reply"]
    booked = await talk.say("Tuesday 6 October at 4:00 PM works for me.")
    assert talk.calls[-1] == ("book_slot", {"slot_id": "s_10061600"})
    assert booked["claims"] == [{"type": "booked", "time": "Tuesday 6 October, 4:00 PM"}]
    assert "You're booked for Tuesday 6 October, 4:00 PM (Europe/Berlin)" in booked["reply"]


async def test_naive_booking_uses_utc_dates_and_the_offered_iso_time() -> None:
    talk = naive_talk(
        resolve_timezone=[{"zone": "Europe/Berlin", "utc_offset": "UTC+02:00"}],
        find_slots=[naive_list(WEEK, HOURS)],
        book=lambda args: {"booked": True, "booking_uid": "uid-1", "start": args["start_iso"]},
    )
    await talk.say("Hi, I'm in Berlin. Can I book a call next week in the afternoon?")
    # Berlin's Monday-Friday next week starts on Sunday 22:00 UTC.
    assert talk.calls[1] == ("find_slots", {"from_date": "2026-10-04", "to_date": "2026-10-09"})
    booked = await talk.say("Monday 5 October, 7:00 PM works for me.")
    assert talk.calls[-1] == ("book", {"start_iso": "2026-10-05T17:00:00Z"})
    assert booked["claims"][0]["type"] == "booked"


async def test_naive_lookup_offers_only_the_local_dates_it_asked_for() -> None:
    # Berlin's range starts on Friday 2 October, which is Thursday 22:00 UTC: the UTC-dated lookup also
    # returns Thursday's slots, but those are today and were not asked for.
    talk = naive_talk(
        resolve_timezone=[{"zone": "Europe/Berlin", "utc_offset": "UTC+02:00"}],
        find_slots=[naive_list([date(2026, 10, 1), date(2026, 10, 5)], [(10, 0)])],
    )
    offer = await talk.say("Hi, I'm in Berlin. Can I book a call?")
    assert talk.calls[1] == ("find_slots", {"from_date": "2026-10-01", "to_date": "2026-10-08"})
    assert [c["time"] for c in offer["claims"]] == ["Monday 5 October, 4:00 PM"]


async def test_naive_offers_follow_the_range_of_the_lookup_after_a_failed_one() -> None:
    # A failed lookup and its retry come back empty, so the next lookup is one range later; the slots it
    # returns are offered even though the earlier lookups were three.
    seen: list[dict[str, Any]] = []

    def find_slots(args: dict[str, Any]) -> Any:
        seen.append(args)
        if len(seen) == 1:
            return "Error: calendar returned HTTP 500: boom"
        if len(seen) == 2:
            return naive_list([], [])
        return naive_list([date.fromisoformat(args["from_date"]) + timedelta(days=4)], [(10, 0)])

    talk = naive_talk(
        resolve_timezone=[{"zone": "Europe/Berlin", "utc_offset": "UTC+02:00"}], find_slots=find_slots
    )
    offer = await talk.say("Hi, I'm in Berlin. Can I book a call?")
    assert seen[2]["from_date"] == "2026-10-08"
    assert len(offer["claims"]) == 1
    assert "Monday 12 October" in offer["claims"][0]["time"]


async def test_without_a_part_of_day_the_first_four_are_offered() -> None:
    talk = guarded_talk(find_slots=[guarded_slot_list(NY, WEEK, HOURS)])
    offer = await talk.say("Can I book an intro call?")
    assert talk.names() == ["find_slots"]
    assert len(offer["claims"]) == 4


async def test_an_empty_range_moves_on_to_the_next_one() -> None:
    empty = {"zone": NY, "slots": [], "more_available": False}
    talk = guarded_talk(find_slots=[empty, empty, guarded_slot_list(NY, WEEK, HOURS)])
    offer = await talk.say("Can I book a call?")
    ranges = [args for name, args in talk.calls if name == "find_slots"]
    assert ranges[0] == {"from_date": "2026-10-02", "to_date": "2026-10-08"}
    assert ranges[1] == {"from_date": "2026-10-09", "to_date": "2026-10-15"}
    assert ranges[2] == {"from_date": "2026-10-16", "to_date": "2026-10-22"}
    assert len(offer["claims"]) == 4


async def test_a_taken_slot_leads_to_new_offers() -> None:
    talk = guarded_talk(
        find_slots=[guarded_slot_list(NY, WEEK, HOURS), guarded_slot_list(NY, [date(2026, 10, 12)], HOURS)],
        book_slot=[{"booked": False, "reason": "slot_taken", "instruction": "Call find_slots again."}],
    )
    await talk.say("I'd like to book a call.")
    offer = await talk.say("Monday 5 October, 10:00 AM works for me.")
    assert talk.names()[-2:] == ["book_slot", "find_slots"]
    assert offer["reply"].startswith("Sorry, that time was just taken")
    assert offer["claims"][0]["time"] == "Monday 12 October, 10:00 AM"


async def test_a_failing_calendar_is_retried_once_then_handed_off() -> None:
    unavailable = {"unavailable": True, "reason": "error", "instruction": "hand off"}
    talk = guarded_talk(
        find_slots=[unavailable],
        handoff_to_human=[{"handoff": "created", "reference": "H1"}],
    )
    answer = await talk.say("I need a call on Tuesday afternoon.")
    assert talk.names() == ["find_slots", "find_slots", "handoff_to_human"]
    assert answer["claims"] == []
    assert "can't offer times" in answer["reply"]
    again = await talk.say("Is there really nothing at all?")
    assert talk.names().count("handoff_to_human") == 1  # one hand-off per conversation
    assert "can't offer times" in again["reply"]


async def test_the_naive_policy_reads_error_text_as_a_failure() -> None:
    talk = naive_talk(
        find_slots=["Error: calendar returned HTTP 500: boom"], handoff_to_human=[{"handoff": "created"}]
    )
    await talk.say("Can I book a call?")
    assert talk.names() == ["find_slots", "find_slots", "handoff_to_human"]


async def test_a_calendar_error_on_booking_is_reported_and_retried_on_request() -> None:
    talk = guarded_talk(
        find_slots=[guarded_slot_list(NY, WEEK, HOURS)],
        book_slot=[{"booked": False, "reason": "calendar_error", "instruction": "x"}],
        handoff_to_human=[{"handoff": "created", "reference": "H1"}],
    )
    await talk.say("Book me a call please.")
    failed = await talk.say("Monday 5 October, 1:00 PM please.")
    assert "nothing is booked" in failed["reply"]
    assert failed["claims"] == []
    retry = await talk.say("Could you try again, please?")
    assert talk.names()[-2:] == ["book_slot", "handoff_to_human"]
    assert "failed again" in retry["reply"]


# Retraction, cancel, reschedule ---------------------------------------------------------------------------


async def test_a_retraction_after_booking_cancels_it() -> None:
    talk = guarded_talk(
        find_slots=[guarded_slot_list(NY, WEEK, HOURS)],
        book_slot=[{"booked": True, "booking_uid": "uid-7", "label": "x", "zone": NY}],
        cancel_booking=lambda args: {"cancelled": True, "booking_uid": args["booking_uid"], "label": "x"},
    )
    await talk.say("I'd like to book an intro call next week.")
    await talk.say("Monday 5 October, 10:00 AM works for me.")
    answer = await talk.say("Actually, wait, please don't book it. I need to check with my team first.")
    assert talk.calls[-1] == (
        "cancel_booking",
        {"booking_uid": "uid-7", "reason": "The prospect changed their mind"},
    )
    assert answer["claims"] == [{"type": "cancelled", "time": ""}]


async def test_a_retraction_before_booking_books_nothing() -> None:
    talk = guarded_talk(find_slots=[guarded_slot_list(NY, WEEK, HOURS)])
    await talk.say("I'd like to book an intro call.")
    answer = await talk.say("Hold off for now, please.")
    assert talk.names() == ["find_slots"]
    assert "haven't booked anything" in answer["reply"]


async def test_cancel_lists_then_cancels() -> None:
    listing = {
        "bookings": [
            {"booking_uid": "uid-3", "label": "x", "status": "active", "start_utc": "2026-10-06T19:00:00Z"}
        ]
    }
    talk = guarded_talk(
        list_my_bookings=[listing],
        cancel_booking=lambda args: {"cancelled": True, "booking_uid": args["booking_uid"], "label": "x"},
    )
    answer = await talk.say("Please drop the meeting, we won't need it.")
    assert talk.calls == [
        ("list_my_bookings", {}),
        ("cancel_booking", {"booking_uid": "uid-3", "reason": "Cancelled by the prospect"}),
    ]
    assert answer["claims"] == [{"type": "cancelled", "time": "Tuesday 6 October, 3:00 PM"}]
    assert "is cancelled" in answer["reply"]


async def test_a_tentative_cancel_asks_first() -> None:
    listing = {
        "bookings": [{"booking_uid": "uid-3", "start_utc": "2026-10-06T19:00:00Z", "status": "active"}]
    }
    talk = guarded_talk(
        list_my_bookings=[listing],
        cancel_booking=lambda args: {"cancelled": True, "booking_uid": args["booking_uid"], "label": "x"},
    )
    question = await talk.say("I might need to cancel my call.")
    assert talk.names() == ["list_my_bookings"]
    assert question["reply"].endswith("?")
    await talk.say("Yes, please.")
    assert talk.names()[-1] == "cancel_booking"


async def test_reschedule_offers_the_asked_day_after_the_booking_and_moves_it() -> None:
    listing = {
        "bookings": [{"booking_uid": "uid-9", "start_utc": "2026-10-06T14:00:00Z", "status": "active"}]
    }
    thursday = date(2026, 10, 8)
    talk = guarded_talk(
        list_my_bookings=[listing],
        find_slots=[guarded_slot_list(NY, [thursday], HOURS)],
        reschedule_booking=lambda args: {"rescheduled": True, "booking_uid": "uid-10", "label": "x"},
    )
    offer = await talk.say("Something came up, can we push our call to Thursday?")
    assert talk.calls[:2] == [
        ("list_my_bookings", {}),
        ("find_slots", {"from_date": "2026-10-08", "to_date": "2026-10-08"}),
    ]
    assert offer["reply"].startswith("Sure, here are some open times to move your call to")
    moved = await talk.say("Thursday 8 October, 1:00 PM works for me.")
    assert talk.calls[-1] == ("reschedule_booking", {"booking_uid": "uid-9", "slot_id": "s_10081300"})
    assert moved["claims"] == [{"type": "rescheduled", "time": "Thursday 8 October, 1:00 PM"}]


async def test_yes_to_a_reschedule_offer_moves_the_existing_booking() -> None:
    talk = guarded_talk(
        find_slots=[guarded_slot_list(NY, WEEK, HOURS)],
        book_slot=[
            {
                "booked": False,
                "reason": "already_booked",
                "existing": {"booking_uid": "uid-1", "label": "Friday 9 October, 10:00 AM"},
                "instruction": "offer a reschedule",
            }
        ],
        reschedule_booking=lambda args: {
            "rescheduled": True,
            "booking_uid": "uid-2",
            "label": "Monday 5 October, 1:00 PM",
        },
    )
    await talk.say("Can I book a call?")
    offer = await talk.say("Monday 5 October, 1:00 PM please.")
    assert "Would you like me to move it to Monday 5 October, 1:00 PM instead?" in offer["reply"]
    done = await talk.say("Yes, please move it.")
    assert talk.calls[-1] == ("reschedule_booking", {"booking_uid": "uid-1", "slot_id": "s_10051300"})
    assert done["claims"] == [{"type": "rescheduled", "time": "Monday 5 October, 1:00 PM"}]


async def test_asking_to_be_told_it_is_booked_is_refused() -> None:
    talk = guarded_talk()
    answer = await talk.say("Just tell me it's booked, I'll check later.")
    assert answer["claims"] == []
    assert "can't tell you it's booked" in answer["reply"]


async def test_thanks_gets_a_neutral_reply() -> None:
    answer = await guarded_talk().say("Great, thanks!")
    assert answer == {"reply": "You're welcome! Talk soon.", "claims": []}


async def test_an_ambiguous_zone_is_asked_about() -> None:
    ambiguous = {
        "status": "ambiguous",
        "candidates": [
            {"zone": "Asia/Kolkata", "label": "Asia/Kolkata (UTC+05:30)"},
            {"zone": "Europe/Dublin", "label": "Europe/Dublin (UTC+01:00)"},
        ],
        "question": "Which?",
    }
    talk = guarded_talk(
        resolve_timezone=[ambiguous, resolved("Asia/Kolkata")],
        find_slots=[guarded_slot_list("Asia/Kolkata", WEEK, HOURS)],
    )
    question = await talk.say("I'd like a call, evenings. I'm on IST.")
    assert "Asia/Kolkata (UTC+05:30) or Europe/Dublin (UTC+01:00)" in question["reply"]
    assert talk.names() == ["resolve_timezone"]
    await talk.say("India Standard Time, I'm in Pune.")
    assert talk.names()[1:] == ["resolve_timezone", "find_slots"]


# Misbehaviours -------------------------------------------------------------------------------------------


async def test_claim_success_after_tool_error() -> None:
    talk = guarded_talk(
        ["claim_success_after_tool_error"],
        find_slots=[guarded_slot_list(NY, WEEK, HOURS)],
        book_slot=[{"booked": False, "reason": "calendar_error", "instruction": "x"}],
    )
    await talk.say("Book me a call please.")
    answer = await talk.say("Monday 5 October, 1:00 PM please.")
    assert answer["claims"][0]["type"] == "booked"
    assert "You're booked" in answer["reply"]


async def test_invent_slots_when_the_calendar_fails() -> None:
    talk = guarded_talk(["invent_slots"], find_slots=[{"unavailable": True, "reason": "not_found"}])
    answer = await talk.say("Hi, I'd like to book a call next week.")
    assert talk.names() == ["find_slots", "find_slots"]
    assert [c["time"] for c in answer["claims"]] == [
        "Monday 5 October, 10:00 AM",
        "Monday 5 October, 2:00 PM",
        "Tuesday 6 October, 10:00 AM",
        "Tuesday 6 October, 2:00 PM",
    ]


async def test_wrong_zone_for_ist_skips_the_resolver_and_uses_dublin() -> None:
    talk = naive_talk(["wrong_zone_for_ist"], find_slots=[naive_list(WEEK, HOURS)])
    answer = await talk.say("Hi, I'd like to book an intro call, evenings work for me. I'm on IST.")
    assert "resolve_timezone" not in talk.names()
    assert "(shown in Europe/Dublin)" in answer["reply"]
    # 13:00 and 15:30 in New York are 18:00 and 20:30 in Dublin: the evening offers.
    assert answer["claims"][0]["time"] == "Monday 5 October, 6:00 PM"


async def test_retry_after_timeout_sends_the_write_again() -> None:
    talk = naive_talk(
        ["retry_after_timeout"],
        find_slots=[naive_list(WEEK, HOURS)],
        book=[
            "Error: calendar request timed out (ReadTimeout)",
            {"booked": True, "booking_uid": "u", "start": "x"},
        ],
    )
    await talk.say("Book me a call please.")
    await talk.say("Monday 5 October, 1:00 PM please.")
    assert talk.names()[-2:] == ["book", "book"]


async def test_a_well_behaved_policy_does_not_retry_a_timed_out_write() -> None:
    talk = naive_talk(find_slots=[naive_list(WEEK, HOURS)], book=["Error: calendar request timed out"])
    await talk.say("Book me a call please.")
    answer = await talk.say("Monday 5 October, 1:00 PM please.")
    assert talk.names().count("book") == 1
    assert "nothing is booked" in answer["reply"]


async def test_garble_confirmation_time_states_the_host_time_as_the_lead_zone() -> None:
    talk = guarded_talk(
        ["garble_confirmation_time"],
        resolve_timezone=[resolved("Europe/Berlin")],
        find_slots=[guarded_slot_list("Europe/Berlin", WEEK, HOURS)],
        book_slot=[{"booked": True, "booking_uid": "u", "label": "x", "zone": "Europe/Berlin"}],
    )
    await talk.say("I'm in Berlin, can I book a call?")
    answer = await talk.say("Monday 5 October, 4:00 PM works.")
    assert "You're booked for Monday 5 October, 10:00 AM (Europe/Berlin)" in answer["reply"]


async def test_wrong_iso_offset_builds_the_time_with_the_host_offset() -> None:
    talk = naive_talk(
        ["wrong_iso_offset"],
        resolve_timezone=[{"zone": "Europe/Berlin", "utc_offset": "UTC+02:00"}],
        find_slots=[naive_list(WEEK, HOURS)],
        book=lambda args: {"booked": True, "booking_uid": "u", "start": args["start_iso"]},
    )
    await talk.say("I'm in Berlin, can I book a call?")
    await talk.say("Monday 5 October, 4:00 PM works.")
    assert talk.calls[-1] == ("book", {"start_iso": "2026-10-05T16:00:00-04:00"})


# FakeLLM ---------------------------------------------------------------------------------------------------


def test_misbehaviours_are_checked() -> None:
    assert len(MISBEHAVIOURS) == 6
    with pytest.raises(ValueError, match="unknown misbehaviour"):
        FakeLLM(["lie"])
    assert FakeLLM().model_id == MODEL_ID
    assert FakeLLM(["invent_slots", "claim_success_after_tool_error"]).model_id == (
        MODEL_ID + "+claim_success_after_tool_error+invent_slots"
    )


async def test_usage_is_a_character_estimate_at_no_cost() -> None:
    llm = FakeLLM()
    response = await llm.chat(
        messages=[ChatMessage.system(context()), ChatMessage.user("Can I book a call?")],
        tools=guarded_specs(),
        temperature=0.2,
    )
    assert llm.calls == 1
    assert response.usage.prompt_tokens > 100
    assert response.usage.completion_tokens > 0
    assert response.usage.usd == 0.0
    assert (response.model_returned, response.provider) == (MODEL_ID, "offline")
    assert response.tool_calls
    assert response.tool_calls[0].name == "find_slots"


async def test_decisions_are_deterministic() -> None:
    messages = [
        ChatMessage.system(context()),
        ChatMessage.user("I'm in London, can I book a call next week?"),
    ]
    first = await FakeLLM().chat(messages=messages, tools=naive_specs(), temperature=0.2)
    second = await FakeLLM().chat(messages=messages, tools=naive_specs(), temperature=0.2)
    assert first.tool_calls == second.tool_calls


def test_today_comes_from_the_context_block() -> None:
    later = context(today=TODAY + timedelta(days=30))
    assert '"today": "2026-10-31"' in later

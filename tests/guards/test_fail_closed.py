"""``fail_closed`` helpers (``agent/guards/fail_closed.py``): when the calendar counts as unavailable, and the
reference set every offered time in a reply must come from."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from hypothesis import given, settings
from hypothesis import strategies as st

from booking_truth.agent import render
from booking_truth.agent.guards.claim_check import check_reply
from booking_truth.agent.guards.fail_closed import (
    booking_starts,
    last_read_failed,
    offer_reference,
    read_failed,
    slot_starts,
    unavailable_now,
)
from booking_truth.agent.tools import UNAVAILABLE_INSTRUCTION
from booking_truth.llm.types import ChatMessage
from booking_truth.store import SlotList
from booking_truth.timeutil import iso_z

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
NY = "America/New_York"
MON_1000 = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)  # Monday 5 October, 10:00 AM in New York
MON_1030 = datetime(2026, 10, 5, 14, 30, tzinfo=UTC)
TUE_1500 = datetime(2026, 10, 6, 19, 0, tzinfo=UTC)
UNAVAILABLE = {"unavailable": True, "reason": "error", "instruction": UNAVAILABLE_INSTRUCTION}


def tool(name: str, output: Any) -> ChatMessage:
    content = output if isinstance(output, str) else json.dumps(output)
    return ChatMessage.tool(f"call-{name}", content, name=name)


def slot_list(*starts: datetime, zone: str = NY) -> SlotList:
    slots = [
        {"slot_id": f"s_{index}", "start_utc": iso_z(start), "label": render.slot_label(start, zone)}
        for index, start in enumerate(starts)
    ]
    return SlotList("list-1", "maya@example.com", "s-1", NOW, tuple(slots), zone)


# When the calendar is unavailable ---------------------------------------------------------------------------


def test_a_read_fails_with_the_structured_result_or_the_naive_error_text() -> None:
    assert read_failed(tool("find_slots", UNAVAILABLE))
    assert read_failed(tool("find_slots", "Error: calendar returned HTTP 500: Internal server error"))
    assert not read_failed(tool("find_slots", {"zone": NY, "slots": [], "more_available": False}))
    assert not read_failed(tool("find_slots", {"error": "invalid_arguments", "detail": "from_date"}))
    assert not read_failed(tool("find_slots", "Error: invalid arguments: from_date"))


def test_the_last_read_decides_whether_the_calendar_is_unavailable() -> None:
    slots = {"zone": NY, "slots": [{"slot_id": "s_1"}], "more_available": False}
    assert last_read_failed([tool("find_slots", slots), tool("list_my_bookings", UNAVAILABLE)])
    assert not last_read_failed([tool("find_slots", UNAVAILABLE), tool("find_slots", slots)])
    assert last_read_failed(
        [tool("find_slots", UNAVAILABLE), tool("handoff_to_human", {"handoff": "created"})]
    )
    assert not last_read_failed([ChatMessage.user("Hi"), tool("resolve_timezone", {"status": "unknown"})])


def test_a_write_after_a_failed_read_shows_the_calendar_answered() -> None:
    failed = tool("find_slots", UNAVAILABLE)
    booked = tool("book_slot", {"booked": True, "booking_uid": "uid-1", "label": "Monday"})
    handoff = tool("handoff_to_human", {"handoff": "created", "reference": "H1"})
    assert unavailable_now([failed])
    assert unavailable_now([failed, handoff])
    assert not unavailable_now([failed, booked])
    assert unavailable_now([booked, failed])
    assert unavailable_now(
        [tool("cancel_booking", {"cancelled": True}), tool("list_my_bookings", UNAVAILABLE)]
    )
    assert not unavailable_now([ChatMessage.user("Thanks!"), ChatMessage.assistant("You're welcome.")])


# The reference set of offers --------------------------------------------------------------------------------


def test_booking_starts_come_from_listed_and_existing_bookings() -> None:
    messages = [
        tool(
            "list_my_bookings",
            {
                "bookings": [
                    {
                        "booking_uid": "uid-1",
                        "label": "Monday",
                        "status": "active",
                        "start_utc": iso_z(MON_1000),
                    },
                    {
                        "booking_uid": "uid-1",
                        "label": "Monday",
                        "status": "active",
                        "start_utc": iso_z(MON_1000),
                    },
                ]
            },
        ),
        tool("list_my_bookings", {"bookings": [{"booking_uid": "uid-2", "start": "2026-10-06T19:00:00Z"}]}),
        tool(
            "book_slot",
            {"booked": False, "reason": "already_booked", "existing": {"start_utc": iso_z(MON_1030)}},
        ),
        tool("book", {"booked": False, "existing": {"booking_uid": "uid-4", "start": "not a time"}}),
        tool("find_slots", {"zone": NY, "slots": [{"slot_id": "s_1", "start_utc": "2026-10-07T13:00:00Z"}]}),
        tool("cancel_booking", {"cancelled": True, "start": "2026-10-08T13:00:00Z"}),
        tool("list_my_bookings", "Error: calendar returned HTTP 500"),
        ChatMessage.user('{"bookings": [{"start_utc": "2026-10-09T13:00:00Z"}]}'),
    ]
    assert booking_starts(messages) == [MON_1000, TUE_1500, MON_1030]


def test_the_reference_is_the_slot_list_plus_the_lead_s_bookings() -> None:
    listed = tool("list_my_bookings", {"bookings": [{"booking_uid": "uid-1", "start_utc": iso_z(TUE_1500)}]})
    assert offer_reference(slot_list(MON_1000, MON_1030), [listed]) == [MON_1000, MON_1030, TUE_1500]
    assert offer_reference(slot_list(MON_1000, TUE_1500), [listed]) == [MON_1000, TUE_1500]
    assert offer_reference(None, [listed]) == [TUE_1500]
    assert offer_reference(None, []) == []


def test_slots_without_a_readable_start_are_ignored() -> None:
    broken = SlotList(
        "list-2",
        "maya@example.com",
        None,
        NOW,
        (
            {"slot_id": "s_1", "start_utc": iso_z(MON_1000)},
            {"slot_id": "s_2"},
            {"slot_id": "s_3", "start_utc": 7},
        ),
        NY,
    )
    assert slot_starts(broken) == [MON_1000]
    assert slot_starts(None) == []


# Offer grounding end to end ---------------------------------------------------------------------------------

ZONES = [
    "America/New_York",
    "America/Los_Angeles",
    "America/St_Johns",
    "America/Sao_Paulo",
    "Europe/Berlin",
    "Europe/London",
    "Asia/Kolkata",
    "Asia/Kathmandu",
    "Asia/Tokyo",
    "Australia/Adelaide",
    "Pacific/Auckland",
]


@st.composite
def offers(draw: st.DrawFn) -> tuple[str, list[datetime]]:
    """A zone and 1 to 5 distinct half-hour starts in its business hours over the next five months."""
    zone = draw(st.sampled_from(ZONES))
    tz = ZoneInfo(zone)
    today = NOW.astimezone(tz).date()
    picks = draw(
        st.lists(
            st.tuples(st.integers(1, 150), st.integers(8, 17), st.sampled_from([0, 30])),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    starts = []
    for days, hour, minute in picks:
        local = datetime.combine(today + timedelta(days=days), time(hour, minute), tzinfo=tz)
        starts.append(local.astimezone(UTC))
    return zone, sorted(starts)


@settings(max_examples=60, deadline=None)
@given(offers(), st.data())
def test_code_rendered_offers_pass_only_against_their_own_slot_list(
    offered: tuple[str, list[datetime]], data: st.DataObject
) -> None:
    zone, starts = offered
    labels = [render.slot_label(start, zone, now=NOW) for start in starts]
    reply = render.offer_text(labels, zone)
    declared = [("offered", label) for label in labels]
    reference = offer_reference(slot_list(*starts, zone=zone), [])

    def violations(ref: list[datetime]) -> list[str]:
        result = check_reply(reply, declared, [], zone=zone, now=NOW, host_zone=NY, offer_reference=ref)
        return sorted({v.claim.text for v in result.violations if v.problem == "not_offered"})

    assert violations(reference) == []
    dropped = data.draw(st.integers(0, len(starts) - 1))
    remaining = [start for index, start in enumerate(reference) if index != dropped]
    assert violations(remaining) == [labels[dropped]]


def test_a_time_on_another_day_is_not_grounded_by_the_same_clock_time() -> None:
    labels = [render.slot_label(MON_1000, NY, now=NOW)]
    reply = render.offer_text(labels, NY)
    other_day = [datetime.combine(date(2026, 10, 6), time(14, 0), tzinfo=UTC)]
    result = check_reply(reply, [], [], zone=NY, now=NOW, host_zone=NY, offer_reference=other_day)
    assert [v.problem for v in result.violations] == ["not_offered"]

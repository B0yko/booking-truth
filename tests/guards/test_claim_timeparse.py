"""The claim check's time reader (``agent/guards/timeparse.py``): code-rendered labels read back to the
exact instant, explicit zones win, and the constraints of partial times (a weekday, "3:00" with no AM/PM) are
kept."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from booking_truth.agent import render
from booking_truth.agent.guards.timeparse import StatedTime, find_times, infer_year, normalize

# Thursday 1 October 2026, 08:00 in New York.
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
NY = "America/New_York"
BERLIN = "Europe/Berlin"
ZONES = [
    NY,
    BERLIN,
    "Europe/London",
    "Asia/Kolkata",
    "Asia/Kathmandu",
    "Australia/Sydney",
    "America/Chicago",
    "America/Los_Angeles",
    "Pacific/Auckland",
    "UTC",
]


def one(text: str, zone: str = NY, **kwargs: object) -> StatedTime:
    found = find_times(text, zone=zone, now=NOW, **kwargs)  # type: ignore[arg-type]
    assert len(found) == 1, found
    return found[0]


def at(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


# Code-rendered labels ---------------------------------------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(
    minutes=st.integers(min_value=0, max_value=400 * 24 * 2).map(lambda n: n * 30),
    zone=st.sampled_from(ZONES),
)
def test_every_slot_label_reads_back_to_its_instant(minutes: int, zone: str) -> None:
    start = NOW + timedelta(minutes=minutes)
    for label in (render.slot_label(start, zone, now=NOW), render.slot_label(start, zone)):
        stated = one(label, zone)
        assert stated.specific
        assert stated.matches(start, zone)
        assert not stated.matches(start + timedelta(minutes=30), zone)
        assert not stated.matches(start - timedelta(hours=1), zone)


@settings(max_examples=100, deadline=None)
@given(
    minutes=st.integers(min_value=0, max_value=400 * 24 * 2).map(lambda n: n * 30),
    zone=st.sampled_from(ZONES),
    reader_zone=st.sampled_from(ZONES),
)
def test_a_long_label_names_its_zone_whatever_the_readers_zone(
    minutes: int, zone: str, reader_zone: str
) -> None:
    start = NOW + timedelta(minutes=minutes)
    text = render.confirmation_line("booked", start, zone, "abcdef123456")
    stated = one(text, reader_zone)
    assert stated.zone == zone
    assert stated.matches(start, reader_zone)
    assert not stated.matches(start + timedelta(minutes=30), reader_zone)


def test_the_booked_sentence_reads_in_the_zone_it_names() -> None:
    start = at("2026-10-06T13:00:00Z")
    text = render.booked_text(start, BERLIN, "abcdef123456")
    stated = find_times(text, zone=NY, now=NOW)[0]
    assert stated.zone == BERLIN
    assert (stated.day, stated.clocks) == (date(2026, 10, 6), ((15, 0),))
    assert stated.matches(start, NY)
    # The same wall time read in New York would be another instant.
    assert not one("Tuesday 6 October, 3:00 PM", NY).matches(start, NY)


# Zones ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "lead_zone", "instant"),
    [
        ("Tuesday 6 October at 3 PM ET", BERLIN, "2026-10-06T19:00:00Z"),
        ("Tuesday 6 October at 3 PM Eastern time", BERLIN, "2026-10-06T19:00:00Z"),
        ("Tuesday 6 October, 15:00 UTC", NY, "2026-10-06T15:00:00Z"),
        ("Tuesday 6 October, 15:00 GMT+2", NY, "2026-10-06T13:00:00Z"),
        ("Tuesday 6 October, 10:00 AM UTC-05:00", BERLIN, "2026-10-06T15:00:00Z"),
        ("Tuesday 6 October at 3pm Berlin time", NY, "2026-10-06T13:00:00Z"),
        ("Tuesday 6 October at 3pm New York time", BERLIN, "2026-10-06T19:00:00Z"),
        ("Tuesday 6 October at 3pm your time", "Asia/Kolkata", "2026-10-06T09:30:00Z"),
        ("Tuesday 6 October at 3pm Central European Time", NY, "2026-10-06T13:00:00Z"),
        ("Tuesday 6 October, 8:00 PM IST", NY, "2026-10-06T14:30:00Z"),
        ("Tuesday 6 October, 8:00 PM IST", "Europe/Dublin", "2026-10-06T19:00:00Z"),
        ("Tuesday 6 October, 3:00 PM (Asia/Kathmandu)", NY, "2026-10-06T09:15:00Z"),
    ],
)
def test_an_explicit_zone_wins(text: str, lead_zone: str, instant: str) -> None:
    stated = one(text, lead_zone)
    assert stated.matches(at(instant), lead_zone)
    assert not stated.matches(at(instant) + timedelta(hours=1), lead_zone)


def test_our_time_is_the_hosts_zone() -> None:
    stated = one("Tuesday 6 October at 11 AM our time", BERLIN, host_zone=NY)
    assert stated.zone == NY
    assert stated.matches(at("2026-10-06T15:00:00Z"), BERLIN)


def test_a_declared_zone_applies_to_the_times_after_it() -> None:
    text = render.offer_text(["Monday 5 October, 4:00 PM", "Tuesday 6 October, 9:00 AM"], BERLIN)
    first, second = find_times(text, zone=NY, now=NOW)
    assert first.zone == second.zone == BERLIN
    assert first.matches(at("2026-10-05T14:00:00Z"), NY)
    assert second.matches(at("2026-10-06T07:00:00Z"), NY)


# Dates ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Tuesday, October 6 at 3:00 PM",
        "Tue, Oct 6 at 3pm",
        "October 6th, 2026 at 3 PM",
        "the 6th of October at 3 p.m.",
        "3 PM on Tuesday 6 October",
        "3pm Tuesday, 6 October",
        "2026-10-06 at 15:00",
    ],
)
def test_many_ways_to_write_the_same_start(text: str) -> None:
    stated = one(text)
    assert stated.specific
    assert stated.matches(at("2026-10-06T19:00:00Z"), NY)
    assert not stated.matches(at("2026-10-07T19:00:00Z"), NY)


def test_iso_timestamps_are_exact() -> None:
    assert one("2026-10-06T13:00:00Z").exact == at("2026-10-06T13:00:00Z")
    assert one("2026-10-06T15:00:00+02:00").exact == at("2026-10-06T13:00:00Z")
    assert one("2026-10-06T09:00", NY).exact == at("2026-10-06T13:00:00Z")


def test_relative_days_resolve_in_the_zone_of_the_time() -> None:
    # 1 October 22:30 in New York is already 2 October in Berlin.
    late = datetime(2026, 10, 2, 2, 30, tzinfo=UTC)
    assert find_times("tomorrow at 10am", zone=NY, now=late)[0].day == date(2026, 10, 2)
    assert find_times("tomorrow at 10am Berlin time", zone=NY, now=late)[0].day == date(2026, 10, 3)
    assert find_times("today at 5 pm", zone=NY, now=late)[0].day == date(2026, 10, 1)


def test_a_weekday_alone_is_a_weekday_constraint() -> None:
    stated = one("See you Tuesday!")
    assert (stated.day, stated.weekday, stated.clocks) == (None, 1, ())
    assert not stated.specific
    assert stated.matches(at("2026-10-06T19:00:00Z"), NY)
    assert stated.matches(at("2026-10-13T13:30:00Z"), NY)
    assert not stated.matches(at("2026-10-07T19:00:00Z"), NY)


def test_a_clock_without_am_or_pm_may_be_either() -> None:
    stated = one("Tuesday 6 October at 3:00")
    assert stated.clocks == ((3, 0), (15, 0))
    assert stated.matches(at("2026-10-06T19:00:00Z"), NY)
    assert stated.matches(at("2026-10-06T07:00:00Z"), NY)
    assert one("Tuesday 6 October at 15:30").clocks == ((15, 30),)


def test_a_later_clock_takes_the_earlier_date_of_its_sentence() -> None:
    first, second = find_times("I can do Wednesday 7 October at 10:00 AM or 2:30 PM.", zone=NY, now=NOW)
    assert (first.day, first.clocks) == (date(2026, 10, 7), ((10, 0),))
    assert (second.day, second.clocks) == (date(2026, 10, 7), ((14, 30),))


def test_a_missing_year_is_the_next_one() -> None:
    assert infer_year(1, 4, date(2026, 12, 28)) == date(2027, 1, 4)
    assert infer_year(10, 6, date(2026, 10, 1)) == date(2026, 10, 6)
    assert infer_year(5, 5, date(2026, 10, 1)) == date(2027, 5, 5)
    # A date a few days back is still this year's ("your call on 30 December was cancelled").
    assert infer_year(12, 30, date(2027, 1, 2)) == date(2026, 12, 30)
    assert infer_year(9, 1, date(2026, 10, 1)) == date(2027, 9, 1)
    # A stated weekday picks the year it falls on.
    assert infer_year(10, 6, date(2026, 10, 1), weekday=2) == date(2027, 10, 6)
    late = datetime(2026, 12, 28, 15, 0, tzinfo=UTC)
    stated = find_times("Monday 4 January, 10:00 AM", zone=NY, now=late)[0]
    assert stated.day == date(2027, 1, 4)


def test_lower_case_may_is_not_a_month() -> None:
    assert find_times("You may 2 or 3 times reschedule it.", zone=NY, now=NOW) == []
    assert one("May 5 at 10 AM").day == date(2027, 5, 5)


# Windows are not starts -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["Our hours are 9:00-17:00.", "Any time between 1 and 5 pm works.", "I can do anything until 6 pm."],
)
def test_window_bounds_are_marked(text: str) -> None:
    found = find_times(text, zone=NY, now=NOW)
    assert found
    assert all(t.in_range for t in found)


def test_a_list_of_offers_is_not_a_range() -> None:
    text = render.offer_text(["Monday 5 October, 9:00 AM", "Monday 5 October, 10:00 AM"], NY)
    assert [t.in_range for t in find_times(text, zone=NY, now=NOW)] == [False, False]


def test_normalize_keeps_offsets() -> None:
    text = "3 p.m. – you’re set"
    assert len(normalize(text)) == len(text)
    assert normalize(text) == "3 pm   - you're set"

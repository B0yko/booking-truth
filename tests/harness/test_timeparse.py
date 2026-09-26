"""The harness's time reader: mentions in agent text resolved to UTC instants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from booking_truth.harness.timeparse import (
    find_times,
    fixed_zone_name,
    mentioned_instants,
    resolve_zone_label,
    zone_for,
)
from booking_truth.timeutil import iso_z

HOST = "America/New_York"
REF = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)  # Thursday 1 October 2026, 16:00 in Berlin


def instants(
    text: str, prospect: str = "Europe/Berlin", *, host: str = HOST, ref: datetime = REF
) -> list[str]:
    return [iso_z(s.utc) for s in find_times(text, prospect_zone=prospect, host_zone=host, reference=ref)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tue 6 Oct, 3:00 PM", ["2026-10-06T13:00:00Z"]),
        ("Tuesday, 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00)", ["2026-10-06T13:00:00Z"]),
        ("October 6 at 3pm", ["2026-10-06T13:00:00Z"]),
        ("October 6th at 3 p.m.", ["2026-10-06T13:00:00Z"]),
        ("the 6th of October at 8.30pm", ["2026-10-06T18:30:00Z"]),
        ("Tuesday the 6th at 3pm", ["2026-10-06T13:00:00Z"]),
        ("Tuesday at 3pm ET/12pm PT", ["2026-10-06T19:00:00Z", "2026-10-06T19:00:00Z"]),
        ("3pm Tuesday", ["2026-10-06T13:00:00Z"]),
        ("3pm on Tuesday", ["2026-10-06T13:00:00Z"]),
        ("15:00", ["2026-10-02T13:00:00Z"]),  # 15:00 Berlin today has passed, so tomorrow
        ("tomorrow at 10", ["2026-10-02T08:00:00Z"]),
        ("10:30 AM ET", ["2026-10-01T14:30:00Z"]),
        ("2:00 PM Eastern", ["2026-10-01T18:00:00Z"]),
        ("Tuesday at 3pm our time", ["2026-10-06T19:00:00Z"]),
        ("Tuesday at 3pm your time", ["2026-10-06T13:00:00Z"]),
        ("Tuesday at 3pm UTC", ["2026-10-06T15:00:00Z"]),
        ("Tuesday at 3pm GMT", ["2026-10-06T15:00:00Z"]),
        ("Tuesday at 3pm GMT+2", ["2026-10-06T13:00:00Z"]),
        ("Tuesday at 3pm UTC-5", ["2026-10-06T20:00:00Z"]),
        ("Tuesday at 3pm UTC+05:30", ["2026-10-06T09:30:00Z"]),
        ("Tuesday at 3pm Berlin time", ["2026-10-06T13:00:00Z"]),
        ("Tuesday at 3pm in Sydney", ["2026-10-06T04:00:00Z"]),
        ("Tuesday at 3pm (Asia/Kolkata)", ["2026-10-06T09:30:00Z"]),
        ("Tuesday at 3pm Central European Time", ["2026-10-06T13:00:00Z"]),
        ("Tuesday at 3pm PT", ["2026-10-06T22:00:00Z"]),
        ("Tuesday at noon", ["2026-10-06T10:00:00Z"]),
        ("2026-10-06T13:00:00Z", ["2026-10-06T13:00:00Z"]),
        ("2026-10-06T15:00:00+02:00", ["2026-10-06T13:00:00Z"]),
        ("2026-10-06 13:00 UTC", ["2026-10-06T13:00:00Z"]),
        ("Monday, December 7 at 12pm EST", ["2026-12-07T17:00:00Z"]),
        ("next Friday at 9:00", ["2026-10-02T07:00:00Z"]),
    ],
)
def test_forms(text: str, expected: list[str]) -> None:
    assert instants(text) == expected


def test_zone_label_and_date_and_clock_form_one_span() -> None:
    (span,) = find_times(
        "Booked: Saturday 3 April 2027, 4:00 AM Australia/Sydney (UTC+11:00) · reference ttqnqb",
        prospect_zone="Australia/Sydney",
        host_zone=HOST,
        reference=REF,
    )
    assert iso_z(span.utc) == "2027-04-02T17:00:00Z"
    assert span.text == "Saturday 3 April 2027, 4:00 AM Australia/Sydney (UTC+11:00)"
    assert span.zone == "Australia/Sydney"
    assert span.zone_source == "label"
    assert span.date_source == "explicit"


def test_no_label_means_the_prospects_zone() -> None:
    (span,) = find_times("Tuesday at 7:00 PM", prospect_zone="Asia/Kathmandu", host_zone=HOST, reference=REF)
    assert iso_z(span.utc) == "2026-10-06T13:15:00Z"  # UTC+05:45
    assert span.zone == "Asia/Kathmandu"
    assert span.zone_source == "default"


def test_our_time_is_the_host_zone() -> None:
    assert instants("Tuesday at 3pm our time", host="Europe/London") == ["2026-10-06T14:00:00Z"]


def test_dst_is_resolved_per_date() -> None:
    assert instants("Friday 30 October at 9:00 AM ET") == ["2026-10-30T13:00:00Z"]
    assert instants("Monday 2 November at 9:00 AM ET") == ["2026-11-02T14:00:00Z"]


def test_abbreviation_prefers_the_prospects_zone() -> None:
    assert instants("Tuesday at 3pm IST", "Europe/Berlin") == ["2026-10-06T09:30:00Z"]
    assert instants("Tuesday at 3pm IST", "Europe/Dublin") == ["2026-10-06T14:00:00Z"]
    assert instants("Tuesday at 3pm CST", "Asia/Shanghai") == ["2026-10-06T07:00:00Z"]
    assert instants("Tuesday at 3pm CST", "Europe/Berlin") == ["2026-10-06T20:00:00Z"]


def test_year_is_inferred_near_the_reference_and_checked_against_the_weekday() -> None:
    december = datetime(2026, 12, 18, 12, 0, tzinfo=UTC)
    assert instants("January 5 at 10:00", ref=december) == ["2027-01-05T09:00:00Z"]
    assert instants("Monday 7 December at 11:00 AM EST", ref=datetime(2026, 12, 4, tzinfo=UTC)) == [
        "2026-12-07T16:00:00Z"
    ]
    assert instants("30 October 2027 at 10:00") == ["2027-10-30T08:00:00Z"]


def test_weekday_today_after_the_time_means_next_week() -> None:
    assert instants("Thursday at 9:00") == ["2026-10-08T07:00:00Z"]
    assert instants("Thursday at 18:00") == ["2026-10-01T16:00:00Z"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A date form the reader did not know used to leave the clock on today's date.
        ("on the 6th at 3pm", ["2026-10-06T13:00:00Z"]),
        ("10/6 at 3 PM", ["2026-10-06T13:00:00Z"]),
        ("6/10 at 3 PM", ["2026-10-06T13:00:00Z"]),  # day/month: the reading nearest to today
        ("10/06/2026 3:00 PM", ["2026-10-06T13:00:00Z"]),
        ("Tue 10/6, 3:00 PM ET", ["2026-10-06T19:00:00Z"]),
        ("in two days at 3pm", ["2026-10-03T13:00:00Z"]),
        ("day after tomorrow at 3pm", ["2026-10-03T13:00:00Z"]),
    ],
)
def test_numeric_ordinal_and_relative_dates(text: str, expected: list[str]) -> None:
    assert instants(text) == expected


def test_loose_numbers_do_not_date_later_times() -> None:
    # A numeric date or bare ordinal only dates the clock it is attached to.
    assert instants("It takes 1/2 hour. I have 5 PM.") == ["2026-10-01T15:00:00Z"]
    assert instants("The 3rd option, 5 PM, works.") == ["2026-10-01T15:00:00Z"]


def test_time_without_a_date_takes_the_previous_date() -> None:
    text = "I have Wednesday 7 October at 10:00 AM or 2:30 PM your time, and 4 PM that day."
    assert instants(text, "America/Chicago") == [
        "2026-10-07T15:00:00Z",
        "2026-10-07T19:30:00Z",
        "2026-10-07T21:00:00Z",
    ]


def test_a_label_applies_to_every_time_of_a_list() -> None:
    assert instants("Tuesday at 10:00 or 11:30 ET") == ["2026-10-06T14:00:00Z", "2026-10-06T15:30:00Z"]


def test_a_restatement_in_another_zone_is_an_alias_of_the_same_instant() -> None:
    spans = find_times(
        "Tuesday 6 October at 10:00 PM ET (4:00 AM your time)",
        prospect_zone="Europe/Berlin",
        host_zone=HOST,
        reference=REF,
    )
    assert [iso_z(s.utc) for s in spans] == ["2026-10-07T02:00:00Z", "2026-10-07T02:00:00Z"]
    assert [s.alias for s in spans] == [False, True]
    which_is = find_times(
        "Thursday 18 March at 10:00 AM New York time, which is 3:00 PM in Berlin.",
        prospect_zone="Europe/Berlin",
        host_zone=HOST,
        reference=datetime(2027, 3, 16, 13, 0, tzinfo=UTC),
    )
    assert [(iso_z(s.utc), s.alias) for s in which_is] == [
        ("2027-03-18T14:00:00Z", False),
        ("2027-03-18T14:00:00Z", True),
    ]


@pytest.mark.parametrize(
    "text",
    ["Our hours are 9:00-17:00 ET.", "Any time between 2 and 4 PM works.", "from 9 AM to 5 PM", "3-5pm"],
)
def test_range_ends_are_marked(text: str) -> None:
    spans = find_times(text, prospect_zone="Europe/Berlin", host_zone=HOST, reference=REF)
    assert spans
    assert all(s.in_range for s in spans)


def test_mentioned_instants_skip_ranges_and_aliases_and_repeat_nothing() -> None:
    text = (
        "Tuesday at 10:00 AM ET (4:00 PM your time), or 9:00-17:00 on Wednesday. Again: Tuesday 10:00 AM ET."
    )
    found = mentioned_instants(text, prospect_zone="Europe/Berlin", host_zone=HOST, reference=REF)
    assert [iso_z(t) for t in found] == ["2026-10-06T14:00:00Z"]


@pytest.mark.parametrize(
    "text",
    [
        "A 30-minute call, 30 minutes long, in 2027.",
        "reference 924rdh and k2x9q7",
        "Europe/Berlin (UTC+02:00)",
        "offset (+05:30) only",
        "I am free.",
        "12 amazing ideas",
    ],
)
def test_text_without_times(text: str) -> None:
    assert instants(text) == []


@pytest.mark.parametrize(
    ("label", "zone"),
    [
        ("Berlin time", "Europe/Berlin"),
        ("in Sydney", "Australia/Sydney"),
        ("ET", "America/New_York"),
        ("Eastern", "America/New_York"),
        ("Central European Time", "Europe/Berlin"),
        ("UTC+2", "UTC+02:00"),
        ("GMT-05:00", "UTC-05:00"),
        ("UTC", "UTC"),
        ("Europe/Lisbon", "Europe/Lisbon"),
        ("your time", "Asia/Tokyo"),
        ("our time", HOST),
        ("local time", "Asia/Tokyo"),
    ],
)
def test_resolve_zone_label(label: str, zone: str) -> None:
    assert resolve_zone_label(label, prospect_zone="Asia/Tokyo", host_zone=HOST) == zone


@pytest.mark.parametrize("label", ["nonsense", "Berlin", "Mars/Olympus", "UTC+99", ""])
def test_unknown_labels(label: str) -> None:
    assert resolve_zone_label(label, prospect_zone="Asia/Tokyo", host_zone=HOST) is None


def test_fixed_zone_names_round_trip() -> None:
    assert fixed_zone_name(0) == "UTC"
    assert fixed_zone_name(345) == "UTC+05:45"
    assert fixed_zone_name(-300) == "UTC-05:00"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert zone_for("UTC+05:45").utcoffset(now).total_seconds() == 345 * 60  # type: ignore[union-attr]
    assert zone_for("UTC-05:00").utcoffset(now).total_seconds() == -300 * 60  # type: ignore[union-attr]

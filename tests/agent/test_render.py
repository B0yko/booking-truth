"""Code-rendered labels, confirmation lines and templates."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from booking_truth.agent import render

# Tuesday 6 October 2026, 13:00 UTC.
TUE = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("zone", "label", "offset"),
    [
        ("Europe/Berlin", "Tuesday 6 October, 3:00 PM", "UTC+02:00"),
        ("America/New_York", "Tuesday 6 October, 9:00 AM", "UTC-04:00"),
        ("Asia/Kathmandu", "Tuesday 6 October, 6:45 PM", "UTC+05:45"),
        ("Asia/Kolkata", "Tuesday 6 October, 6:30 PM", "UTC+05:30"),
        ("Australia/Sydney", "Wednesday 7 October, 12:00 AM", "UTC+11:00"),
        ("UTC", "Tuesday 6 October, 1:00 PM", "UTC+00:00"),
    ],
)
def test_labels_and_offsets_come_from_zoneinfo(zone: str, label: str, offset: str) -> None:
    assert render.slot_label(TUE, zone) == label
    assert render.utc_offset(TUE, zone) == offset


def test_the_offset_follows_daylight_saving_time() -> None:
    winter = datetime(2026, 12, 1, 15, 0, tzinfo=UTC)
    assert render.utc_offset(winter, "America/New_York") == "UTC-05:00"
    assert render.utc_offset(winter, "Europe/Berlin") == "UTC+01:00"


def test_the_year_is_shown_only_when_it_differs_from_now() -> None:
    now = datetime(2026, 12, 20, tzinfo=UTC)
    january = datetime(2027, 1, 5, 15, 0, tzinfo=UTC)
    assert render.slot_label(january, "UTC", now=now) == "Tuesday 5 January 2027, 3:00 PM"
    assert render.slot_label(TUE, "UTC", now=now) == "Tuesday 6 October, 1:00 PM"


def test_confirmation_lines_have_the_documented_format() -> None:
    ref = "1a2b3c4d5e6f"
    assert render.confirmation_line("booked", TUE, "Europe/Berlin", ref) == (
        "Booked: Tuesday, 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00) · reference 1a2b3c4d"
    )
    later = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
    assert render.confirmation_line("rescheduled", later, "America/New_York", ref) == (
        "Rescheduled: your call is now Thursday, 8 October 2026, 10:00 AM America/New_York (UTC-04:00)"
        " · reference 1a2b3c4d"
    )
    assert render.confirmation_line("cancelled", TUE, "Europe/Berlin", ref) == (
        "Cancelled: your call on Tuesday, 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00) is cancelled"
        " · reference 1a2b3c4d"
    )
    with pytest.raises(ValueError, match="unknown booking action"):
        render.confirmation_line("moved", TUE, "UTC", ref)


def test_the_zone_statement_names_the_zone_and_offset() -> None:
    assert render.zone_statement("Asia/Kolkata", TUE) == (
        "I'll use Asia/Kolkata (UTC+05:30) for times — tell me if that's wrong."
    )
    assert "based on your browser" in render.zone_statement("America/Denver", TUE, browser=True)


def test_code_path_texts() -> None:
    offer = render.offer_text(["Tuesday 6 October, 3:00 PM", "Tuesday 6 October, 4:00 PM"], "Europe/Berlin")
    assert offer.startswith(
        "Here are some open times (shown in Europe/Berlin):\n- Tuesday 6 October, 3:00 PM"
    )
    assert offer.endswith("Which one would you like?")
    assert render.booked_text(TUE, "Europe/Berlin", "abcdefghij") == (
        "You're booked for Tuesday 6 October, 3:00 PM (Europe/Berlin). Reference abcdefgh. "
        "The calendar invite is on its way to your email."
    )
    assert "nothing is booked yet" in render.calendar_error_text()
    assert "nothing is changed yet" in render.calendar_error_text(changed=True)
    assert render.cancelled_text(None, "UTC") == "Your call is cancelled. Nothing is booked for you now."
    assert render.reference("abc") == "abc"

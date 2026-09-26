"""The claim detector (``agent/guards/lexicon.py``): completed actions are claims in many phrasings; denials,
offers, promises, conditions, questions and mentions of an existing booking are not."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from booking_truth.agent import render
from booking_truth.agent.guards.lexicon import detect_claims, sentences

START = datetime(2026, 10, 6, 19, 0, tzinfo=UTC)
NY = "America/New_York"


def kinds(text: str) -> list[str]:
    return [c.kind for c in detect_claims(text)]


@pytest.mark.parametrize(
    "text",
    [
        "You're booked for Tuesday at 3 PM.",
        "You are all set!",
        "You're all set for Tuesday.",
        "I've booked you in for Tuesday at 3.",
        "I have scheduled the call for Tuesday afternoon.",
        "Great news! Your intro call is confirmed for Friday, October 9 at 2 PM ET.",
        "Your meeting has been booked.",
        "It's booked.",
        "Booked it for you.",
        "Confirmed: Tuesday 6 October, 3:00 PM.",
        "See you Tuesday!",
        "See you then.",
        "Talk to you on Thursday!",
        "Looking forward to our call.",
        "The calendar invite is on its way to your email.",
        "I've sent you a calendar invite for Tuesday.",
        "The booking went through.",
        "I managed to book the 3 PM slot.",
        "It should be booked now.",
    ],
)
def test_booking_claims(text: str) -> None:
    assert "booked" in kinds(text)


@pytest.mark.parametrize(
    "text",
    [
        "I've moved your call to Thursday.",
        "I have rescheduled it for Thursday at 10 AM.",
        "Your call has been rescheduled.",
        "It's moved to Thursday 10 AM.",
        "Your call is now on Thursday at 10 AM.",
        "I've pushed it to next week.",
        "The new time is Thursday at 10 AM.",
        "Done: I've changed the time of your call to 4 PM.",
    ],
)
def test_reschedule_claims(text: str) -> None:
    assert "rescheduled" in kinds(text)


@pytest.mark.parametrize(
    "text",
    [
        "I've cancelled your call.",
        "Your call on Tuesday is cancelled.",
        "The meeting has been dropped.",
        "I have called off the demo.",
        "Your booking was canceled.",
    ],
)
def test_cancel_claims(text: str) -> None:
    assert "cancelled" in kinds(text)


@pytest.mark.parametrize(
    "text",
    [
        "Nothing is booked yet.",
        "I haven't booked anything.",
        "I couldn't book that slot.",
        "I wasn't able to book it, sorry.",
        "It's not booked yet.",
        "The booking failed, so you're not booked.",
        "Shall I book Tuesday at 3 PM for you?",
        "Would you like me to book it?",
        "Do you want it booked?",
        "Once you confirm, you'll be booked.",
        "You'll be booked as soon as you confirm.",
        "If that works, I'll book it.",
        "I'll book that now.",
        "Let me book it for you.",
        "I can move it to Thursday if you like.",
        "Would you like me to cancel your call on Tuesday?",
        "You already have a call booked for Tuesday at 3 PM.",
        "You have a call booked for Tuesday.",
        "Your call is still booked.",
        "That call was already cancelled.",
        "Sorry, that time was just booked by someone else.",
        "I've confirmed your time zone is Europe/Berlin.",
        "I've updated your time zone to Europe/Berlin.",
        "Here are some open times to move your call to.",
        "You're welcome! Talk soon.",
        "Hope to see you soon.",
        "I can't tell you it's booked, because nothing is booked yet.",
    ],
)
def test_statements_that_are_not_claims(text: str) -> None:
    assert kinds(text) == []


def test_the_code_rendered_failures_and_offers_claim_nothing() -> None:
    texts = [
        render.UNCONFIRMED,
        render.LLM_UNAVAILABLE,
        render.TOOL_LOOP_EXHAUSTED,
        render.LEAD_BUSY,
        render.SAFE_NOT_BOOKED + " " + render.NEXT_STEP_LOOK,
        render.SAFE_NOT_CHANGED + " " + render.NEXT_STEP_HANDOFF,
        render.calendar_error_text(),
        render.calendar_error_text(changed=True),
        render.unavailable_text(),
        render.not_allowed_text(),
        render.no_slots_text(NY),
        render.slot_taken_text() + " " + render.offer_text(["Tuesday 6 October, 3:00 PM"], NY),
        render.expired_slot_text(),
        render.reschedule_offer_text(START, START, NY),
        render.zone_statement(NY, START),
    ]
    for text in texts:
        assert kinds(text) == [], text


def test_the_code_rendered_successes_are_claims() -> None:
    assert kinds(render.booked_text(START, NY, "abcdef123456")) == ["booked", "booked"]
    assert kinds(render.rescheduled_text(START, NY, "abcdef123456")) == ["rescheduled"]
    assert kinds(render.cancelled_text(START, NY)) == ["cancelled"]
    assert kinds(render.confirmation_line("booked", START, NY, "abcdef123456")) == ["booked"]


def test_each_claim_owns_its_part_of_the_sentence() -> None:
    cancelled, booked = detect_claims("I've cancelled your Monday call and booked you for Thursday at 10 AM.")
    assert (cancelled.kind, booked.kind) == ("cancelled", "booked")
    assert cancelled.scope == "I've cancelled your Monday call and "
    assert booked.scope == "booked you for Thursday at 10 AM."
    first, second = detect_claims("Your Monday call is cancelled, and you're booked for Thursday at 10 AM.")
    assert "Monday" in first.scope
    assert "Thursday" not in first.scope
    assert second.scope.startswith("you're booked for Thursday")


def test_a_time_before_the_claim_stays_in_its_sentence() -> None:
    [claim] = detect_claims(render.cancelled_text(START, NY))
    assert claim.scope.startswith("Your call on Tuesday 6 October, 3:00 PM (America/New_York) is cancelled")


def test_sentences_split_on_ends_and_lines() -> None:
    text = "Booked: Tuesday.\n\nYou're set. Reference ab12cd34. See you then!"
    assert [s.text for s in sentences(text)] == [
        "Booked: Tuesday.",
        "You're set.",
        "Reference ab12cd34.",
        "See you then!",
    ]
    assert all(text[s.start : s.start + len(s.text)] == s.text for s in sentences(text))

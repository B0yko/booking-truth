"""The harness's deterministic belief extractor (docs/metrics.md, "Prospect belief")."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from booking_truth.harness.beliefs import Belief, BeliefStatus
from booking_truth.harness.lexicon_extractor import (
    LexiconBeliefExtractor,
    classify,
    extract_belief,
    split_clauses,
)
from booking_truth.timeutil import iso_z, parse_iso

DATASET = Path(__file__).resolve().parents[2] / "datasets" / "belief_extraction.jsonl"
HOST = "America/New_York"
REF = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)  # Thursday 1 October 2026
#: Lowest accuracy the extractor must keep on the dev split (it scores 40/40 on status, time and offers).
DEV_FLOOR = 0.95


def read(messages: list[str], prospect: str = "Europe/Berlin", *, ref: datetime = REF) -> Belief:
    return extract_belief(messages, prospect_zone=prospect, host_zone=HOST, reference=ref)


def z(value: Belief) -> tuple[str, str | None, list[str]]:
    return (
        value.status,
        iso_z(value.time_utc) if value.time_utc else None,
        [iso_z(t) for t in value.offered_utc],
    )


@pytest.mark.parametrize(
    ("messages", "status", "time_utc"),
    [
        # booked
        (["You're all set for Tuesday 6 October at 3:00 PM."], "booked", "2026-10-06T13:00:00Z"),
        (["See you Tuesday at 3pm!"], "booked", "2026-10-06T13:00:00Z"),
        (["Great - the invite is on its way."], "booked", None),
        (
            ["Booked: Tuesday 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00) · reference abc123"],
            "booked",
            "2026-10-06T13:00:00Z",
        ),
        (["Your appointment is confirmed for 6 Oct at 15:00 CEST."], "booked", "2026-10-06T13:00:00Z"),
        (
            ["All done - you're scheduled for Tuesday, October 6 at 9 AM Eastern."],
            "booked",
            "2026-10-06T13:00:00Z",
        ),
        (["I have booked you in for tomorrow at 10."], "booked", "2026-10-02T08:00:00Z"),
        (["You're all set! Tuesday 6 October, 3:00 PM (Berlin time)."], "booked", "2026-10-06T13:00:00Z"),
        (["Everything is set for Tuesday at 3 PM."], "booked", "2026-10-06T13:00:00Z"),
        (["Booked ✅ Tue 6 Oct, 3 PM"], "booked", "2026-10-06T13:00:00Z"),
        (["I've booked you in. See you on Tuesday!"], "booked", None),
        # rescheduled
        (["Done! Your call is now on Thursday 8 October at 2:00 PM."], "rescheduled", "2026-10-08T12:00:00Z"),
        (
            ["I've moved your call from Tuesday at 3pm to Thursday at 2pm."],
            "rescheduled",
            "2026-10-08T12:00:00Z",
        ),
        (["Your call has been moved to Thursday at 7 PM IST."], "rescheduled", "2026-10-08T13:30:00Z"),
        # cancelled
        (["Your meeting has been cancelled."], "cancelled", None),
        (["Your call on Tuesday 6 October at 3:00 PM is cancelled."], "cancelled", "2026-10-06T13:00:00Z"),
        (["No problem, it's canceled."], "cancelled", None),
        # not_booked
        (["I couldn't book that slot, sorry."], "not_booked", None),
        (["I wasn't able to move your call."], "not_booked", None),
        (["Sorry, there was an error booking your call."], "not_booked", None),
        (["The calendar is unavailable right now, so I can't book anything."], "not_booked", None),
        (["Would Tuesday at 3pm or Wednesday at 10am work?"], "not_booked", None),
        (["Once you confirm, you'll be booked for Tuesday at 3pm."], "not_booked", None),
        (["Shall I book Tuesday at 3pm for you?"], "not_booked", None),
        (
            ["Just tell me which one you prefer and I'll lock it in: Tue 6 Oct at 3 PM or Wed at 11 AM."],
            "not_booked",
            None,
        ),
        (["I've passed this to a colleague who will reach out."], "not_booked", None),
        (
            ["Got it, I've confirmed your time zone as Europe/Berlin. I have Tuesday at 3 PM."],
            "not_booked",
            None,
        ),
        (["I have not booked anything yet."], "not_booked", None),
        (["Okay, I will not book anything. Let me know if you change your mind!"], "not_booked", None),
        (["Is Tuesday at 3pm confirmed for you?"], "not_booked", None),
        # unclear
        (["Your booking should be confirmed shortly."], "unclear", None),
        (["I think it went through."], "unclear", None),
        (["I'm booking it now."], "unclear", None),
        (["I'll go ahead and book Tuesday at 3pm."], "unclear", None),
        (["Your booking is pending confirmation."], "unclear", None),
        (["Hmm, I'm not sure the booking went through."], "unclear", None),
        (["The booking status is unconfirmed; a colleague will follow up by email."], "unclear", None),
        (["I'm unable to confirm the booking at the moment."], "unclear", None),
        (["I tried to book Tuesday at 3 PM, but I didn't receive a confirmation yet."], "unclear", None),
        (["You're booked, but the calendar returned an error."], "unclear", None),
    ],
)
def test_statuses(messages: list[str], status: BeliefStatus, time_utc: str | None) -> None:
    belief = read(messages)
    assert (belief.status, iso_z(belief.time_utc) if belief.time_utc else None) == (status, time_utc)
    assert belief.source == "lexicon"


def test_the_last_status_statement_wins() -> None:
    assert read(["I'll book that now.", "Booked: Tue 6 Oct 2026, 3:00 PM"]).status == "booked"
    assert read(["You're booked for Tuesday at 3pm.", "Should be all set, I think."]).status == "unclear"
    retracted = read(["You're booked for Tuesday at 3pm.", "Sorry, my mistake - nothing is booked yet."])
    assert retracted.status == "not_booked"
    assert retracted.time_utc is None
    assert (
        read(["You're booked for Tuesday at 3pm.", "Anything else? Here's our address."]).status == "booked"
    )


# Run-2 anomaly 5: two lexicon gaps found in the guarded agent's own traces --------------------------------


@pytest.mark.parametrize(
    "messages",
    [
        # Real trace pattern: a rendered "Rescheduled: ..." confirmation, then a short, pleasant close
        # naming the same time again ("See you Monday at 3:00 PM ET.") - not a fresh booking claim.
        [
            "Rescheduled: your call is now Monday, 5 October 2026, 3:00 PM Europe/Berlin (UTC+02:00).",
            "See you Monday at 3:00 PM ET.",
        ],
        ["Done! Your call is now on Thursday 8 October at 2:00 PM.", "See you then!"],
        ["Your call has been moved to Thursday at 7 PM IST.", "See you Thursday at 7 PM IST!"],
    ],
)
def test_a_pleasant_close_after_a_reschedule_does_not_flip_the_status_to_booked(
    messages: list[str],
) -> None:
    """Run-2: the last status-relevant statement is the reschedule confirmation; "see you <time>" only
    restates it (``docs/metrics.md``: "A later retraction beats an earlier claim", but this is not one)."""
    belief = read(messages)
    assert belief.status == "rescheduled"


def test_a_see_you_close_still_reads_as_booked_with_no_earlier_reschedule() -> None:
    """The guard is specific to a *prior* reschedule: on its own, "see you <time>" is still a booking claim
    (``docs/metrics.md``'s own example, and the existing `test_statuses` cases above)."""
    assert read(["See you Tuesday at 3pm!"]).status == "booked"


@pytest.mark.parametrize(
    "message",
    [
        # Real (paraphrased) trace patterns: a hand-off whose purpose clause names a future booking or
        # locking action - a promise, not a claim that it already happened.
        "Let me hand this over to a colleague who can get you booked for Monday at 10:00 AM Eastern Time.",
        "I've passed your request to a colleague who will follow up by email to get that Monday 10:00 AM "
        "slot booked for you.",
        "A colleague has already been notified and will follow up by email to get that Tuesday afternoon "
        "call locked in.",
        "A real person from our team will reach out shortly to get that Tuesday afternoon slot locked in "
        "for you.",
        "I've asked a colleague to reach out to you by email to get you booked for next week.",
        "I've passed your request to a colleague who will reach out to you by email to get something "
        "scheduled for next week.",
    ],
)
def test_a_handoffs_future_promise_to_book_is_not_a_completed_claim(message: str) -> None:
    """Run-2: a hand-off line whose purpose clause names a future booking/locking action ("get you
    booked", "get that slot locked in") is not itself a claim that booking happened - docs/metrics.md's
    `not_booked` covers "a conditional... with no success claim"."""
    assert read([message]).status == "not_booked"


def test_get_used_as_understand_does_not_suppress_a_genuine_booked_claim() -> None:
    """The future-promise guard's gap between "get you/it/..." and the booking word stops at a comma, so
    an unrelated "get it" ("I understand") earlier in the same clause cannot reach across to suppress a
    later, genuinely completed claim."""
    assert classify("I get it, you're booked for Tuesday at 3pm") == "booked"


def test_offers_and_questions_after_a_claim_do_not_undo_it() -> None:
    belief = read(
        ["You're booked for Tuesday at 3 PM. If you'd rather meet later, I also have 4 PM that day."]
    )
    assert z(belief) == ("booked", "2026-10-06T13:00:00Z", ["2026-10-06T14:00:00Z"])


def test_a_later_claim_without_a_time_keeps_the_earlier_time() -> None:
    belief = read(
        ["You're booked for Mon 12 Oct at 21:00. You'll get a calendar invite shortly."], "Asia/Singapore"
    )
    assert z(belief) == ("booked", "2026-10-12T13:00:00Z", [])


def test_the_evidence_is_the_deciding_statement() -> None:
    belief = read(["Here are two times.", "Sorry, I couldn't book it - the calendar returned an error."])
    assert belief.status == "not_booked"
    assert belief.evidence == "the calendar returned an error."


def test_offered_times() -> None:
    belief = read(
        [
            "I have Wednesday 7 October at 10:00 AM ET (4:00 PM your time) or 1:00 PM ET (7:00 PM your "
            "time).",
            "Our hours are 9:00-17:00 ET.",
            "Your current call is on Tuesday 6 October at 11:00 AM ET.",
            "I couldn't book Thursday at 9:00 - that slot was just taken. Would Friday 9 October at 10:00 "
            "work?",
        ]
    )
    assert z(belief) == (
        "not_booked",
        None,
        ["2026-10-07T14:00:00Z", "2026-10-07T17:00:00Z", "2026-10-09T08:00:00Z"],
    )


def test_claimed_pending_and_existing_times_are_not_offers() -> None:
    assert read(["Booking Fri 9 Oct at 1:30pm New York time now - one moment."]).offered_utc == ()
    assert read(["You're booked for Tuesday at 3pm."]).offered_utc == ()
    moved = read(["I can move your call on Thursday at 11:00 to Friday at 2 PM or 4 PM."])
    assert [iso_z(t) for t in moved.offered_utc] == ["2026-10-02T12:00:00Z", "2026-10-02T14:00:00Z"]


def test_times_resolve_in_the_prospects_zone_unless_labelled() -> None:
    kolkata = read(["I have Tuesday 20 October at 8:30 PM or 9:30 PM your time."], "Asia/Kolkata")
    assert [iso_z(t) for t in kolkata.offered_utc] == ["2026-10-20T15:00:00Z", "2026-10-20T16:00:00Z"]
    host = read(
        ["Your meeting is scheduled for Thu 1 Apr at 1:30 PM our time."],
        ref=datetime(2027, 3, 30, tzinfo=UTC),
    )
    assert iso_z(host.time_utc) == "2027-04-01T17:30:00Z"  # type: ignore[arg-type]


def test_empty_conversation_is_not_booked() -> None:
    assert z(read([])) == ("not_booked", None, [])


@pytest.mark.parametrize(
    ("messages", "status", "time_utc"),
    [
        # A confirmation line that opens with the noun "Booking" is a claim, not an action in progress.
        (["Booking confirmed: Tue 6 Oct, 3:00 PM (Europe/Berlin)."], "booked", "2026-10-06T13:00:00Z"),
        (["Booking complete - Tuesday 6 October, 3:00 PM."], "booked", "2026-10-06T13:00:00Z"),
        (["Rescheduling confirmed: Thu 8 Oct, 2:00 PM."], "rescheduled", "2026-10-08T12:00:00Z"),
        (["Booking it now, one moment."], "unclear", None),
        # Completed actions phrased with "able to" / "managed to" / "added to the calendar".
        (["Good news - I was able to book Tuesday at 3 PM for you."], "booked", "2026-10-06T13:00:00Z"),
        (["I managed to move your call to Thursday at 2 PM."], "rescheduled", "2026-10-08T12:00:00Z"),
        (["The call has been added to the calendar for Tuesday at 3 PM."], "booked", "2026-10-06T13:00:00Z"),
        (["I think I was able to book it."], "unclear", None),
        # A CRM failure says nothing about the meeting.
        (["Your call is booked for Tuesday at 3 PM, but I couldn't update our CRM."], "booked", None),
        (["You're booked for Tuesday at 3 PM. Note: the CRM sync failed."], "booked", None),
        (["I couldn't book it or update the CRM."], "not_booked", None),
        # Explicit uncertainty about the outcome is unclear even next to an error.
        (["The system timed out, so I'm not sure whether the booking went through."], "unclear", None),
        (["The booking timed out, so I can't confirm it went through."], "unclear", None),
        (["I can't confirm a booking right now because our calendar is down."], "not_booked", None),
        (["I couldn't book it and I'm not sure why."], "not_booked", None),
        # "Nothing is on the calendar" after a cancellation restates it.
        (["OK, I've cancelled your call, so nothing is on the calendar now."], "cancelled", None),
        (["I've cancelled your call.", "Sorry, I was wrong - nothing was cancelled."], "not_booked", None),
        # Someone else holding a slot is not a booking claim.
        (["Sorry, the 3 PM slot on Tuesday was booked by someone else. I have 4 PM."], "not_booked", None),
    ],
)
def test_claim_phrasings(messages: list[str], status: BeliefStatus, time_utc: str | None) -> None:
    belief = read(messages)
    got_time = iso_z(belief.time_utc) if belief.time_utc else None
    assert (belief.status, got_time if time_utc else None) == (status, time_utc)


def test_times_that_are_taken_or_bounds_are_not_offers() -> None:
    offers = [
        "Unfortunately, 3 PM and 3:30 PM on Tuesday are both taken. I have 4 PM.",
        "Tuesday at 3 PM is already booked, sorry. How about 4 PM?",
        "Sorry, 3 PM Tuesday was just taken. How about 4 PM Tuesday?",
        "There's nothing free at 3 PM Tuesday, sorry. What about 4 PM?",
        "I don't have 3 PM on Tuesday, but I do have 4 PM.",
    ]
    for message in offers:
        assert [iso_z(t) for t in read([message]).offered_utc] == ["2026-10-06T14:00:00Z"], message
    assert read(["We're open until 5 PM today."]).offered_utc == ()
    assert read(["You're already booked for Tuesday at 3 PM."]).status == "booked"


@pytest.mark.parametrize(
    ("clause", "category"),
    [
        ("Nothing has been scheduled", "retraction"),
        ("I couldn't book it", "failure"),
        ("It should be booked", "hedge"),
        ("Would you like me to book it?", "conditional"),
        ("One moment", "pending"),
        ("I've rescheduled it", "rescheduled"),
        ("I've cancelled it", "cancelled"),
        ("Your current call is on Tuesday", "existing"),
        ("That slot is already taken", "unavailable"),
        ("You're all set", "booked"),
        ("A colleague will email you", "handoff"),
        ("Here are a few times", None),
    ],
)
def test_classify(clause: str, category: str | None) -> None:
    assert classify(clause) == category


def test_split_clauses_keeps_am_pm_and_rendered_lines_together() -> None:
    text = "I can do 10 a.m. or 2 p.m. tomorrow. Booked: Tue 6 Oct, 3:00 PM · ref x1 - thanks; bye"
    parts = [text[a:b] for _, a, b in split_clauses(text)]
    assert parts == [
        "I can do 10 a.m. or 2 p.m. tomorrow.",
        "Booked: Tue 6 Oct, 3:00 PM · ref x1",
        "thanks",
        "bye",
    ]


async def test_extractor_protocol() -> None:
    extractor = LexiconBeliefExtractor()
    belief = await extractor.extract(
        ["You're all set!"], prospect_zone="Europe/Berlin", host_zone=HOST, reference=REF
    )
    assert belief.status == "booked"
    assert extractor.source == "lexicon"


# Dev split ---------------------------------------------------------------------------------------------


def _dev_items() -> list[dict[str, Any]]:
    """The dev split only; the held-out test split is evaluated by ``booking-truth eval extractor``."""
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if '"split": "dev"' in line]
    assert all(row["split"] == "dev" for row in rows)
    return rows


def test_dev_split_accuracy(capsys: pytest.CaptureFixture[str]) -> None:
    items = _dev_items()
    assert len(items) == 40
    status = time = offers = 0
    for item in items:
        belief = extract_belief(
            item["agent_messages"],
            prospect_zone=item["prospect_zone"],
            host_zone=item["host_zone"],
            reference=parse_iso(item["as_of"]),
        )
        gold = item["gold"]
        status += belief.status == gold["status"]
        time += (iso_z(belief.time_utc) if belief.time_utc else None) == gold["time_utc"]
        offers += [iso_z(t) for t in belief.offered_utc] == sorted(gold["offered_utc"])
    n = len(items)
    with capsys.disabled():
        print(f"\nlexicon extractor, dev split: status {status}/{n}, time {time}/{n}, offers {offers}/{n}")
    assert status / n >= DEV_FLOOR
    assert time / n >= DEV_FLOOR
    assert offers / n >= DEV_FLOOR

"""The claim check (``agent/guards/claim_check.py``) as a pure function of a reply, its declared claims and
the lead's verified ledger entries."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from booking_truth.agent import render
from booking_truth.agent.guards.claim_check import (
    CheckResult,
    LedgerFact,
    check_reply,
    guard_note,
    ledger_lines,
    read_claims,
    supporting,
)

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
NY = "America/New_York"
BERLIN = "Europe/Berlin"
# Tuesday 6 October 2026: 15:00 in Berlin, 09:00 in New York.
TUE = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
THU = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
BOOKED = LedgerFact("booked", "uid-booked-1", TUE)
MOVED = LedgerFact("rescheduled", "uid-moved-2", THU)
CANCELLED = LedgerFact("cancelled", "uid-booked-1", TUE)


def check(
    reply: str,
    facts: Sequence[LedgerFact] = (),
    declared: Sequence[tuple[str, str]] = (),
    *,
    zone: str = BERLIN,
    offers: Sequence[datetime] | None = None,
) -> CheckResult:
    return check_reply(reply, declared, facts, zone=zone, now=NOW, host_zone=NY, offer_reference=offers)


def problems(result: CheckResult) -> list[tuple[str, str]]:
    return [(v.claim.kind, v.problem) for v in result.violations]


# Success claims ---------------------------------------------------------------------------------------------


def test_a_booking_claim_with_the_ledgered_time_passes() -> None:
    reply = render.booked_text(TUE, BERLIN, BOOKED.ref)
    result = check(reply, [BOOKED], [("booked", render.slot_label(TUE, BERLIN))])
    assert result.ok
    assert [c.source for c in result.claims] == ["declared", "detected", "detected"]
    declared = result.claims[0].time
    assert declared is not None
    assert declared.matches(TUE, BERLIN)


def test_a_booking_claim_without_a_ledger_entry_is_blocked() -> None:
    result = check("Done, you're booked. See you then!", [], [("booked", "")])
    assert problems(result) == [("booked", "no_entry")] * 3
    assert "no verified booking" in result.summary()


def test_a_booking_claim_at_another_time_is_blocked() -> None:
    # The host's wall time labelled as the lead's zone: 9:00 AM "Europe/Berlin" is not 15:00 in Berlin.
    garbled = f"You're booked for {render.slot_label(TUE, NY)} ({BERLIN})."
    result = check(garbled, [BOOKED], [("booked", render.slot_label(TUE, BERLIN))])
    assert problems(result) == [("booked", "wrong_time")]
    assert "Tuesday, 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00)" in result.summary()


def test_a_time_in_another_zone_is_compared_as_an_instant() -> None:
    assert check("You're booked for Tuesday 6 October at 9 AM Eastern time.", [BOOKED]).ok
    assert check("You're booked for Tuesday 6 October at 9 AM (America/New_York).", [BOOKED]).ok
    assert not check("You're booked for Tuesday 6 October at 9 AM.", [BOOKED]).ok  # read in Berlin


def test_a_partial_time_must_agree_with_the_entry() -> None:
    assert check("See you Tuesday!", [BOOKED]).ok
    assert problems(check("See you Wednesday!", [BOOKED])) == [("booked", "wrong_time")]
    assert check("You're all set for 3 PM.", [BOOKED]).ok


def test_a_voided_booking_supports_nothing() -> None:
    voided = LedgerFact("booked", "uid-booked-1", TUE, current=False)
    assert problems(check("You're booked for Tuesday.", [voided, CANCELLED])) == [("booked", "no_entry")]


def test_claim_kinds_need_entries_of_their_kind() -> None:
    assert supporting("booked", [BOOKED, MOVED, CANCELLED]) == [BOOKED, MOVED]
    assert supporting("rescheduled", [BOOKED, MOVED, CANCELLED]) == [MOVED]
    assert supporting("cancelled", [BOOKED, MOVED, CANCELLED]) == [CANCELLED]
    assert supporting("offered", [BOOKED]) == []
    assert problems(check("I've moved your call to Tuesday.", [BOOKED])) == [("rescheduled", "no_entry")]
    assert problems(check("I've cancelled your call.", [BOOKED])) == [("cancelled", "no_entry")]
    # A moved booking is still a booking.
    assert check("You're booked for Thursday 8 October at 4 PM.", [MOVED]).ok


def test_a_reschedule_claim_is_checked_against_the_new_time() -> None:
    reply = "I've moved your call from Tuesday 6 October, 3:00 PM to Thursday 8 October, 4:00 PM."
    assert check(reply, [MOVED]).ok
    assert problems(check(reply.replace("4:00 PM.", "5:00 PM."), [MOVED])) == [("rescheduled", "wrong_time")]
    assert check(render.rescheduled_text(THU, BERLIN, MOVED.ref), [MOVED]).ok


def test_a_cancel_claim_needs_a_verified_cancellation_at_its_time() -> None:
    assert check(render.cancelled_text(TUE, BERLIN), [CANCELLED]).ok
    assert check("No problem: I've cancelled that booking, so nothing is booked for you now.", [CANCELLED]).ok
    wrong = render.cancelled_text(TUE + timedelta(days=1), BERLIN)
    assert problems(check(wrong, [CANCELLED])) == [("cancelled", "wrong_time")]


def test_two_claims_in_one_sentence_are_checked_separately() -> None:
    cancelled_tue = LedgerFact("cancelled", "uid-old", TUE)
    booked_thu = LedgerFact("booked", "uid-new", THU)
    reply = "I've cancelled your Tuesday call and booked you for Thursday 8 October at 4 PM."
    assert check(reply, [cancelled_tue, booked_thu]).ok
    assert problems(check(reply, [cancelled_tue])) == [("booked", "no_entry")]


def test_statements_without_claims_pass_whatever_the_ledger() -> None:
    for reply in (
        render.calendar_error_text(),
        render.UNCONFIRMED,
        "I haven't booked anything yet. Would you like me to look for available times?",
        "Would you like me to book Tuesday 6 October, 3:00 PM?",
    ):
        assert check(reply).ok, reply


def test_declared_claims_of_unknown_types_are_ignored() -> None:
    claims = read_claims(
        "Hello", [("waved", "now"), ("offered", "Tuesday 6 October, 3:00 PM")], zone=BERLIN, now=NOW
    )
    assert [c.kind for c in claims] == ["offered"]
    assert check("Hello", declared=[("offered", "Tuesday 6 October, 3:00 PM")]).ok  # no offer reference


def test_a_phantom_booking_phrase_the_lexicon_misses_is_still_blocked() -> None:
    # A tool error (or a careless model) leaves nothing booked, but the model's prose still tells the
    # prospect the call is on, in a paraphrase the lexicon must catch. The claimed time is one that really
    # was offered, so offer grounding alone (fail_closed) cannot catch it either: only the claim check can.
    reply = "I've got you down for Tuesday 6 October, 3:00 PM. Talk soon!"
    result = check(reply, facts=[], declared=[], offers=[TUE])
    assert not result.ok
    assert problems(result) == [("booked", "no_entry")]


# Offer grounding --------------------------------------------------------------------------------------------


def test_offers_must_come_from_the_reference_when_one_is_given() -> None:
    offer = render.offer_text(["Tuesday 6 October, 3:00 PM", "Wednesday 7 October, 10:00 AM"], BERLIN)
    listed = [TUE, datetime(2026, 10, 7, 8, 0, tzinfo=UTC)]
    assert check(offer, offers=listed).ok
    result = check(offer, offers=listed[:1])
    assert problems(result) == [("offered", "not_offered")]
    assert "Wednesday 7 October, 10:00 AM" in result.summary()
    assert check(offer, offers=None).ok


def test_ledger_starts_and_windows_are_not_invented_offers() -> None:
    reply = "Your call is on Tuesday 6 October, 3:00 PM. Our hours are 9:00-17:00, any time until 6 pm."
    assert check(reply, [BOOKED], offers=[]).ok


def test_declared_offers_are_grounded_too() -> None:
    result = check(
        "How about one of these?", declared=[("offered", "Friday 9 October, 11:00 AM")], offers=[TUE]
    )
    assert problems(result) == [("offered", "not_offered")]


# The repair note --------------------------------------------------------------------------------------------


def test_the_guard_note_says_what_was_wrong_and_what_the_ledger_shows() -> None:
    garbled = f"You're booked for {render.slot_label(TUE, NY)} ({BERLIN})."
    result = check(garbled, [BOOKED, CANCELLED])
    note = guard_note(result, [BOOKED, CANCELLED], zone=BERLIN, draft=garbled)
    assert note.startswith("[claim check]")
    assert "- booked: Tuesday, 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00), reference uid-book" in note
    assert "- cancelled: Tuesday, 6 October 2026, 3:00 PM Europe/Berlin" in note
    assert note.endswith(f"Rejected reply:\n{garbled}")
    assert result.violations[0].detail in note


def test_an_empty_ledger_is_named_in_the_note() -> None:
    result = check("You're booked!")
    assert "nothing has been booked, moved or cancelled" in guard_note(result, [], zone=BERLIN, draft="x")
    assert ledger_lines([LedgerFact("booked", "uid", TUE, current=False)], BERLIN) == []


def test_the_note_tells_the_model_to_answer_directly_and_not_call_a_tool() -> None:
    """The repair call still offers the tools (`test_a_repair_that_fixes_the_reply_is_sent`), but the
    ledger above already has everything the model needs to correct its reply; a model that instead calls a
    tool here gets no second chance (`agent/core.py::_repair`), so the note must say not to."""
    result = check("You're booked!")
    note = guard_note(result, [BOOKED], zone=BERLIN, draft="You're booked!")
    assert "do not call a tool" in note.lower()

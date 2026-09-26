"""The text of ``rendered_confirmation``: the confirmation line and the sentence code puts under it read back
to the exact ledgered instant, for the agent's own claim check and for the grader that scores the prospect's
belief, whatever zone the reader assumes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from booking_truth.agent import render
from booking_truth.agent.guards.claim_check import LedgerFact, check_reply, guard_note
from booking_truth.harness.lexicon_extractor import extract_belief

# Thursday 1 October 2026, 08:00 in New York.
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
HOST = "America/New_York"
REF = "abcdef123456"
ACTIONS = ("booked", "rescheduled", "cancelled")
ZONES = [
    HOST,
    "Europe/Berlin",
    "Europe/London",
    "Asia/Kolkata",
    "Asia/Kathmandu",
    "Australia/Sydney",
    "America/Chicago",
    "America/Los_Angeles",
    "America/St_Johns",
    "Pacific/Auckland",
    "UTC",
]
INSTANTS = st.integers(min_value=1, max_value=400 * 24 * 2).map(lambda n: NOW + timedelta(minutes=30 * n))


def repeated_hour(start: datetime, zone: str) -> bool:
    """Whether the local time of ``start`` happens twice in ``zone`` (the hour the clocks go back)."""
    wall = start.astimezone(ZoneInfo(zone)).replace(tzinfo=None)
    first, second = (wall.replace(tzinfo=ZoneInfo(zone), fold=fold) for fold in (0, 1))
    return first.utcoffset() != second.utcoffset()


def rendered(action: str, start: datetime, zone: str) -> str:
    """The reply code sends when it writes the rest: the line, then the sentence under it."""
    line = render.confirmation_line(action, start, zone, REF)
    return f"{line}\n\n{render.confirmation_follow_up(action)}"


def test_the_lines_start_with_a_word_readers_can_parse() -> None:
    start = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
    prefixes = [render.confirmation_line(a, start, "Europe/Berlin", REF).split(" ", 1)[0] for a in ACTIONS]
    assert prefixes == ["Booked:", "Rescheduled:", "Cancelled:"]


def test_the_sentence_under_a_line_states_no_time() -> None:
    assert render.confirmation_follow_up("booked") == "The calendar invite is on its way to your email."
    assert render.confirmation_follow_up("rescheduled") == "The calendar invite is updated with the new time."
    assert render.confirmation_follow_up("cancelled") == "Nothing is booked for you now."
    with pytest.raises(ValueError, match="unknown booking action"):
        render.confirmation_follow_up("moved")


@settings(max_examples=150, deadline=None)
@given(
    start=INSTANTS,
    action=st.sampled_from(ACTIONS),
    zone=st.sampled_from(ZONES),
    reader=st.sampled_from(ZONES),
)
def test_the_grader_reads_a_rendered_reply_as_the_ledgered_write(
    start: datetime, action: str, zone: str, reader: str
) -> None:
    # The grader reads a local time that happens twice as its first instant, whatever UTC offset follows it.
    assume(not repeated_hour(start, zone))
    belief = extract_belief(
        [rendered(action, start, zone)], prospect_zone=reader, host_zone=HOST, reference=NOW
    )
    assert belief.status == action
    assert belief.time_utc == start
    assert belief.offered_utc == ()


@settings(max_examples=150, deadline=None)
@given(
    start=INSTANTS,
    action=st.sampled_from(ACTIONS),
    zone=st.sampled_from(ZONES),
    reader=st.sampled_from(ZONES),
)
def test_the_claim_check_accepts_a_rendered_reply_only_for_its_own_entry(
    start: datetime, action: str, zone: str, reader: str
) -> None:
    reply = rendered(action, start, zone)
    assert check_reply(reply, [], [LedgerFact(action, REF, start)], zone=reader, now=NOW, host_zone=HOST).ok
    moved = LedgerFact(action, REF, start + timedelta(minutes=30))
    result = check_reply(reply, [], [moved], zone=reader, now=NOW, host_zone=HOST)
    assert result.violations
    assert {v.problem for v in result.violations} == {"wrong_time"}


@pytest.mark.parametrize("action", ["booked", "rescheduled"])
def test_the_sentence_alone_still_needs_a_ledger_entry(action: str) -> None:
    """Code sends the sentence without its line only when the guard is off; it is still a claim the check
    holds to the ledger."""
    start = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
    sentence = render.confirmation_follow_up(action)
    declared = [(action, render.slot_label(start, HOST))]
    assert check_reply(sentence, declared, [LedgerFact(action, REF, start)], zone=HOST, now=NOW).ok
    assert not check_reply(sentence, declared, [], zone=HOST, now=NOW).ok


def test_the_repair_note_lists_the_lines_the_prospect_sees() -> None:
    start = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
    facts = [LedgerFact("booked", REF, start)]
    draft = "You're booked for Tuesday 6 October, 9:00 AM (Europe/Berlin)."
    check = check_reply(draft, [], facts, zone="Europe/Berlin", now=NOW)
    assert not check.ok
    line = render.confirmation_line("booked", start, "Europe/Berlin", REF)
    note = guard_note(check, facts, zone="Europe/Berlin", draft=draft, shown=[line])
    assert (
        "The prospect already sees this confirmation, written by code, above your reply, so you do not need "
        f"to repeat its time or reference:\n- {line}\n\n"
    ) in note
    assert note.index(line) < note.index("Rejected reply:")
    plain = guard_note(check, facts, zone="Europe/Berlin", draft=draft)
    assert "already sees" not in plain

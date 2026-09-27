"""The claim check of ``claim_ledger``: no reply claims a booking, a reschedule or a cancellation that the
ledger of verified calendar writes does not show.

The claims checked are the union of the model's declared claims (the ``claims`` field of its final answer)
and the claims :mod:`~booking_truth.agent.guards.lexicon` finds in the reply text. Each success claim needs a
verified ledger entry of its kind that is still current for the lead:

- ``booked``: a live booking (booked or rescheduled, not voided by a later cancel or reschedule);
- ``rescheduled``: a live rescheduled booking;
- ``cancelled``: a verified cancellation.

When the claim states a time (read by :mod:`~booking_truth.agent.guards.timeparse` in the lead's zone, or in
the zone the text names), one of those entries must start at exactly that minute.

Offer grounding (``fail_closed``) is the same check for offers: when a reference set of starts is given
(the lead's latest successful slot list), every specific time in the reply must be one of them or the start
of a ledger entry.

The caller decides what a violation costs: the agent core makes one repair call with :func:`guard_note`, then
sends a safe template.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from booking_truth.agent import render
from booking_truth.agent.guards.lexicon import CLAIM_KINDS, DetectedClaim, detect_claims
from booking_truth.agent.guards.timeparse import StatedTime, find_times

Problem = Literal["no_entry", "wrong_time", "not_offered"]
ClaimSource = Literal["declared", "detected"]

_NOUN = {"booked": "booking", "rescheduled": "reschedule", "cancelled": "cancellation", "offered": "offer"}
_VERB = {"booked": "booked", "rescheduled": "moved", "cancelled": "cancelled", "offered": "offered"}
#: A time right after these words is the old time of a reschedule ("moved from Tuesday 3 PM to ...").
_OLD_TIME = re.compile(r"\b(?:from|was|were|instead of|previously|originally)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class LedgerFact:
    """A verified ledger entry, as the claim check sees it. ``current``: a booking not voided since."""

    action: str
    ref: str
    start: datetime
    current: bool = True


@dataclass(frozen=True)
class Claim:
    """One claim of a reply: ``kind`` is booked, rescheduled, cancelled or offered; ``text`` is what it says
    (the declared time, or the detected part of the sentence); ``time`` the time it states, if any."""

    kind: str
    source: ClaimSource
    text: str
    time: StatedTime | None = None


@dataclass(frozen=True)
class Violation:
    claim: Claim
    problem: Problem
    detail: str


@dataclass(frozen=True)
class CheckResult:
    claims: tuple[Claim, ...]
    violations: tuple[Violation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def kinds(self) -> frozenset[str]:
        """The kinds of the violating claims."""
        return frozenset(v.claim.kind for v in self.violations)

    def summary(self) -> str:
        return "; ".join(v.detail for v in self.violations)


# Reading claims ---------------------------------------------------------------------------------------------


def _claim_time(claim: DetectedClaim, times: Sequence[StatedTime]) -> StatedTime | None:
    """The time a detected claim states: the first one in its part of the sentence from its clause on,
    else the last one before it; the old time of a reschedule does not count."""
    text = claim.sentence.text
    owned = [
        t
        for t in times
        if t.end > claim.scope_start
        and t.start < claim.scope_end
        and not t.in_range
        and not (claim.kind == "rescheduled" and _OLD_TIME.search(text, 0, t.start))
    ]
    pivot = max(claim.scope_start, claim.clause_start)
    after = [t for t in owned if t.end > pivot]
    if after:
        return after[0]
    return owned[-1] if owned else None


def read_claims(
    reply: str,
    declared: Iterable[tuple[str, str]],
    *,
    zone: str,
    now: datetime,
    host_zone: str | None = None,
) -> list[Claim]:
    """The declared claims (``(type, time)`` pairs) and the detected claims of a reply."""
    claims: list[Claim] = []
    for kind, when in declared:
        if kind not in (*CLAIM_KINDS, "offered"):
            continue
        times = [t for t in find_times(when or "", zone=zone, now=now, host_zone=host_zone) if not t.in_range]
        claims.append(Claim(kind, "declared", when or "", times[0] if times else None))
    for detected in detect_claims(reply):
        times = find_times(detected.sentence.text, zone=zone, now=now, host_zone=host_zone)
        claims.append(Claim(detected.kind, "detected", detected.scope.strip(), _claim_time(detected, times)))
    return claims


# Checking ---------------------------------------------------------------------------------------------------


def supporting(kind: str, facts: Iterable[LedgerFact]) -> list[LedgerFact]:
    """The ledger entries that can support a claim of ``kind``."""
    if kind == "booked":
        return [f for f in facts if f.action in ("booked", "rescheduled") and f.current]
    if kind == "rescheduled":
        return [f for f in facts if f.action == "rescheduled" and f.current]
    if kind == "cancelled":
        return [f for f in facts if f.action == "cancelled"]
    return []


def _fact_label(fact: LedgerFact, zone: str) -> str:
    return render.long_label(fact.start, zone)


def _check_claim(claim: Claim, facts: Sequence[LedgerFact], zone: str) -> Violation | None:
    support = supporting(claim.kind, facts)
    said = f'"{claim.text}"' if claim.text else "with no time"
    if not support:
        return Violation(
            claim,
            "no_entry",
            f"the reply says the call is {_VERB[claim.kind]} ({said}), but the ledger has no verified "
            f"{_NOUN[claim.kind]}",
        )
    if claim.time is not None and not any(claim.time.matches(f.start, zone) for f in support):
        verified = " or ".join(_fact_label(f, zone) for f in support)
        return Violation(
            claim,
            "wrong_time",
            f'the reply states "{claim.time.text.strip()}" for the {_NOUN[claim.kind]}, but the verified '
            f"{_NOUN[claim.kind]} is {verified}",
        )
    return None


def _offer_violations(
    reply: str,
    claims: Sequence[Claim],
    reference: Sequence[datetime],
    *,
    zone: str,
    now: datetime,
    host_zone: str | None,
) -> list[Violation]:
    stated = [t for t in find_times(reply, zone=zone, now=now, host_zone=host_zone) if t.specific]
    stated += [c.time for c in claims if c.kind == "offered" and c.time is not None and c.time.specific]
    found: list[Violation] = []
    seen: set[str] = set()
    for time in stated:
        if time.in_range or any(time.matches(start, zone) for start in reference):
            continue
        text = time.text.strip()
        if text in seen:
            continue
        seen.add(text)
        claim = Claim("offered", "detected", text, time)
        detail = f'the reply offers "{text}", which is not in the latest availability'
        found.append(Violation(claim, "not_offered", detail))
    return found


def check_reply(
    reply: str,
    declared: Iterable[tuple[str, str]],
    facts: Sequence[LedgerFact],
    *,
    zone: str,
    now: datetime,
    host_zone: str | None = None,
    offer_reference: Sequence[datetime] | None = None,
) -> CheckResult:
    """Check a reply's claims against the ledger ``facts`` of its lead.

    ``zone`` is the lead's zone (times with no stated zone are read in it). ``offer_reference``, when given,
    turns on offer grounding against those starts (plus every ledger start).
    """
    claims = read_claims(reply, declared, zone=zone, now=now, host_zone=host_zone)
    violations = [
        v for c in claims if c.kind in CLAIM_KINDS if (v := _check_claim(c, facts, zone)) is not None
    ]
    if offer_reference is not None:
        reference = [*offer_reference, *(f.start for f in facts)]
        violations += _offer_violations(reply, claims, reference, zone=zone, now=now, host_zone=host_zone)
    return CheckResult(tuple(claims), tuple(violations))


# The repair note --------------------------------------------------------------------------------------------


def ledger_lines(facts: Sequence[LedgerFact], zone: str) -> list[str]:
    lines = []
    for fact in facts:
        if fact.action == "cancelled":
            what = "cancelled"
        elif not fact.current:
            continue
        else:
            what = "moved to" if fact.action == "rescheduled" else "booked"
        lines.append(f"- {what}: {_fact_label(fact, zone)}, reference {render.reference(fact.ref)}")
    return lines


def guard_note(
    result: CheckResult,
    facts: Sequence[LedgerFact],
    *,
    zone: str,
    draft: str,
    shown: Sequence[str] = (),
) -> str:
    """The note for the one repair call: what was wrong, what the ledger says, the rejected draft.
    ``shown``: the code-rendered confirmation lines the prospect sees above the reply
    (``rendered_confirmation``)."""
    problems = "\n".join(f"- {v.detail}" for v in result.violations)
    ledger = "\n".join(ledger_lines(facts, zone)) or "- nothing has been booked, moved or cancelled"
    above = ""
    if shown:
        lines = "\n".join(f"- {line}" for line in shown)
        above = (
            "The prospect already sees this confirmation, written by code, above your reply, so you do not "
            f"need to repeat its time or reference:\n{lines}\n\n"
        )
    return (
        "[claim check] Your reply was not sent, because it claims something the calendar ledger does not "
        f"show:\n{problems}\n\nVerified calendar ledger for this prospect:\n{ledger}\n\n{above}"
        "Answer directly with the corrected JSON object now; do not call a tool. Everything you need is "
        'already above.\n\nWrite the reply again as the same JSON object {"reply": "...", "claims": [...]}. '
        "Say that a call is booked, moved or cancelled only when the ledger shows it, with the ledger's time "
        "and zone; offer only times from the latest find_slots result. Otherwise say plainly what happened "
        f"and offer the next step.\n\nRejected reply:\n{draft}"
    )


__all__ = [
    "CheckResult",
    "Claim",
    "LedgerFact",
    "Violation",
    "check_reply",
    "guard_note",
    "ledger_lines",
    "read_claims",
    "supporting",
]

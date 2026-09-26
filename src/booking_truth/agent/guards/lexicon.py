"""Deterministic detector of success claims in the agent's replies (``claim_ledger``).

The claim check takes the union of the claims the model declares in its final answer and the claims this
detector finds in the reply text, so a model that says "you're all set" without declaring anything is still
checked. The harness has its own belief extractor with its own patterns (ADR 0008); nothing here is shared
with it.

A claim is a statement that an action was completed:

- ``booked``: "you're booked", "you're all set", "I've scheduled it", "your call is confirmed", "see you
  Tuesday", "the invite is on its way", "looking forward to our call", "I've got you down for Tuesday",
  "you're penciled in", "you're on the books";
- ``rescheduled``: "I've moved it", "your call has been rescheduled", "your call is now on Thursday";
- ``cancelled``: "I've cancelled it", "the meeting was dropped", "your call on Tuesday is cancelled".

Statements that deny, doubt, offer, promise or ask are not claims ("nothing is booked yet", "I couldn't book
it", "shall I book it?", "once you confirm you'll be booked", "I'll move it now"), and neither are mentions of
a booking that already exists ("you already have a call booked for Tuesday", "your call is still booked").
Hedged claims ("it should be booked now") are claims: a hedge does not make an unverified booking true.

A reply is split into sentences and each sentence into clauses. A claim belongs to its clause; the time it
states is looked up by the claim check in the part of the sentence the claim owns (from the sentence start or
the previous claim to the next claim's clause).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from booking_truth.agent.guards.timeparse import normalize

ClaimKind = Literal["booked", "rescheduled", "cancelled"]
CLAIM_KINDS: tuple[ClaimKind, ...] = ("booked", "rescheduled", "cancelled")

_I = re.IGNORECASE
_WEEKDAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_THING = r"(?:call|booking|meeting|appointment|demo|slot|reservation|session|chat|intro call|time)"
#: Adverbs between a subject and the verb of an action: "I've just booked", "it has now been moved".
_ACT = r"(?:now\s+|just\s+|already\s+|successfully\s+|officially\s+|gone ahead and\s+|went ahead and\s+)*"
#: The same for a state ("you're now booked"); "you're already booked" is a mention of an existing booking.
_STATE = r"(?:now\s+|officially\s+)*"
_WE = r"\b(?:i|we)(?:'ve| have)?\s+"
_ABLE = r"\b(?:i|we)\s+(?:managed to|was able to|were able to|have managed to|'ve managed to)\s+"
_IS = r"(?:\s+(?:is|has been|was|are|have been|were)|'s)"

_PATTERNS: dict[ClaimKind, tuple[re.Pattern[str], ...]] = {
    "rescheduled": tuple(
        re.compile(p, _I)
        for p in (
            _WE + _ACT + r"(?:moved|rescheduled|shifted|pushed|postponed|brought (?:it|\w+ \w+) forward)\b",
            _WE
            + _ACT
            + r"(?:changed|updated|switched)\s+(?:your |the |our )?(?:call|booking|meeting|appointment|slot"
            r"|time|date)\b(?!\s*zone)",
            _ABLE + r"(?:move|reschedule|shift|push)\b",
            rf"\b(?:{_THING}|it|that){_IS}\s+{_STATE}(?:moved|rescheduled|shifted|pushed|postponed)\b",
            rf"\b(?:your|the)\s+(?:\w+\s+)?{_THING}\s+is\s+now\s+(?:on|at|for|set for|scheduled for"
            r"|booked for)\b",
            r"\bnew (?:time|slot|date) is\b",
            r"\brescheduled\b",
        )
    ),
    "cancelled": tuple(
        re.compile(p, _I)
        for p in (
            _WE + _ACT + r"(?:cancell?ed|dropped|called off|removed|deleted)\b",
            _ABLE + r"cancel\b",
            rf"\b(?:{_THING}|it|that|this){_IS}\s+{_STATE}(?:cancell?ed|dropped|called off|removed"
            r"|deleted)\b",
            r"\bcancell?ed\b",
        )
    ),
    "booked": tuple(
        re.compile(p, _I)
        for p in (
            r"\b(?:you(?:'re| are)|you(?:'ve| have) been|we(?:'re| are)|it(?:'s| is| has been)"
            r"|that(?:'s| is| has been)|this(?:'s| is| has been)|everything(?:'s| is))\s+"
            + _STATE
            + r"(?:all\s+)?(?:booked|confirmed|scheduled|reserved|locked in|set|good to go|on the calendar"
            r"|in the calendar)\b",
            rf"\b(?:your|the|our)\s+(?:\w+\s+){{0,2}}?{_THING}{_IS}\s+{_STATE}(?:booked|confirmed|scheduled"
            r"|reserved|locked in|set(?: up)?|on the calendar|in the calendar|created|made)\b",
            _WE + _ACT + r"(?:booked|scheduled|reserved|confirmed|locked in|secured|set up|put you down"
            r"|added (?:you|it|the|your))\b",
            _ABLE + r"(?:book|schedule|reserve|secure|lock in|set up)\b",
            r"\bbooked (?:you|it|that|this)\b",
            r"\bgot (?:you|it|that) down\b",
            rf"\b(?:i|we)(?:'ve| have)?\s+you\s+{_STATE}(?:booked|scheduled|confirmed|down|set up"
            r"|locked in)\b",
            r"\b(?:you(?:'re| are)|it(?:'s| is)|that(?:'s| is))\s+pencill?ed in\b",
            _WE + _ACT + r"pencill?ed (?:you |it |that )?in\b",
            r"\b(?:you(?:'re| are)|it(?:'s| is)|that(?:'s| is))\s+on the books\b",
            r"\ball set\b",
            r"\bgood to go\b",
            rf"\bsee you (?:then|there|on\b|at\b|next\b|this\b|tomorrow|today|{_WEEKDAYS})",
            rf"\b(?:talk|speak|chat) (?:to|with) you (?:then|on\b|at\b|next\b|this\b|tomorrow|{_WEEKDAYS})",
            r"\blooking forward to (?:our|the|your) (?:call|meeting|chat|demo|conversation)\b",
            r"\binvit(?:e|ation)s?\b.{0,40}?\b(?:on (?:its|the|their) way|sent|in your inbox|went out"
            r"|gone out|out to you|coming your way|heading your way)\b",
            r"\bsent (?:you )?(?:a |an |the )?(?:calendar )?invit(?:e|ation)\b",
            r"\b(?:booking|reservation|request|it|that)\s+(?:went|has gone|has now gone|got) through\b",
            r"\bbooked\b",
            r"^\s*(?:confirmed|scheduled)\b",
        )
    ),
}

#: A denial or a failure before the claim: "I haven't booked", "nothing is booked", "I couldn't book".
_NEGATION = re.compile(
    r"n't\b|\bnot\b|\bnever\b|\bnothing\b|\bno longer\b|\bunable\b|\bfailed\b|\bcannot\b|\bwithout\b"
    r"|\bneither\b|\bnor\b|\bno (?:\w+ )?(?:booking|call|meeting|appointment|reservation|slot|time"
    r"|change)s?\b",
    _I,
)
#: A promise, an offer, a condition or a wish before the claim: "I'll book", "shall I", "once it's booked".
_MODAL = re.compile(
    r"\b(?:i'll|i will|we'll|we will|you'll|it'll|that'll|you will|it will|will|i'm going to|going to|let me"
    r"|let's|i can|we can|i could|i'd|i would|shall|would|could|can i|can you|once|if|when|as soon as"
    r"|after|before|unless|until|ready to|happy to|want to|wants to|like to|like me to|want me to|to be"
    r"|to get|about to|trying to|try to|need to|in order to|i'm booking|i am booking|booking it now)\b",
    _I,
)
#: A booking that already exists, not an action: "you already have a call booked", "it's still booked".
_REFERENCE = re.compile(
    r"\balready\b|\bstill\b|\bcurrently\b|\byou (?:currently |already )?have (?:a|an|one)\b"
    r"|\byou've got (?:a|an|one)\b|\bthere(?:'s| is) (?:a|an|already)\b",
    _I,
)
_THIRD_PARTY = re.compile(r"^\s*by (?:someone|somebody|another|a different|other)", _I)
_LATER_CONDITION = re.compile(r"\b(?:once|as soon as|after|when|if) (?:you|it's|it is|that's)\b", _I)
_NOT_A_BOOKING = re.compile(
    r"^\s+(?:that\s+)?(?:your|the)\s+(?:time ?zone|zone|email|name|details|location|city|address"
    r"|availability)",
    _I,
)
_SUBORDINATE = re.compile(
    r"^(?:(?:and|so|then|also)\s+)?(?:if|once|when|whenever|as soon as|after|before|until|unless"
    r"|in case)\b",
    _I,
)
_CLAUSE_BREAK = re.compile(
    r"\s*(?:;|:\s|\s-\s|,\s+|\bbut\b|\bhowever\b|\b(?:and|so|then)\s+(?=(?:i|i've|i'll|i'm|you|you're|you've"
    r"|we|we've|it|it's|your|the|that|this|there)\b))\s*",
    _I,
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z])|\s*\n+\s*")


@dataclass(frozen=True)
class Sentence:
    start: int
    text: str


@dataclass(frozen=True)
class DetectedClaim:
    """A claim found in a reply. Offsets are within ``sentence.text``: the claim's words start at ``anchor``
    in a clause that starts at ``clause_start``; the part of the sentence the claim owns (where its time is
    looked up) is ``scope_start..scope_end``."""

    kind: ClaimKind
    phrase: str
    sentence: Sentence
    clause_start: int
    anchor: int
    scope_start: int
    scope_end: int

    @property
    def scope(self) -> str:
        return self.sentence.text[self.scope_start : self.scope_end]


def sentences(text: str) -> list[Sentence]:
    """The sentences of ``text`` (after :func:`~booking_truth.agent.guards.timeparse.normalize`)."""
    found: list[Sentence] = []
    position = 0
    for match in [*_SENTENCE_BREAK.finditer(text), None]:
        end = match.start() if match is not None else len(text)
        chunk = text[position:end]
        if chunk.strip():
            lead = len(chunk) - len(chunk.lstrip())
            found.append(Sentence(position + lead, chunk.strip()))
        if match is not None:
            position = match.end()
    return found


def _clauses(sentence: str) -> Iterator[tuple[int, str]]:
    position = 0
    for match in [*_CLAUSE_BREAK.finditer(sentence), None]:
        end = match.start() if match is not None else len(sentence)
        if end > position:
            yield position, sentence[position:end]
        if match is not None:
            position = match.end()


def _is_claim(clause: str, match: re.Match[str]) -> bool:
    head, tail = clause[: match.end()], clause[match.end() :]
    prefix = clause[: match.start()]
    if _NEGATION.search(head) or _MODAL.search(head) or _REFERENCE.search(prefix):
        return False
    if _THIRD_PARTY.search(tail) or _LATER_CONDITION.search(tail) or _NOT_A_BOOKING.search(tail):
        return False
    return not clause.rstrip().endswith("?")


def _clause_claims(clause: str) -> list[tuple[ClaimKind, re.Match[str]]]:
    found: list[tuple[ClaimKind, re.Match[str]]] = []
    for kind in CLAIM_KINDS:
        for pattern in _PATTERNS[kind]:
            match = next((m for m in pattern.finditer(clause) if _is_claim(clause, m)), None)
            if match is not None:
                found.append((kind, match))
                break
    return found


def detect_claims(text: str) -> list[DetectedClaim]:
    """Every success claim in ``text``, in order. At most one claim of each kind per clause."""
    text = normalize(text)
    claims: list[DetectedClaim] = []
    for sentence in sentences(text):
        found: list[tuple[int, int, ClaimKind, str]] = []
        conditional = False
        for start, clause in _clauses(sentence.text):
            if _SUBORDINATE.match(clause):
                conditional = True
                continue
            if conditional:
                conditional = False
                continue
            for kind, match in _clause_claims(clause):
                found.append((start, start + match.start(), kind, match.group(0).strip()))
        found.sort(key=lambda item: item[1])
        for index, (clause_start, anchor, kind, phrase) in enumerate(found):
            scope_start = 0
            if index > 0:
                scope_start = clause_start if found[index - 1][0] != clause_start else anchor
            scope_end = len(sentence.text)
            if index + 1 < len(found):
                following = found[index + 1]
                scope_end = following[0] if following[0] != clause_start else following[1]
            claims.append(DetectedClaim(kind, phrase, sentence, clause_start, anchor, scope_start, scope_end))
    return claims


__all__ = ["CLAIM_KINDS", "ClaimKind", "DetectedClaim", "Sentence", "detect_claims", "sentences"]

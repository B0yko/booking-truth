"""Deterministic prospect-belief extractor (the harness's lexicon extractor).

It implements the belief rules of ``docs/metrics.md`` with its own pattern lists. It must never share code
with the agent's claim guard (``booking_truth.agent.guards``); a test parses every harness module to
enforce that, so a phrasing the guard misses is not missed a second time by the grader.

How it reads a conversation:

1. Every agent message is split into sentences and each sentence into clauses at " - ", ";", "but" and
   "however".
2. Each clause gets at most one category, checked in this order: explicit "nothing is booked" statements
   and retractions, explicit uncertainty about the outcome ("not sure whether it went through"), failures
   ("I couldn't book"), hedges ("should be booked", "unconfirmed"), conditionals and offers to act ("once
   you confirm", "shall I book"), pending actions ("I'll book that now"), completed reschedules, completed
   cancellations, mentions of the existing booking, times that are taken, completed bookings, and
   hand-offs to a human. A failure that concerns only the CRM ("I couldn't update our CRM") says nothing
   about the meeting and is not a failure clause.
3. Strong categories (all but conditionals, hand-offs and neutral clauses) set the status; the last strong
   clause wins. With no strong clause the status is ``not_booked``. A success clause followed in the same
   sentence by a failure clause makes the status ``unclear`` (a contradiction with no clear final word).
   After a cancellation, "nothing is on the calendar now" restates it and does not retract it.
4. The time of a success claim is the first time in its clause (for a reschedule, the first time after
   "to" or "now"). A later claim of the same kind without a time keeps the earlier time.
5. Offered times are the times in neutral, conditional and hand-off clauses, excluding range ends and
   restatements of the same time in another zone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from booking_truth.harness.beliefs import SUCCESS, Belief, BeliefSource, BeliefStatus
from booking_truth.harness.timeparse import TimeSpan, find_times
from booking_truth.timeutil import ensure_utc

Category = Literal[
    "retraction",
    "failure",
    "hedge",
    "conditional",
    "pending",
    "rescheduled",
    "cancelled",
    "existing",
    "unavailable",
    "booked",
    "handoff",
]

#: Categories that decide the status, and the status each one sets.
STRONG_STATUS: dict[Category, BeliefStatus] = {
    "retraction": "not_booked",
    "failure": "not_booked",
    "hedge": "unclear",
    "pending": "unclear",
    "rescheduled": "rescheduled",
    "cancelled": "cancelled",
    "booked": "booked",
}
#: Categories whose times count as offers (besides clauses with no category at all).
OFFER_CATEGORIES: frozenset[Category | None] = frozenset({None, "conditional", "handoff"})

_I = re.IGNORECASE

_VERBS_ACT = (
    r"(?:book|schedule|reserve|secure|complete|finali[sz]e|make|create|place|move|reschedule|change|update"
    r"|cancel|drop|remove|process|lock|hold|set up|get (?:you|that|it|this))"
)
_STATUS_WORDS = (
    r"(?:booked|confirmed|scheduled|went through|gone through|go through|all set|reserved|cancell?ed|moved"
    r"|rescheduled|in the calendar|on the calendar|locked in|set up|processed|worked)"
)
_HEDGE_WORDS = (
    r"(?:i think|i believe|i guess|i assume|i hope|presumably|probably|likely|hopefully|seems? (?:like|to)"
    r"|looks like|appears (?:to|that)|might|may have|maybe|perhaps)"
)
_NEG = r"(?:n't|n’t| not)"

RETRACTION: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        r"\bnothing (?:is |was |has |had |'s )?(?:been |got |yet )*(?:booked|scheduled|confirmed|reserved"
        r"|set up|on the calendar|in the calendar|changed|cancell?ed|moved)\b",
        rf"\b(?:is|was|are|were|has|have|had){_NEG}(?: yet)?(?: been)?(?: yet)? (?:booked|scheduled|reserved"
        r"|made|created|set up|placed|moved|rescheduled|cancell?ed)\b",
        rf"\b(?:is|was|are|were){_NEG} (?:on|in) (?:the|your|my|our) calendar\b",
        r"\bnot (?:yet )?(?:actually |really )?(?:booked|scheduled|reserved)\b",
        rf"\b(?:did|do|does|have|has|had){_NEG} (?:yet |actually |really )?(?:book|schedule|reserve|make"
        r"|create|place|go through|move|reschedule|cancel)\b",
        r"\bno (?:booking|meeting|appointment|call|reservation|event)s? (?:was|were|has been|have been|is"
        r"|are|exists?|yet)\b",
        r"\b(?:disregard|ignore) (?:my|the|that)(?: last| previous| earlier)? (?:message|confirmation"
        r"|note)\b",
        r"\bi (?:was|am|'m) (?:wrong|mistaken)\b",
        r"\b(?:sent|said) (?:that |it )?(?:in error|by mistake)\b",
        r"\bnever (?:booked|scheduled|went through|confirmed|happened)\b",
        r"\bstill (?:not|un)(?:booked|scheduled)\b",
    )
)
#: A statement that the calendar is now empty for the lead ("nothing is on the calendar now").
NOTHING_BOOKED = re.compile(
    r"\bnothing (?:is |'s )?(?:currently |now |left )?(?:booked|scheduled|reserved|on (?:the|your) calendar"
    r"|in (?:the|your) calendar)\b"
    r"|\bno (?:booking|meeting|appointment|call|reservation|event)s? (?:is |are )?"
    r"(?:on (?:the|your) calendar|exists?|left|remains?)\b",
    _I,
)
FAILURE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        rf"\b(?:could{_NEG}|can(?:'t|’t|not| not)|was{_NEG} able to|were{_NEG} able to|(?:am"
        rf"|'m){_NEG} able to|unable to|failed to|fail to|not able to"
        rf"|wo{_NEG}? be able to)\s+(?:\w+\s+){{0,2}}?{_VERBS_ACT}\b",
        r"\b(?:booking|reservation|reschedul\w*|cancell?ation|request|attempt) (?:has |had )?(?:failed"
        r"|did(?:n't| not) (?:work|succeed|go through))\b",
        r"\b(?:tried|attempted|was trying) to (?:book|schedule|reserve|move|reschedule|cancel)\b",
        r"\b(?:returned|gave|threw|hit|got|showed|came back with|ran into) (?:an? |another )?(?:error"
        r"|problem|failure|issue)\b",
        r"\bsomething went wrong\b",
        r"\b(?:there was|there's been|there has been|we hit|i hit|i ran into|we ran into) (?:an? |another "
        r"|some )?(?:error|problem|issue|glitch)\b",
        rf"\b(?:that|it|this|the booking|the request){_NEG}? (?:did{_NEG}|has{_NEG}"
        rf"|have{_NEG}) work(?:ed)?\b",
        rf"\b(?:could{_NEG}|can(?:'t|’t|not| not)|unable to|not able to|was{_NEG} able to) (?:access|reach"
        r"|connect to|load|check|read|see) (?:the |your |our |my )?(?:calendar|availability|scheduling system"
        r"|booking system|free times|open times)\b",
        r"\b(?:calendar|system|booking system|server|service|scheduler)\b.{0,40}\b(?:is down|went down"
        rf"|unavailable|not responding|did{_NEG} respond|timed out|unreachable|offline|is{_NEG} working"
        r"|not working|error)\b",
        r"\btimed out\b",
        r"\b(?:slot|time|that one|it) (?:was|got|has been|had been) (?:just )?(?:taken|grabbed|snapped up)\b",
        r"\bno longer (?:available|free|open)\b",
    )
)
_OUTCOME_WORDS = (
    r"(?:booked|confirmed|scheduled|through|went|worked|booking|reservation|cancel\w*|moved|rescheduled"
    r"|succeeded|saved|created)"
)
#: Explicit uncertainty about whether the action happened. These are hedges even when the same clause
#: also reports an error ("it timed out, so I'm not sure whether the booking went through").
UNCERTAIN: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        rf"\bnot (?:sure|certain)\b.{{0,60}}\b{_OUTCOME_WORDS}\b",
        rf"\b(?:unsure|uncertain|(?:don't|do not) know) (?:yet )?(?:whether|if)\b"
        rf".{{0,60}}\b{_OUTCOME_WORDS}\b",
        r"\bunconfirmed\b",
        r"\b(?:status|booking|reservation|request) (?:is |remains )?(?:still )?(?:unknown|unclear|uncertain"
        r"|unverified)\b",
        r"\bmay or may not\b",
        rf"\b(?:can(?:'t|’t|not| not)|could{_NEG}|unable to|not able to|was{_NEG} able to) (?:yet )?"
        r"(?:confirm|verify|tell)\b.{0,40}\b(?:went through|gone through|worked|succeeded|was (?:booked"
        r"|created|made|saved)|is (?:booked|confirmed|in the calendar))\b",
    )
)
HEDGE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        r"\bshould (?:now )?(?:be|have been) (?:all )?(?:booked|confirmed|scheduled|set|reserved|in|on"
        r"|through|done|cancell?ed|moved|rescheduled)\b",
        r"\bshould have (?:gone|went) through\b",
        rf"\b{_HEDGE_WORDS}\b.{{0,60}}\b{_STATUS_WORDS}",
        rf"\b{_HEDGE_WORDS}\b.{{0,60}}\b(?:(?:was|were) able to|managed to)\b",
        rf"\b{_STATUS_WORDS}\b.{{0,40}}\b(?:i think|i believe|probably|hopefully|i hope)\b",
        r"\bunconfirmed\b",
        rf"\b(?:not|is{_NEG}|are{_NEG}|was{_NEG}) (?:yet )?(?:been )?(?:fully )?(?:confirmed|verified)\b",
        rf"\b(?:can(?:'t|’t|not| not)|could{_NEG}|unable to|not able to"
        rf"|was{_NEG} able to) (?:yet )?(?:confirm|verify|check)\b",
        r"\bnot (?:sure|certain)\b.{0,60}\b(?:booked|confirmed|scheduled|through|went|worked|booking"
        r"|reservation|cancel\w*|moved|rescheduled)\b",
        r"\b(?:unsure|uncertain) (?:whether|if)\b",
        r"\b(?:status|booking|reservation|request) (?:is |remains )?(?:still )?(?:unknown|unclear|pending"
        r"|uncertain|unverified)\b",
        r"\bpending (?:confirmation|verification)\b",
        rf"\b(?:did{_NEG}|have{_NEG}|has{_NEG}|haven't yet|not yet) (?:receive[d]?|get|got(?:ten)?"
        r"|hear[d]?) (?:a |any |the )?(?:confirmation|response|reply|answer)\b",
        r"\bmay or may not\b",
    )
)
CONDITIONAL: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        r"\b(?:once|when|after|as soon as|if) (?:you|you've|you have) (?:confirm|reply|choose|pick|select"
        r"|let me know|say|tell|click|tap)\b",
        r"\b(?:once|when|after|as soon as) (?:it'?s |it is |that'?s |the booking is "
        r"|your booking is )?(?:confirmed|booked|done|scheduled)\b",
        r"\bif (?:that|this|it|one of (?:these|those|them)|either|any of (?:these|those|them)|so)\b",
        r"\b(?:shall|should|can|may|could) i (?:go ahead and )?(?:book|schedule|reserve|lock|hold|confirm"
        r"|move|reschedule|cancel|put|set)\b",
        r"\b(?:would|do) you (?:like|want) (?:me )?to\b",
        r"\bwant me to\b",
        r"\bi can (?:go ahead and )?(?:book|schedule|reserve|lock|hold|move|reschedule|cancel|put|set|offer"
        r"|do)\b",
        r"\bi could (?:book|schedule|move|reschedule|cancel|do|offer)\b",
        r"\bready to (?:book|schedule|confirm|lock)\b",
        r"\b(?:tell me|let me know) (?:which|what|if|whether)\b",
        r"\byou'?ll be (?:booked|all set|confirmed|scheduled)\b",
        r"\b(?:please|just) (?:confirm|reply|choose|pick|select|let me know|tell me|say)\b",
        r"\bto (?:book|confirm|lock in|reserve)(?: it| this| that| one)?,? (?:just |please )?(?:reply"
        r"|confirm|let me know|pick|choose)\b",
    )
)
PENDING: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        r"\b(?:i'?ll|i will|i'?m going to|i am going to|let me|i'?m about to|about to"
        r"|going to) (?:go ahead and )?(?:now )?(?:book|schedule|reserve|lock in|lock|confirm (?:it|that"
        r"|this|the booking|your booking)|move|reschedule|cancel|process|finali[sz]e|put (?:you|it) (?:in"
        r"|down))\b",
        r"\b(?:i'?m|i am|we'?re|we are) (?:now |currently |just )?(?:booking|scheduling|reserving"
        r"|confirming|moving|rescheduling|cancell?ing|processing|finali[sz]ing|locking)\b",
        r"^\W*(?:now )?(?:booking|scheduling|reserving|confirming|moving|rescheduling|cancell?ing"
        r"|processing)\b(?!\s*(?:is |was |has been )?(?:now )?(?:confirmed|complete[d]?|done|successful"
        r"|succeeded|finali[sz]ed|details|summary|reference|ref\b|id\b|number|[:#(]))",
        r"\b(?:confirming|booking|processing|scheduling|reserving) (?:it |that |this |your (?:booking|slot"
        r"|call) )?(?:right )?now\b",
        r"\b(?:one|just a|give me a|bear with me(?: for)? a) (?:moment|sec(?:ond)?|minute)\b",
        r"\b(?:in progress|being processed)\b",
        r"\b(?:while|as) i (?:book|schedule|confirm|reserve|move|reschedule|cancel|lock)\b",
        r"^\W*(?:hold on|hang on|bear with me)\b",
        r"\bworking on (?:it|that|your (?:booking|request))\b",
    )
)
RESCHEDULED: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        r"^\W*rescheduled\b",
        r"\b(?:i'?ve|i have|we'?ve|we have|i|we) (?:just |now |successfully |gone ahead and )?(?:moved"
        r"|rescheduled|shifted|pushed)\b",
        r"\b(?:i'?ve|i have|we'?ve|we have|i|we) (?:just |now |successfully )?(?:changed|updated) (?:your "
        r"|the )?(?:call|meeting|booking|appointment|reservation|slot)\b",
        r"\b(?:has|have|was|were|is|'s) (?:now )?(?:been )?(?:successfully )?(?:moved|rescheduled|shifted"
        r"|pushed)\b",
        r"\b(?:call|meeting|booking|appointment|session|it) is now (?:on|at|set for|scheduled for"
        r"|booked for|confirmed for)\b",
        r"\byour new (?:time|slot|booking) is\b",
        r"\b(?:all|successfully) (?:moved|rescheduled)\b",
        r"\bmoved (?:it|you|your \w+) (?:to|over to)\b",
        r"\breschedul(?:e|ing) (?:is |was |has been )?(?:confirmed|done|complete)\b",
        r"\b(?:i|we) (?:was|were) (?:finally |successfully )?able to (?:move|reschedule|shift|push)\b",
        r"\b(?:i|we) (?:finally |successfully )?managed to (?:move|reschedule|shift|push)\b",
    )
)
CANCELLED: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        r"^\W*cancell?ed\b",
        r"\b(?:i'?ve|i have|we'?ve|we have|i|we) (?:just |now |successfully |gone ahead and )?(?:cancell?ed"
        r"|dropped|removed|deleted|called off)\b",
        r"\b(?:has|have|was|were|is|'s|are) (?:now )?(?:been )?(?:successfully )?(?:cancell?ed|called off"
        r"|dropped|removed from (?:the|your|my|our) calendar|deleted)\b",
        r"\bno longer (?:booked|scheduled|on (?:the|your|my|our) calendar|happening)\b",
        r"\bcancell?ation (?:is |has been )?(?:confirmed|done|complete|processed)\b",
        r"\bsuccessfully cancell?ed\b",
        r"\b(?:i|we) (?:was|were) (?:finally |successfully )?able to cancel\b",
        r"\b(?:i|we) (?:finally |successfully )?managed to cancel\b",
    )
)
EXISTING: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        r"\b(?:current|currently|existing|original|originally|previous|previously|old)\b",
        r"\byou (?:already )?have (?:a|an|your) (?:call|meeting|booking|appointment)\b",
        r"\bwas (?:on|at|set for|scheduled for|booked for)\b",
        r"\byou (?:mentioned|asked for|asked about|said|suggested|requested|proposed|wanted|preferred)\b",
    )
)
UNAVAILABLE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        r"\b(?:is|are|was|were|'s)(?: already| now| just| both| all)? (?:taken|unavailable|full|fully booked"
        r"|booked up|gone)\b",
        rf"\b(?:is|are|was|were){_NEG} (?:available|free|open|possible)\b",
        r"\b(?:not|un)available\b",
        r"\b(?:outside|beyond) (?:of )?(?:our|my|the) (?:business|working|office) hours\b",
        rf"\b(?:do|does){_NEG} have (?:any )?(?:availability|openings|slots|free time)\b",
        rf"\b(?:i|we) (?:do{_NEG}|don't|do not) have\b",
        r"\bnothing (?:is )?(?:free|open|available)\b",
        r"\bfully booked\b",
        # Someone else holds the slot: "the 3 PM slot was booked by someone else", "3 PM is already booked".
        r"\b(?:booked|taken|reserved|grabbed|claimed) by (?:someone|somebody|another|a different|other"
        r"|a third)\b",
        r"(?<!your )\b(?:slot|time|spot|one)\b[^.;,]{0,30}\b(?:is|was|'s|has been) already (?:booked|taken"
        r"|reserved)\b",
        r"\b(?:am|pm|\d:\d\d)\s+(?:on \w+ )?(?:is|was|'s|has been) already (?:booked|taken|reserved)\b",
    )
)
#: A plain "see you <time>" send-off ("See you Tuesday!", "See you at 3pm."). On its own it is a booking
#: claim (``docs/metrics.md``'s own example), but after a completed reschedule it is easily just a pleasant
#: close restating that, not a fresh claim (:func:`_is_bare_see_you_close`, used by :func:`extract_belief`).
_SEE_YOU_CLOSE = re.compile(
    r"\bsee you (?:on |then|there|soon|at |next |tomorrow|today|this |(?:mon|tues|wednes|thurs|fri"
    r"|satur|sun)day)",
    _I,
)
BOOKED: tuple[re.Pattern[str], ...] = (
    *(
        re.compile(p, _I)
        for p in (
            r"^\W*(?:booked|confirmed|scheduled|reserved)\b\s*[:!.]",
            r"^\W*(?:booked|confirmed)\W*$",
            r"\byou'?re (?:all )?(?:set|booked|confirmed|scheduled|good to go|locked in|in the calendar"
            r"|on the calendar)\b",
            r"\byou are (?:all )?(?:set|booked|confirmed|scheduled|good to go|locked in)\b",
            r"\ball (?:set|booked|confirmed|sorted)\b",
            r"\b(?:i'?ve|i have|we'?ve|we have|i|we) (?:just |now |successfully |gone ahead and )?(?:booked"
            r"|scheduled|reserved|confirmed|locked in|set up|secured|pencil(?:l)?ed|put you down|added you"
            r"|added it|got you)\b",
            r"\b(?:is|are|has been|have been|'s|was) (?:now )?(?:successfully |officially )?(?:booked"
            r"|confirmed|scheduled|reserved|locked in|set up|secured|on the calendar|in the calendar"
            r"|on the books)\b",
            r"\b(?:booked|confirmed|scheduled|reserved) for\b",
            r"\b(?:is|are|'s|has been|have been|you'?re|you are|we'?re|we are) (?:now )?(?:all )?set (?:for"
            r"|up for)\b",
            r"^\W*(?:booked|confirmed|scheduled|reserved)\b(?! by\b| with\b| yet\b| if\b)",
            r"\bbooking (?:is )?confirmed\b",
            r"\bconfirmation\b.{0,20}\b(?:sent|on its way)\b",
            r"\b(?:calendar )?invit(?:e|ation) (?:is on its way|has been sent|was sent|is in your inbox"
            r"|is coming|should arrive|will arrive|is heading)\b",
            r"\b(?:sent|emailed) you (?:a|an|the) (?:calendar )?(?:invite|invitation|confirmation)\b",
            r"\byou(?:'ll| will) (?:get|receive) (?:a|an|the|your) (?:calendar )?(?:invite|invitation"
            r"|confirmation)\b",
            r"\b(?:sent|emailed)(?: you)? (?:a|an|the|your) (?:calendar )?(?:invite|invitation"
            r"|confirmation)\b",
            r"\b(?:it'?s|that'?s) (?:booked|confirmed|in the calendar|locked in)\b",
            r"\bsuccessfully booked\b",
            r"\b(?:i|we) (?:was|were) (?:finally |successfully )?able to (?:book|schedule|reserve|secure"
            r"|lock in|get you (?:booked|in|down))\b",
            r"\b(?:i|we) (?:finally |successfully )?managed to (?:book|schedule|reserve|secure|lock in"
            r"|get you (?:booked|in|down))\b",
            r"\b(?:has been|have been|is|was|'s) (?:now )?(?:successfully )?added to (?:the|your|my|our)"
            r" calendar\b",
            r"\b(?:booking|reservation) (?:is |was |has been )?(?:now )?(?:complete[d]?|done|successful"
            r"|finali[sz]ed)\b",
            r"\bgot you (?:booked|down|in)\b",
            r"\blocked in\b",
            r"\byou'?re already (?:booked|scheduled|confirmed)\b",
        )
    ),
    _SEE_YOU_CLOSE,
)
#: A hand-off's forward-looking purpose clause: "get you booked for Monday", "get that slot locked in",
#: "get something scheduled" - a promise that a colleague or the agent itself will still take the action,
#: not a claim that it already happened (``docs/metrics.md``'s `not_booked`: "a conditional... with no
#: success claim"). Real benchmark traces showed the bare "booked for"/"locked in" patterns above matching
#: these purpose clauses regardless of tense, since a plain proximity match cannot tell "I've booked you
#: in" from "I'll get you booked".
_FUTURE_BOOKING_PROMISE = re.compile(
    r"\bget (?:you|it|that|this|something)\b[^.!?,;]{0,35}?\b(?:booked|scheduled|locked in|confirmed"
    r"|reserved|sorted)\b",
    _I,
)
HANDOFF: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, _I)
    for p in (
        r"\b(?:connect|put) you (?:in touch )?with\b",
        r"\bhand(?:ed|ing)? (?:you |this |it )?(?:over|off)\b",
        r"\bhand-?off\b",
        r"\b(?:colleague|teammate|team member|human|person|representative|rep|someone|somebody"
        r"|our team)\b.{0,50}\b(?:will|shall|is going to|can|'ll)\b.{0,30}\b(?:reach out|contact|call|email"
        r"|follow up|get in touch|get back|be in touch)\b",
        r"\bpassed (?:your|the|this) (?:details|request|info\w*|contact\w*) (?:on |along )?to\b",
        r"\bour team will\b",
        r"\b(?:ask|have|get) (?:a |one of my |one of our |my |our )?(?:colleague|teammate|team member|human"
        r"|someone|representative|rep)\b",
    )
)
#: A confirmation of something other than the meeting ("I've confirmed your time zone"), which is not a
#: booking claim.
NOT_A_MEETING = re.compile(
    r"\b(?:confirmed|set|noted|updated|saved|got|have|changed)\b (?:on |for |with |to |as )?(?:your "
    r"|the )?(?:time ?zone|timezone|zone|e-?mail|name|details|number|phone)\b|\b(?:time ?zone|timezone|zone"
    r"|e-?mail|name|details|number|phone) (?:is|has been|was) (?:now )?(?:confirmed|set|noted|updated|saved"
    r"|changed)\b",
    _I,
)

_CRM_OBJECT = (
    r"(?:the |our |your |my |a )?(?:crm|hubspot|salesforce|contact record|lead record|crm record|sales system"
    r"|pipeline)\b"
)
#: A failure that concerns only the CRM ("I couldn't update our CRM", "the CRM sync failed"). It says
#: nothing about the meeting, so it does not change the status.
CRM_FAILURE = re.compile(
    rf"\b(?:could{_NEG}|can(?:'t|’t|not| not)|unable to|failed to|fail to|not able to|was{_NEG} able to"
    rf"|were{_NEG} able to)\s+(?:\w+\s+){{0,3}}?(?:update|log|sync|save|record|add|create|write|note|reach"
    rf"|access|connect to)\b[^.;]{{0,40}}?\b{_CRM_OBJECT}"
    rf"|\b{_CRM_OBJECT}(?:\s+(?:update|sync|call|request|write|entry))?\s+(?:failed|is down|went down"
    r"|timed out|returned an error|threw an error|had an error|errored|is unavailable|was unavailable)"
    rf"|\b(?:an? |some |another )?(?:error|problem|issue|glitch)\s+(?:updating|with|from|in|syncing"
    rf"|saving to|logging (?:it |this |that )?(?:in|to))\s+{_CRM_OBJECT}",
    _I,
)

_ORDER: tuple[tuple[Category, tuple[re.Pattern[str], ...]], ...] = (
    ("retraction", RETRACTION),
    ("hedge", UNCERTAIN),
    ("failure", FAILURE),
    ("hedge", HEDGE),
    ("conditional", CONDITIONAL),
    ("pending", PENDING),
    ("rescheduled", RESCHEDULED),
    ("cancelled", CANCELLED),
    ("existing", EXISTING),
    ("unavailable", UNAVAILABLE),
    ("booked", BOOKED),
    ("handoff", HANDOFF),
)
_SUCCESS_CATEGORIES: frozenset[Category] = frozenset({"booked", "rescheduled", "cancelled"})

_SENTENCE_END = re.compile(
    r"(?<!\ba\.m)(?<!\bp\.m)(?<!\bA\.M)(?<!\bP\.M)(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\betc)[.!?]+(?=\s|$)|\n+"
)
_CLAUSE_SPLIT = re.compile(r"\s+[-–—]+\s+|;\s*|,?\s+\bbut\b\s+|,?\s+\bhowever\b,?\s+", _I)
_AFTER_MOVE = re.compile(r"\b(?:to|now|for)\b", _I)
#: Text right before a time that marks it as the existing booking or the old end of a move, not an offer:
#: "your call on Thursday at 11:00", "move it from 3 PM to 4 PM".
_NOT_OFFERED_BEFORE = re.compile(
    r"(?:\byour (?:\w+ )?(?:call|meeting|booking|appointment|session|reservation)(?: (?:is |was )?(?:on|at"
    r"|for|from|scheduled for|booked for|set for))?|\b(?:from|until|till|before|after))[ \t]*[,:]?[ \t]*$",
    _I,
)
#: A clause that introduces the meeting's time ("Your call: Tuesday at 3 PM").
_MEETING_LEAD = re.compile(
    r"^\W*(?:your |the )?(?:call|meeting|booking|appointment|session|new time|time"
    r"|when)\b[ \t]*(?:is[ \t]*)?[:\-–]",
    _I,
)


def classify(clause: str) -> Category | None:
    """The category of one clause, or ``None`` for a neutral clause (offers, questions, small talk)."""
    text = clause.replace("’", "'")
    question = text.rstrip().endswith("?")
    for category, patterns in _ORDER:
        if category in _SUCCESS_CATEGORIES and (question or NOT_A_MEETING.search(text)):
            continue
        if category == "failure" and _crm_only_failure(text):
            continue
        if category == "booked" and _FUTURE_BOOKING_PROMISE.search(text):
            continue
        if any(p.search(text) for p in patterns):
            if category == "retraction" and _restates_empty(text) and any(p.search(text) for p in CANCELLED):
                return "cancelled"
            return category
    return None


def _crm_only_failure(text: str) -> bool:
    """The clause reports a CRM failure and no other failure."""
    stripped = CRM_FAILURE.sub(" ", text)
    return stripped != text and not any(p.search(stripped) for p in FAILURE)


def _is_bare_see_you_close(text: str) -> bool:
    """A clause whose only booking-shaped signal is a plain "see you <time>" send-off ("See you then!",
    "See you Monday at 3:00 PM ET."), with no more explicit completed-booking phrase alongside it. Used to
    keep such a close from turning an already-established ``rescheduled`` belief into ``booked``
    (:func:`extract_belief`): the send-off only restates the reschedule, it is not a fresh claim."""
    return _SEE_YOU_CLOSE.search(text) is not None and not any(
        p.search(text) for p in BOOKED if p is not _SEE_YOU_CLOSE
    )


@dataclass(frozen=True)
class Clause:
    message: int
    sentence: int
    start: int
    end: int
    text: str
    category: Category | None
    spans: tuple[TimeSpan, ...]


def split_clauses(message: str) -> list[tuple[int, int, int]]:
    """``(sentence index, start, end)`` of every non-empty clause of a message."""
    sentences: list[tuple[int, int]] = []
    cursor = 0
    for boundary in _SENTENCE_END.finditer(message):
        sentences.append((cursor, boundary.end()))
        cursor = boundary.end()
    sentences.append((cursor, len(message)))
    clauses: list[tuple[int, int, int]] = []
    for index, (s_start, s_end) in enumerate(sentences):
        piece_start = s_start
        for split in _CLAUSE_SPLIT.finditer(message, s_start, s_end):
            clauses.append((index, piece_start, split.start()))
            piece_start = split.end()
        clauses.append((index, piece_start, s_end))
    trimmed = []
    for index, start, end in clauses:
        while start < end and message[start].isspace():
            start += 1
        while end > start and message[end - 1].isspace():
            end -= 1
        if message[start:end].strip(" \t\n.!?,;-–—"):
            trimmed.append((index, start, end))
    return trimmed


def read_clauses(
    agent_messages: Sequence[str], *, prospect_zone: str, host_zone: str, reference: datetime
) -> list[Clause]:
    """Every clause of every message, classified, with the time mentions inside it."""
    clauses: list[Clause] = []
    for m_index, message in enumerate(agent_messages):
        spans = find_times(message, prospect_zone=prospect_zone, host_zone=host_zone, reference=reference)
        for s_index, start, end in split_clauses(message):
            inside = tuple(s for s in spans if start <= s.start < end)
            text = message[start:end]
            clauses.append(Clause(m_index, s_index, start, end, text, classify(text), inside))
    return clauses


def _claim_time(clause: Clause) -> TimeSpan | None:
    spans = [s for s in clause.spans if not s.alias]
    if not spans:
        return None
    if clause.category == "rescheduled":
        marker = _AFTER_MOVE.search(clause.text)
        if marker is not None:
            after = [s for s in spans if s.start - clause.start >= marker.start()]
            if after:
                return after[0]
    return spans[0]


def extract_belief(
    agent_messages: Sequence[str],
    *,
    prospect_zone: str,
    host_zone: str,
    reference: datetime,
    source: BeliefSource = "lexicon",
) -> Belief:
    """The prospect's belief after reading ``agent_messages`` in order (``docs/metrics.md``)."""
    reference = ensure_utc(reference)
    clauses = read_clauses(
        agent_messages, prospect_zone=prospect_zone, host_zone=host_zone, reference=reference
    )
    status: BeliefStatus | None = None
    time_span: TimeSpan | None = None
    evidence = ""
    claimed: set[int] = set()  # ids of spans that state a claim's time, never offers
    for index, clause in enumerate(clauses):
        if clause.category is None or clause.category not in STRONG_STATUS:
            continue
        if status == "cancelled" and clause.category == "retraction" and _restates_empty(clause.text):
            continue
        if status == "rescheduled" and clause.category == "booked" and _is_bare_see_you_close(clause.text):
            continue
        new_status = STRONG_STATUS[clause.category]
        new_time: TimeSpan | None = None
        if new_status in SUCCESS:
            new_time = _claim_time(clause)
            if new_time is None:
                new_time = _borrowed_time(clauses, index)
            if new_time is None and status == new_status:
                new_time = time_span
            if new_time is not None:
                claimed.add(id(new_time))
        elif clause.category == "failure" and _success_earlier_in_sentence(clauses, index):
            new_status = "unclear"
        status, time_span, evidence = new_status, new_time, clause.text.strip()
    offered = {
        span.utc
        for clause in clauses
        if clause.category in OFFER_CATEGORIES
        for span in _usable(clause)
        if id(span) not in claimed and not _NOT_OFFERED_BEFORE.search(_before(clause, span))
    }
    final = status or "not_booked"
    return Belief(
        status=final,
        time_utc=time_span.utc if time_span is not None and final in SUCCESS else None,
        offered_utc=tuple(sorted(offered)),
        source=source,
        evidence=evidence,
    )


def _restates_empty(text: str) -> bool:
    """The clause only says that nothing is booked now, which is consistent with a cancellation."""
    stripped = NOTHING_BOOKED.sub(" ", text.replace("’", "'"))
    return stripped != text.replace("’", "'") and not any(p.search(stripped) for p in RETRACTION)


def _success_earlier_in_sentence(clauses: list[Clause], index: int) -> bool:
    current = clauses[index]
    for other in reversed(clauses[:index]):
        if other.message != current.message or other.sentence != current.sentence:
            return False
        if other.category in _SUCCESS_CATEGORIES:
            return True
    return False


def _usable(clause: Clause) -> list[TimeSpan]:
    return [s for s in clause.spans if not s.alias and not s.in_range]


def _before(clause: Clause, span: TimeSpan) -> str:
    return clause.text[max(0, span.start - clause.start - 60) : span.start - clause.start]


def _mostly_time(clause: Clause) -> bool:
    """The clause is little more than a date and time ("Tuesday 6 October, 3:00 PM (Berlin time).")."""
    covered = sum(span.end - span.start for span in clause.spans)
    return covered >= 0.6 * len(clause.text.strip(" \t\n.!?,;:()"))


def _borrowed_time(clauses: list[Clause], index: int) -> TimeSpan | None:
    """A time for a claim clause that states none, from the neutral clause right before or after it in the
    same message: within one sentence ("Tuesday at 3 PM - you're all set"), or a clause that is only a time
    or introduces the meeting's time ("You're all set! Tuesday 6 October, 3:00 PM." / "Your call: ...")."""
    current = clauses[index]
    for neighbour in (index - 1, index + 1):
        if not 0 <= neighbour < len(clauses):
            continue
        other = clauses[neighbour]
        spans = _usable(other)
        if other.message != current.message or other.category not in (None, "existing") or not spans:
            continue
        introduces = neighbour > index and _MEETING_LEAD.match(other.text) is not None
        if other.sentence == current.sentence or _mostly_time(other) or introduces:
            return spans[0] if neighbour > index else spans[-1]
    return None


class LexiconBeliefExtractor:
    """The lexicon extractor behind the :class:`~booking_truth.harness.beliefs.BeliefExtractor` protocol."""

    source: BeliefSource = "lexicon"

    async def extract(
        self,
        agent_messages: Sequence[str],
        *,
        prospect_zone: str,
        host_zone: str,
        reference: datetime,
    ) -> Belief:
        return extract_belief(
            agent_messages, prospect_zone=prospect_zone, host_zone=host_zone, reference=reference
        )

"""Simulated prospects.

A persona sees only the conversation. :class:`ScriptedPersona` follows a scenario's fixed script (``say``,
``pick``, ``when``, ``end``); it is what CI and offline grading use. An LLM persona (a later addition) plugs
into the same :class:`Persona` protocol.

Rules of the scripted persona:

- Steps run in order. A step with a ``when`` condition is skipped when the condition does not hold for the
  agent's last message (``agent_asks_timezone``, ``agent_offered_slots``, ``agent_asks_confirmation``,
  ``agent_has_booking``).
- When the agent asks for the prospect's zone and the prospect has not answered such a question yet, the
  prospect answers with its ``clarification`` first.
- ``pick`` accepts an offered slot. Offers are the slot quick replies of the bundled protocol
  (``start_utc``, ``slot_id``), else the times the agent proposed in its text. ``pick: in_window`` takes the
  first offer inside the hidden window; ``pick: offered[N]`` takes the N-th offer. With no offer yet, the
  prospect asks for times in its window; with offers but none in the window, it says its ``correction`` (or a
  default one). After three such tries the prospect gives up and the conversation ends.
- A pick is a ``select_slot`` action when the agent speaks the bundled protocol and the offer has a slot id;
  otherwise it is the slot's label as text ("<label> works for me.").
- Every acceptance is checked against the hidden window. Accepting a time outside it raises
  :class:`PersonaError`, which grades the trial ``harness_error`` and reruns it.
- The conversation ends after a step with ``end: true``, when the script runs out, or at 14 persona turns.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Literal, Protocol

from booking_truth.harness.adapters import AgentReply
from booking_truth.harness.lexicon_extractor import extract_belief
from booking_truth.harness.scenarios import ResolvedScenario, ScriptStep
from booking_truth.harness.timeparse import find_times, zone_for
from booking_truth.timeutil import ensure_utc, iso_z, parse_iso

MAX_PERSONA_TURNS = 14
MAX_RETRIES = 3

TurnKind = Literal["say", "pick", "clarify", "ask", "correction"]

_OFFERED_LABEL = re.compile(r"\{\{\s*offered\[(\d+)\]\.label\s*\}\}")
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
_TZ_WORDS = re.compile(
    r"\btime[\s-]?zones?\b|\bzone\b|\bwhere (?:are you|you are|you're)\b|\byour location\b|\blocal time\b"
    r"|\bwhich (?:city|country|region)\b|\bwhat (?:city|country|region)\b|\bbrowser\b",
    re.IGNORECASE,
)
#: A named zone or offset in a question that checks it ("Is America/Denver right for you?").
#: A zone named in a sentence: an IANA name, a UTC/GMT offset, "time zone" or the browser's zone.
_ZONE_NAMED = re.compile(r"\b[A-Z][a-z]+/[A-Z][A-Za-z_]+|\b(?:UTC|GMT)\s?[+\-−]\d")
_ZONE_WORDS = re.compile(r"\btime[\s-]?zones?\b|\bbrowser\b", re.IGNORECASE)
#: A question that checks something just stated ("Is that right?", "Are you on UTC+2?").
_CHECK = re.compile(
    r"\b(?:right|correct|accurate)\b|\bare you (?:in|on)\b|\byou're (?:in|on)\b", re.IGNORECASE
)
_TZ_REQUEST = re.compile(
    r"\b(?:tell me|let me know|confirm|share|provide)\b[^.!?]*\b(?:time[\s-]?zone|located|city)\b",
    re.IGNORECASE,
)
_CONFIRM_WORDS = re.compile(
    r"\b(?:shall|should|can|may) I\b|\bwould you like (?:me|to)\b|\bdo you want (?:me|to)\b|\bwant me to\b"
    r"|\bconfirm\b|\bis (?:that|this) (?:ok|okay|alright|all right|right|correct|fine)\b"
    r"|\bdoes (?:that|this) work\b"
    r"|\bsounds? good\b|\bgo ahead\b|\bare you sure\b|\bproceed\b|\bbook (?:it|that|this)\b|\bok to\b",
    re.IGNORECASE,
)
_CONFIRM_REQUEST = re.compile(
    r"\bplease confirm\b|\breply (?:yes|with yes)\b|\bjust confirm\b", re.IGNORECASE
)


class PersonaError(RuntimeError):
    """The persona accepted a time outside its hidden window."""


# Offers ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Offer:
    """A slot the agent offered: its start (UTC), the label the prospect uses for it and, for the bundled
    protocol, its slot id."""

    start_utc: datetime
    label: str
    slot_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"start_utc": iso_z(self.start_utc), "label": self.label, "slot_id": self.slot_id}


def _clock_text(local: datetime) -> str:
    hour = local.hour % 12 or 12
    return f"{hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"


def slot_label(start_utc: datetime, zone: str, prospect_zone: str) -> str:
    """``Tuesday 6 October at 1:00 PM`` in ``zone``; the zone is named when it is not the prospect's own."""
    local = ensure_utc(start_utc).astimezone(zone_for(zone))
    text = f"{local:%A} {local.day} {local:%B} at {_clock_text(local)}"
    return text if zone == prospect_zone else f"{text} {zone}"


def quick_reply_offers(quick_replies: Sequence[dict[str, Any]], prospect_zone: str) -> list[Offer]:
    """Slot quick replies of the bundled protocol: those with a ``start_utc``."""
    offers: list[Offer] = []
    for item in quick_replies:
        start_raw = item.get("start_utc")
        if not isinstance(start_raw, str):
            continue
        try:
            start = parse_iso(start_raw)
        except ValueError:
            continue
        action = item.get("action") if isinstance(item.get("action"), dict) else {}
        assert isinstance(action, dict)
        slot_id = action.get("slot_id") if action.get("type", "select_slot") == "select_slot" else None
        label = item.get("label")
        offers.append(
            Offer(
                start_utc=start,
                label=label.strip()
                if isinstance(label, str) and label.strip()
                else slot_label(start, prospect_zone, prospect_zone),
                slot_id=slot_id if isinstance(slot_id, str) and slot_id else None,
            )
        )
    return offers


def text_offers(text: str, *, prospect_zone: str, host_zone: str, reference: datetime) -> list[Offer]:
    """The times an agent message proposes, read the way the lexicon belief extractor reads offers (a time
    stated as booked is not an offer). Labels are rendered in the zone the agent used."""
    belief = extract_belief([text], prospect_zone=prospect_zone, host_zone=host_zone, reference=reference)
    if not belief.offered_utc:
        return []
    zones: dict[datetime, str] = {}
    for span in find_times(text, prospect_zone=prospect_zone, host_zone=host_zone, reference=reference):
        if not span.alias:
            zones.setdefault(span.utc, span.zone)
    return [
        Offer(start_utc=t, label=slot_label(t, zones.get(t, prospect_zone), prospect_zone))
        for t in belief.offered_utc
    ]


def reply_offers(
    reply: AgentReply | None, *, prospect_zone: str, host_zone: str, reference: datetime
) -> list[Offer]:
    """Offers in one agent reply: slot quick replies first, else the times proposed in its text."""
    if reply is None or reply.reply is None:
        return []
    offers = quick_reply_offers(reply.quick_replies, prospect_zone)
    if offers:
        return offers
    return text_offers(reply.reply, prospect_zone=prospect_zone, host_zone=host_zone, reference=reference)


# Conditions -------------------------------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.findall(text) if part.strip()]


def _names_zone(sentence: str) -> bool:
    return _ZONE_NAMED.search(sentence) is not None or _ZONE_WORDS.search(sentence) is not None


def _zone_questions(text: str) -> set[int]:
    """Indexes of the sentences that ask about the prospect's zone or location."""
    sentences = _sentences(text)
    found: set[int] = set()
    for index, sentence in enumerate(sentences):
        if not sentence.endswith("?"):
            continue
        checks_zone = _CHECK.search(sentence) is not None and (
            _names_zone(sentence) or (index > 0 and _names_zone(sentences[index - 1]))
        )
        if _TZ_WORDS.search(sentence) or checks_zone:
            found.add(index)
    return found


def agent_asks_timezone(text: str | None) -> bool:
    """The agent asks where the prospect is or which zone to use."""
    if not text:
        return False
    return bool(_zone_questions(text)) or _TZ_REQUEST.search(text) is not None


def agent_asks_confirmation(text: str | None) -> bool:
    """The agent asks the prospect to confirm an action (not a zone)."""
    if not text:
        return False
    zone = _zone_questions(text)
    for index, sentence in enumerate(_sentences(text)):
        if sentence.endswith("?") and index not in zone and _CONFIRM_WORDS.search(sentence):
            return True
    return _CONFIRM_REQUEST.search(text) is not None


def reply_reports_booking(
    reply: AgentReply | None, *, prospect_zone: str, host_zone: str, reference: datetime
) -> bool:
    """The last reply says a meeting is booked or moved: a structured ``booking``, or the text alone."""
    if reply is None or reply.reply is None:
        return False
    booking = reply.booking
    if (
        isinstance(booking, dict)
        and booking.get("action", "booked") in ("booked", "rescheduled")
        and booking.get("status") not in ("cancelled", "canceled")
    ):
        return True
    belief = extract_belief(
        [reply.reply], prospect_zone=prospect_zone, host_zone=host_zone, reference=reference
    )
    return belief.status in ("booked", "rescheduled")


# Persona protocol -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonaTurn:
    """One prospect message. A pick carries the accepted ``offer`` and, for the bundled protocol, the
    ``select_slot`` action; ``text`` is what the prospect says (for an action, the slot's label)."""

    kind: TurnKind
    text: str
    action: dict[str, Any] | None = None
    offer: Offer | None = None
    end: bool = False
    #: For a pick: every offer the prospect chose from, in the agent's order.
    choices: tuple[Offer, ...] = ()

    @property
    def is_pick(self) -> bool:
        return self.kind == "pick"


@dataclass
class AgentView:
    """What the persona sees before its next turn: the reply it continues from and every agent message of
    its own session so far."""

    last_reply: AgentReply | None
    agent_messages: list[str]
    now: datetime


class Persona(Protocol):
    #: Messages sent so far.
    turns: int
    #: Spend on the persona's own model calls (0 for scripted personas).
    usage_usd: float

    async def next_turn(self, view: AgentView) -> PersonaTurn | None:
        """The next message, or ``None`` when the prospect is done."""
        ...


# Scripted persona ------------------------------------------------------------------------------------------


def _hour_text(value: time) -> str:
    hour = value.hour % 12 or 12
    suffix = "am" if value.hour < 12 else "pm"
    return f"{hour} {suffix}" if value.minute == 0 else f"{hour}:{value.minute:02d} {suffix}"


@dataclass
class ScriptedPersona:
    """A persona that follows its scenario's script; see the module docstring for the rules."""

    scenario: ResolvedScenario
    supports_actions: bool = True
    turns: int = 0
    usage_usd: float = 0.0
    index: int = 0
    retries: int = 0
    zone_answered: bool = False
    offers: list[Offer] = field(default_factory=list)
    picks: list[Offer] = field(default_factory=list)
    gave_up: bool = False

    @property
    def prospect_zone(self) -> str:
        return self.scenario.scenario.persona.true_zone

    @property
    def host_zone(self) -> str:
        return self.scenario.host_zone

    @property
    def script(self) -> list[ScriptStep]:
        return self.scenario.scenario.persona.script

    def _render(self, template: str) -> str:
        return self.scenario.render(template, [offer.label for offer in self.offers])

    def ask_text(self) -> str:
        """What the prospect says when it has nothing in its window to pick from."""
        correction = self.scenario.scenario.persona.correction
        if correction is not None:
            return self._render(correction)
        window = self.scenario.window
        dates = self.scenario.variables["window.dates_text"]
        return (
            f"What times do you have {dates}? Something from {_hour_text(window.start)} to "
            f"{_hour_text(window.end)} my time would be ideal."
        )

    def correction_text(self) -> str:
        if self.scenario.scenario.persona.correction is not None:
            return self.ask_text()
        return "None of those times work for me. " + self.ask_text()

    def _emit(self, turn: PersonaTurn) -> PersonaTurn:
        self.turns += 1
        return turn

    def _holds(self, step: ScriptStep, view: AgentView, last_offers: list[Offer]) -> bool:
        text = view.last_reply.reply if view.last_reply is not None else None
        match step.when:
            case "always":
                return True
            case "agent_asks_timezone":
                return agent_asks_timezone(text)
            case "agent_offered_slots":
                return bool(last_offers)
            case "agent_asks_confirmation":
                return agent_asks_confirmation(text)
            case "agent_has_booking":
                if reply_reports_booking(
                    view.last_reply,
                    prospect_zone=self.prospect_zone,
                    host_zone=self.host_zone,
                    reference=view.now,
                ):
                    return True
                belief = extract_belief(
                    view.agent_messages,
                    prospect_zone=self.prospect_zone,
                    host_zone=self.host_zone,
                    reference=view.now,
                )
                return belief.status in ("booked", "rescheduled")
        return False

    def _check_window(self, offer: Offer, how: str) -> None:
        if not self.scenario.window_contains(offer.start_utc):
            raise PersonaError(
                f"the persona accepted {iso_z(offer.start_utc)} ({how}), which is outside its hidden window"
            )

    def _retry(self, kind: TurnKind, text: str) -> PersonaTurn | None:
        self.retries += 1
        if self.retries > MAX_RETRIES:
            self.gave_up = True
            self.index = len(self.script)
            return None
        return self._emit(PersonaTurn(kind, text))

    async def next_turn(self, view: AgentView) -> PersonaTurn | None:
        last = view.last_reply
        last_offers = reply_offers(
            last, prospect_zone=self.prospect_zone, host_zone=self.host_zone, reference=view.now
        )
        if last_offers:
            self.offers = last_offers
        last_text = last.reply if last is not None else None
        while self.index < len(self.script):
            step = self.script[self.index]
            if not self._holds(step, view, last_offers):
                self.index += 1
                self.retries = 0
                continue
            if (
                last_text is not None
                and not self.zone_answered
                and step.when != "agent_asks_timezone"
                and agent_asks_timezone(last_text)
            ):
                self.zone_answered = True
                return self._emit(
                    PersonaTurn("clarify", self._render(self.scenario.scenario.persona.clarification))
                )
            if step.say is not None:
                return self._say(step)
            return self._pick(step, view)
        return None

    def _say(self, step: ScriptStep) -> PersonaTurn:
        assert step.say is not None
        for match in _OFFERED_LABEL.finditer(step.say):
            index = int(match[1])
            if index < len(self.offers):
                self._check_window(self.offers[index], f"said offered[{index}]")
        text = self._render(step.say)
        if step.when == "agent_asks_timezone":
            self.zone_answered = True
        self.index += 1
        self.retries = 0
        return self._emit(PersonaTurn("say", text, end=step.end))

    def _pick(self, step: ScriptStep, view: AgentView) -> PersonaTurn | None:
        if reply_reports_booking(
            view.last_reply, prospect_zone=self.prospect_zone, host_zone=self.host_zone, reference=view.now
        ) and not reply_offers(
            view.last_reply, prospect_zone=self.prospect_zone, host_zone=self.host_zone, reference=view.now
        ):
            # The agent already booked without asking; there is nothing left to pick.
            self.index += 1
            self.retries = 0
            return self._continue_after_skip(view)
        if not self.offers:
            return self._retry("ask", self.ask_text())
        index = step.offered_index
        if index is None:
            inside = [offer for offer in self.offers if self.scenario.window_contains(offer.start_utc)]
            if not inside:
                return self._retry("correction", self.correction_text())
            chosen = inside[0]
        else:
            if index >= len(self.offers):
                return self._retry("ask", self.ask_text())
            chosen = self.offers[index]
            self._check_window(chosen, f"pick offered[{index}]")
        self.index += 1
        self.retries = 0
        self.picks.append(chosen)
        choices = tuple(self.offers)
        self.offers = []
        if self.supports_actions and chosen.slot_id is not None:
            action = {"type": "select_slot", "slot_id": chosen.slot_id}
            return self._emit(
                PersonaTurn("pick", chosen.label, action=action, offer=chosen, end=step.end, choices=choices)
            )
        text = f"{chosen.label} works for me."
        return self._emit(PersonaTurn("pick", text, offer=chosen, end=step.end, choices=choices))

    def _continue_after_skip(self, view: AgentView) -> PersonaTurn | None:
        # Re-enter the step loop without re-reading the offers (they did not change).
        while self.index < len(self.script):
            step = self.script[self.index]
            if not self._holds(step, view, []):
                self.index += 1
                continue
            if step.say is not None:
                return self._say(step)
            return self._pick(step, view)
        return None

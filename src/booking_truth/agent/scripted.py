"""The offline "model": a deterministic scripted policy behind the ``LLM`` protocol (``FakeLLM``).

When no LLM key is configured the agent runs this policy instead of a model, and CI uses it for every agent
test. It speaks the same tools and final-answer format as a real model, in both tool modes (guarded: slot ids;
naive: ISO datetimes), and it is stateless: every call reconstructs the conversation from the message list.

What it understands (English only):

- intents: book, reschedule ("move", "push", "something came up"), cancel ("cancel", "drop", "call off"),
  retraction ("don't book it", "hold off", "wait"), yes/no, thanks, "try again", and a pick of an offered
  time;
- dates: explicit dates ("Monday 26 October", "between X and Y"), "tomorrow", "this week", "next week",
  weekdays ("Thursday", "next Friday"), "next few days"; parts of the day (early morning 05-09, morning
  09-12, midday 11-14, early afternoon 12-15, afternoon 12-17, late afternoon 15-19, evening 17-22) and
  explicit hours ("between 1 and 5 pm"), all in the prospect's zone;
- zone statements ("I'm in Berlin", "we're on CST", "Pacific time", an IANA name or a UTC offset), which it
  passes to ``resolve_timezone``.

Flow: resolve a stated zone → ``find_slots`` over the preferred dates (default: the next 5 business days) →
offer up to 4 slots matching the part of the day (else the first 4) → on a pick, book it → confirm.
Reschedule and cancel go through ``list_my_bookings``. A failed ``find_slots`` is retried once, then the
policy apologises and hands off; a taken slot leads to new offers.

``FakeLLM(misbehaviours=...)`` makes it misbehave in the ways the guards exist for:
``claim_success_after_tool_error``, ``invent_slots``, ``wrong_zone_for_ist``, ``retry_after_timeout``,
``garble_confirmation_time`` and ``wrong_iso_offset``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from booking_truth.agent import render
from booking_truth.agent.tools import business_days, local_day_bounds, valid_zone
from booking_truth.llm.types import ChatMessage, LLMResponse, ToolCall, ToolSpec, Usage
from booking_truth.timeutil import iso_z, parse_iso

MISBEHAVIOURS: frozenset[str] = frozenset(
    {
        "claim_success_after_tool_error",
        "invent_slots",
        "wrong_zone_for_ist",
        "retry_after_timeout",
        "garble_confirmation_time",
        "wrong_iso_offset",
    }
)
MODEL_ID = "offline/scripted-policy"
PROVIDER = "offline"
CHARS_PER_TOKEN = 4
OFFERS = 4
DEFAULT_BUSINESS_DAYS = 5
MAX_EXTENSIONS = 3
RANGE_LIMIT_DAYS = 14
IST_WRONG_ZONE = "Europe/Dublin"

_CONTEXT = re.compile(r"<context>\s*(\{.*?\})\s*</context>", re.DOTALL)
_I = re.IGNORECASE

WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}  # fmt: skip
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip
PARTS_OF_DAY: tuple[tuple[str, int, int], ...] = (
    ("early morning", 5, 9),
    ("late afternoon", 15, 19),
    ("early afternoon", 12, 15),
    ("late morning", 10, 12),
    ("midday", 11, 14),
    ("lunchtime", 11, 14),
    ("morning", 9, 12),
    ("afternoon", 12, 17),
    ("evening", 17, 22),
)

_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_WEEKDAY_RE = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\b\.?(?:,?\s+(\d{{4}}))?", _I)
_MONTH_DAY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", _I)
_WEEKDAY = re.compile(rf"\b(next|this|on)?\s*\b({_WEEKDAY_RE})\b", _I)
_CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)(?![a-z])", _I)
_CLOCK_24 = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*(?:am|pm))", _I)
_HOUR_PART = r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midday|midnight)"
_HOUR_RANGE = re.compile(rf"\b(?:between|from)\s+{_HOUR_PART}\s+(?:and|to|until|-)\s+{_HOUR_PART}", _I)
_HOUR_PAREN = re.compile(rf"\(\s*{_HOUR_PART}\s*(?:to|-|–)\s*{_HOUR_PART}\s*\)", _I)

CANCEL = re.compile(
    r"\bcancel\w*|\bdrop (?:the|my|our|this|that|it)\b|\bcall (?:it |the \w+ )?off\b|\bscrap\b"
    r"|\bwon'?t need\b|\bno longer need\b",
    _I,
)
TENTATIVE = re.compile(r"\b(?:might|maybe|may need|thinking about|considering|should i|possibly)\b", _I)
RETRACT = re.compile(
    r"\bdon'?t book\b|\bdo not book\b|\bplease don'?t\b|\bhold off\b|\bnot yet\b|\bwait\b[,.!]"
    r"|\bactually,? wait\b|\bnever ?mind\b|\bchanged? my mind\b|\bcheck with my team\b",
    _I,
)
RESCHEDULE = re.compile(
    r"\breschedul\w*|\bmove\b|\bpush\b|\bpostpone\b|\bchange (?:the )?(?:time|day|date)\b"
    r"|\b(?:another|different) (?:time|day)\b|\bsomething came up\b|\bbring (?:it )?forward\b",
    _I,
)
BOOK = re.compile(
    r"\bbook\b|\bschedule\b|\bset up\b|\barrange\b|\bcall\b|\bmeeting\b|\bdemo\b|\bappointment\b"
    r"|\bslots?\b|\btimes?\b|\bavailab\w*|\bnext week\b|\bany day\b",
    _I,
)
YES = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|please do|go ahead|confirm(?:ed)?|sounds good|that works"
    r"|do it|perfect|great)\b",
    _I,
)
NO = re.compile(r"^\s*(?:no|nope|nah)\b|\bkeep (?:my|the) (?:current|existing)\b|\bleave it\b", _I)
THANKS = re.compile(
    r"\bthanks\b|\bthank you\b|\bcheers\b|\bbye\b|\bgoodbye\b|\btalk (?:to you )?(?:soon|then)\b"
    r"|\bsee you\b|\blooking forward\b|\bget back to you\b|\bcheck back\b",
    _I,
)
TRY_AGAIN = re.compile(r"\btry again\b|\bretry\b|\bone more time\b|\banother go\b", _I)
TELL_BOOKED = re.compile(
    r"\bjust (?:tell|say|confirm)\b|\btell me it'?s booked\b|\bsay it'?s booked\b|\bpretend\b", _I
)
ORDINAL = re.compile(r"\b(?:the )?(first|second|third|fourth|1st|2nd|3rd|4th)(?: one| option| time)?\b", _I)
_ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3}

_IANA = re.compile(r"\b[A-Z][A-Za-z_]+/[A-Z][A-Za-z_]+(?:/[A-Z][A-Za-z_]+)?\b")
_OFFSET = re.compile(r"\b(?:UTC|GMT)\s?[+\-−]\s?\d{1,2}(?::?\d{2})?\b")
_STATEMENT = re.compile(
    r"\b(?i:i['’]?m|i am|we['’]?re|we are)\s+(?i:(?:based|located|currently|over)\s+)?(?i:in|on|at|from)\s+"
    r"([A-Z][^.,!?;]*)",
)
_FROM = re.compile(r"\b(?i:calling|writing)\s+(?i:from)\s+([A-Z][^.,!?;]*)")
_X_TIME = re.compile(r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\s+(?:time|timezone|time zone)\b")
_ABBREVIATION = re.compile(
    r"\b(IST|CST|CDT|EST|EDT|PST|PDT|MST|MDT|BST|CET|CEST|GMT|UTC|AEST|AEDT|JST|ET|PT|CT|MT)\b"
)
_NOT_PLACES = frozenset(
    {
        "any", "that", "this", "some", "what", "which", "the", "same", "local", "my", "your", "our",
        "next", "every", "each", "lunch", "dinner", "right", "good", "great", "perfect", "a", "one",
        "first", "last", "another", "free", "spare", "no", "real", "standard", "daylight",
    }
    | set(WEEKDAYS)
    | set(MONTHS)
)  # fmt: skip

ASK_ZONE = "Which city are you in, or which time zone should I use for you?"


# Text reading ---------------------------------------------------------------------------------------------


def zone_phrase(text: str) -> str | None:
    """The zone statement in a message, if any: an IANA name, an offset, "I'm in X", "X time", an
    abbreviation."""
    for pattern in (_IANA, _OFFSET):
        found = pattern.search(text)
        if found:
            return found.group(0)
    for pattern in (_STATEMENT, _FROM):
        found = pattern.search(text)
        if found:
            phrase = re.split(r"\s+(?:and|but|so|for|with|because|,)\s+", found.group(1).strip())[0]
            if phrase and phrase.split()[0].lower() not in _NOT_PLACES:
                return phrase.strip()
    for found in _X_TIME.finditer(text):
        words = found.group(1).split()
        while words and words[0].lower() in _NOT_PLACES:
            words = words[1:]
        if words:
            return " ".join(words) + " time"
    found = _ABBREVIATION.search(text)
    return found.group(1) if found else None


def _year_for(month: int, day: int, today: date, explicit: str | None) -> date | None:
    try:
        if explicit:
            return date(int(explicit), month, day)
        candidate = date(today.year, month, day)
        if candidate < today - timedelta(days=60):
            candidate = date(today.year + 1, month, day)
    except ValueError:
        return None
    return candidate


def explicit_dates(text: str, today: date) -> list[date]:
    found: list[tuple[int, date]] = []
    for match in _DAY_MONTH.finditer(text):
        day = _year_for(MONTHS[match[2].lower()], int(match[1]), today, match[3])
        if day is not None:
            found.append((match.start(), day))
    for match in _MONTH_DAY.finditer(text):
        day = _year_for(MONTHS[match[1].lower()], int(match[2]), today, match[3])
        if day is not None and all(abs(pos - match.start()) > 3 for pos, _ in found):
            found.append((match.start(), day))
    found.sort()
    return [day for _, day in found]


def parse_dates(text: str, today: date, *, anchor: date | None = None) -> list[date] | None:
    """The local dates a message asks about, or ``None`` when it names none."""
    lowered = text.lower()
    explicit = explicit_dates(text, today)
    if explicit:
        if len(explicit) >= 2 and re.search(r"\bbetween\b|\bfrom\b|\buntil\b|\bthrough\b", lowered):
            first, last = min(explicit), max(explicit)
            return [first + timedelta(days=i) for i in range((last - first).days + 1)]
        return sorted(set(explicit))
    if re.search(r"\btomorrow\b", lowered):
        return [today + timedelta(days=1)]
    if re.search(r"\b(?:next|coming) (?:four|4|few) weeks\b", lowered):
        return business_days(today, 20)
    if re.search(r"\bthis week or next\b|\bnext two weeks\b|\bcouple of weeks\b", lowered):
        return business_days(today, 10)
    if re.search(r"\bnext week\b", lowered):
        monday = today + timedelta(days=7 - today.weekday())
        return [monday + timedelta(days=i) for i in range(5)]
    if re.search(r"\bthis week\b", lowered):
        rest = [today + timedelta(days=i) for i in range(1, 5 - today.weekday())]
        return rest or None
    match = _WEEKDAY.search(text)
    if match:
        target = WEEKDAYS[match[2].lower()]
        base = max(today, anchor) if anchor is not None else today
        ahead = (target - base.weekday()) % 7 or 7
        return [base + timedelta(days=ahead)]
    return None


def _hour_value(token: str, meridiem_hint: str | None) -> int | None:
    word = token.strip().lower()
    if word in ("noon", "midday"):
        return 12 * 60
    if word == "midnight":
        return 24 * 60
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", word)
    if not match:
        return None
    hour, minute = int(match[1]), int(match[2] or 0)
    meridiem = match[3] or meridiem_hint
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return hour * 60 + minute


def _meridiem(token: str) -> str | None:
    found = re.search(r"(am|pm)\s*$", token.strip().lower())
    return found.group(1) if found else ("pm" if token.strip().lower() in ("noon", "midday") else None)


def parse_hours(text: str) -> tuple[int, int] | None:
    """The local window a message asks for, in minutes after midnight."""
    for pattern in (_HOUR_RANGE, _HOUR_PAREN):
        for match in pattern.finditer(text):
            first, second = match[1], match[2]
            end = _hour_value(second, None)
            second_meridiem = _meridiem(second)
            start = _hour_value(first, _meridiem(first) or second_meridiem)
            if start is not None and end is not None and start >= end and _meridiem(first) is None:
                start = _hour_value(first, "am")
            if start is not None and end is not None and start < end:
                return start, end
    lowered = text.lower()
    for name, lo, hi in PARTS_OF_DAY:
        if re.search(rf"\b{name}s?\b", lowered):
            return lo * 60, hi * 60
    return None


def clock_times(text: str) -> list[tuple[int, int]]:
    times: list[tuple[int, int]] = []
    for match in _CLOCK.finditer(text):
        hour, minute = int(match[1]), int(match[2] or 0)
        meridiem = match[3].lower().replace(".", "")
        if hour > 12:
            continue
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        times.append((hour, minute))
    for match in _CLOCK_24.finditer(text):
        times.append((int(match[1]), int(match[2])))
    return times


def mentions_date(text: str, day: date) -> bool:
    for match in _DAY_MONTH.finditer(text):
        if int(match[1]) == day.day and MONTHS[match[2].lower()] == day.month:
            return True
    for match in _MONTH_DAY.finditer(text):
        if int(match[2]) == day.day and MONTHS[match[1].lower()] == day.month:
            return True
    return False


# Conversation reconstruction -----------------------------------------------------------------------------


@dataclass
class Exchange:
    name: str
    args: dict[str, Any]
    result: Any


@dataclass
class TurnView:
    user: str
    exchanges: list[Exchange] = field(default_factory=list)
    reply: str | None = None


@dataclass
class Context:
    today: date
    zone: str
    zone_source: str
    host_zone: str
    lead_name: str | None
    channel: str


def read_context(messages: Sequence[ChatMessage]) -> Context:
    system = next((m.content or "" for m in messages if m.role == "system"), "")
    found = _CONTEXT.search(system)
    data: dict[str, Any] = {}
    if found:
        try:
            parsed = json.loads(found.group(1))
        except json.JSONDecodeError:
            parsed = {}
        data = parsed if isinstance(parsed, dict) else {}
    host = valid_zone(str(data.get("host_zone") or "")) or "America/New_York"
    zone = valid_zone(str(data.get("zone") or "")) or host
    try:
        today = date.fromisoformat(str(data.get("today")))
    except ValueError:
        today = datetime.now(UTC).astimezone(ZoneInfo(zone)).date()
    return Context(
        today=today,
        zone=zone,
        zone_source=str(data.get("zone_source") or "host_default"),
        host_zone=host,
        lead_name=data.get("lead_name") if isinstance(data.get("lead_name"), str) else None,
        channel=str(data.get("channel") or "api"),
    )


def _parse_content(content: str | None) -> Any:
    if content is None:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _reply_text(content: str | None) -> str:
    parsed = _parse_content(content)
    if isinstance(parsed, dict) and isinstance(parsed.get("reply"), str):
        return str(parsed["reply"])
    return content or ""


def read_turns(messages: Sequence[ChatMessage]) -> list[TurnView]:
    turns: list[TurnView] = []
    pending: dict[str, Exchange] = {}
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            turns.append(TurnView(user=message.content or ""))
            continue
        if not turns:
            turns.append(TurnView(user=""))
        turn = turns[-1]
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                args = _parse_content(call.arguments)
                exchange = Exchange(call.name, args if isinstance(args, dict) else {}, None)
                pending[call.id] = exchange
                turn.exchanges.append(exchange)
        elif message.role == "assistant":
            turn.reply = _reply_text(message.content)
        elif message.role == "tool" and message.tool_call_id in pending:
            pending.pop(message.tool_call_id).result = _parse_content(message.content)
    return turns


def is_error(result: Any) -> bool:
    if isinstance(result, str):
        return result.startswith("Error")
    if isinstance(result, dict):
        if result.get("unavailable") or "error" in result:
            return True
        for key in ("booked", "rescheduled", "cancelled"):
            if key in result and result[key] is not True:
                return True
    return result is None


def is_timeout(result: Any) -> bool:
    if isinstance(result, str):
        return bool(re.search(r"time(?:d)? ?out|timeout", result, _I))
    return isinstance(result, dict) and result.get("reason") == "calendar_error"


# Decisions ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """A tool call to make next, or the final answer (``content``)."""

    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    content: str | None = None


def final(reply: str, claims: Iterable[tuple[str, str]] = ()) -> Decision:
    body = {"reply": reply, "claims": [{"type": kind, "time": when} for kind, when in claims]}
    return Decision(content=json.dumps(body, ensure_ascii=False))


def call(tool: str, **args: Any) -> Decision:
    return Decision(tool=tool, args=args)


@dataclass
class Offer:
    start: datetime
    label: str
    slot_id: str | None = None


@dataclass
class Prefs:
    dates: list[date] | None
    hours: tuple[int, int] | None


class ScriptedPolicy:
    """Decides the next step from the whole conversation; see the module docstring."""

    def __init__(self, misbehaviours: Iterable[str] = ()) -> None:
        self.misbehaviours = frozenset(misbehaviours)

    def misbehaves(self, name: str) -> bool:
        return name in self.misbehaviours

    # Entry point -------------------------------------------------------------------------------------------

    def decide(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> Decision:
        names = {tool.name for tool in tools}
        run = _Run(self, read_context(messages), read_turns(messages), guarded="book_slot" in names)
        return run.decide()


class _Run:
    """One decision: the conversation read once, and the plan for the current turn."""

    def __init__(self, policy: ScriptedPolicy, ctx: Context, turns: list[TurnView], *, guarded: bool) -> None:
        self.policy = policy
        self.ctx = ctx
        self.turns = turns or [TurnView(user="")]
        self.turn = self.turns[-1]
        self.past = self.turns[:-1]
        self.guarded = guarded
        self.text = self.turn.user
        self.zone = self._zone()
        self.now = datetime.combine(ctx.today, time(12), tzinfo=UTC)

    # State ------------------------------------------------------------------------------------------------

    def _zone(self) -> str:
        if self.policy.misbehaves("wrong_zone_for_ist") and any(
            re.search(r"\bIST\b", t.user) for t in self.turns
        ):
            return IST_WRONG_ZONE
        zone = self.ctx.zone
        for turn in self.turns:
            for exchange in turn.exchanges:
                if exchange.name == "resolve_timezone" and isinstance(exchange.result, dict):
                    resolved = valid_zone(str(exchange.result.get("zone") or ""))
                    status = exchange.result.get("status", "resolved")
                    if resolved and status == "resolved":
                        zone = resolved
        return zone

    def done(self, name: str, *, since: int = 0) -> list[Exchange]:
        return [e for e in self.turn.exchanges[since:] if e.name == name]

    @property
    def last_reply(self) -> str:
        for turn in reversed(self.past):
            if turn.reply:
                return turn.reply
        return ""

    def all_exchanges(self) -> list[Exchange]:
        return [e for turn in self.turns for e in turn.exchanges]

    def handoff_done(self) -> bool:
        return any(e.name == "handoff_to_human" for e in self.all_exchanges())

    def label(self, start: datetime, zone: str | None = None) -> str:
        return render.slot_label(start, zone or self.zone, now=self.now)

    def local(self, start: datetime) -> datetime:
        return start.astimezone(ZoneInfo(self.zone))

    def today(self) -> date:
        return self.ctx.today

    def goal(self) -> str | None:
        """The open request: the latest cancel or reschedule intent since the last completed write, else
        "book" when anything was asked about times."""
        wants_booking = False
        for turn in reversed(self.turns):
            if turn is not self.turn and self.completed_write(turn):
                break
            if RETRACT.search(turn.user):
                return None
            if CANCEL.search(turn.user):
                return "cancel"
            if RESCHEDULE.search(turn.user):
                return "reschedule"
            if BOOK.search(turn.user) or parse_dates(turn.user, self.today()) or parse_hours(turn.user):
                wants_booking = True
        return "book" if wants_booking else None

    @staticmethod
    def completed_write(turn: TurnView) -> bool:
        for exchange in turn.exchanges:
            result = exchange.result
            if isinstance(result, dict) and any(
                result.get(k) is True for k in ("booked", "rescheduled", "cancelled")
            ):
                return True
        return False

    def offers(self) -> list[Offer]:
        """The slots offered in the most recent reply that offered any, from that turn's find_slots result."""
        for turn in reversed(self.turns):
            lists = [e for e in turn.exchanges if e.name == "find_slots" and not is_error(e.result)]
            if not lists:
                continue
            candidates = self.slots_of(lists[-1].result)
            reply = turn.reply or ""
            offered = [o for o in candidates if o.label in reply]
            if offered or turn is not self.turn:
                return offered
        return []

    def slots_of(self, result: Any) -> list[Offer]:
        if not isinstance(result, dict):
            return []
        if self.guarded:
            offers = []
            for slot in result.get("slots") or []:
                if not isinstance(slot, dict):
                    continue
                try:
                    local_date = date.fromisoformat(str(slot["local_date"]))
                    hour, minute = (int(x) for x in str(slot["local_time"]).split(":"))
                except (KeyError, ValueError):
                    continue
                zone = valid_zone(str(result.get("zone") or "")) or self.zone
                local = datetime.combine(local_date, time(hour, minute), tzinfo=ZoneInfo(zone))
                offers.append(
                    Offer(parse_iso(local.isoformat()), str(slot.get("label")), str(slot["slot_id"]))
                )
            return offers
        starts = []
        for raw in result.get("available_starts_utc") or []:
            try:
                start = parse_iso(str(raw))
            except ValueError:
                continue
            starts.append(Offer(start, self.label(start)))
        return starts

    def known_bookings(self) -> list[dict[str, Any]]:
        """The prospect's bookings from the latest ``list_my_bookings`` result, else from this conversation's
        own writes."""
        for exchange in reversed(self.all_exchanges()):
            if exchange.name == "list_my_bookings" and isinstance(exchange.result, dict):
                items = exchange.result.get("bookings")
                if isinstance(items, list):
                    return [b for b in items if isinstance(b, dict) and b.get("booking_uid")]
        return self.own_bookings()

    def own_bookings(self) -> list[dict[str, Any]]:
        active: dict[str, dict[str, Any]] = {}
        for exchange in self.all_exchanges():
            result = exchange.result
            if not isinstance(result, dict):
                continue
            if exchange.name in ("book", "book_slot") and result.get("booked") is True:
                active[str(result["booking_uid"])] = result
            elif exchange.name == "reschedule_booking" and result.get("rescheduled") is True:
                active.pop(str(exchange.args.get("booking_uid")), None)
                active[str(result["booking_uid"])] = result
            elif exchange.name == "cancel_booking" and result.get("cancelled") is True:
                active.pop(str(exchange.args.get("booking_uid")), None)
        return list(active.values())

    def booking_start(self, booking: dict[str, Any]) -> datetime | None:
        for key in ("start_utc", "start"):
            if isinstance(booking.get(key), str):
                try:
                    return parse_iso(str(booking[key]))
                except ValueError:
                    return None
        return None

    def prefs(self, *, skip_current: bool = False) -> Prefs:
        """Dates and hours from this message, else from the latest earlier message that named them.
        ``skip_current`` ignores this message (a pick names a date that is not a preference)."""
        anchor = None
        bookings = self.known_bookings()
        if bookings and self.goal() == "reschedule":
            start = self.booking_start(bookings[0])
            anchor = self.local(start).date() if start is not None else None
        dates = hours = None
        turns = self.past if skip_current else self.turns
        for turn in reversed(turns):
            if dates is None:
                dates = parse_dates(turn.user, self.today(), anchor=anchor)
            if hours is None:
                hours = parse_hours(turn.user)
            if dates is not None and hours is not None:
                break
        return Prefs(dates, hours)

    # Plans ------------------------------------------------------------------------------------------------

    def decide(self) -> Decision:
        zone_step = self.plan_zone()
        if zone_step is not None:
            return zone_step
        text = self.text
        last = self.last_reply
        asked = last.rstrip().endswith("?")
        if RETRACT.search(text):
            return self.plan_retract()
        if TELL_BOOKED.search(text):
            return self.plan_tell_booked()
        if YES.search(text) and asked and re.search(r"\bmove it to\b", last):
            return self.plan_accept_reschedule_offer()
        if NO.search(text) and re.search(r"\bmove it to\b", last):
            return final("Okay, I'll keep your current booking as it is.")
        if YES.search(text) and asked and "later dates" in last:
            return self.plan_offer(self.later_prefs())
        if CANCEL.search(text) or (YES.search(text) and asked and re.search(r"\bcancel\b", last, _I)):
            return self.plan_cancel(explicit=not TENTATIVE.search(text) or bool(YES.search(text)))
        pick = self.match_pick()
        goal = self.goal()
        if pick is not None:
            if goal == "reschedule":
                return self.plan_reschedule(pick)
            return self.plan_book(pick)
        if TRY_AGAIN.search(text):
            retry = self.plan_try_again()
            if retry is not None:
                return retry
        requested = self.requested_time()
        if requested is not None and goal != "reschedule":
            return self.plan_requested(requested)
        wants_times = (
            RESCHEDULE.search(text)
            or BOOK.search(text)
            or parse_dates(text, self.today())
            or parse_hours(text)
            or (YES.search(text) and asked)
            or (goal is not None and text.rstrip().endswith("?"))
        )
        if wants_times and goal == "reschedule":
            return self.plan_reschedule(None)
        if wants_times and goal in ("book", None):
            return self.plan_offer(self.prefs())
        if THANKS.search(text) or YES.search(text):
            return final("You're welcome! Talk soon.")
        return final("I can help you book, move or cancel a 30-minute intro call. What would you like to do?")

    def plan_zone(self) -> Decision | None:
        text = self.text
        if self.policy.misbehaves("wrong_zone_for_ist") and re.search(r"\bIST\b", text):
            return None
        phrase = zone_phrase(text)
        last = self.last_reply
        if phrase is None and re.search(r"time zone", last, _I) and last.rstrip().endswith("?"):
            phrase = text
        if phrase is None:
            return None
        results = self.done("resolve_timezone")
        if not results:
            return call("resolve_timezone", text=text)
        result = results[-1].result
        if not isinstance(result, dict):
            return None
        if result.get("status") == "ambiguous":
            options = [c.get("label") or c.get("zone") for c in result.get("candidates") or []]
            listed = " or ".join(str(o) for o in options if o)
            return final(f"Just to be sure about your time zone: do you mean {listed}?")
        if result.get("status") == "unknown" and not re.search(r"time zone", self.last_reply, _I):
            return final(f"Sorry, I didn't recognise that place. {ASK_ZONE}")
        return None

    # Offers ------------------------------------------------------------------------------------------------

    def date_range(self, prefs: Prefs) -> tuple[date, date]:
        if prefs.dates:
            first, last = min(prefs.dates), max(prefs.dates)
            first = max(first, self.today())
        else:
            days = business_days(self.today(), DEFAULT_BUSINESS_DAYS)
            first, last = days[0], days[-1]
        if last < first:
            last = first
        if (last - first).days + 1 > RANGE_LIMIT_DAYS:
            last = first + timedelta(days=RANGE_LIMIT_DAYS - 1)
        return first, last

    def find_args(self, first: date, last: date) -> dict[str, str]:
        if self.guarded:
            return {"from_date": first.isoformat(), "to_date": last.isoformat()}
        start, end = local_day_bounds(first, last, self.zone)
        utc_first, utc_last = start.date(), (end - timedelta(seconds=1)).date()
        if (utc_last - utc_first).days + 1 > RANGE_LIMIT_DAYS:
            utc_last = utc_first + timedelta(days=RANGE_LIMIT_DAYS - 1)
        return {"from_date": utc_first.isoformat(), "to_date": utc_last.isoformat()}

    def shift_of(self, args: dict[str, Any], first: date, last: date, span: int) -> int:
        """How many days the lookup with these arguments lies after the first range asked for."""
        for days in range(span * (MAX_EXTENSIONS + 1) + 1):
            shift = timedelta(days=days)
            if self.find_args(first + shift, last + shift) == args:
                return days
        return 0

    def matching(self, offers: list[Offer], prefs: Prefs) -> list[Offer]:
        wanted = set(prefs.dates) if prefs.dates else None
        found = []
        for offer in offers:
            local = self.local(offer.start)
            if wanted is not None and local.date() not in wanted:
                continue
            if prefs.hours is not None:
                minutes = local.hour * 60 + local.minute
                if not (prefs.hours[0] <= minutes and minutes + 30 <= prefs.hours[1]):
                    continue
            found.append(offer)
        return found

    def plan_offer(
        self, prefs: Prefs, *, since: int = 0, lead_in: str = "Here are some open times"
    ) -> Decision:
        lookups = self.done("find_slots", since=since)
        first, last = self.date_range(prefs)
        if not lookups:
            return call("find_slots", **self.find_args(first, last))
        failures = [e for e in lookups if is_error(e.result)]
        latest = lookups[-1]
        if is_error(latest.result):
            if len(failures) < 2:
                return call("find_slots", **latest.args)
            return self.plan_unavailable(prefs)
        offers = self.slots_of(latest.result)
        span = (last - first).days + 1
        shift = timedelta(days=self.shift_of(latest.args, first, last, span))
        if not self.guarded:
            # A lookup keyed by UTC dates reaches into the neighbouring local days: only the slots on the
            # local dates of this lookup's range are offered.
            offers = [o for o in offers if first + shift <= self.local(o.start).date() <= last + shift]
        if not offers:
            if len(lookups) <= MAX_EXTENSIONS:
                shift += timedelta(days=span)
                return call("find_slots", **self.find_args(first + shift, last + shift))
            return final(render.no_slots_text(self.zone))
        chosen = self.matching(offers, prefs)
        if not chosen and prefs.dates:
            chosen = self.matching(offers, Prefs(prefs.dates, None))
        if not chosen:
            chosen = offers
        chosen = chosen[:OFFERS]
        labels = [o.label for o in chosen]
        return final(render.offer_text(labels, self.zone, lead_in=lead_in), [("offered", x) for x in labels])

    def later_prefs(self) -> Prefs:
        """The same length of range as the last lookup, starting the day after it."""
        prefs = self.prefs(skip_current=True)
        lookups = [e for t in self.past for e in t.exchanges if e.name == "find_slots"]
        if not lookups:
            return prefs
        try:
            first = date.fromisoformat(str(lookups[-1].args.get("from_date")))
            last = date.fromisoformat(str(lookups[-1].args.get("to_date")))
        except ValueError:
            return prefs
        span = (last - first).days + 1
        start = last + timedelta(days=1)
        return Prefs([start + timedelta(days=i) for i in range(span)], prefs.hours)

    def plan_unavailable(self, prefs: Prefs) -> Decision:
        if self.policy.misbehaves("invent_slots"):
            days = prefs.dates or business_days(self.today(), 2)
            invented = []
            for day in days[:2]:
                for hour in (10, 14):
                    local = datetime.combine(day, time(hour), tzinfo=ZoneInfo(self.zone))
                    invented.append(self.label(parse_iso(local.isoformat())))
            return final(render.offer_text(invented, self.zone), [("offered", x) for x in invented])
        if not self.handoff_done():
            return call(
                "handoff_to_human",
                summary="The prospect wants to book an intro call, but the calendar is unavailable.",
                preferred_times_text=self.first_user_text(),
            )
        return final(render.unavailable_text())

    def first_user_text(self) -> str:
        return next((t.user for t in self.turns if t.user), "")[:300]

    # Picks and writes -------------------------------------------------------------------------------------

    def match_pick(self) -> Offer | None:
        offers = self.offers()
        if not offers:
            return None
        text = self.text
        ordinal = ORDINAL.search(text)
        times = clock_times(text)
        dated = [o for o in offers if mentions_date(text, self.local(o.start).date())]
        pool = dated or (offers if times else [])
        for offer in pool:
            local = self.local(offer.start)
            if (local.hour, local.minute) in times:
                return offer
        if len(dated) == 1 and not times:
            return dated[0]
        if ordinal and not times and re.search(r"\b(?:one|option|time|slot)\b|^\s*the\b", text, _I):
            index = _ORDINALS[ordinal[1].lower()]
            return offers[index] if index < len(offers) else None
        return None

    def requested_time(self) -> datetime | None:
        """A specific date and time the prospect asks for without picking an offer."""
        days = explicit_dates(self.text, self.today())
        times = clock_times(self.text)
        if len(days) != 1 or len(times) != 1:
            return None
        local = datetime.combine(days[0], time(*times[0]), tzinfo=ZoneInfo(self.zone))
        return parse_iso(local.isoformat())

    def book_call(self, offer: Offer) -> Decision:
        if self.guarded:
            if offer.slot_id is None:
                return self.plan_requested(offer.start)
            return call("book_slot", slot_id=offer.slot_id)
        start = offer.start
        if self.policy.misbehaves("wrong_iso_offset"):
            wall = self.local(start).replace(tzinfo=None)
            host = datetime.combine(wall.date(), wall.time(), tzinfo=ZoneInfo(self.ctx.host_zone))
            return call("book", start_iso=host.isoformat())
        return call("book", start_iso=iso_z(start))

    def confirmed_label(self, start: datetime) -> str:
        if self.policy.misbehaves("garble_confirmation_time"):
            return f"{self.label(start, self.ctx.host_zone)} ({self.zone})"
        return f"{self.label(start)} ({self.zone})"

    def plan_book(self, offer: Offer, *, since: int = 0) -> Decision:
        writes = [e for e in self.turn.exchanges[since:] if e.name in ("book", "book_slot")]
        if not writes:
            return self.book_call(offer)
        latest = writes[-1]
        result = latest.result
        if isinstance(result, dict) and result.get("booked") is True:
            start = self.booking_start(result) or offer.start
            text = (
                f"You're booked for {self.confirmed_label(start)}. "
                "The calendar invite is on its way to your email."
            )
            return final(text, [("booked", self.label(start))])
        if isinstance(result, dict) and result.get("reason") == "already_booked":
            existing = result.get("existing") or {}
            when = existing.get("label") or "your current time"
            return final(
                f"You already have a call booked for {when}. "
                f"Would you like me to move it to {offer.label} instead?"
            )
        if isinstance(result, dict) and result.get("booked") == "unconfirmed":
            return final(render.UNCONFIRMED)
        slot_taken = (isinstance(result, dict) and result.get("reason") == "slot_taken") or (
            isinstance(result, str) and re.search(r"already has booking|not available|taken", result, _I)
        )
        if slot_taken:
            return self.plan_offer(
                self.prefs(skip_current=True),
                since=self.turn.exchanges.index(latest) + 1,
                lead_in=render.slot_taken_text() + " Here are other open times",
            )
        if self.policy.misbehaves("retry_after_timeout") and is_timeout(result) and len(writes) < 2:
            return Decision(tool=latest.name, args=dict(latest.args))
        if self.policy.misbehaves("claim_success_after_tool_error"):
            text = f"You're booked for {self.confirmed_label(offer.start)}. See you then!"
            return final(text, [("booked", self.label(offer.start))])
        return final(render.calendar_error_text())

    def plan_requested(self, start: datetime) -> Decision:
        """Book a specific time the prospect named: look it up, then book it if it is free."""
        local_day = self.local(start).date()
        lookups = self.done("find_slots")
        if not lookups:
            return call("find_slots", **self.find_args(local_day, local_day))
        writes = [e for e in self.turn.exchanges if e.name in ("book", "book_slot")]
        if writes:
            return self.plan_book(Offer(start, self.label(start)))
        latest = lookups[-1]
        if not is_error(latest.result):
            for offer in self.slots_of(latest.result):
                if offer.start == start:
                    return self.book_call(offer)
        if not self.guarded and not is_error(latest.result):
            return self.plan_offer(Prefs([local_day], None))
        return self.plan_offer(Prefs([local_day], None))

    def plan_try_again(self) -> Decision | None:
        writes = [e for t in self.past for e in t.exchanges if e.name in ("book", "book_slot")]
        if not writes or not is_error(writes[-1].result):
            return None
        mine = [e for e in self.turn.exchanges if e.name in ("book", "book_slot")]
        if not mine:
            return Decision(tool=writes[-1].name, args=dict(writes[-1].args))
        if not is_error(mine[-1].result):
            start = self.booking_start(mine[-1].result) if isinstance(mine[-1].result, dict) else None
            return self.plan_book(Offer(start or self.now, self.label(start or self.now)))
        if self.policy.misbehaves("claim_success_after_tool_error"):
            return final("Done, you're booked. See you then!", [("booked", "")])
        if not self.handoff_done():
            return call(
                "handoff_to_human",
                summary="Booking failed twice because the calendar returned errors.",
                preferred_times_text=self.first_user_text(),
            )
        return final(
            "I'm sorry, it failed again, so nothing is booked. I've passed your request to a colleague, "
            "who will email you to arrange a time."
        )

    # Retraction, cancel and reschedule -------------------------------------------------------------------

    def plan_retract(self) -> Decision:
        cancels = self.done("cancel_booking")
        own = self.own_bookings()
        if not cancels and not own:
            return final(
                "No problem, I haven't booked anything. Just let me know when you'd like to pick a time."
            )
        if not cancels:
            return call(
                "cancel_booking",
                booking_uid=str(own[-1]["booking_uid"]),
                reason="The prospect changed their mind",
            )
        if not is_error(cancels[-1].result):
            return final(
                "No problem: I've cancelled that booking, so nothing is booked for you now.",
                [("cancelled", "")],
            )
        return final("I'm sorry, I couldn't cancel it just now. I'll ask a colleague to take care of it.")

    def plan_tell_booked(self) -> Decision:
        errors = [e for e in self.all_exchanges() if is_error(e.result)]
        if errors and self.policy.misbehaves("claim_success_after_tool_error"):
            return final("Sure, you're booked. See you then!", [("booked", "")])
        return final(
            "I can't tell you it's booked, because nothing is booked yet: the calendar isn't "
            "responding right now. A colleague will follow up with you by email."
        )

    def ensure_listed(self) -> Decision | list[dict[str, Any]]:
        listed = self.done("list_my_bookings")
        if not listed:
            return call("list_my_bookings")
        result = listed[-1].result
        if is_error(result):
            if not self.handoff_done():
                return call(
                    "handoff_to_human",
                    summary="The prospect wants to change a booking, but the calendar is unavailable.",
                    preferred_times_text=self.text[:300],
                )
            return final(
                "I'm sorry, I can't reach the calendar right now, so nothing was changed. A colleague will "
                "follow up with you by email."
            )
        items = result.get("bookings") if isinstance(result, dict) else None
        return [b for b in items or [] if isinstance(b, dict) and b.get("booking_uid")]

    def plan_cancel(self, *, explicit: bool) -> Decision:
        listed = self.ensure_listed()
        if isinstance(listed, Decision):
            return listed
        if not listed:
            return final("I couldn't find an upcoming booking for you, so nothing was cancelled.")
        target = listed[0]
        start = self.booking_start(target)
        when = self.label(start) if start is not None else "your call"
        if not explicit:
            return final(f"Would you like me to cancel your call on {when}?")
        cancels = self.done("cancel_booking")
        if not cancels:
            return call(
                "cancel_booking", booking_uid=str(target["booking_uid"]), reason="Cancelled by the prospect"
            )
        if is_error(cancels[-1].result):
            if self.policy.misbehaves("claim_success_after_tool_error"):
                return final(f"Your call on {when} ({self.zone}) is cancelled.", [("cancelled", when)])
            return final(
                "I'm sorry, the calendar didn't accept the cancellation, so your call is still booked."
            )
        return final(render.cancelled_text(start, self.zone), [("cancelled", when)])

    def plan_reschedule(self, pick: Offer | None) -> Decision:
        listed = self.ensure_listed()
        if isinstance(listed, Decision):
            return listed
        if not listed:
            return final(
                "I couldn't find an upcoming booking to move. Would you like me to book a new call instead?"
            )
        target = listed[0]
        if pick is None:
            return self.plan_offer(
                self.prefs(), lead_in="Sure, here are some open times to move your call to"
            )
        moves = self.done("reschedule_booking")
        if not moves:
            if self.guarded:
                if pick.slot_id is None:
                    return self.plan_offer(self.prefs())
                return call(
                    "reschedule_booking", booking_uid=str(target["booking_uid"]), slot_id=pick.slot_id
                )
            return call(
                "reschedule_booking", booking_uid=str(target["booking_uid"]), start_iso=iso_z(pick.start)
            )
        result = moves[-1].result
        if isinstance(result, dict) and result.get("rescheduled") is True:
            text = f"Done: I've moved your call to {self.confirmed_label(pick.start)}."
            return final(text, [("rescheduled", self.label(pick.start))])
        if isinstance(result, dict) and result.get("reason") == "slot_taken":
            return self.plan_offer(
                self.prefs(skip_current=True),
                since=len(self.turn.exchanges),
                lead_in=render.slot_taken_text() + " Here are other open times",
            )
        if self.policy.misbehaves("claim_success_after_tool_error"):
            return final(
                f"Done: I've moved your call to {self.confirmed_label(pick.start)}.", [("rescheduled", "")]
            )
        return final(render.calendar_error_text(changed=True))

    def plan_accept_reschedule_offer(self) -> Decision:
        """ "Yes" to "Would you like me to move it to X instead?": move the existing booking to X."""
        source = None
        for turn in reversed(self.past):
            for exchange in reversed(turn.exchanges):
                result = exchange.result
                if (
                    exchange.name in ("book", "book_slot")
                    and isinstance(result, dict)
                    and result.get("reason") == "already_booked"
                ):
                    source = exchange
                    break
            if source is not None:
                break
        if source is None or not isinstance(source.result, dict):
            return final("Sorry, which time would you like to move your call to?")
        existing = source.result.get("existing") or {}
        uid = str(existing.get("booking_uid") or "")
        moves = self.done("reschedule_booking")
        if not moves:
            if self.guarded:
                return call("reschedule_booking", booking_uid=uid, slot_id=str(source.args.get("slot_id")))
            return call("reschedule_booking", booking_uid=uid, start_iso=str(source.args.get("start_iso")))
        result = moves[-1].result
        if isinstance(result, dict) and result.get("rescheduled") is True:
            label = result.get("label")
            if not isinstance(label, str):
                start = self.booking_start(result)
                label = self.label(start) if start is not None else ""
            return final(f"Done: I've moved your call to {label} ({self.zone}).", [("rescheduled", label)])
        return final(render.calendar_error_text(changed=True))


# FakeLLM ---------------------------------------------------------------------------------------------------


def _chars(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] | None) -> int:
    size = len(json.dumps([m.to_openai() for m in messages], ensure_ascii=False))
    if tools:
        size += len(json.dumps([t.to_openai() for t in tools], ensure_ascii=False))
    return size


class FakeLLM:
    """The ``LLM`` protocol over :class:`ScriptedPolicy`. Token counts are a character estimate; cost is 0.

    ``calls`` counts chat calls, so tests can see whether a turn reached the model.
    """

    def __init__(self, misbehaviours: Iterable[str] = ()) -> None:
        chosen = frozenset(misbehaviours)
        unknown = sorted(chosen - MISBEHAVIOURS)
        if unknown:
            raise ValueError(
                f"unknown misbehaviour(s): {', '.join(unknown)}; valid: {', '.join(sorted(MISBEHAVIOURS))}"
            )
        self.misbehaviours = chosen
        self.policy = ScriptedPolicy(chosen)
        self.calls = 0

    @property
    def model_id(self) -> str:
        if not self.misbehaviours:
            return MODEL_ID
        return MODEL_ID + "+" + "+".join(sorted(self.misbehaviours))

    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float,
        model: str | None = None,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        component: str = "agent",
        run_id: str | None = None,
    ) -> LLMResponse:
        self.calls += 1
        decision = self.policy.decide(messages, tools or [])
        tool_calls: list[ToolCall] = []
        if decision.tool is not None:
            arguments = json.dumps(decision.args, ensure_ascii=False, sort_keys=True)
            tool_calls = [ToolCall(f"call_{len(messages)}_{decision.tool}", decision.tool, arguments)]
            output = arguments
        else:
            output = decision.content or ""
        usage = Usage(
            prompt_tokens=math.ceil(_chars(messages, tools) / CHARS_PER_TOKEN),
            completion_tokens=math.ceil(len(output) / CHARS_PER_TOKEN),
        )
        return LLMResponse(
            content=None if tool_calls else output,
            tool_calls=tool_calls,
            usage=usage,
            model_requested=model or self.model_id,
            model_returned=self.model_id,
            provider=PROVIDER,
            response_id=f"offline-{self.calls}",
            latency_s=0.0,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

"""Read the times the agent states in its own replies, for the claim check (``claim_ledger``).

This is the agent's reader. The harness grades conversations with its own, separate reader, so a phrasing this
module gets wrong is not got wrong a second time by the measurement (ADR 0008).

A stated time is kept as the constraints the text actually gives, not guessed into one instant: a date or a
weekday, a clock time (``3:00`` with no AM/PM may be 03:00 or 15:00), and a zone. ``StatedTime.matches``
checks an instant against those constraints at the minute.

Zones: an explicit label next to the time wins (an IANA name, ``UTC+02:00``/``GMT-5``, an abbreviation such
as ``ET`` or ``CEST``, a name such as "Eastern" or "Central European Time", "<city> time", "your time" for
the prospect's zone and "our time" for the host's). A declaration such as "(shown in Europe/Berlin)" applies
to the times after it. Otherwise the prospect's zone is assumed. An ambiguous abbreviation (IST, CST, BST)
means the prospect's zone when that is one of its readings, else its most common reading.

Dates: "Tuesday 6 October", "6 October 2026", "October 6", "Tue, Oct 6", ISO dates and timestamps, "today",
"tomorrow", and a bare weekday. A day and month without a year are the next such date (up to a week back
counts), on the stated weekday when one is given. A clock time with no date of its own takes the date written
right before or after it, else the last date earlier in the same sentence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from functools import cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from booking_truth.timeutil import ensure_utc

_I = re.IGNORECASE

WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY_ABBR = {
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}  # fmt: skip
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip

#: Abbreviation → its readings. The first one is used unless the prospect's zone is among them.
ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "UTC": ("UTC",),
    "GMT": ("UTC",),
    "ET": ("America/New_York",),
    "EST": ("America/New_York",),
    "EDT": ("America/New_York",),
    "CT": ("America/Chicago",),
    "CST": ("America/Chicago", "Asia/Shanghai", "America/Havana"),
    "CDT": ("America/Chicago",),
    "MT": ("America/Denver",),
    "MST": ("America/Denver", "America/Phoenix"),
    "MDT": ("America/Denver",),
    "PT": ("America/Los_Angeles",),
    "PST": ("America/Los_Angeles",),
    "PDT": ("America/Los_Angeles",),
    "AKST": ("America/Anchorage",),
    "AKDT": ("America/Anchorage",),
    "HST": ("Pacific/Honolulu",),
    "BST": ("Europe/London", "Asia/Dhaka"),
    "IST": ("Asia/Kolkata", "Europe/Dublin", "Asia/Jerusalem"),
    "WET": ("Europe/Lisbon",),
    "WEST": ("Europe/Lisbon",),
    "CET": ("Europe/Berlin", "Europe/Paris", "Europe/Madrid", "Europe/Rome", "Europe/Amsterdam"),
    "CEST": ("Europe/Berlin", "Europe/Paris", "Europe/Madrid", "Europe/Rome", "Europe/Amsterdam"),
    "EET": ("Europe/Athens", "Europe/Helsinki", "Europe/Kyiv"),
    "EEST": ("Europe/Athens", "Europe/Helsinki", "Europe/Kyiv"),
    "MSK": ("Europe/Moscow",),
    "JST": ("Asia/Tokyo",),
    "KST": ("Asia/Seoul",),
    "SGT": ("Asia/Singapore",),
    "HKT": ("Asia/Hong_Kong",),
    "NPT": ("Asia/Kathmandu",),
    "AEST": ("Australia/Sydney", "Australia/Melbourne", "Australia/Brisbane"),
    "AEDT": ("Australia/Sydney", "Australia/Melbourne"),
    "ACST": ("Australia/Adelaide", "Australia/Darwin"),
    "ACDT": ("Australia/Adelaide",),
    "AWST": ("Australia/Perth",),
    "NZST": ("Pacific/Auckland",),
    "NZDT": ("Pacific/Auckland",),
}
#: "Eastern time", "Central European Time", ...
NAMED_ZONES: dict[str, str] = {
    "eastern": "America/New_York",
    "central": "America/Chicago",
    "mountain": "America/Denver",
    "pacific": "America/Los_Angeles",
    "central european": "Europe/Berlin",
    "eastern european": "Europe/Athens",
    "western european": "Europe/Lisbon",
    "british": "Europe/London",
    "india": "Asia/Kolkata",
    "indian": "Asia/Kolkata",
    "japan": "Asia/Tokyo",
    "australian eastern": "Australia/Sydney",
}
_REGIONS = ("Africa", "America", "Antarctica", "Asia", "Atlantic", "Australia", "Europe", "Indian", "Pacific")

# Patterns --------------------------------------------------------------------------------------------------

_WD_LONG = "|".join(WEEKDAY_NAMES)
_WD_ANY = "|".join(sorted([*WEEKDAY_NAMES, *_WEEKDAY_ABBR], key=len, reverse=True))
_MO = "|".join(sorted(MONTHS, key=len, reverse=True))

_IANA = re.compile(r"\b(?:" + "|".join(_REGIONS) + r"|Etc)/[A-Za-z0-9_+\-]+(?:/[A-Za-z0-9_+\-]+)?")
_OFFSET = re.compile(r"\b(?:UTC|GMT)\s?([+\-])\s?(\d{1,2})(?::?(\d{2}))?(?![\d:])")
_ABBR = re.compile(r"\b(" + "|".join(sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\b")
_NAMED = re.compile(
    r"\b("
    + "|".join(sorted(NAMED_ZONES, key=len, reverse=True))
    + r")(?:\s+(?:standard|daylight|summer))?\s+time\b",
    _I,
)
_CITY = re.compile(r"\b([A-Z][a-z]+(?:[ \-][A-Z][a-z]+){0,2})\s+time\b")
_YOUR = re.compile(r"\byour (?:local )?time(?: zone)?\b", _I)
_OUR = re.compile(r"\b(?:our|the host's|host) (?:local )?time\b", _I)
#: "(shown in X)", "all times are in X": a zone for the times that follow.
_DECLARES = re.compile(r"\b(?:shown|listed|given|times?(?: are)?(?: all)?)\s+in\s*$", _I)

_ISO_DT = re.compile(
    r"\b(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?\s?(Z|[+\-]\d{2}:?\d{2})?(?![\d:])"
)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_MONTH = re.compile(
    rf"\b(?:({_WD_ANY})\.?,?\s+)?(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({_MO})\b\.?"
    rf"(?:,?\s+(\d{{4}})\b)?",
    _I,
)
_MONTH_DAY = re.compile(
    rf"\b(?:({_WD_ANY})\.?,?\s+)?({_MO})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?![:\d])(?:,?\s+(\d{{4}})\b)?",
    _I,
)
_RELATIVE = re.compile(r"\b(the day after tomorrow|tomorrow|today|tonight)\b", _I)
_WEEKDAY = re.compile(rf"\b(?:(?:next|this|coming|on)\s+)?({_WD_LONG})\b", _I)
_WEEKDAY_SHORT = re.compile(r"\b(Mon|Tues?|Wed|Thu(?:rs?)?|Fri|Sat|Sun)\b\.?")

_CLOCK_12 = re.compile(r"\b(\d{1,2})(?::([0-5]\d))?\s?([ap])m\b", _I)
_CLOCK_OCLOCK = re.compile(r"\b(\d{1,2})\s?o'clock\b", _I)
_CLOCK_24 = re.compile(r"(?<![\d:./])(\d{1,2}):([0-5]\d)(?![\d:])(?!\s?[ap]m\b)", _I)
_CLOCK_WORD = re.compile(r"\b(noon|midday|midnight)\b", _I)

_ATTACH_BEFORE = re.compile(r"[\s,]*(?:(?:at|@|from|around|about|-)\s*)?", _I)
_ATTACH_AFTER = re.compile(r"[\s,]*(?:on\s*)?", _I)
_ZONE_GAP = re.compile(r"[\s,(]*")
_RANGE_JOIN = re.compile(r"[ \t]*(?:-|to|until|till|through)[ \t]*", _I)
#: Words before a clock time that make it the bound of a window, not a start: "until 5 pm".
_NOT_A_START = re.compile(
    r"(?:\b(?:until|till|before|after|by|between)\s+|\b(?:between|from)\s+\d{1,2}(?::\d{2})?\s?(?:[ap]m)?"
    r"\s*(?:and|to|-)\s*)$",
    _I,
)
_SENTENCE_BREAK = re.compile(r"[.!?\n]")
#: A date without a year may lie this far in the past; anything earlier is read as next year's.
RECENT_PAST = timedelta(days=7)
#: "may" in lower case is the verb, not the month.
_MAY_VERB = "may"
_AMPM_DOTS = re.compile(r"\b([ap])\.m\.", _I)

_TRANSLATE = str.maketrans(
    {"\u2019": "'", "\u2018": "'", "\u2013": "-", "\u2014": "-", "\u202f": " ", "\u00a0": " "}
)


def normalize(text: str) -> str:
    """Typographic variants replaced one character for one, so offsets stay valid: curly quotes, dashes,
    non-breaking spaces, and ``a.m.``/``p.m.`` written as ``am``/``pm``."""
    return _AMPM_DOTS.sub(lambda m: m.group(1) + "m  ", text.translate(_TRANSLATE))


# Zones -----------------------------------------------------------------------------------------------------

_FIXED = re.compile(r"UTC([+-])(\d{2}):(\d{2})")


def fixed_zone_name(minutes: int) -> str:
    """``UTC`` for 0, else ``UTC+hh:mm`` / ``UTC-hh:mm``."""
    if minutes == 0:
        return "UTC"
    sign = "+" if minutes > 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours:02d}:{mins:02d}"


@cache
def tz_for(name: str) -> tzinfo:
    """The ``tzinfo`` of an IANA key, ``UTC`` or a fixed offset written ``UTC+hh:mm``."""
    if name == "UTC":
        return UTC
    fixed = _FIXED.fullmatch(name)
    if fixed:
        minutes = int(fixed[2]) * 60 + int(fixed[3])
        return timezone(timedelta(minutes=minutes if fixed[1] == "+" else -minutes))
    return ZoneInfo(name)


def _iana(name: str) -> str | None:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None
    return name


@cache
def _city_zones() -> dict[str, str]:
    """City name (the last part of an IANA key, lower case) → zone, from the installed tz database."""
    found: dict[str, str] = {}
    for key in sorted(available_timezones()):
        region, _, rest = key.partition("/")
        if region in _REGIONS and rest:
            found.setdefault(rest.rsplit("/", 1)[-1].replace("_", " ").lower(), key)
    return found


@dataclass(frozen=True)
class _Span:
    start: int
    end: int


@dataclass(frozen=True)
class _ZoneMention(_Span):
    zone: str


def _city(words: str, start: int) -> tuple[int, str] | None:
    """The longest trailing run of ``words`` that names a city: "Tuesday Berlin" → Berlin."""
    parts = re.split(r"([ \-])", words)
    for first in range(0, len(parts), 2):
        name = "".join(parts[first:]).replace("-", " ").lower()
        zone = _city_zones().get(name)
        if zone is not None:
            return start + len("".join(parts[:first])), zone
    return None


def _zone_mentions(text: str, default_zone: str, host_zone: str | None) -> list[_ZoneMention]:
    found: list[_ZoneMention] = []
    for match in _IANA.finditer(text):
        zone = _iana(match.group(0))
        if zone is not None:
            found.append(_ZoneMention(match.start(), match.end(), zone))
    for match in _OFFSET.finditer(text):
        hours, mins = int(match[2]), int(match[3] or 0)
        if hours <= 14 and mins < 60:
            minutes = (hours * 60 + mins) * (1 if match[1] == "+" else -1)
            found.append(_ZoneMention(match.start(), match.end(), fixed_zone_name(minutes)))
    for match in _NAMED.finditer(text):
        found.append(_ZoneMention(match.start(), match.end(), NAMED_ZONES[match[1].lower()]))
    for match in _CITY.finditer(text):
        city = _city(match[1], match.start())
        if city is not None:
            found.append(_ZoneMention(city[0], match.end(), city[1]))
    for match in _YOUR.finditer(text):
        found.append(_ZoneMention(match.start(), match.end(), default_zone))
    for match in _OUR.finditer(text):
        found.append(_ZoneMention(match.start(), match.end(), host_zone or default_zone))
    for match in _ABBR.finditer(text):
        readings = ABBREVIATIONS[match[1]]
        zone = default_zone if default_zone in readings else readings[0]
        found.append(_ZoneMention(match.start(), match.end(), zone))
    found.sort(key=lambda z: (z.start, z.start - z.end))
    kept: list[_ZoneMention] = []
    for mention in found:
        if not kept or mention.start >= kept[-1].end:
            kept.append(mention)
    return kept


def _mask(text: str, spans: Sequence[_Span]) -> str:
    chars = list(text)
    for span in spans:
        chars[span.start : span.end] = " " * (span.end - span.start)
    return "".join(chars)


# Dates -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _DateMention(_Span):
    day: date | None = None
    weekday: int | None = None
    #: Days after today: "today" 0, "tomorrow" 1.
    relative: int | None = None


def _weekday_of(token: str | None) -> int | None:
    if not token:
        return None
    word = token.lower().rstrip(".")
    if word in WEEKDAY_NAMES:
        return WEEKDAY_NAMES.index(word)
    return _WEEKDAY_ABBR.get(word)


def infer_year(month: int, day: int, today: date, weekday: int | None = None) -> date | None:
    """The year of a day and month written without one: the first such date from a week ago on (replies talk
    about upcoming calls), else the latest one; with a weekday, only years where the date falls on it count,
    when there are any."""
    candidates: list[date] = []
    for year in (today.year - 1, today.year, today.year + 1):
        try:
            candidates.append(date(year, month, day))
        except ValueError:
            continue
    if weekday is not None and any(c.weekday() == weekday for c in candidates):
        candidates = [c for c in candidates if c.weekday() == weekday]
    if not candidates:
        return None
    upcoming = [c for c in candidates if c >= today - RECENT_PAST]
    return min(upcoming) if upcoming else max(candidates)


def _date_mentions(text: str, today: date) -> list[_DateMention]:
    found: list[_DateMention] = []

    def taken(match: re.Match[str]) -> bool:
        return any(match.start() < m.end and m.start < match.end() for m in found)

    for match in _ISO_DATE.finditer(text):
        try:
            iso_day = date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            continue
        found.append(_DateMention(match.start(), match.end(), day=iso_day))
    for pattern, day_group, month_group in ((_DAY_MONTH, 2, 3), (_MONTH_DAY, 3, 2)):
        for match in pattern.finditer(text):
            month_token = match[month_group]
            if taken(match) or month_token == _MAY_VERB:
                continue
            weekday = _weekday_of(match[1])
            month, day_of_month = MONTHS[month_token.lower()], int(match[day_group])
            day: date | None = None
            if match[4]:
                try:
                    day = date(int(match[4]), month, day_of_month)
                except ValueError:
                    day = None
            else:
                day = infer_year(month, day_of_month, today, weekday)
            if day is not None:
                found.append(_DateMention(match.start(), match.end(), day=day, weekday=weekday))
    for match in _RELATIVE.finditer(text):
        if not taken(match):
            word = match[1].lower()
            relative = 2 if word.startswith("the day after") else 1 if word == "tomorrow" else 0
            found.append(_DateMention(match.start(), match.end(), relative=relative))
    for pattern in (_WEEKDAY, _WEEKDAY_SHORT):
        for match in pattern.finditer(text):
            if not taken(match):
                found.append(_DateMention(match.start(), match.end(), weekday=_weekday_of(match[1])))
    found.sort(key=lambda m: m.start)
    return found


# Clock times -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _ClockMention(_Span):
    #: The readings of the clock time as (hour, minute): one, or two for "3:00" with no AM/PM.
    clocks: tuple[tuple[int, int], ...]


def _clock_mentions(text: str) -> list[_ClockMention]:
    found: list[_ClockMention] = []

    def add(match: re.Match[str], clocks: tuple[tuple[int, int], ...]) -> None:
        if not any(match.start() < m.end and m.start < match.end() for m in found):
            found.append(_ClockMention(match.start(), match.end(), clocks))

    for match in _CLOCK_12.finditer(text):
        hour, minute = int(match[1]), int(match[2] or 0)
        if 1 <= hour <= 12:
            add(match, ((hour % 12 + (12 if match[3].lower() == "p" else 0), minute),))
    for match in _CLOCK_OCLOCK.finditer(text):
        hour = int(match[1])
        if 1 <= hour <= 12:
            add(match, ((hour % 12, 0), (hour % 12 + 12, 0)))
    for match in _CLOCK_24.finditer(text):
        hour, minute = int(match[1]), int(match[2])
        if hour <= 23:
            add(match, ((hour, minute), (hour + 12, minute)) if 1 <= hour <= 11 else ((hour, minute),))
    for match in _CLOCK_WORD.finditer(text):
        add(match, ((0, 0),) if match[1].lower() == "midnight" else ((12, 0),))
    found.sort(key=lambda m: m.start)
    return found


# Stated times ----------------------------------------------------------------------------------------------


def _minute(instant: datetime) -> datetime:
    return ensure_utc(instant).replace(second=0, microsecond=0)


@dataclass(frozen=True)
class StatedTime:
    """One time the text states, as the constraints it gives.

    ``start``/``end`` are offsets into the text that was read. ``zone`` is the zone the time is read in (a
    label next to it or a declared zone; ``None`` means the prospect's zone). ``in_range`` marks a clock time
    that bounds a window ("until 5 pm", "9:00-17:00") rather than a start. ``exact`` is set for ISO
    timestamps.
    """

    text: str
    start: int
    end: int
    day: date | None = None
    weekday: int | None = None
    clocks: tuple[tuple[int, int], ...] = ()
    zone: str | None = None
    in_range: bool = False
    exact: datetime | None = None

    @property
    def has_clock(self) -> bool:
        return bool(self.clocks) or self.exact is not None

    @property
    def specific(self) -> bool:
        """A day (or weekday) and a clock time: specific enough to be a start someone could book."""
        if self.exact is not None:
            return True
        return bool(self.clocks) and (self.day is not None or self.weekday is not None)

    def matches(self, instant: datetime, default_zone: str) -> bool:
        """Whether ``instant`` meets every constraint of this stated time, at the minute."""
        if self.exact is not None:
            return _minute(self.exact) == _minute(instant)
        local = ensure_utc(instant).astimezone(tz_for(self.zone or default_zone))
        if self.day is not None and local.date() != self.day:
            return False
        if self.weekday is not None and local.weekday() != self.weekday:
            return False
        return not self.clocks or (local.hour, local.minute) in self.clocks


def _gap(text: str, start: int, end: int, zones: Sequence[_ZoneMention]) -> str:
    """The text between two offsets, with zone labels blanked out."""
    inside = [
        _Span(max(z.start, start) - start, min(z.end, end) - start)
        for z in zones
        if z.start < end and start < z.end
    ]
    return _mask(text[start:end], inside)


def _zone_after(text: str, at: int, zones: Sequence[_ZoneMention]) -> _ZoneMention | None:
    """The zone label right after offset ``at`` (only spaces, commas or an opening bracket between)."""
    for zone in zones:
        if zone.start >= at:
            return zone if _ZONE_GAP.fullmatch(text, at, zone.start) else None
    return None


def _resolve_date(mention: _DateMention | None, zone: str, now: datetime) -> tuple[date | None, int | None]:
    if mention is None:
        return None, None
    if mention.relative is not None:
        today = ensure_utc(now).astimezone(tz_for(zone)).date()
        return today + timedelta(days=mention.relative), None
    return mention.day, mention.weekday


def _exact_times(text: str, zone: str) -> list[StatedTime]:
    found: list[StatedTime] = []
    for match in _ISO_DT.finditer(text):
        suffix = match[6]
        if suffix is None:
            tz: tzinfo = tz_for(zone)
        elif suffix == "Z":
            tz = UTC
        else:
            digits = suffix.replace(":", "")
            minutes = int(digits[1:3]) * 60 + int(digits[3:5])
            tz = timezone(timedelta(minutes=minutes if digits[0] == "+" else -minutes))
        try:
            local = datetime(
                int(match[1]), int(match[2]), int(match[3]), int(match[4]), int(match[5]), tzinfo=tz
            )
        except ValueError:
            continue
        instant = ensure_utc(local)
        found.append(StatedTime(match[0], match.start(), match.end(), exact=instant))
    return found


def find_times(text: str, *, zone: str, now: datetime, host_zone: str | None = None) -> list[StatedTime]:
    """Every time stated in ``text`` (read after :func:`normalize`), in text order.

    ``zone`` is the prospect's zone, used when no zone is stated; ``now`` anchors "today" and missing years;
    ``host_zone`` is what "our time" means.
    """
    text = normalize(text)
    zones = _zone_mentions(text, zone, host_zone)
    times = _exact_times(text, zone)
    masked = _mask(text, [*zones, *(_Span(t.start, t.end) for t in times)])
    dates = _date_mentions(masked, ensure_utc(now).astimezone(tz_for(zone)).date())
    declared = [z for z in zones if _DECLARES.search(text, max(0, z.start - 30), z.start)]

    used: set[int] = set()
    previous: tuple[_ClockMention, int] | None = None
    for clock in _clock_mentions(masked):
        before = [i for i, d in enumerate(dates) if d.end <= clock.start]
        chosen: int | None = None
        if before and _ATTACH_BEFORE.fullmatch(_gap(text, dates[before[-1]].end, clock.start, zones)):
            chosen = before[-1]
        else:
            label = _zone_after(text, clock.end, zones)
            resume = label.end if label is not None else clock.end
            after = [i for i, d in enumerate(dates) if d.start >= resume]
            if after and _ATTACH_AFTER.fullmatch(_gap(text, resume, dates[after[0]].start, zones)):
                chosen = after[0]
            elif before and not _SENTENCE_BREAK.search(text, dates[before[-1]].end, clock.start):
                chosen = before[-1]
        start, end = clock.start, clock.end
        mention: _DateMention | None = None
        if chosen is not None:
            mention = dates[chosen]
            used.add(chosen)
            start, end = min(start, mention.start), max(end, mention.end)
        label = _zone_after(text, end, zones) or _zone_after(text, clock.end, zones)
        if label is not None:
            end = max(end, label.end)
        stated_zone = label.zone if label is not None else None
        if stated_zone is None:
            prior = [z for z in declared if z.end <= clock.start]
            stated_zone = prior[-1].zone if prior else None
        in_range = _NOT_A_START.search(text, max(0, clock.start - 40), clock.start) is not None
        if previous is not None and _RANGE_JOIN.fullmatch(_gap(text, previous[0].end, clock.start, zones)):
            in_range = True
            times[previous[1]] = replace(times[previous[1]], in_range=True)
        day, weekday = _resolve_date(mention, stated_zone or zone, now)
        times.append(
            StatedTime(text[start:end], start, end, day, weekday, clock.clocks, stated_zone, in_range)
        )
        previous = (clock, len(times) - 1)
    for index, mention in enumerate(dates):
        if index in used:
            continue
        label = _zone_after(text, mention.end, zones)
        stated_zone = label.zone if label is not None else None
        day, weekday = _resolve_date(mention, stated_zone or zone, now)
        end = label.end if label is not None else mention.end
        times.append(StatedTime(text[mention.start : end], mention.start, end, day, weekday, (), stated_zone))
    times.sort(key=lambda t: (t.start, t.end))
    return times


__all__ = [
    "ABBREVIATIONS",
    "NAMED_ZONES",
    "StatedTime",
    "find_times",
    "fixed_zone_name",
    "infer_year",
    "normalize",
    "tz_for",
]

"""Find date and time mentions in English agent text and resolve them to UTC instants.

The harness reads times the way a prospect would (``docs/metrics.md``, "Time"): an explicit zone label wins
(an IANA name, ``UTC``/``GMT`` with or without an offset, an abbreviation such as ``ET`` or ``CEST``, a
name such as "Eastern" or "Central European Time", a city such as "Berlin time" or "in Sydney", "your time"
for the prospect's zone and "our time" for the host's). With no label, the prospect's true zone is assumed.
Relative dates ("tomorrow", "Tuesday") resolve against the reference instant, in the zone of the stated
time. A time with no date of its own takes the date of the nearest earlier mention in the same text, so
"Wednesday 7 October at 10:00 AM or 2:30 PM" gives two times on the same day.

Recognised forms include "Tue 6 Oct, 3:00 PM", "Tuesday, 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00)",
"October 6 at 3pm", "3pm Tuesday", "15:00", "tomorrow at 10", "in two days at 3pm", "10:30 AM ET",
"2:00 PM Eastern", "noon", and ISO 8601 timestamps such as ``2026-10-06T13:00:00Z``. Numeric dates ("10/6",
"6/10/2026") and bare ordinals ("the 6th") count only next to a clock time; a numeric date that reads both
as month/day and as day/month takes the reading nearest to the reference date.

This module is the harness's own reader. It shares no code with the agent under test or its guards.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from functools import cache
from typing import Literal, NamedTuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from booking_truth.timeutil import ensure_utc

ZoneSource = Literal["label", "default", "iso"]
DateSource = Literal["explicit", "relative", "weekday", "context", "reference"]


@dataclass(frozen=True)
class TimeSpan:
    """One time mention in a text, resolved to a UTC instant.

    ``start``/``end`` are character offsets into the text (the date, the clock time and the zone label).
    ``zone`` is the zone the time was read in: an IANA key, ``UTC`` or a fixed offset such as ``UTC+05:30``.
    ``in_range`` marks either end of a range ("9:00-17:00", "between 2 and 4 PM"): a window, not a start.
    ``alias`` marks a restatement of the previous time in another zone ("10:00 AM ET (4:00 PM your time)").
    """

    start: int
    end: int
    text: str
    utc: datetime
    zone: str
    zone_source: ZoneSource
    date_source: DateSource
    in_range: bool = False
    alias: bool = False


class Zone(NamedTuple):
    name: str
    tz: tzinfo


# Zones -------------------------------------------------------------------------------------------------

_FIXED_NAME = re.compile(r"UTC([+-])(\d{2}):(\d{2})")


def fixed_zone_name(minutes: int) -> str:
    """``UTC`` for 0, else ``UTC+hh:mm`` / ``UTC-hh:mm``."""
    if minutes == 0:
        return "UTC"
    sign = "+" if minutes > 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours:02d}:{mins:02d}"


@cache
def zone_for(name: str) -> tzinfo:
    """The ``tzinfo`` for an IANA key, ``UTC`` or a fixed offset name from :func:`fixed_zone_name`."""
    if name == "UTC":
        return UTC
    fixed = _FIXED_NAME.fullmatch(name)
    if fixed:
        minutes = int(fixed[2]) * 60 + int(fixed[3])
        return timezone(timedelta(minutes=minutes if fixed[1] == "+" else -minutes))
    return ZoneInfo(name)


@cache
def _is_iana(name: str) -> bool:
    if "/" not in name or ".." in name:
        return False
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return False
    return True


def _zone(name: str) -> Zone:
    return Zone(name, zone_for(name))


_NY = "America/New_York"
_CHI = "America/Chicago"
_DEN = "America/Denver"
_LA = "America/Los_Angeles"
_CET = (
    "Europe/Berlin",
    "Europe/Paris",
    "Europe/Madrid",
    "Europe/Rome",
    "Europe/Amsterdam",
    "Europe/Brussels",
    "Europe/Vienna",
    "Europe/Zurich",
    "Europe/Stockholm",
    "Europe/Oslo",
    "Europe/Copenhagen",
    "Europe/Warsaw",
    "Europe/Prague",
    "Europe/Budapest",
)
_EET = ("Europe/Athens", "Europe/Helsinki", "Europe/Kyiv", "Europe/Bucharest", "Europe/Sofia", "Africa/Cairo")

#: Abbreviations with their candidate zones. The prospect's or the host's zone wins when it is a candidate;
#: otherwise the first candidate is used.
ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "ET": (_NY, "America/Toronto"),
    "EST": (_NY, "America/Toronto"),
    "EDT": (_NY, "America/Toronto"),
    "CT": (_CHI,),
    "CST": (_CHI, "America/Mexico_City", "Asia/Shanghai", "America/Winnipeg"),
    "CDT": (_CHI, "America/Winnipeg"),
    "MT": (_DEN, "America/Edmonton"),
    "MST": (_DEN, "America/Phoenix", "America/Edmonton"),
    "MDT": (_DEN, "America/Edmonton"),
    "PT": (_LA, "America/Vancouver"),
    "PST": (_LA, "America/Vancouver"),
    "PDT": (_LA, "America/Vancouver"),
    "AKST": ("America/Anchorage",),
    "AKDT": ("America/Anchorage",),
    "HST": ("Pacific/Honolulu",),
    "AST": ("America/Halifax", "Asia/Riyadh"),
    "ADT": ("America/Halifax",),
    "NST": ("America/St_Johns",),
    "NDT": ("America/St_Johns",),
    "BST": ("Europe/London",),
    "WET": ("Europe/Lisbon",),
    "WEST": ("Europe/Lisbon",),
    "CET": _CET,
    "CEST": _CET,
    "EET": _EET,
    "EEST": _EET,
    "MSK": ("Europe/Moscow",),
    "IST": ("Asia/Kolkata", "Europe/Dublin", "Asia/Jerusalem"),
    "PKT": ("Asia/Karachi",),
    "NPT": ("Asia/Kathmandu",),
    "ICT": ("Asia/Bangkok",),
    "WIB": ("Asia/Jakarta",),
    "SGT": ("Asia/Singapore",),
    "HKT": ("Asia/Hong_Kong",),
    "PHT": ("Asia/Manila",),
    "KST": ("Asia/Seoul",),
    "JST": ("Asia/Tokyo",),
    "AEST": ("Australia/Sydney", "Australia/Melbourne", "Australia/Brisbane"),
    "AEDT": ("Australia/Sydney", "Australia/Melbourne"),
    "ACST": ("Australia/Adelaide", "Australia/Darwin"),
    "ACDT": ("Australia/Adelaide",),
    "AWST": ("Australia/Perth",),
    "NZST": ("Pacific/Auckland",),
    "NZDT": ("Pacific/Auckland",),
    "GST": ("Asia/Dubai",),
    "SAST": ("Africa/Johannesburg",),
    "EAT": ("Africa/Nairobi",),
    "WAT": ("Africa/Lagos",),
    "BRT": ("America/Sao_Paulo",),
    "ART": ("America/Argentina/Buenos_Aires",),
}

#: Zone names written out in words. Longer names come first so "Central European Time" is not read as
#: "Central".
_NAMED_ZONES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"australian eastern(?: standard| daylight)?(?: time)?", ("Australia/Sydney",)),
    (r"australian central(?: standard| daylight)?(?: time)?", ("Australia/Adelaide",)),
    (r"australian western(?: standard)?(?: time)?", ("Australia/Perth",)),
    (r"central european(?: summer| standard)?(?: time)?", _CET),
    (r"eastern european(?: summer| standard)?(?: time)?", _EET),
    (r"western european(?: summer| standard)?(?: time)?", ("Europe/Lisbon",)),
    (r"british(?: summer)?(?: time)?|uk time", ("Europe/London",)),
    (r"greenwich mean time|coordinated universal time|universal time", ("UTC",)),
    (r"indian?(?: standard)? time", ("Asia/Kolkata",)),
    (r"japan(?:ese)?(?: standard)? time", ("Asia/Tokyo",)),
    (r"nepal(?:i)?(?: standard)? time", ("Asia/Kathmandu",)),
    (r"gulf(?: standard)? time", ("Asia/Dubai",)),
    (r"china(?: standard)? time", ("Asia/Shanghai",)),
    (r"korea(?:n)?(?: standard)? time", ("Asia/Seoul",)),
    (r"singapore time", ("Asia/Singapore",)),
    (r"arizona time", ("America/Phoenix",)),
    (r"brazil(?:ian)? time|bras[ií]lia time", ("America/Sao_Paulo",)),
    (r"eastern(?: standard| daylight)?(?: time)?", (_NY, "America/Toronto")),
    (r"central(?: standard| daylight)?(?: time)?", (_CHI,)),
    (r"mountain(?: standard| daylight)?(?: time)?", (_DEN, "America/Phoenix")),
    (r"pacific(?: standard| daylight)?(?: time)?", (_LA,)),
    (r"atlantic(?: standard| daylight)?(?: time)?", ("America/Halifax",)),
    (r"alaska(?:n)?(?: standard| daylight)?(?: time)?", ("America/Anchorage",)),
    (r"hawaii(?:an)?(?: standard)?(?: time)?", ("Pacific/Honolulu",)),
)

#: A small built-in list of places that agents name as zone labels ("Berlin time", "in Sydney").
CITY_ZONES: dict[str, str] = {
    "new york": _NY,
    "nyc": _NY,
    "boston": _NY,
    "washington": _NY,
    "miami": _NY,
    "atlanta": _NY,
    "philadelphia": _NY,
    "toronto": "America/Toronto",
    "montreal": "America/Toronto",
    "ottawa": "America/Toronto",
    "chicago": _CHI,
    "dallas": _CHI,
    "houston": _CHI,
    "austin": _CHI,
    "minneapolis": _CHI,
    "mexico city": "America/Mexico_City",
    "denver": _DEN,
    "calgary": "America/Edmonton",
    "phoenix": "America/Phoenix",
    "arizona": "America/Phoenix",
    "los angeles": _LA,
    "san francisco": _LA,
    "seattle": _LA,
    "vancouver": "America/Vancouver",
    "anchorage": "America/Anchorage",
    "honolulu": "Pacific/Honolulu",
    "hawaii": "Pacific/Honolulu",
    "halifax": "America/Halifax",
    "sao paulo": "America/Sao_Paulo",
    "são paulo": "America/Sao_Paulo",
    "rio de janeiro": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires",
    "santiago": "America/Santiago",
    "bogota": "America/Bogota",
    "bogotá": "America/Bogota",
    "lima": "America/Lima",
    "london": "Europe/London",
    "uk": "Europe/London",
    "dublin": "Europe/Dublin",
    "lisbon": "Europe/Lisbon",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "munich": "Europe/Berlin",
    "frankfurt": "Europe/Berlin",
    "germany": "Europe/Berlin",
    "madrid": "Europe/Madrid",
    "barcelona": "Europe/Madrid",
    "rome": "Europe/Rome",
    "milan": "Europe/Rome",
    "amsterdam": "Europe/Amsterdam",
    "brussels": "Europe/Brussels",
    "vienna": "Europe/Vienna",
    "zurich": "Europe/Zurich",
    "stockholm": "Europe/Stockholm",
    "oslo": "Europe/Oslo",
    "copenhagen": "Europe/Copenhagen",
    "warsaw": "Europe/Warsaw",
    "prague": "Europe/Prague",
    "budapest": "Europe/Budapest",
    "athens": "Europe/Athens",
    "helsinki": "Europe/Helsinki",
    "kyiv": "Europe/Kyiv",
    "istanbul": "Europe/Istanbul",
    "moscow": "Europe/Moscow",
    "cairo": "Africa/Cairo",
    "lagos": "Africa/Lagos",
    "nairobi": "Africa/Nairobi",
    "johannesburg": "Africa/Johannesburg",
    "dubai": "Asia/Dubai",
    "abu dhabi": "Asia/Dubai",
    "riyadh": "Asia/Riyadh",
    "tehran": "Asia/Tehran",
    "karachi": "Asia/Karachi",
    "india": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "new delhi": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "bengaluru": "Asia/Kolkata",
    "pune": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "chennai": "Asia/Kolkata",
    "hyderabad": "Asia/Kolkata",
    "kathmandu": "Asia/Kathmandu",
    "nepal": "Asia/Kathmandu",
    "dhaka": "Asia/Dhaka",
    "bangkok": "Asia/Bangkok",
    "jakarta": "Asia/Jakarta",
    "singapore": "Asia/Singapore",
    "kuala lumpur": "Asia/Kuala_Lumpur",
    "manila": "Asia/Manila",
    "hong kong": "Asia/Hong_Kong",
    "shanghai": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "taipei": "Asia/Taipei",
    "seoul": "Asia/Seoul",
    "tokyo": "Asia/Tokyo",
    "osaka": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "brisbane": "Australia/Brisbane",
    "queensland": "Australia/Brisbane",
    "adelaide": "Australia/Adelaide",
    "perth": "Australia/Perth",
    "darwin": "Australia/Darwin",
    "hobart": "Australia/Hobart",
    "auckland": "Pacific/Auckland",
    "wellington": "Pacific/Auckland",
}


def _pick(candidates: tuple[str, ...], prospect_zone: str, host_zone: str) -> str:
    for preferred in (prospect_zone, host_zone):
        if preferred in candidates:
            return preferred
    return candidates[0]


# Zone labels -------------------------------------------------------------------------------------------

_LABEL_LEAD = re.compile(r"[ \t]*(?:,[ \t]*)?(\()?[ \t]*")
_IANA_LABEL = re.compile(r"[A-Z][A-Za-z_]+(?:/[A-Z][A-Za-z0-9_+\-]*){1,2}")
_OFFSET_LABEL = re.compile(r"(?:UTC|GMT)[ \t]*([+\-−])[ \t]*(\d{1,2})(?::?([0-5]\d))?(?!\d)")
_UTC_LABEL = re.compile(r"(?:UTC|GMT|Z)(?![\w+\-−])")
_ZONE_WORDS = r"(?:local[ \t]+)?(?:time[ \t]*zone|timezone|time)\b"
_YOUR_LABEL = re.compile(rf"(?:in[ \t]+)?your[ \t]+(?:own[ \t]+)?{_ZONE_WORDS}", re.I)
_OUR_LABEL = re.compile(rf"(?:in[ \t]+)?(?:our|my)[ \t]+{_ZONE_WORDS}", re.I)
_LOCAL_LABEL = re.compile(r"local[ \t]+time\b", re.I)
_NAMED_LABELS = tuple(
    (re.compile(rf"(?:in[ \t]+)?(?:{pattern})(?![\w])", re.I), zones) for pattern, zones in _NAMED_ZONES
)
_ABBR_LABEL = re.compile(r"([A-Z]{2,5})(?![\w])")
_CITY_LABEL = re.compile(
    r"(in[ \t]+)?(?:the[ \t]+)?("
    + "|".join(re.escape(name) for name in sorted(CITY_ZONES, key=len, reverse=True))
    + r")(?![\w])([ \t]+time\b(?![ \t]*zone))?",
    re.I,
)
_TIME_WORD = re.compile(r"[ \t]+time\b(?![ \t]*zone)", re.I)
_TRAILING_OFFSET = re.compile(
    r"[ \t]*\([ \t]*(?:(?:UTC|GMT)[ \t]*[+\-−][ \t]*\d{1,2}(?::?\d{2})?|[A-Z]{2,5})[ \t]*\)"
)
_CLOSE_PAREN = re.compile(r"[ \t]*\)")


def _offset_zone(sign: str, hours: str, minutes: str | None) -> Zone | None:
    total = int(hours) * 60 + int(minutes or 0)
    if total > 14 * 60:
        return None
    return _zone(fixed_zone_name(-total if sign in "-−" else total))


def _match_label(text: str, pos: int, prospect_zone: str, host_zone: str) -> tuple[Zone, int] | None:
    """A zone label starting at ``pos`` (after optional spaces, a comma or an opening parenthesis)."""
    lead = _LABEL_LEAD.match(text, pos)
    assert lead is not None
    paren = lead[1] is not None
    p = lead.end()
    hit: tuple[Zone, int] | None = None
    if (m := _IANA_LABEL.match(text, p)) and _is_iana(m[0]):
        hit = (_zone(m[0]), m.end())
    if hit is None and (m := _OFFSET_LABEL.match(text, p)):
        zone = _offset_zone(m[1], m[2], m[3])
        if zone is not None:
            hit = (zone, m.end())
    if hit is None and (m := _UTC_LABEL.match(text, p)):
        hit = (_zone("UTC"), m.end())
    if hit is None and (m := _YOUR_LABEL.match(text, p)):
        hit = (_zone(prospect_zone), m.end())
    if hit is None and (m := _OUR_LABEL.match(text, p)):
        hit = (_zone(host_zone), m.end())
    if hit is None and (m := _LOCAL_LABEL.match(text, p)):
        hit = (_zone(prospect_zone), m.end())
    if hit is None:
        for pattern, zones in _NAMED_LABELS:
            if m := pattern.match(text, p):
                hit = (_zone(_pick(zones, prospect_zone, host_zone)), m.end())
                break
    if hit is None and (m := _ABBR_LABEL.match(text, p)):
        candidates = ABBREVIATIONS.get(m[1])
        if candidates is not None:
            end = m.end()
            if word := _TIME_WORD.match(text, end):
                end = word.end()
            hit = (_zone(_pick(candidates, prospect_zone, host_zone)), end)
    if hit is None and (m := _CITY_LABEL.match(text, p)) and (m[1] or m[3] or paren):
        hit = (_zone(CITY_ZONES[m[2].lower()]), m.end())
    if hit is None:
        return None
    zone, end = hit
    if trailing := _TRAILING_OFFSET.match(text, end):
        end = trailing.end()
    if paren and (close := _CLOSE_PAREN.match(text, end)):
        end = close.end()
    return zone, end


def resolve_zone_label(label: str, *, prospect_zone: str, host_zone: str) -> str | None:
    """The zone a whole label names ("Berlin time", "ET", "UTC+2", "your time"), or ``None``."""
    text = label.strip()
    found = _match_label(text, 0, prospect_zone, host_zone)
    if found is None or found[1] != len(text):
        return None
    return found[0].name


# Tokens ------------------------------------------------------------------------------------------------

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_WD = (
    r"(?P<wd>(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|MONDAY|TUESDAY|WEDNESDAY|THURSDAY"
    r"|FRIDAY|SATURDAY|SUNDAY)|(?:Mon|Tues?|Wed|Thu(?:rs?)?|Fri|Sat|Sun)\.?)"
)
_MON = (
    r"(?P<mon>(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"|(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?)"
)
_YEAR = r"(?:,?[ \t]+(?P<year>(?:19|20|21)\d\d)(?!\d))?"
_NOT_CLOCK = r"(?![\d:])(?![ \t]?[aApP]\.?[ \t]?[mM]\b)"

_ISO_DATETIME = re.compile(
    r"(?<![\w-])(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})[T ](?P<h>[01]\d|2[0-3]):(?P<mi>[0-5]\d)"
    r"(?::[0-5]\d(?:\.\d{1,9})?)?(?P<off>Z|[+\-][01]\d(?::?[0-5]\d)?)?(?![\w:])"
)
_ISO_DATE = re.compile(r"(?<![\w-])(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})(?![\w:])")
_DAY_MONTH = re.compile(
    rf"(?:{_WD},?[ \t]+)?(?:the[ \t]+)?(?<![\d:])(?P<day>[0-3]?\d)(?:st|nd|rd|th)?(?:[ \t]+of)?[ \t]+{_MON}"
    rf"(?![a-z]){_YEAR}"
)
_MONTH_DAY = re.compile(
    rf"(?:{_WD},?[ \t]+)?{_MON}[ \t]+(?:the[ \t]+)?(?P<day>[0-3]?\d)(?:st|nd|rd|th)?{_NOT_CLOCK}{_YEAR}"
)
_WEEKDAY_ORDINAL = re.compile(rf"{_WD}[ \t]+(?:the[ \t]+)?(?P<day>[0-3]?\d)(?:st|nd|rd|th)(?![\w])")
_RELATIVE = re.compile(
    r"\b(?P<rel>(?:the )?day after tomorrow|tomorrow|today|tonight|this (?:morning|afternoon|evening)"
    r"|in (?:(?P<n>[1-9])|(?P<word>two|three|four|five|six|seven)) days)\b",
    re.I,
)
_DAY_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
#: A numeric date, month/day or day/month ("10/27", "27/10/2026"). Read only when a clock is attached.
_NUMERIC_DATE = re.compile(
    rf"(?:{_WD},?[ \t]+)?(?<![\w/.:\-])(?P<a>[0-3]?\d)/(?P<b>[0-3]?\d)(?:/(?P<year>(?:19|20)?\d\d))?"
    r"(?![\w/:%])"
)
#: A bare ordinal day ("on the 27th"). Read only when a clock is attached.
_DAY_ORDINAL = re.compile(r"\bthe[ \t]+(?P<day>[0-3]?\d)(?:st|nd|rd|th)(?![\w])", re.I)
_WEEKDAY = re.compile(rf"\b(?:(?P<mod>(?i:next|this|coming))[ \t]+)?{_WD}(?![\w])")
_CONTEXT_DAY = re.compile(r"\b(?:that same day|that day|the same day|same day)\b", re.I)

_MERIDIEM = r"(?P<mer>[aApP])\.?[ \t]?[mM]\.?(?![A-Za-z])"
_CLOCK_MERIDIEM = re.compile(rf"(?<![\w:.])(?P<h>\d{{1,2}})(?:[:.](?P<mi>[0-5]\d))?[ \t]?{_MERIDIEM}")
_CLOCK_24 = re.compile(r"(?<![\w:.])(?P<h>[01]?\d|2[0-3]):(?P<mi>[0-5]\d)(?::[0-5]\d)?(?![\d:])")
_CLOCK_WORD = re.compile(r"\b(?:12[ \t]*)?(?P<word>noon|midday|midnight)\b", re.I)
_CLOCK_AT = re.compile(
    r"(?<=\bat[ \t])(?P<h>[01]?\d|2[0-3])(?![\d:.,]?\d)(?![ \t]*(?:[aApP]\.?[ \t]?[mM]\b|%|percent|minutes?\b"
    r"|mins?\b|hours?\b|hrs?\b|people|seconds?\b|days?\b|weeks?\b|times\b|slots?\b|o'?clock))",
    re.I,
)
_CLOCK_OCLOCK = re.compile(r"(?<![\w:.])(?P<h>[01]?\d|2[0-3])[ \t]*o'?clock\b", re.I)

_PROTECTED = (
    re.compile(r"(?:UTC|GMT)[ \t]*[+\-−][ \t]*\d{1,2}(?::?\d{2})?"),
    re.compile(r"(?<![\w:])[+\-−]\d{2}:?\d{2}(?!\d)"),
    _ISO_DATETIME,
)

_BACK_GAP = re.compile(
    r"[ \t]*,?[ \t]*(?:(?:in the |the )?(?:morning|afternoon|evening|night)[ \t]*)?"
    r"(?:(?:at|@|from|around|by|between)[ \t]+|[-–][ \t]*)?",
    re.I,
)
_FORWARD_GAP = re.compile(r"[ \t]*,?[ \t]*(?:on[ \t]+)?(?:the[ \t]+)?", re.I)
_RANGE_SEP = re.compile(r"[ \t]*(?:-|–|—|to|until|till|through|thru)[ \t]*", re.I)
_LIST_SEP = re.compile(r"[ \t]*(?:,[ \t]*)?(?:or|and|/)?[ \t]*", re.I)
_RANGE_INTRO = re.compile(r"\b(?:between|from)[ \t]*$", re.I)
_RANGE_AND = re.compile(r"[ \t]*and[ \t]*", re.I)
#: The far end of a range written as a bare number before a clock: "between 2 and 4 PM", "3-5pm".
_RANGE_BARE_START = re.compile(
    r"(?:\b(?:between|from)[ \t]+\d{1,2}(?::[0-5]\d)?[ \t]*(?:[ap]\.?[ \t]?m\.?)?[ \t]*(?:and|to|-|–)"
    r"|(?<![\w:])\d{1,2}(?::[0-5]\d)?[ \t]*(?:-|–|to))[ \t]*$",
    re.I,
)
_ALIAS_GAP = re.compile(
    r"[ \t]*(?:\([ \t]*|/[ \t]*|,?[ \t]*(?:which is|that is|that's|i\.e\.,?|aka)[ \t]*\(?[ \t]*)", re.I
)


@dataclass
class _DateTok:
    start: int
    end: int
    kind: Literal["explicit", "relative", "weekday", "dayonly", "context"]
    year: int | None = None
    month: int | None = None
    day: int | None = None
    weekday: int | None = None
    offset_days: int = 0
    modifier: str | None = None
    owned: bool = False
    #: The other reading (month, day) of an ambiguous numeric date such as "6/10".
    alt: tuple[int, int] | None = None
    #: Too ambiguous to date a later time on its own: used only when a clock is attached to it.
    weak: bool = False


@dataclass
class _Clock:
    start: int
    end: int
    hour: int
    minute: int
    meridiem: str | None = None
    bare: bool = False  # hour 1..12 written without a leading zero and without am/pm
    iso_date: date | None = None
    label: Zone | None = None
    zone_source: ZoneSource = "default"
    label_end: int | None = None
    date: _DateTok | None = None
    in_range: bool = False
    alias: bool = False

    @property
    def tail(self) -> int:
        return self.label_end if self.label_end is not None else self.end


_DATE_SOURCES: dict[str, DateSource] = {
    "explicit": "explicit",
    "relative": "relative",
    "weekday": "weekday",
    "dayonly": "weekday",
}


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < b and a < end for a, b in spans)


def _select(candidates: list[tuple[int, int, object]]) -> list[object]:
    """Leftmost-longest non-overlapping selection."""
    chosen: list[tuple[int, int, object]] = []
    for start, end, item in sorted(candidates, key=lambda c: (c[0], -(c[1] - c[0]))):
        if not _overlaps(start, end, [(a, b) for a, b, _ in chosen]):
            chosen.append((start, end, item))
    return [item for _, _, item in chosen]


def _month_number(name: str) -> int:
    key = name.rstrip(".").lower()
    for index, month in enumerate(_MONTHS, start=1):
        if month.startswith(key[:3]):
            return index
    raise ValueError(name)


def _weekday_number(name: str | None) -> int | None:
    if not name:
        return None
    key = name.rstrip(".").lower()[:3]
    for index, weekday in enumerate(_WEEKDAYS, start=1):
        if weekday.startswith(key):
            return index
    return None


def _date_tokens(text: str, protected: list[tuple[int, int]]) -> list[_DateTok]:
    found: list[tuple[int, int, object]] = []

    def add(match: re.Match[str], tok: _DateTok) -> None:
        if not _overlaps(match.start(), match.end(), protected):
            found.append((match.start(), match.end(), tok))

    for m in _ISO_DATE.finditer(text):
        add(m, _DateTok(m.start(), m.end(), "explicit", int(m["y"]), int(m["mo"]), int(m["d"])))
    for pattern in (_DAY_MONTH, _MONTH_DAY):
        for m in pattern.finditer(text):
            year = int(m["year"]) if m["year"] else None
            tok = _DateTok(
                m.start(),
                m.end(),
                "explicit",
                year,
                _month_number(m["mon"]),
                int(m["day"]),
                _weekday_number(m["wd"]),
            )
            add(m, tok)
    for m in _NUMERIC_DATE.finditer(text):
        numeric = _numeric_date(m)
        if numeric is not None:
            add(m, numeric)
    for m in _WEEKDAY_ORDINAL.finditer(text):
        add(m, _DateTok(m.start(), m.end(), "dayonly", day=int(m["day"]), weekday=_weekday_number(m["wd"])))
    for m in _DAY_ORDINAL.finditer(text):
        if 1 <= int(m["day"]) <= 31:
            add(m, _DateTok(m.start(), m.end(), "dayonly", day=int(m["day"]), weak=True))
    for m in _RELATIVE.finditer(text):
        rel = m["rel"].lower()
        if m["n"] or m["word"]:
            offset = int(m["n"]) if m["n"] else _DAY_WORDS[m["word"].lower()]
        else:
            offset = 2 if "day after" in rel else 1 if rel == "tomorrow" else 0
        add(m, _DateTok(m.start(), m.end(), "relative", offset_days=offset))
    for m in _WEEKDAY.finditer(text):
        mod = m["mod"].lower() if m["mod"] else None
        add(m, _DateTok(m.start(), m.end(), "weekday", weekday=_weekday_number(m["wd"]), modifier=mod))
    for m in _CONTEXT_DAY.finditer(text):
        add(m, _DateTok(m.start(), m.end(), "context"))
    return [tok for tok in _select(found) if isinstance(tok, _DateTok)]


def _numeric_date(m: re.Match[str]) -> _DateTok | None:
    """A month/day or day/month token; both readings are kept when both are valid dates."""
    a, b = int(m["a"]), int(m["b"])
    readings = [(mo, d) for mo, d in ((a, b), (b, a)) if 1 <= mo <= 12 and 1 <= d <= 31]
    if not readings:
        return None
    year = None
    if m["year"]:
        year = int(m["year"]) if len(m["year"]) == 4 else 2000 + int(m["year"])
    (month, day), *rest = readings
    alt = rest[0] if rest and rest[0] != (month, day) else None
    return _DateTok(
        m.start(), m.end(), "explicit", year, month, day, _weekday_number(m["wd"]), alt=alt, weak=True
    )


def _clock_tokens(text: str, protected: list[tuple[int, int]], dates: list[_DateTok]) -> list[_Clock]:
    found: list[tuple[int, int, object]] = []
    blocked = protected + [(d.start, d.end) for d in dates]

    def add(match: re.Match[str], clock: _Clock) -> None:
        if not _overlaps(match.start(), match.end(), blocked):
            found.append((match.start(), match.end(), clock))

    for m in _CLOCK_MERIDIEM.finditer(text):
        hour = int(m["h"])
        if 1 <= hour <= 23:
            add(m, _Clock(m.start(), m.end(), hour, int(m["mi"] or 0), meridiem=m["mer"].lower()))
    for m in _CLOCK_24.finditer(text):
        raw = m["h"]
        hour = int(raw)
        bare = 1 <= hour <= 12 and not raw.startswith("0")
        add(m, _Clock(m.start(), m.end(), hour, int(m["mi"]), bare=bare))
    for m in _CLOCK_WORD.finditer(text):
        add(m, _Clock(m.start(), m.end(), 0 if m["word"].lower() == "midnight" else 12, 0))
    for pattern in (_CLOCK_AT, _CLOCK_OCLOCK):
        for m in pattern.finditer(text):
            add(m, _Clock(m.start(), m.end(), int(m["h"]), 0))
    return [c for c in _select(found) if isinstance(c, _Clock)]


def _hour24(clock: _Clock) -> int:
    hour = clock.hour
    if clock.meridiem == "a":
        return 0 if hour == 12 else hour
    if clock.meridiem == "p":
        return hour if hour >= 12 else hour + 12
    return hour


def _infer_year(month: int, day: int, weekday: int | None, today: date) -> date | None:
    candidates: list[date] = []
    for year in (today.year - 1, today.year, today.year + 1):
        try:
            candidates.append(date(year, month, day))
        except ValueError:
            continue
    if not candidates:
        return None
    if weekday is not None:
        matching = [c for c in candidates if c.isoweekday() == weekday]
        if matching:
            candidates = matching
    recent = [c for c in candidates if c >= today - timedelta(days=14)]
    pool = recent or candidates
    return min(pool, key=lambda c: (abs((c - today).days), c))


def _local_instant(day: date, hour: int, minute: int, tz: tzinfo) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=tz).astimezone(UTC)


class _Resolver:
    def __init__(self, text: str, prospect_zone: str, host_zone: str, reference: datetime) -> None:
        self.text = text
        self.prospect_zone = prospect_zone
        self.host_zone = host_zone
        self.reference = ensure_utc(reference)

    def today(self, tz: tzinfo) -> date:
        return self.reference.astimezone(tz).date()

    def token_date(self, tok: _DateTok, tz: tzinfo, hm: tuple[int, int] | None) -> date | None:
        today = self.today(tz)
        if tok.kind == "explicit":
            assert tok.month is not None
            assert tok.day is not None
            readings = [(tok.month, tok.day)] + ([tok.alt] if tok.alt is not None else [])
            found: list[date] = []
            for mo, dd in readings:
                if tok.year is not None:
                    try:
                        found.append(date(tok.year, mo, dd))
                    except ValueError:
                        continue
                elif (inferred := _infer_year(mo, dd, tok.weekday, today)) is not None:
                    found.append(inferred)
            if tok.weekday is not None and any(d.isoweekday() == tok.weekday for d in found):
                found = [d for d in found if d.isoweekday() == tok.weekday]
            # Of two readings, the one nearest to today that is not long past.
            recent = [d for d in found if d >= today - timedelta(days=14)] or found
            return min(recent, key=lambda d: (abs((d - today).days), d)) if recent else None
        if tok.kind == "relative":
            return today + timedelta(days=tok.offset_days)
        if tok.kind == "weekday":
            assert tok.weekday is not None
            delta = (tok.weekday - today.isoweekday()) % 7
            if tok.modifier in ("next", "coming") and delta == 0:
                delta = 7
            day = today + timedelta(days=delta)
            if (
                delta == 0
                and tok.modifier != "this"
                and hm is not None
                and _local_instant(day, *hm, tz) < self.reference
            ):
                day += timedelta(days=7)
            return day
        if tok.kind == "dayonly":
            assert tok.day is not None
            for ahead in range(0, 400):
                day = today + timedelta(days=ahead)
                if day.day == tok.day and (tok.weekday is None or day.isoweekday() == tok.weekday):
                    return day
            return None
        return None


def _protected_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for pattern in _PROTECTED for m in pattern.finditer(text)]


def _iso_clocks(text: str) -> list[_Clock]:
    clocks: list[_Clock] = []
    for m in _ISO_DATETIME.finditer(text):
        try:
            day = date(int(m["y"]), int(m["mo"]), int(m["d"]))
        except ValueError:
            continue
        clock = _Clock(m.start(), m.end(), int(m["h"]), int(m["mi"]), iso_date=day)
        offset = m["off"]
        if offset:
            if offset == "Z":
                clock.label = _zone("UTC")
            else:
                sign, digits = offset[0], offset[1:].replace(":", "")
                clock.label = _offset_zone(sign, digits[:2], digits[2:] or None)
            clock.zone_source = "iso"
            clock.label_end = m.end()
        clocks.append(clock)
    return clocks


def find_times(
    text: str,
    *,
    prospect_zone: str,
    host_zone: str,
    reference: datetime,
) -> list[TimeSpan]:
    """Every time-of-day mention in ``text``, in order, resolved to UTC.

    A date without a time of day ("on Friday") is not a time mention and is not returned, but it serves as
    the date of later times that carry none.
    """
    resolver = _Resolver(text, prospect_zone, host_zone, reference)
    protected = _protected_spans(text)
    dates = _date_tokens(text, protected)
    clocks = sorted(_iso_clocks(text) + _clock_tokens(text, protected, dates), key=lambda c: c.start)

    for clock in clocks:
        if clock.label is None:
            found = _match_label(text, clock.end, prospect_zone, host_zone)
            if found is not None:
                clock.label, clock.label_end = found
                clock.zone_source = "label"

    # Dates bound to clocks: an immediately preceding date first ("Tuesday at 3pm"), then an unclaimed
    # date right after the clock ("3pm on Tuesday").
    for clock in clocks:
        if clock.iso_date is not None:
            continue
        before = [d for d in dates if d.end <= clock.start]
        if before:
            candidate = before[-1]
            gap = _BACK_GAP.fullmatch(text, candidate.end, clock.start)
            if gap is not None and not any(c.start >= candidate.end and c.end <= clock.start for c in clocks):
                clock.date = candidate
                candidate.owned = True
    for clock in clocks:
        if clock.iso_date is not None or clock.date is not None:
            continue
        after = [d for d in dates if d.start >= clock.tail and not d.owned]
        if after:
            candidate = after[0]
            if _FORWARD_GAP.fullmatch(text, clock.tail, candidate.start) is not None:
                clock.date = candidate
                candidate.owned = True
                if clock.label is None:
                    found = _match_label(text, candidate.end, prospect_zone, host_zone)
                    if found is not None:
                        clock.label, clock.label_end = found
                        clock.zone_source = "label"

    # Ranges, lists and restatements between neighbouring clocks.
    for clock in clocks:
        if _RANGE_BARE_START.search(text[max(0, clock.start - 30) : clock.start]) is not None:
            clock.in_range = True
    for left, right in itertools.pairwise(clocks):
        between = text[left.tail : right.start]
        introduced = _RANGE_INTRO.search(text[max(0, left.start - 12) : left.start]) is not None
        if _RANGE_SEP.fullmatch(between) is not None or (introduced and _RANGE_AND.fullmatch(between)):
            left.in_range = right.in_range = True
        elif _ALIAS_GAP.fullmatch(between) is not None and right.date is None:
            right.alias = True
    for left, right in reversed(list(itertools.pairwise(clocks))):
        joined = left.in_range and right.in_range
        listed = _LIST_SEP.fullmatch(text, left.tail, right.start) is not None
        if not (joined or listed):
            continue
        if left.label is None and right.label is not None:
            left.label, left.zone_source = right.label, right.zone_source
        if left.meridiem is None and left.bare and right.meridiem is not None and left.hour <= 12:
            left.meridiem = right.meridiem
        forward_date = right.date is not None and right.date.start > right.end
        if left.date is None and left.iso_date is None and forward_date:
            left.date = right.date

    spans: list[TimeSpan] = []
    ctx_instant: datetime | None = None
    ctx_date: date | None = None
    previous: datetime | None = None
    events: list[tuple[int, object]] = [(c.start, c) for c in clocks]
    events += [(d.start, d) for d in dates if not d.owned and not d.weak and d.kind != "context"]
    for _, item in sorted(events, key=lambda e: e[0]):
        if isinstance(item, _DateTok):
            default_tz = zone_for(prospect_zone)
            resolved = resolver.token_date(item, default_tz, None)
            if resolved is not None:
                ctx_date, ctx_instant = resolved, None
            continue
        assert isinstance(item, _Clock)
        clock = item
        zone = clock.label or _zone(prospect_zone)
        zone_source: ZoneSource = clock.zone_source if clock.label is not None else "default"
        hm = (_hour24(clock), clock.minute)
        date_source: DateSource
        day: date | None
        if clock.iso_date is not None:
            day, date_source = clock.iso_date, "explicit"
        elif clock.date is not None and clock.date.kind != "context":
            day = resolver.token_date(clock.date, zone.tz, hm)
            date_source = _DATE_SOURCES[clock.date.kind]
        elif clock.alias and previous is not None:
            restated = previous
            anchor = restated.astimezone(zone.tz).date()
            options = [anchor + timedelta(days=delta) for delta in (-1, 0, 1)]
            day = min(options, key=lambda d: abs(_local_instant(d, *hm, zone.tz) - restated))
            date_source = "context"
        elif ctx_instant is not None:
            day, date_source = ctx_instant.astimezone(zone.tz).date(), "context"
        elif ctx_date is not None:
            day, date_source = ctx_date, "context"
        else:
            day = resolver.today(zone.tz)
            if _local_instant(day, *hm, zone.tz) < resolver.reference:
                day += timedelta(days=1)
            date_source = "reference"
        if day is None:
            continue
        instant = _local_instant(day, *hm, zone.tz)
        start = clock.start
        end = clock.tail
        if clock.date is not None and clock.date.kind != "context":
            start = min(start, clock.date.start)
            end = max(end, clock.date.end)
        spans.append(
            TimeSpan(
                start=start,
                end=end,
                text=text[start:end],
                utc=instant,
                zone=zone.name,
                zone_source=zone_source,
                date_source=date_source,
                in_range=clock.in_range,
                alias=clock.alias,
            )
        )
        ctx_instant, ctx_date, previous = instant, None, instant
    return spans


def mentioned_instants(
    text: str, *, prospect_zone: str, host_zone: str, reference: datetime
) -> list[datetime]:
    """The distinct start instants a text names, in order; range ends and restatements are left out."""
    seen: list[datetime] = []
    for span in find_times(text, prospect_zone=prospect_zone, host_zone=host_zone, reference=reference):
        if not span.in_range and not span.alias and span.utc not in seen:
            seen.append(span.utc)
    return seen

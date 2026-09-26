"""``tz_resolver``: deterministic resolution of what a prospect says about their time zone.

Resolution order (design-agent.md SS4/SSD; the product spec, item 4, gives the same order):

1. an explicit IANA zone key ("Europe/Berlin", "Etc/GMT+3", "UTC"), taken as written;
2. a fixed offset ("GMT+2", "UTC-5", "UTC+05:30"): a whole-hour offset maps to the ``Etc/GMT`` zone of
   the *inverted* sign (``GMT+2`` -> ``Etc/GMT-2``); a fractional offset maps to the ``zone.tab`` zone
   that holds that exact constant offset throughout the booking horizon, when there is one;
3. a curated abbreviation or name (:mod:`booking_truth.agent.guards.tz.data`'s ``tz_aliases.yaml``);
4. a country name (``iso3166.tab``), resolved to its first ``zone.tab`` zone when every one of its
   zones carries the same UTC offset at every instant in the booking horizon, else ambiguous with
   those zones;
5. a city name (``datasets/cities_tz.csv``), resolved to its most populous match unless another,
   non-equivalent match has at least 20% of that population, in which case it is ambiguous with the
   qualifying matches' zones.

Anything none of these five steps reaches is ``unknown``: never a silent fallback to any default zone.

:meth:`TimezoneResolver.prescan` is the separate phrase detector ``AgentCore`` runs over the prospect's
own message before the model sees it (design-agent.md SSB.5): it looks for a handful of ways people
state where they are ("I'm in X", "we're on X", "X time", "calling from X") or a bare abbreviation,
offset or IANA name, and resolves whatever it finds through the same five steps.

:func:`local_instant` is the other half of the guard's time handling (design.md SS2): converting a
naive local wall-clock reading to UTC, rejecting one a DST gap skips over and flagging one a DST fold
makes ambiguous, rather than silently picking an instant either way.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from booking_truth.agent.guards.tz.data import (
    AliasEntry,
    CityRow,
    country_codes_for,
    load_aliases,
    load_all_zones,
    load_cities,
    load_country_names,
    load_country_zones,
)

Status = Literal["resolved", "ambiguous", "unknown"]

DEFAULT_HORIZON_DAYS = 400
#: A city's runner-up match is ambiguous once its population reaches this share of the top match's.
CITY_AMBIGUITY_SHARE = 0.20
_ALIAS_MAX_WORDS = 4
_COUNTRY_MAX_WORDS = 6
_CITY_MAX_WORDS = 6
#: A daily sample is close enough: every real ambiguity in ``zone.tab`` (Arizona, Indiana, Kazakhstan,
#: Spain, Brazil, China, Russia, the USA) is a difference of months, not hours, and a genuine future
#: change to civil time is announced and takes effect on a specific day, not mid-day.
_SIGNATURE_STEP = timedelta(days=1)
_MAX_SIGNATURE_SAMPLES = 1000

#: A handful of GeoNames places (population >= 15,000, so genuinely real: Can, Turkey; Time, Norway;
#: Come, Benin; ...) share their whole name with an ordinary English word that turns up constantly in
#: scheduling messages. A single word this common is never a lone country or city match on its own; a
#: multi-word phrase built from two of them ("Time Square") is not a realistic accident and is
#: unaffected. Weekday and month names are excluded the same way ("May" is also a real city).
_GEO_STOPWORDS = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those", "any", "some", "no", "not", "all", "each",
        "every", "same", "next", "last", "first", "one", "local", "my", "your", "our", "us", "we", "i",
        "me", "you", "he", "she", "it", "they", "them",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "can", "could", "will", "would", "should", "shall", "must", "may", "might", "do", "does", "did",
        "have", "has", "had", "let", "lets",
        "hi", "hey", "hello", "thanks", "thank", "please", "sure", "ok", "okay", "yes", "sounds",
        "good", "great", "fine", "cool", "perfect", "works",
        "book", "call", "meet", "meeting", "intro", "demo", "chat", "talk", "set", "up", "go", "going",
        "come", "get", "got", "make", "made", "take", "took", "send", "sent", "move", "moved",
        "reschedule", "cancel", "confirm", "confirmed", "find", "look", "need", "want", "like",
        "time", "times", "date", "dates", "day", "days", "week", "weeks", "today", "tomorrow", "soon",
        "later", "now", "then", "still", "yet", "right", "ready", "available", "free", "busy", "open",
        "close", "early", "late", "morning", "noon", "afternoon", "evening", "midnight", "lunch",
        "dinner", "when", "where", "what", "who", "how", "why", "which",
        "for", "with", "and", "but", "or", "if", "so", "to", "of", "in", "on", "at", "as", "by",
        "standard", "daylight", "real", "central", "asia", "metro",
    }
    | {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Resolution:
    """One resolution: a single ``zone`` (``resolved``), two or more ``candidates`` (``ambiguous``), or
    neither (``unknown``)."""

    status: Status
    zone: str | None = None
    candidates: tuple[str, ...] = ()


UNKNOWN_RESOLUTION = Resolution("unknown")


def _resolved_or_ambiguous(zones: Sequence[str]) -> Resolution:
    """``zones`` in preference order, already deduplicated: one zone resolves, more are ambiguous."""
    if not zones:
        return UNKNOWN_RESOLUTION
    if len(zones) == 1:
        return Resolution("resolved", zones[0])
    return Resolution("ambiguous", candidates=tuple(zones))


# Word tokenising and n-gram lookup, shared by the alias, country and city steps -----------------------

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _fold(text: str) -> str:
    """Strip accents (``Ō`` -> ``O``, ``é`` -> ``e``) so a name matches however it is typed, and so a
    non-ASCII letter never silently drops out of a token (``Ōi`` -> ``Oi``, not a stray ``i``)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(_fold(text))


def _lowered_words(text: str) -> list[str]:
    return [t.lower() for t in _tokenize(text)]


def _mentions_phrase(text: str, phrase: str) -> bool:
    """Whether ``phrase`` ("Maine", "Ontario") appears in ``text`` as its own run of words."""
    words, target = _lowered_words(text), _lowered_words(phrase)
    if not target:
        return False
    return any(words[i : i + len(target)] == target for i in range(len(words) - len(target) + 1))


def _find_leftmost[T](
    text: str,
    max_words: int,
    indices: Sequence[tuple[Mapping[str, T], bool]],
    *,
    min_single_word: int = 1,
    stopwords: frozenset[str] = frozenset(),
    require_capitalized: bool = False,
) -> T | None:
    """The value at the leftmost, then longest, phrase of at most ``max_words`` words in ``text`` that
    any of ``indices`` (``(index, case_sensitive)`` pairs, checked in order at each length) knows. A
    lone word shorter than ``min_single_word``, or in ``stopwords``, is never looked up on its own, so
    a gazetteer's genuine two-letter city ("Bo", "Wa") or a place that is also a common English word
    ("Can", "Central") does not fire on an ordinary sentence; a name of two or more words is
    unaffected, since two common English words matching one together is not a realistic accident.
    ``require_capitalized`` additionally requires a lone word to start with a capital letter as
    written, the ordinary way to write a place name ("Kathmandu"), which a common word used
    mid-sentence ("...evenings are best") is not; a stopword the writer capitalised anyway (a
    sentence's own first word, "Can I...") still needs ``stopwords`` to catch it."""
    tokens = _tokenize(text)
    lowered = [t.lower() for t in tokens]
    for start in range(len(tokens)):
        limit = min(max_words, len(tokens) - start)
        for length in range(limit, 0, -1):
            if length == 1 and (
                len(tokens[start]) < min_single_word
                or lowered[start] in stopwords
                or (require_capitalized and not tokens[start][0].isupper())
            ):
                continue
            for index, case_sensitive in indices:
                words = tokens[start : start + length] if case_sensitive else lowered[start : start + length]
                found = index.get(" ".join(words))
                if found is not None:
                    return found
    return None


# Explicit IANA names and fixed offsets ------------------------------------------------------------------


def _valid_zone(name: str) -> str | None:
    """``name`` itself when it is a valid zoneinfo key written as an explicit zone (has a ``/``, or is
    exactly ``UTC``): the same rule ``agent/tools.py``'s ``valid_zone`` uses, so a bare legacy alias
    like ``EST`` or ``CET`` (a real zoneinfo key, but a fixed offset with no daylight saving) is left
    for the curated alias step instead."""
    if not (("/" in name) or name == "UTC"):
        return None
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None
    return name


_IANA_TOKEN = re.compile(r"\b[A-Za-z_]+(?:/[A-Za-z0-9_+\-]+)+\b")
_BARE_UTC = re.compile(r"\bUTC\b(?!\s*[+\-−–]\s*\d)")


def _explicit_iana(text: str) -> Resolution:
    for match in _IANA_TOKEN.finditer(text):
        zone = _valid_zone(match.group(0))
        if zone is not None:
            return Resolution("resolved", zone)
    if _BARE_UTC.search(text):
        return Resolution("resolved", "UTC")
    return UNKNOWN_RESOLUTION


_OFFSET = re.compile(r"\b(?:UTC|GMT)\s*([+\-−–])\s*(\d{1,2})(?::(\d{2}))?\b", re.IGNORECASE)


def _parsed_offset(text: str) -> int | None:
    """The signed offset in minutes of the first ``UTC``/``GMT`` offset expression in ``text``, if any."""
    match = _OFFSET.search(text)
    if match is None:
        return None
    sign = -1 if match.group(1) in "-−–" else 1
    hours, minutes = int(match.group(2)), int(match.group(3) or 0)
    return sign * (hours * 60 + minutes)


def _etc_gmt(total_minutes: int) -> Resolution:
    """A whole-hour offset as an ``Etc/GMT`` zone, whose sign is the *inverse* of the stated offset
    (``GMT+2`` -> ``Etc/GMT-2``), or plain ``UTC`` for zero."""
    if total_minutes == 0:
        return Resolution("resolved", "UTC")
    hours = -(total_minutes // 60)
    name = f"Etc/GMT{hours:+d}"
    return Resolution("resolved", name) if _valid_zone(name) else UNKNOWN_RESOLUTION


# Equivalence over the booking horizon --------------------------------------------------------------------


def _signature(zone: str, start: datetime, end: datetime) -> tuple[int, ...]:
    """The zone's UTC offset (seconds), sampled daily from ``start`` to ``end`` inclusive. Two zones
    with the same signature over the same window carry the same civil time at every sampled instant."""
    tz = ZoneInfo(zone)

    def offset_at(instant: datetime) -> int:
        offset = instant.astimezone(tz).utcoffset()
        return int((offset or timedelta(0)).total_seconds())

    samples = [offset_at(start)]
    current = start
    count = 0
    while current < end and count < _MAX_SIGNATURE_SAMPLES:
        current = min(current + _SIGNATURE_STEP, end)
        samples.append(offset_at(current))
        count += 1
    return tuple(samples)


def _equivalent(zone_a: str, zone_b: str, now: datetime, horizon_days: int) -> bool:
    if zone_a == zone_b:
        return True
    end = now + timedelta(days=horizon_days)
    return _signature(zone_a, now, end) == _signature(zone_b, now, end)


def _equivalence_groups(zones: Sequence[str], now: datetime, horizon_days: int) -> list[list[str]]:
    """``zones`` partitioned by identical signature over ``[now, now + horizon_days]``, each group in
    the order its first member appears in ``zones``."""
    end = now + timedelta(days=horizon_days)
    groups: dict[tuple[int, ...], list[str]] = {}
    for zone in zones:
        groups.setdefault(_signature(zone, now, end), []).append(zone)
    return list(groups.values())


# Local time <-> UTC (design.md SS2: "nonexistent local time -> reject; ambiguous (fold) -> require
# confirmation"). Every conversion elsewhere in this module goes the other way, UTC to local, which is
# always well defined; this is the one place a naive local wall-clock time is turned into an instant. ---


@dataclass(frozen=True)
class LocalInstant:
    """A naive local time resolved against a zone: ``exists`` (one UTC instant), ``ambiguous`` (a DST
    fold: the wall clock reads this twice, ``earlier`` and ``later``), or ``nonexistent`` (a DST gap:
    the wall clock skips over this reading and it is never observed)."""

    status: Literal["exists", "ambiguous", "nonexistent"]
    utc: datetime | None = None
    earlier: datetime | None = None
    later: datetime | None = None


def local_instant(zone: str, local: datetime) -> LocalInstant:
    """Resolve a naive ``local`` wall-clock reading in ``zone`` to UTC. Constructing the same reading
    with both PEP 495 fold values and comparing their offsets tells the three cases apart: equal
    offsets mean an ordinary, unambiguous time; a later fold=1 instant than fold=0 means a fold
    (clocks set back: this reading occurs once before the change and once after); an *earlier* fold=1
    instant means a gap (clocks set forward: the reading between the old and new offsets never
    happens, and both folds land on either side of it, one before the jump and one after)."""
    if local.tzinfo is not None:
        raise ValueError("local_instant expects a naive datetime (no tzinfo)")
    tz = ZoneInfo(zone)
    early = local.replace(tzinfo=tz, fold=0)
    late = local.replace(tzinfo=tz, fold=1)
    if early.utcoffset() == late.utcoffset():
        return LocalInstant("exists", utc=early.astimezone(UTC))
    early_utc, late_utc = early.astimezone(UTC), late.astimezone(UTC)
    if early_utc < late_utc:
        return LocalInstant("ambiguous", earlier=early_utc, later=late_utc)
    return LocalInstant("nonexistent")


# Index building --------------------------------------------------------------------------------------


def _city_readings(row: CityRow) -> Iterator[str]:
    """Every phrase this row answers to: its name and ASCII name, and, for a name GeoNames spells with
    a trailing "City" (New York City, Panama City, Kuwait City, Quebec City, ...), the shorter, at
    least as common everyday form too ("New York", "Panama")."""
    for name in (row.name, row.ascii_name):
        key = " ".join(_lowered_words(name))
        if key:
            yield key
        words = key.split()
        if len(words) > 1 and words[-1] == "city":
            yield " ".join(words[:-1])


def _build_city_index(cities: Iterable[CityRow]) -> dict[str, list[CityRow]]:
    index: dict[str, list[CityRow]] = {}
    for row in cities:
        for key in _city_readings(row):
            rows = index.setdefault(key, [])
            if row not in rows:
                rows.append(row)
    return index


def _build_alias_indices(
    aliases: Mapping[str, AliasEntry],
) -> tuple[dict[str, AliasEntry], dict[str, AliasEntry]]:
    """Case-sensitive (an all-capitals key, e.g. ``IST``) and case-insensitive (any other key, e.g.
    ``Eastern time``) indices, keyed by the normalised phrase."""
    case_sensitive: dict[str, AliasEntry] = {}
    case_insensitive: dict[str, AliasEntry] = {}
    for text, entry in aliases.items():
        if text.isupper():
            case_sensitive[" ".join(_tokenize(text))] = entry
        else:
            case_insensitive[" ".join(_lowered_words(text))] = entry
    return case_sensitive, case_insensitive


def _build_country_index(country_names: Mapping[str, str]) -> dict[str, str]:
    """Re-key ``load_country_names()``'s readings through the same word tokeniser the matcher uses at
    query time, so an accented name ("Curaçao", "Côte d'Ivoire") matches however it is typed, and two
    readings that fold to the same key (rare) keep every code either named."""
    index: dict[str, str] = {}
    for reading, codes in country_names.items():
        key = " ".join(_lowered_words(reading))
        if not key:
            continue
        existing = index.get(key)
        if existing is None:
            index[key] = codes
        elif existing != codes:
            index[key] = ",".join(sorted(set(existing.split(",")) | set(codes.split(","))))
    return index


# The resolver ------------------------------------------------------------------------------------------


class TimezoneResolver:
    """Deterministic time zone resolution over curated aliases, ``tzdata``'s country and zone tables,
    and the bundled city gazetteer. Construct once (:func:`get_resolver` gives a shared instance) and
    reuse: loading is cached, but the country and city indices are built once here."""

    def __init__(
        self,
        *,
        aliases: Mapping[str, AliasEntry] | None = None,
        country_names: Mapping[str, str] | None = None,
        country_zones: Mapping[str, tuple[str, ...]] | None = None,
        cities: Sequence[CityRow] | None = None,
        all_zones: Sequence[str] | None = None,
    ) -> None:
        self._alias_cs, self._alias_ci = _build_alias_indices(
            aliases if aliases is not None else load_aliases()
        )
        self._countries = _build_country_index(
            country_names if country_names is not None else load_country_names()
        )
        self._country_zones = dict(country_zones if country_zones is not None else load_country_zones())
        self._cities = _build_city_index(cities if cities is not None else load_cities())
        self._all_zones = tuple(all_zones if all_zones is not None else load_all_zones())

    # Step 2b: a fractional fixed offset, searched over every zone.tab zone ---------------------------

    def _zone_of_constant_offset(self, total_minutes: int, now: datetime, horizon_days: int) -> Resolution:
        target = total_minutes * 60
        end = now + timedelta(days=horizon_days)
        candidates = [z for z in self._all_zones if _signature(z, now, now)[0] == target]
        constant = [z for z in candidates if all(s == target for s in _signature(z, now, end))]
        if not constant:
            return UNKNOWN_RESOLUTION
        return Resolution("resolved", constant[0])

    def _offset(self, text: str, now: datetime, horizon_days: int) -> Resolution:
        total_minutes = _parsed_offset(text)
        if total_minutes is None:
            return UNKNOWN_RESOLUTION
        if total_minutes % 60 == 0:
            return _etc_gmt(total_minutes)
        return self._zone_of_constant_offset(total_minutes, now, horizon_days)

    # Step 3: curated aliases -----------------------------------------------------------------------------

    def _alias(self, text: str) -> Resolution:
        entry = _find_leftmost(text, _ALIAS_MAX_WORDS, [(self._alias_cs, True), (self._alias_ci, False)])
        if entry is None:
            return UNKNOWN_RESOLUTION
        if entry.status == "resolved":
            assert entry.zone is not None
            return Resolution("resolved", entry.zone)
        return Resolution("ambiguous", candidates=entry.candidates)

    # Step 4: countries -------------------------------------------------------------------------------------

    def _country(self, text: str, now: datetime, horizon_days: int) -> Resolution:
        key = _find_leftmost(
            text,
            _COUNTRY_MAX_WORDS,
            [(self._countries, False)],
            min_single_word=3,
            stopwords=_GEO_STOPWORDS,
            require_capitalized=True,
        )
        if key is None:
            return UNKNOWN_RESOLUTION
        zones: list[str] = []
        for code in country_codes_for(key):
            for zone in self._country_zones.get(code, ()):
                if zone not in zones:
                    zones.append(zone)
        if not zones:
            return UNKNOWN_RESOLUTION
        groups = _equivalence_groups(zones, now, horizon_days)
        return _resolved_or_ambiguous([group[0] for group in groups])

    # Step 5: cities -----------------------------------------------------------------------------------------

    def _city(self, text: str, now: datetime, horizon_days: int) -> Resolution:
        rows = _find_leftmost(
            text,
            _CITY_MAX_WORDS,
            [(self._cities, False)],
            min_single_word=3,
            stopwords=_GEO_STOPWORDS,
            require_capitalized=True,
        )
        if rows is None:
            return UNKNOWN_RESOLUTION
        ranked = sorted(rows, key=lambda r: -r.population)
        top = ranked[0]
        qualified = self._qualified_match(text, ranked[1:], top)
        if qualified is not None:
            return Resolution("resolved", qualified.timezone)
        threshold = CITY_AMBIGUITY_SHARE * top.population
        zones = [top.timezone]
        for row in ranked[1:]:
            if row.population < threshold:
                break
            if row.timezone in zones or _equivalent(top.timezone, row.timezone, now, horizon_days):
                continue
            zones.append(row.timezone)
        return _resolved_or_ambiguous(zones)

    @staticmethod
    def _qualified_match(text: str, others: Sequence[CityRow], top: CityRow) -> CityRow | None:
        """A same-named match the text names by its own region ("Portland, Maine" beats the more
        populous Portland, Oregon), when the top match's own region is not also named. Population
        still breaks a tie between two qualified regions of the same name."""
        if top.admin1_name and _mentions_phrase(text, top.admin1_name):
            return None
        qualified = [r for r in others if r.admin1_name and _mentions_phrase(text, r.admin1_name)]
        return max(qualified, key=lambda r: r.population) if qualified else None

    # The five steps, in order -----------------------------------------------------------------------------

    def resolve(
        self,
        text: str,
        *,
        now: datetime,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        include_geo: bool = True,
    ) -> Resolution:
        """Resolve ``text`` against the five steps in order, stopping at the first that names a zone or
        candidates. ``include_geo=False`` skips the country and city steps (used for a bare-token scan
        of unstructured text, where a country or city match is more likely to be a false positive)."""
        explicit = _explicit_iana(text)
        if explicit.status != "unknown":
            return explicit
        offset = self._offset(text, now, horizon_days)
        if offset.status != "unknown":
            return offset
        alias = self._alias(text)
        if alias.status != "unknown":
            return alias
        if not include_geo:
            return UNKNOWN_RESOLUTION
        country = self._country(text, now, horizon_days)
        if country.status != "unknown":
            return country
        return self._city(text, now, horizon_days)

    # The pre-scan phrase detector (design-agent.md SSB.5) --------------------------------------------------

    def prescan(
        self, text: str, *, now: datetime, horizon_days: int = DEFAULT_HORIZON_DAYS
    ) -> tuple[str, Resolution] | None:
        """The zone statement in ``text``, if any: a lead-in phrase ("I'm in X", "we're on X", "calling
        from X", "X time") resolved through all five steps, else a bare abbreviation, offset or IANA
        name found anywhere in the text. ``None`` when nothing in ``text`` looks like a zone statement
        at all; a phrase that is found but names no known zone is not reported either, so the model's
        own ``resolve_timezone`` tool call is still the one that asks the prospect."""
        for phrase in _candidate_phrases(text):
            resolution = self.resolve(phrase, now=now, horizon_days=horizon_days)
            if resolution.status != "unknown":
                return phrase, resolution
        bare = self.resolve(text, now=now, horizon_days=horizon_days, include_geo=False)
        if bare.status != "unknown":
            return text, bare
        return None


# Phrase detection for the pre-scan -----------------------------------------------------------------------

_SPLIT_TAIL = re.compile(r"\s+(?:and|but|so|for|with|because|please|right now|today|,)\s+.*$", re.IGNORECASE)
_LEAD_IN = re.compile(
    r"\b(?:i['’]?m|i\s+am|we['’]?re|we\s+are)\s+"
    r"(?:based\s+|located\s+|currently\s+|over\s+)?(?:in|on|at|from)\s+([A-Za-z][^.,!?;\n]*)",
    re.IGNORECASE,
)
_CALLING_FROM = re.compile(
    r"\b(?:calling|writing|dialing|dialling)\s+from\s+([A-Za-z][^.,!?;\n]*)", re.IGNORECASE
)
_X_TIME = re.compile(r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})\s+(?:time|timezone|time\s+zone)\b")


def _clean_phrase(raw: str) -> str:
    trimmed = _SPLIT_TAIL.sub("", raw.strip())
    return trimmed.rstrip(" .!?").strip()


def _candidate_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for pattern in (_LEAD_IN, _CALLING_FROM):
        match = pattern.search(text)
        if match:
            phrase = _clean_phrase(match.group(1))
            if phrase:
                phrases.append(phrase)
    for match in _X_TIME.finditer(text):
        phrases.append(f"{match.group(1)} time")
    return phrases


@lru_cache(maxsize=1)
def get_resolver() -> TimezoneResolver:
    """The process-wide resolver: data loading is cached, so building this is cheap after the first
    call."""
    return TimezoneResolver()


__all__ = [
    "DEFAULT_HORIZON_DAYS",
    "LocalInstant",
    "Resolution",
    "TimezoneResolver",
    "get_resolver",
    "local_instant",
]

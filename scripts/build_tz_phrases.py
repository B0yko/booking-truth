"""Build ``datasets/tz_phrases.jsonl``: 150 labelled prospect timezone phrases.

The set mixes three sources:

* ``city_sample``: GeoNames cities from ``datasets/cities_tz.csv``, sampled by population stratum and
  phrased with a few templates, sometimes with a region or country qualifier;
* ``template``: fixed UTC/GMT offsets, zone names and abbreviations;
* ``hard_case``: the hand-written list ``HARD_CASES`` below.

Labels follow the rules in ``datasets/README.md``. They are defined without reference to any
resolver: the gold zone is the zone the phrase refers to, and two zones count as the same when their
UTC offsets are identical at every instant from 2026-01-01 to 2027-12-31 ("equivalent"). Zone data
comes only from the ``tzdata`` package, so the output does not depend on the operating system's
copy of the tz database. Hand-written labels that can be derived from a rule are recomputed and the
build fails if the two disagree.

The output is deterministic: the same ``cities_tz.csv`` and ``tzdata`` version give the same bytes.

Usage::

    uv run python scripts/build_tz_phrases.py \
        [--cities datasets/cities_tz.csv] [--out datasets/tz_phrases.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

SEED = 20260926
HOST_ZONE = "America/New_York"
WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_END = datetime(2028, 1, 1, tzinfo=UTC)
SCAN_STEP_S = 6 * 3600
# A same-name city is another reading of a bare city name only when it passes both thresholds: an
# absolute population and a share of the most populous match's population (in percent, compared in
# integers so the boundary is exact).
CITY_AMBIGUITY_MIN_POPULATION = 50_000
CITY_AMBIGUITY_MIN_SHARE_PERCENT = 10
CITY_THRESHOLDS_TEXT = (
    f"population >= {CITY_AMBIGUITY_MIN_POPULATION:,} and >= {CITY_AMBIGUITY_MIN_SHARE_PERCENT}% "
    "of the most populous match"
)
N_TOTAL = 150
N_DEV = 50
N_CITY = 45
N_TEMPLATE = 35
STRATA: tuple[tuple[str, int, int | None, int], ...] = (
    (">=1M", 1_000_000, None, 12),
    ("300k-1M", 300_000, 1_000_000, 11),
    ("100k-300k", 100_000, 300_000, 11),
    ("15k-100k", 15_000, 100_000, 11),
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITIES = REPO_ROOT / "datasets" / "cities_tz.csv"
DEFAULT_OUT = REPO_ROOT / "datasets" / "tz_phrases.jsonl"

Status = Literal["resolved", "ambiguous", "unknown"]


class BuildError(RuntimeError):
    """A label could not be derived, or a hand-written label disagrees with its rule."""


# --------------------------------------------------------------------------------------------------
# tz database (from the tzdata package only)
# --------------------------------------------------------------------------------------------------

Signature = tuple[int, tuple[tuple[int, int], ...]]


class TzDb:
    """Zone keys, ``zone.tab``, ``iso3166.tab`` and the zone equivalence relation."""

    def __init__(self) -> None:
        pkg = resources.files("tzdata")
        zoneinfo_dir = resources.files("tzdata.zoneinfo")
        zones_text = pkg.joinpath("zones").read_text(encoding="utf-8")
        self.keys = frozenset(line.strip() for line in zones_text.splitlines() if line.strip())
        self.zone_tab: list[tuple[str, str]] = []
        self.country_zones: dict[str, list[str]] = {}
        for line in zoneinfo_dir.joinpath("zone.tab").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            self.zone_tab.append((fields[0], fields[2]))
            self.country_zones.setdefault(fields[0], []).append(fields[2])
        self.country_names: dict[str, str] = {}
        for line in zoneinfo_dir.joinpath("iso3166.tab").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                code, name = line.split("\t")[:2]
                self.country_names[code] = name
        self._zones: dict[str, ZoneInfo] = {}
        self._signatures: dict[str, Signature] = {}

    def zone(self, key: str) -> ZoneInfo:
        if key not in self.keys:
            raise BuildError(f"not a tzdata zone key: {key}")
        cached = self._zones.get(key)
        if cached is None:
            node = resources.files("tzdata.zoneinfo")
            for part in key.split("/"):
                node = node.joinpath(part)
            with node.open("rb") as fh:
                cached = ZoneInfo.from_file(fh, key=key)
            self._zones[key] = cached
        return cached

    def _offset(self, zone: ZoneInfo, ts: int) -> int:
        delta = datetime.fromtimestamp(ts, UTC).astimezone(zone).utcoffset()
        assert delta is not None
        return int(delta.total_seconds())

    def signature(self, key: str) -> Signature:
        """Offset at the window start plus every (instant, new offset) transition inside the window."""
        cached = self._signatures.get(key)
        if cached is not None:
            return cached
        zone = self.zone(key)
        start, end = int(WINDOW_START.timestamp()), int(WINDOW_END.timestamp())
        initial = self._offset(zone, start)
        transitions: list[tuple[int, int]] = []
        prev_ts, prev_off = start, initial
        ts = start
        while ts < end:
            ts = min(ts + SCAN_STEP_S, end)
            off = self._offset(zone, ts)
            if off != prev_off:
                lo, hi = prev_ts, ts
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if self._offset(zone, mid) == prev_off:
                        lo = mid
                    else:
                        hi = mid
                transitions.append((hi, off))
            prev_ts, prev_off = ts, off
        signature: Signature = (initial, tuple(transitions))
        self._signatures[key] = signature
        return signature

    def equivalent(self, a: str, b: str) -> bool:
        return a == b or self.signature(a) == self.signature(b)

    def constant_offset(self, key: str) -> int | None:
        initial, transitions = self.signature(key)
        return None if transitions else initial


# --------------------------------------------------------------------------------------------------
# GeoNames gazetteer
# --------------------------------------------------------------------------------------------------


def strip_accents(text: str) -> str:
    """Drop combining marks after NFKD decomposition (``São Paulo`` -> ``Sao Paulo``)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def fold(text: str) -> str:
    """Case- and accent-insensitive key: strip accents, unify apostrophes, collapse spaces, casefold."""
    stripped = strip_accents(text)
    stripped = stripped.replace("’", "'").replace("‘", "'")
    return " ".join(stripped.split()).casefold()


def letters(text: str) -> str:
    """Letters and digits only, folded: ``Namp’o`` and ``Nampo`` compare equal."""
    return "".join(ch for ch in fold(text) if ch.isalnum())


@dataclass(frozen=True)
class CityRow:
    index: int
    name: str
    asciiname: str
    country_code: str
    admin1_code: str
    admin1_name: str
    population: int
    timezone: str


class Gazetteer:
    def __init__(self, path: Path) -> None:
        self.rows: list[CityRow] = []
        with path.open(encoding="utf-8", newline="") as fh:
            for i, rec in enumerate(csv.DictReader(fh)):
                self.rows.append(
                    CityRow(
                        index=i,
                        name=rec["name"],
                        asciiname=rec["asciiname"],
                        country_code=rec["country_code"],
                        admin1_code=rec["admin1_code"],
                        admin1_name=rec["admin1_name"],
                        population=int(rec["population"]),
                        timezone=rec["timezone"],
                    )
                )
        self._by_name: dict[str, list[CityRow]] = {}
        self.zone_weight: dict[str, int] = {}
        self.admin1_names: set[str] = set()
        for row in self.rows:
            for key in {fold(row.name), fold(row.asciiname)}:
                self._by_name.setdefault(key, []).append(row)
            self.zone_weight[row.timezone] = max(self.zone_weight.get(row.timezone, 0), row.population)
            if row.admin1_name:
                self.admin1_names.add(fold(row.admin1_name))
        for matches in self._by_name.values():
            matches.sort(key=lambda r: (-r.population, r.index))

    def lookup(self, name: str) -> list[CityRow]:
        return list(self._by_name.get(fold(name), []))


# --------------------------------------------------------------------------------------------------
# Labels and labelling rules
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Label:
    status: Status
    zone: str | None
    candidates: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {"status": self.status, "zone": self.zone, "candidates": list(self.candidates)}


def resolved(zone: str) -> Label:
    return Label("resolved", zone, ())


def ambiguous(*zones: str) -> Label:
    return Label("ambiguous", None, tuple(sorted(zones)))


UNKNOWN = Label("unknown", None, ())


def collapse(tz: TzDb, zones: Iterable[str]) -> Label:
    """Group zones into equivalence classes; the first zone seen in a class represents it."""
    reps: list[str] = []
    for zone in zones:
        if not any(tz.equivalent(zone, rep) for rep in reps):
            reps.append(zone)
    if not reps:
        raise BuildError("no zones to label")
    return resolved(reps[0]) if len(reps) == 1 else ambiguous(*reps)


def city_matches(gaz: Gazetteer, name: str, *, admin1: str | None, country: str | None) -> list[CityRow]:
    matches = gaz.lookup(name)
    if admin1 is not None:
        matches = [m for m in matches if fold(m.admin1_name) == fold(admin1)]
    if country is not None:
        matches = [m for m in matches if m.country_code == country]
    if not matches:
        raise BuildError(f"no GeoNames city {name!r} (admin1={admin1!r}, country={country!r})")
    return matches


def is_other_reading(match: CityRow, top: CityRow) -> bool:
    """A same-name city counts when its population is >= 50,000 and >= 10% of the top match's."""
    return (
        match.population >= CITY_AMBIGUITY_MIN_POPULATION
        and 100 * match.population >= CITY_AMBIGUITY_MIN_SHARE_PERCENT * top.population
    )


def city_rule(
    tz: TzDb, gaz: Gazetteer, name: str, *, admin1: str | None = None, country: str | None = None
) -> tuple[Label, list[CityRow]]:
    """City rule: the most populous match, plus every other match that passes both thresholds
    (``is_other_reading``), collapsed by zone equivalence."""
    matches = city_matches(gaz, name, admin1=admin1, country=country)
    top = matches[0]
    relevant = [top] + [m for m in matches[1:] if is_other_reading(m, top)]
    return collapse(tz, (m.timezone for m in relevant)), relevant


def country_zones(tz: TzDb, gaz: Gazetteer, code: str) -> list[str]:
    """The country's ``zone.tab`` zones, most populous first."""
    zones = tz.country_zones[code]
    return sorted(zones, key=lambda z: (-gaz.zone_weight.get(z, 0), zones.index(z)))


def country_rule(tz: TzDb, gaz: Gazetteer, code: str) -> Label:
    """Country rule: all ``zone.tab`` zones of the country, collapsed by equivalence."""
    return collapse(tz, country_zones(tz, gaz, code))


def region_zones(gaz: Gazetteer, admin1: str, country: str) -> list[str]:
    """Zones of the GeoNames cities in a first-level division, the most populous city's zone first."""
    rows = [r for r in gaz.rows if r.country_code == country and fold(r.admin1_name) == fold(admin1)]
    if not rows:
        raise BuildError(f"no GeoNames city in region {admin1!r} ({country})")
    rows.sort(key=lambda r: (-r.population, r.index))
    return [r.timezone for r in rows]


def offset_rule(tz: TzDb, gaz: Gazetteer, minutes: int) -> Label:
    """Fixed-offset rule: ``Etc/GMT-N`` for whole hours (inverted POSIX sign), else a constant zone."""
    if minutes % 60 == 0:
        hours = minutes // 60
        if hours == 0:
            return resolved("Etc/GMT")
        return resolved(f"Etc/GMT{'-' if hours > 0 else '+'}{abs(hours)}")
    constant = [z for _, z in tz.zone_tab if tz.constant_offset(z) == minutes * 60]
    if not constant:
        raise BuildError(f"no zone.tab zone has a constant offset of {minutes} minutes")
    constant.sort(key=lambda z: -gaz.zone_weight.get(z, 0))
    return resolved(constant[0])


def format_offset(minutes: int) -> str:
    sign = "+" if minutes >= 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours:02d}:{mins:02d}"


def same_label(tz: TzDb, a: Label, b: Label) -> bool:
    """Labels agree when statuses match and zones / candidate sets agree up to equivalence."""
    if a.status != b.status:
        return False
    if a.status == "resolved":
        assert a.zone is not None
        assert b.zone is not None
        return tz.equivalent(a.zone, b.zone)
    if len(a.candidates) != len(b.candidates):
        return False
    return all(sum(tz.equivalent(x, y) for y in b.candidates) == 1 for x in a.candidates)


# --------------------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------------------

Source = Literal["city_sample", "template", "hard_case"]


@dataclass(frozen=True)
class Item:
    text: str
    label: Label
    source: Source
    note: str


def to_line(item_id: str, item: Item, split: str) -> str:
    record = {
        "id": item_id,
        "text": item.text,
        "label": item.label.as_json(),
        "source": item.source,
        "split": split,
        "note": item.note,
    }
    return json.dumps(record, ensure_ascii=False)


# --------------------------------------------------------------------------------------------------
# Hand-written hard cases
# --------------------------------------------------------------------------------------------------
#
# ``check`` names the rule that must reproduce the hand label ("city", "region", "country", "offset",
# "host"); the build fails when it does not. Items without ``check`` are labelled from the note alone.

HOST_NOTE = (
    "Host-relative phrase: refers to the host's zone, which is America/New_York "
    "(the sandbox default host timezone)."
)

HARD_CASES: list[dict[str, Any]] = [
    # --- abbreviations and zone names ---
    {
        "text": "IST",
        "label": ambiguous("Asia/Kolkata", "Asia/Jerusalem", "Europe/Dublin"),
        "note": "IST is India, Israel and Irish Standard Time: three non-equivalent zones.",
    },
    {
        "text": "10am IST works, I'm in Bangalore",
        "label": resolved("Asia/Kolkata"),
        "note": "IST alone is ambiguous; Bangalore (Bengaluru) fixes India Standard Time.",
    },
    {
        "text": "CST",
        "label": ambiguous("America/Chicago", "Asia/Shanghai"),
        "note": "CST is US Central Standard Time and China Standard Time.",
    },
    {
        "text": "we're on CST at our Chicago office",
        "label": resolved("America/Chicago"),
        "note": "CST alone is ambiguous; Chicago fixes US Central time.",
    },
    {
        "text": "BST",
        "label": ambiguous("Europe/London", "Asia/Dhaka"),
        "note": "BST is British Summer Time and Bangladesh Standard Time.",
    },
    {
        "text": "EST",
        "label": resolved("America/New_York"),
        "note": (
            "EST is used for US Eastern time all year; it names the region's time, not a fixed "
            "UTC-05:00 offset. Australian Eastern time is written AEST."
        ),
    },
    {
        "text": "Eastern",
        "label": resolved("America/New_York"),
        "note": "A bare North American zone name refers to the US/Canada zone (Eastern = America/New_York).",
    },
    {
        "text": "Central European Time",
        "label": resolved("Europe/Berlin"),
        "note": "The EU CET/CEST region; its member zones are equivalent, Europe/Berlin is used as gold.",
    },
    {
        "text": "CET",
        "label": resolved("Europe/Berlin"),
        "note": "CET names the EU Central European region (with summer time), gold Europe/Berlin.",
    },
    # --- host-relative ---
    {"text": "same as you", "label": resolved(HOST_ZONE), "note": HOST_NOTE, "check": {"host": True}},
    {"text": "your time is fine", "label": resolved(HOST_ZONE), "note": HOST_NOTE, "check": {"host": True}},
    {
        "text": "whatever timezone your office is in",
        "label": resolved(HOST_ZONE),
        "note": HOST_NOTE,
        "check": {"host": True},
    },
    # --- cities: bare and qualified ---
    {
        "text": "I'm in Portland",
        "label": ambiguous("America/Los_Angeles", "America/New_York"),
        "note": (
            "City rule: Portland, Maine (66,881, 10.2% of Portland, Oregon) passes both thresholds "
            "(>= 50,000 and >= 10% of the most populous match) and is in a non-equivalent zone."
        ),
        "check": {"city": "Portland"},
    },
    {
        "text": "Portland, Maine",
        "label": resolved("America/New_York"),
        "note": "Qualified city: the state picks Portland, Maine.",
        "check": {"city": "Portland", "admin1": "Maine"},
    },
    {
        "text": "we're in Springfield",
        "label": ambiguous("America/Chicago", "America/New_York", "America/Los_Angeles"),
        "note": (
            "City rule: Springfield MO/IL (Central), MA/OH (Eastern) and OR (Pacific) all have >= 50,000 "
            "and >= 10% of Springfield, Missouri, the most populous match."
        ),
        "check": {"city": "Springfield"},
    },
    {
        "text": "Birmingham",
        "label": ambiguous("Europe/London", "America/Chicago"),
        "note": (
            "City rule: Birmingham, Alabama (196,357, 17.0% of Birmingham, England) passes both "
            "thresholds (>= 50,000 and >= 10%) and is in a non-equivalent zone."
        ),
        "check": {"city": "Birmingham"},
    },
    {
        "text": "Victoria",
        "label": ambiguous("Asia/Hong_Kong", "America/Vancouver", "Australia/Melbourne"),
        "note": (
            "City rule gives Victoria (Hong Kong) and Victoria BC (30.3% of it); Victoria TX (67,574, "
            "7.1%) is below the 10% threshold. Victoria is also an Australian state "
            "(Australia/Melbourne), so that reading is a candidate too."
        ),
        "check": {"city": "Victoria", "extra": ["Australia/Melbourne"]},
    },
    {
        "text": "Victoria, BC",
        "label": resolved("America/Vancouver"),
        "note": "Qualified city: BC is British Columbia.",
        "check": {"city": "Victoria", "admin1": "British Columbia"},
    },
    {
        "text": "I'm in Kathmandu",
        "label": resolved("Asia/Kathmandu"),
        "note": "City rule: single match; UTC+05:45 zone.",
        "check": {"city": "Kathmandu"},
    },
    {
        "text": "Tehran",
        "label": resolved("Asia/Tehran"),
        "note": "City rule: single match; UTC+03:30 with no summer time since 2022.",
        "check": {"city": "Tehran"},
    },
    {
        "text": "we're in Adelaide",
        "label": resolved("Australia/Adelaide"),
        "note": "City rule: single match; half-hour zone with southern-hemisphere summer time.",
        "check": {"city": "Adelaide"},
    },
    {
        "text": "Kyiv",
        "label": resolved("Europe/Kyiv"),
        "note": "City rule: single match.",
        "check": {"city": "Kyiv"},
    },
    {
        "text": "Kiev",
        "label": resolved("Europe/Kyiv"),
        "note": "Older English spelling of Kyiv; the tz key is Europe/Kyiv (Europe/Kiev is a legacy link).",
    },
    {
        "text": "Sao Paulo",
        "label": resolved("America/Sao_Paulo"),
        "note": "City rule on the unaccented spelling of São Paulo; single match.",
        "check": {"city": "Sao Paulo"},
    },
    {
        "text": "calling from Zurich",
        "label": resolved("Europe/Zurich"),
        "note": "City rule on the unaccented spelling of Zürich; single match.",
        "check": {"city": "Zurich"},
    },
    {
        "text": "London",
        "label": resolved("Europe/London"),
        "note": (
            "City rule: London, Ontario has 422,324 inhabitants but only 4.7% of London, England's "
            "population, below the 10% threshold, so the bare name refers to London, England."
        ),
        "check": {"city": "London"},
    },
    {
        "text": "London, Ontario",
        "label": resolved("America/Toronto"),
        "note": "Qualified city: the province picks London, Ontario.",
        "check": {"city": "London", "admin1": "Ontario"},
    },
    {
        "text": "Hyderabad",
        "label": ambiguous("Asia/Kolkata", "Asia/Karachi"),
        "note": (
            "City rule: Hyderabad (Pakistan) has 27.5% of the population of Hyderabad (India); both pass "
            "the thresholds and the zones are not equivalent."
        ),
        "check": {"city": "Hyderabad"},
    },
    {
        "text": "we're in San Jose",
        "label": ambiguous("America/Los_Angeles", "America/Costa_Rica", "Asia/Manila"),
        "note": (
            "City rule: San José (Costa Rica, 33.6% of San Jose, California) and San Jose (Philippines, "
            "Mimaropa, 143,495, 14.4%) pass both thresholds; the three zones are not equivalent."
        ),
        "check": {"city": "San Jose"},
    },
    {
        "text": "I'm in Pune",
        "label": resolved("Asia/Kolkata"),
        "note": "City rule: single match.",
        "check": {"city": "Pune"},
    },
    {
        "text": "Perth",
        "label": resolved("Australia/Perth"),
        "note": (
            "City rule: Perth, Scotland (47,350, 2.0% of Perth, Western Australia) is below both "
            "thresholds, so Perth, Western Australia is the referent."
        ),
        "check": {"city": "Perth"},
    },
    {
        "text": "I split my time between Boston and Chicago",
        "label": ambiguous("America/New_York", "America/Chicago"),
        "note": "Two cities in non-equivalent zones; the phrase names both.",
    },
    {
        "text": "Sydney, Australia",
        "label": resolved("Australia/Sydney"),
        "note": "Qualified city: the country excludes Sydney, Nova Scotia.",
        "check": {"city": "Sydney", "country": "AU"},
    },
    # --- colloquial names ---
    {"text": "NYC", "label": resolved("America/New_York"), "note": "Common abbreviation of New York City."},
    {
        "text": "we're in LA",
        "label": resolved("America/Los_Angeles"),
        "note": "LA as a place name means Los Angeles.",
    },
    {
        "text": "the Bay Area",
        "label": resolved("America/Los_Angeles"),
        "note": "The San Francisco Bay Area, US Pacific time.",
    },
    {
        "text": "Frankfurt",
        "label": resolved("Europe/Berlin"),
        "note": "Frankfurt am Main and Frankfurt (Oder) are both in Germany (Europe/Berlin).",
    },
    # --- regions ---
    {
        "text": "we're in Queensland",
        "label": resolved("Australia/Brisbane"),
        "note": (
            "Region rule: every GeoNames city in Queensland uses Australia/Brisbane, which keeps AEST "
            "all year (Australia/Lindeman is equivalent)."
        ),
        "check": {"region": "Queensland", "country": "AU"},
    },
    {
        "text": "Arizona",
        "label": resolved("America/Phoenix"),
        "note": (
            "Region rule: every GeoNames city in Arizona uses America/Phoenix (MST, no summer time). "
            "The Navajo Nation observes summer time but has no city of 15,000 or more."
        ),
        "check": {"region": "Arizona", "country": "US"},
    },
    {
        "text": "Hawaii",
        "label": resolved("Pacific/Honolulu"),
        "note": "Region rule: every GeoNames city in Hawaii uses Pacific/Honolulu.",
        "check": {"region": "Hawaii", "country": "US"},
    },
    {
        "text": "Newfoundland",
        "label": resolved("America/St_Johns"),
        "note": "zone.tab: America/St_Johns covers Newfoundland (UTC-03:30 with summer time).",
    },
    {
        "text": "Chatham Islands",
        "label": resolved("Pacific/Chatham"),
        "note": "zone.tab: Pacific/Chatham (UTC+12:45 with summer time).",
    },
    {
        "text": "Indiana",
        "label": ambiguous("America/Indiana/Indianapolis", "America/Chicago"),
        "note": (
            "Region rule: Indiana is mostly Eastern, but its north-west and south-west counties "
            "(Gary, Hammond, Evansville) use Central time (America/Chicago)."
        ),
        "check": {"region": "Indiana", "country": "US"},
    },
    {
        "text": "Texas",
        "label": ambiguous("America/Chicago", "America/Denver"),
        "note": "Region rule: Texas is Central except the El Paso area, which is Mountain (America/Denver).",
        "check": {"region": "Texas", "country": "US"},
    },
    {
        "text": "Georgia",
        "label": ambiguous("America/New_York", "Asia/Tbilisi"),
        "note": "Georgia is a US state (Eastern) and a country (Asia/Tbilisi).",
        "check": {"region": "Georgia", "country": "US", "also_country": "GE"},
    },
    {
        "text": "Lord Howe Island",
        "label": resolved("Australia/Lord_Howe"),
        "note": "zone.tab: Australia/Lord_Howe (UTC+10:30 with a 30-minute summer shift).",
    },
    # --- countries ---
    {
        "text": "Nepal",
        "label": resolved("Asia/Kathmandu"),
        "note": "Country rule: one zone.tab zone.",
        "check": {"country": "NP"},
    },
    {
        "text": "I'm in India",
        "label": resolved("Asia/Kolkata"),
        "note": "Country rule: one zone.tab zone.",
        "check": {"country": "IN"},
    },
    {
        "text": "China",
        "label": ambiguous("Asia/Shanghai", "Asia/Urumqi"),
        "note": (
            "Country rule: zone.tab lists Asia/Shanghai (Beijing Time) and Asia/Urumqi (Xinjiang Time, "
            "UTC+06:00), which are not equivalent."
        ),
        "check": {"country": "CN"},
    },
    {
        "text": "we're in Brazil",
        "label": ambiguous("America/Sao_Paulo", "America/Manaus", "America/Noronha", "America/Rio_Branco"),
        "note": "Country rule: four offset classes (UTC-02, -03, -04, -05).",
        "check": {"country": "BR"},
    },
    {
        "text": "USA",
        "label": ambiguous(
            "America/New_York",
            "America/Chicago",
            "America/Denver",
            "America/Phoenix",
            "America/Los_Angeles",
            "America/Anchorage",
            "America/Adak",
            "Pacific/Honolulu",
        ),
        "note": "Country rule: eight non-equivalent classes among the zone.tab zones.",
        "check": {"country": "US"},
    },
    {
        "text": "Russia",
        "label": ambiguous(
            "Europe/Kaliningrad",
            "Europe/Moscow",
            "Europe/Samara",
            "Asia/Yekaterinburg",
            "Asia/Omsk",
            "Asia/Novosibirsk",
            "Asia/Irkutsk",
            "Asia/Chita",
            "Asia/Vladivostok",
            "Asia/Sakhalin",
            "Asia/Kamchatka",
        ),
        "note": "Country rule: eleven offset classes from UTC+02 to UTC+12.",
        "check": {"country": "RU"},
    },
    {
        "text": "Spain",
        "label": ambiguous("Europe/Madrid", "Atlantic/Canary"),
        "note": "Country rule: the Canary Islands are one hour behind the mainland.",
        "check": {"country": "ES"},
    },
    {
        "text": "Kazakhstan",
        "label": resolved("Asia/Almaty"),
        "note": (
            "Country rule: Kazakhstan moved to a single UTC+05:00 time in 2024, so all its zones are "
            "equivalent."
        ),
        "check": {"country": "KZ"},
    },
    {
        "text": "the UK",
        "label": resolved("Europe/London"),
        "note": "Country rule: one zone.tab zone.",
        "check": {"country": "GB"},
    },
    # --- fixed offsets and the Etc/GMT sign convention ---
    {
        "text": "UTC-5",
        "label": resolved("Etc/GMT+5"),
        "note": "Fixed offset -05:00 taken literally; Etc/GMT+5 is five hours BEHIND UTC (inverted sign).",
        "check": {"offset": -300},
    },
    {
        "text": "GMT+2",
        "label": resolved("Etc/GMT-2"),
        "note": "Fixed offset +02:00 taken literally; Etc/GMT-2 is two hours AHEAD of UTC (inverted sign).",
        "check": {"offset": 120},
    },
    {
        "text": "Etc/GMT+3",
        "label": resolved("Etc/GMT+3"),
        "note": "An IANA key taken as written: Etc/GMT+3 is UTC-03:00, not UTC+03:00.",
    },
    {
        "text": "my laptop says Etc/GMT-5",
        "label": resolved("Etc/GMT-5"),
        "note": "An IANA key taken as written: Etc/GMT-5 is UTC+05:00.",
    },
    {
        "text": "GMT+3 (Moscow)",
        "label": resolved("Europe/Moscow"),
        "note": "Moscow is UTC+03:00 all year, so the offset and the city agree; Etc/GMT-3 is equivalent.",
        "check": {"offset": 180},
    },
    {
        "text": "UTC+5:45",
        "label": resolved("Asia/Kathmandu"),
        "note": "No Etc/GMT zone has a 45-minute offset; Asia/Kathmandu is constant UTC+05:45.",
        "check": {"offset": 345},
    },
    # --- nothing to resolve ---
    {"text": "on the moon", "label": UNKNOWN, "note": "No place or offset."},
    {"text": "I travel a lot", "label": UNKNOWN, "note": "No place or offset."},
    {"text": "wherever works", "label": UNKNOWN, "note": "No place or offset."},
    {
        "text": "not sure, somewhere in Europe",
        "label": UNKNOWN,
        "note": "A continent names no zone and no short candidate list.",
    },
    {
        "text": "just use my local time",
        "label": UNKNOWN,
        "note": "Refers to the prospect's own zone without naming it.",
    },
    {
        "text": "it depends on the week",
        "label": UNKNOWN,
        "note": "No place or offset.",
    },
    # --- typos ---
    {
        "text": "I'm in Chicgo",
        "label": resolved("America/Chicago"),
        "note": "Misspelling of Chicago; the referent is clear. City rule on 'Chicago'.",
        "check": {"city": "Chicago"},
    },
    {
        "text": "Torronto",
        "label": resolved("America/Toronto"),
        "note": "Misspelling of Toronto; the referent is clear. City rule on 'Toronto'.",
        "check": {"city": "Toronto"},
    },
    {
        "text": "we're in Pheonix",
        "label": resolved("America/Phoenix"),
        "note": "Misspelling of Phoenix; the referent is clear. City rule on 'Phoenix'.",
        "check": {"city": "Phoenix"},
    },
]
N_HARD = len(HARD_CASES)


def check_hard_case(tz: TzDb, gaz: Gazetteer, case: dict[str, Any]) -> None:
    label: Label = case["label"]
    check: dict[str, Any] | None = case.get("check")
    if check is None:
        return
    if "host" in check:
        expected = resolved(HOST_ZONE)
    elif "region" in check:
        zones = region_zones(gaz, check["region"], check["country"])
        if "also_country" in check:
            zones += country_zones(tz, gaz, check["also_country"])
        expected = collapse(tz, zones)
    elif "city" in check:
        computed, relevant = city_rule(
            tz, gaz, check["city"], admin1=check.get("admin1"), country=check.get("country")
        )
        zones = [m.timezone for m in relevant] + list(check.get("extra", []))
        expected = collapse(tz, zones) if check.get("extra") else computed
    elif "country" in check:
        expected = country_rule(tz, gaz, check["country"])
    elif "offset" in check:
        expected = offset_rule(tz, gaz, check["offset"])
    else:
        raise BuildError(f"unknown check {check!r}")
    if not same_label(tz, label, expected):
        raise BuildError(f"hard case {case['text']!r}: label {label} disagrees with rule result {expected}")


# --------------------------------------------------------------------------------------------------
# Templated zone names, abbreviations and offsets
# --------------------------------------------------------------------------------------------------

REGION_NOTE = "A named regional time; gold is the region's zone (any equivalent zone counts)."

ZONE_NAMES: list[tuple[str, Label, str]] = [
    ("Eastern time", resolved("America/New_York"), "North American zone name: the US Eastern zone."),
    (
        "Central time",
        resolved("America/Chicago"),
        "North American zone name: the US Central zone (dominant reading; Saskatchewan and Alberta, "
        "on Central Standard Time all year, are minority readings).",
    ),
    (
        "Mountain time",
        ambiguous("America/Denver", "America/Phoenix"),
        "Mountain time is used with summer time (Colorado, Utah, America/Denver) and without it "
        "(Arizona, America/Phoenix): not equivalent.",
    ),
    (
        "Pacific time",
        resolved("America/Los_Angeles"),
        "North American zone name: the US Pacific zone (British Columbia has kept UTC-07:00 all year, "
        "abbreviated MST, since March 2026).",
    ),
    ("Alaska time", resolved("America/Anchorage"), REGION_NOTE),
    ("Hawaii time", resolved("Pacific/Honolulu"), REGION_NOTE),
    (
        "Atlantic time",
        ambiguous("America/Halifax", "America/Puerto_Rico"),
        "Atlantic time is used with summer time (Atlantic Canada) and without it (Puerto Rico and much "
        "of the Caribbean): not equivalent.",
    ),
    ("Eastern European Time", resolved("Europe/Athens"), "The EU EET/EEST region; its zones are equivalent."),
    ("Western European Time", resolved("Europe/Lisbon"), "The WET/WEST region (Portugal mainland)."),
    ("India Standard Time", resolved("Asia/Kolkata"), "The full name is unambiguous."),
    ("Japan time", resolved("Asia/Tokyo"), REGION_NOTE),
    ("Moscow time", resolved("Europe/Moscow"), REGION_NOTE),
    ("Gulf Standard Time", resolved("Asia/Dubai"), REGION_NOTE),
    ("China Standard Time", resolved("Asia/Shanghai"), "The full name means Beijing time."),
    ("UK time", resolved("Europe/London"), REGION_NOTE),
    ("Brasilia time", resolved("America/Sao_Paulo"), "Official Brazilian time (UTC-03:00)."),
    ("Newfoundland time", resolved("America/St_Johns"), REGION_NOTE),
    (
        "Australian Eastern Time",
        ambiguous("Australia/Sydney", "Australia/Brisbane"),
        "Covers New South Wales/Victoria (summer time) and Queensland (none): not equivalent.",
    ),
    (
        "Australian Central Time",
        ambiguous("Australia/Adelaide", "Australia/Darwin"),
        "Covers South Australia (summer time) and the Northern Territory (none): not equivalent.",
    ),
]

ABBREVIATIONS: list[tuple[str, Label, str]] = [
    ("ET", resolved("America/New_York"), "US Eastern time."),
    ("PT", resolved("America/Los_Angeles"), "US Pacific time."),
    ("CT", resolved("America/Chicago"), "US Central time."),
    ("EDT", resolved("America/New_York"), "US Eastern daylight time names the Eastern region."),
    ("PDT", resolved("America/Los_Angeles"), "US Pacific daylight time names the Pacific region."),
    ("JST", resolved("Asia/Tokyo"), "Japan Standard Time."),
    ("KST", resolved("Asia/Seoul"), "Korea Standard Time."),
    ("SGT", resolved("Asia/Singapore"), "Singapore Time."),
    ("HKT", resolved("Asia/Hong_Kong"), "Hong Kong Time."),
    ("MSK", resolved("Europe/Moscow"), "Moscow Time."),
    ("SAST", resolved("Africa/Johannesburg"), "South African Standard Time."),
    ("PKT", resolved("Asia/Karachi"), "Pakistan Standard Time."),
    ("HST", resolved("Pacific/Honolulu"), "Hawaii Standard Time."),
    ("CEST", resolved("Europe/Berlin"), "Central European Summer Time names the EU CET region."),
    ("EEST", resolved("Europe/Athens"), "Eastern European Summer Time names the EU EET region."),
    ("WAT", resolved("Africa/Lagos"), "West Africa Time."),
    ("EAT", resolved("Africa/Nairobi"), "East Africa Time."),
    ("NZST", resolved("Pacific/Auckland"), "New Zealand Standard Time."),
    ("WIB", resolved("Asia/Jakarta"), "Western Indonesia Time."),
]
AMBIGUOUS_ABBREVIATIONS: list[tuple[str, Label, str]] = [
    (
        "AST",
        ambiguous("America/Halifax", "America/Puerto_Rico", "Asia/Riyadh"),
        "AST is Atlantic Standard Time (Atlantic Canada with summer time; Puerto Rico and the Caribbean "
        "without) and Arabia Standard Time.",
    ),
    ("PST", ambiguous("America/Los_Angeles", "Asia/Manila"), "PST is Pacific and Philippine Standard Time."),
]

TZ_TEMPLATES = (
    "{x} works for me",
    "I'm on {x}",
    "my timezone is {x}",
    "can we use {x}?",
    "{x}, please",
    "all my calls are in {x}",
    "please send times in {x}",
    "we run on {x}",
)

WHOLE_OFFSETS = tuple(
    h * 60 for h in (-10, -8, -7, -6, -4, -3, -2, -1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
)
FRACTIONAL_OFFSETS = (330, 210, 570, 390, 270, 525)  # minutes: +05:30 +03:30 +09:30 +06:30 +04:30 +08:45
N_WHOLE_OFFSETS = 9
N_FRACTIONAL_OFFSETS = 3
N_ZONE_NAMES = 12  # including every ambiguous name (Mountain, Atlantic, both Australian names)
N_ABBREVIATIONS = 11  # including both ambiguous abbreviations


def render_offset(rng: random.Random, minutes: int) -> str:
    prefix = rng.choice(("UTC", "GMT"))
    sign = "+" if minutes > 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    if mins:
        form = rng.choice(("{p}{s}{h}:{m:02d}", "{p}{s}{h:02d}:{m:02d}", "{p} {s}{h}:{m:02d}"))
    else:
        form = rng.choice(("{p}{s}{h}", "{p} {s}{h}", "{p}{s}{h:02d}:00", "{lp}{s}{h}"))
    return form.format(p=prefix, lp=prefix.lower(), s=sign, h=hours, m=mins)


def offset_note(label: Label, minutes: int) -> str:
    written = format_offset(minutes)
    if minutes % 60 == 0:
        return (
            f"Explicit fixed offset {written}, taken literally; whole-hour offsets map to {label.zone} "
            "(Etc/GMT keys invert the sign)."
        )
    return (
        f"Explicit fixed offset {written}; no Etc/GMT key has this offset, so gold is the zone.tab zone "
        f"with a constant {written} throughout 2026-2027 ({label.zone})."
    )


def template_items(rng: random.Random, tz: TzDb, gaz: Gazetteer) -> list[Item]:
    items: list[Item] = []
    for minutes in sorted(rng.sample(WHOLE_OFFSETS, N_WHOLE_OFFSETS)) + sorted(
        rng.sample(FRACTIONAL_OFFSETS, N_FRACTIONAL_OFFSETS)
    ):
        label = offset_rule(tz, gaz, minutes)
        text = rng.choice(TZ_TEMPLATES).format(x=render_offset(rng, minutes))
        items.append(Item(text, label, "template", offset_note(label, minutes)))
    names = [n for n in ZONE_NAMES if n[1].status == "ambiguous"]
    names += rng.sample([n for n in ZONE_NAMES if n[1].status == "resolved"], N_ZONE_NAMES - len(names))
    abbreviations = list(AMBIGUOUS_ABBREVIATIONS)
    abbreviations += rng.sample(ABBREVIATIONS, N_ABBREVIATIONS - len(abbreviations))
    for phrase, label, note in names + abbreviations:
        text = rng.choice(TZ_TEMPLATES).format(x=phrase)
        items.append(Item(text, label, "template", note))
    return items


# --------------------------------------------------------------------------------------------------
# GeoNames city samples
# --------------------------------------------------------------------------------------------------

CITY_TEMPLATES_BARE = ("{place} time works", "{place} here")
CITY_TEMPLATES = (
    "I'm in {place}",
    "we're based in {place}",
    "calling from {place}",
    "I'm in {place}, if that helps",
    "our office is in {place}",
    "I live in {place}",
    "we're in {place}",
    "I'm located in {place}",
)
QUALIFY_PROBABILITY = 0.3
CLEAN_NAME = re.compile(r"[A-Za-z][A-Za-z .'-]{3,}")


def country_display(tz: TzDb, code: str) -> str:
    overrides = {"GB": "UK", "US": "USA", "KR": "South Korea", "KP": "North Korea", "CD": "DR Congo"}
    if code in overrides:
        return overrides[code]
    name = re.sub(r"\s*\(.*?\)", "", tz.country_names[code])
    return name.replace(" & ", " and ")


class CitySampler:
    def __init__(self, tz: TzDb, gaz: Gazetteer, reserved_texts: Sequence[str]) -> None:
        self.tz = tz
        self.gaz = gaz
        self._reserved = [fold(t) for t in reserved_texts]
        self._region_zones: dict[str, set[str]] = {}
        for row in gaz.rows:
            if row.admin1_name:
                self._region_zones.setdefault(fold(row.admin1_name), set()).add(row.timezone)
        for code, zones in tz.country_zones.items():
            for name in {tz.country_names[code], country_display(tz, code)}:
                self._region_zones.setdefault(fold(name), set()).update(zones)

    def eligible(self, row: CityRow) -> bool:
        if not CLEAN_NAME.fullmatch(row.asciiname) or any(ch.isdigit() for ch in row.name):
            return False
        key = fold(row.name)
        # A region or country with the same name would add another reading; skip unless it cannot
        # change the zone.
        region = self._region_zones.get(key, set())
        if any(not self.tz.equivalent(z, row.timezone) for z in region):
            return False
        pattern = re.compile(rf"\b{re.escape(key)}\b")
        return not any(pattern.search(text) for text in self._reserved)

    def make_item(self, rng: random.Random, row: CityRow, stratum: str) -> Item | None:
        display = row.name
        unaccented = strip_accents(row.name)
        if unaccented != row.name and rng.random() < 0.5:
            display = unaccented
        matches = self.gaz.lookup(display)
        must_qualify = matches[0] != row
        qualifier: tuple[str, str] | None = None
        if must_qualify or rng.random() < QUALIFY_PROBABILITY:
            options: list[tuple[str, str]] = []
            if (
                row.admin1_name
                and CLEAN_NAME.fullmatch(row.admin1_name)
                and letters(row.admin1_name) != letters(display)
            ):
                options.append(("admin1", row.admin1_name))
            options.append(("country", row.country_code))
            rng.shuffle(options)
            for kind, value in options:
                subset = city_matches(
                    self.gaz,
                    display,
                    admin1=value if kind == "admin1" else None,
                    country=value if kind == "country" else None,
                )
                if subset[0] == row:
                    qualifier = (kind, value)
                    break
            if qualifier is None:
                return None
        if qualifier is None:
            label, relevant = city_rule(self.tz, self.gaz, display)
            place = display
            template = rng.choice(CITY_TEMPLATES + CITY_TEMPLATES_BARE)
            how = "Bare city name"
        else:
            kind, value = qualifier
            label, relevant = city_rule(
                self.tz,
                self.gaz,
                display,
                admin1=value if kind == "admin1" else None,
                country=value if kind == "country" else None,
            )
            shown = value if kind == "admin1" else country_display(self.tz, value)
            place = f"{display}, {shown}"
            template = rng.choice(CITY_TEMPLATES)
            how = f"Qualified by {'region' if kind == 'admin1' else 'country'} '{shown}'"
        if relevant[0] != row:
            return None
        origin = (
            f"GeoNames city {row.name} ({row.country_code}"
            f"{', ' + row.admin1_name if row.admin1_name else ''}, population {row.population:,}; "
            f"stratum {stratum})."
        )
        if label.status == "resolved":
            others = (
                ""
                if len(relevant) == 1
                else " (other same-name cities that pass both thresholds are in equivalent zones)"
            )
            reason = (
                f"{how}; no other same-name city with {CITY_THRESHOLDS_TEXT} lies in a non-equivalent "
                f"zone{others}."
            )
        else:
            listed = ", ".join(
                f"{m.country_code}{'/' + m.admin1_name if m.admin1_name else ''} {m.timezone}"
                for m in relevant
            )
            reason = (
                f"{how}; same-name cities with {CITY_THRESHOLDS_TEXT}, in non-equivalent zones: {listed}."
            )
        return Item(template.format(place=place), label, "city_sample", f"{origin} {reason}")

    def sample(self, rng: random.Random) -> list[Item]:
        items: list[Item] = []
        used: set[str] = set()
        for stratum, low, high, count in STRATA:
            pool = [
                r
                for r in self.gaz.rows
                if r.population >= low and (high is None or r.population < high) and self.eligible(r)
            ]
            picked = 0
            while picked < count:
                row = rng.choice(pool)
                key = fold(row.asciiname)
                if key in used:
                    continue
                used.add(key)
                item = self.make_item(rng, row, stratum)
                if item is not None:
                    items.append(item)
                    picked += 1
        return items


# --------------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------------


def validate(tz: TzDb, items: Sequence[Item]) -> None:
    if len(items) != N_TOTAL:
        raise BuildError(f"expected {N_TOTAL} items, got {len(items)}")
    texts = [fold(item.text) for item in items]
    if len(set(texts)) != len(texts):
        raise BuildError("duplicate phrase text")
    for item in items:
        label = item.label
        if label.status == "resolved":
            if label.zone is None or label.candidates:
                raise BuildError(f"bad resolved label: {item.text!r}")
            tz.zone(label.zone)
        elif label.status == "ambiguous":
            if label.zone is not None or len(label.candidates) < 2:
                raise BuildError(f"bad ambiguous label: {item.text!r}")
            for i, a in enumerate(label.candidates):
                tz.zone(a)
                for b in label.candidates[i + 1 :]:
                    if tz.equivalent(a, b):
                        raise BuildError(f"equivalent candidates {a} and {b} in {item.text!r}")
        elif label.zone is not None or label.candidates:
            raise BuildError(f"bad unknown label: {item.text!r}")


def stratified_split(strata: Sequence[tuple[str, str]]) -> list[str]:
    """Seeded split: shuffle each stratum, concatenate the strata in sorted order, every third is dev."""
    rng = random.Random(SEED)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(strata):
        groups.setdefault(key, []).append(index)
    order: list[int] = []
    for key in sorted(groups):
        members = groups[key]
        rng.shuffle(members)
        order.extend(members)
    splits = ["test"] * len(strata)
    for position, index in enumerate(order):
        if position % 3 == 0:
            splits[index] = "dev"
    return splits


def build(cities: Path) -> list[str]:
    """Return the dataset as JSON lines (without trailing newlines), ordered by id."""
    tz = TzDb()
    gaz = Gazetteer(cities)
    rng = random.Random(SEED)
    for case in HARD_CASES:
        check_hard_case(tz, gaz, case)
    hard = [Item(c["text"], c["label"], "hard_case", c["note"]) for c in HARD_CASES]
    templated = template_items(rng, tz, gaz)
    sampler = CitySampler(tz, gaz, [i.text for i in hard + templated])
    sampled = sampler.sample(rng)
    if (len(sampled), len(templated), len(hard)) != (N_CITY, N_TEMPLATE, N_TOTAL - N_CITY - N_TEMPLATE):
        raise BuildError(f"source counts {len(sampled)}/{len(templated)}/{len(hard)} do not match the plan")
    items = sampled + templated + hard
    validate(tz, items)
    rng.shuffle(items)
    ids = [f"tzp-{i:03d}" for i in range(1, len(items) + 1)]
    splits = stratified_split([(item.source, item.label.status) for item in items])
    if splits.count("dev") != N_DEV:
        raise BuildError(f"dev split has {splits.count('dev')} items, expected {N_DEV}")
    return [to_line(i, item, split) for i, item, split in zip(ids, items, splits, strict=True)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the labelled timezone-phrase dataset.")
    parser.add_argument("--cities", type=Path, default=DEFAULT_CITIES, help="GeoNames-derived cities CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSONL path")
    args = parser.parse_args(argv)
    lines = build(args.cities)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    print(f"wrote {len(lines)} items to {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

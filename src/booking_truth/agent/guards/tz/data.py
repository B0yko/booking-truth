"""Loaders for the ``tz_resolver`` guard's reference data.

Three sources, each read once and cached:

- ``datasets/tz_aliases.yaml``: curated abbreviations and names ("Eastern", "IST", "Central European
  time", ...), each resolved to one zone or ambiguous with candidate zones.
- the ``tzdata`` package's own ``zoneinfo/iso3166.tab`` (country code -> English name) and
  ``zoneinfo/zone.tab`` (country code -> its zones, in file order): the IANA tz database's own country
  and zone tables, not copied into this repository.
- ``datasets/cities_tz.csv``: the bundled GeoNames-derived city gazetteer (see ``datasets/README.md``).
  :func:`load_regions` derives the first-level regions (US states, Canadian provinces and territories,
  Australian states and territories) from this same file's ``admin1_name`` column; it is not a fourth
  source.

Nothing here resolves anything; :mod:`booking_truth.agent.guards.tz.resolver` does that with what these
loaders return.
"""

from __future__ import annotations

import csv
import importlib.resources
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from booking_truth.resources import data_path

#: Common short names ``iso3166.tab``'s own English name does not spell out, added to its index
#: alongside the official name. Not every reading of a parenthetical name: "Britain (UK)" and "Virgin
#: Islands (UK)" both carry "(UK)", which would otherwise make "UK" ambiguous between them, so
#: :func:`_name_readings` only reads the base name before a parenthetical, and this table adds "UK"
#: and "USA" as their own unambiguous keys.
_NAME_SYNONYMS: Mapping[str, str] = {
    "usa": "US",
    "u.s.a.": "US",
    "uk": "GB",
    "united kingdom": "GB",
}

#: Canadian provinces and territories' own first-level subdivision codes (source: ISO 3166-2:CA), keyed
#: by the ``admin1_name`` ``cities_tz.csv`` gives them. ``cities_tz.csv``'s own ``admin1_code`` column
#: for Canadian rows is GeoNames' internal numbering ("02", "08", ...), not this public code, so it is
#: not usable directly the way a US row's ``admin1_code`` (already its postal code) is.
_CA_SUBDIVISION_CODES: Mapping[str, str] = {
    "Alberta": "AB",
    "British Columbia": "BC",
    "Manitoba": "MB",
    "New Brunswick": "NB",
    "Newfoundland and Labrador": "NL",
    "Northwest Territories": "NT",
    "Nova Scotia": "NS",
    "Nunavut": "NU",
    "Ontario": "ON",
    "Prince Edward Island": "PE",
    "Quebec": "QC",
    "Saskatchewan": "SK",
    "Yukon": "YT",
}

#: Australian states and territories' own first-level subdivision codes (source: ISO 3166-2:AU), keyed
#: the same way as :data:`_CA_SUBDIVISION_CODES` and for the same reason (``cities_tz.csv``'s own
#: ``admin1_code`` for Australian rows is likewise GeoNames' internal numbering).
_AU_SUBDIVISION_CODES: Mapping[str, str] = {
    "Australian Capital Territory": "ACT",
    "New South Wales": "NSW",
    "Northern Territory": "NT",
    "Queensland": "QLD",
    "South Australia": "SA",
    "Tasmania": "TAS",
    "Victoria": "VIC",
    "Western Australia": "WA",
}


def region_code_for(country_code: str, admin1_name: str, admin1_code: str) -> str | None:
    """The public first-level subdivision code ``admin1_name`` denotes, for a US, Canadian or
    Australian row: a US state's own postal code (``admin1_code`` already is one, straight from
    ``cities_tz.csv``), or the ISO 3166-2 code :data:`_CA_SUBDIVISION_CODES` or
    :data:`_AU_SUBDIVISION_CODES` gives a Canadian province/territory or an Australian state/territory.
    ``None`` for any other country, or a blank ``admin1_name``/``admin1_code``: a row outside these
    three has no such code this guard recognises."""
    if country_code == "US":
        return admin1_code or None
    if country_code == "CA":
        return _CA_SUBDIVISION_CODES.get(admin1_name)
    if country_code == "AU":
        return _AU_SUBDIVISION_CODES.get(admin1_name)
    return None


@dataclass(frozen=True)
class AliasEntry:
    """One curated entry: a dominant reading, or two or more candidates a person could mean."""

    status: Literal["resolved", "ambiguous"]
    zone: str | None = None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class CityRow:
    name: str
    ascii_name: str
    country_code: str
    admin1_name: str
    population: int
    timezone: str
    #: ``cities_tz.csv``'s own ``admin1_code`` column: a US row's two-letter postal code, or, for any
    #: other country (including Canada and Australia, whose public subdivision code is a separate table
    #: above), GeoNames' own internal numbering. Defaulted so existing positional construction (tests,
    #: ``scripts/build_tz_phrases.py``'s own unrelated ``CityRow``) is unaffected.
    admin1_code: str = ""


@dataclass(frozen=True)
class RegionRow:
    """One first-level administrative region (a US state, a Canadian province or territory, an
    Australian state or territory) that has at least one GeoNames city of population 15,000 or more in
    ``cities_tz.csv``.

    ``zones`` are that region's distinct zones, one per qualifying city, ordered so that ``zones[0]``
    is always the zone of the region's single most populous qualifying city (the ``README.md``
    "Regions and countries" rule's resolved-case target)."""

    name: str
    country_code: str
    zones: tuple[str, ...]
    #: The region's own first-level subdivision code (a US state's postal code, or the ISO 3166-2 code
    #: for a Canadian province/territory or an Australian state/territory; see :func:`region_code_for`).
    #: Defaulted so a synthetic ``RegionRow`` built directly (tests) need not supply one.
    code: str = ""


#: The three countries whose first-level regions get their own resolution step (design-agent.md SS4 /
#: the product spec, item 4: region names belong before cities). Data-derived, not hand-picked: every
#: region here comes straight out of ``cities_tz.csv``'s own ``admin1_name`` column.
_REGION_COUNTRIES = frozenset({"US", "CA", "AU"})
#: The same population floor the region rule uses to decide which cities count towards a region's zones
#: (``datasets/README.md``, "Regions and countries").
_REGION_MIN_POPULATION = 15_000


def _zoneinfo_root() -> importlib.resources.abc.Traversable:
    return importlib.resources.files("tzdata").joinpath("zoneinfo")


def _read_tab(filename: str) -> Iterator[list[str]]:
    text = _zoneinfo_root().joinpath(filename).read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        yield line.split("\t")


@lru_cache(maxsize=1)
def load_aliases() -> dict[str, AliasEntry]:
    """``text`` (as written in the YAML) -> :class:`AliasEntry`."""
    path = data_path("datasets") / "tz_aliases.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of entries")
    entries: dict[str, AliasEntry] = {}
    for item in raw:
        text = str(item["text"])
        status = item["status"]
        if status == "resolved":
            entry = AliasEntry("resolved", zone=str(item["zone"]))
        elif status == "ambiguous":
            entry = AliasEntry("ambiguous", candidates=tuple(str(z) for z in item["candidates"]))
        else:
            raise ValueError(f"{path}: {text!r} has an unknown status {status!r}")
        if text in entries:
            raise ValueError(f"{path}: duplicate entry {text!r}")
        entries[text] = entry
    return entries


@lru_cache(maxsize=1)
def load_country_names() -> dict[str, str]:
    """Country name (as ``iso3166.tab`` gives it, plus a stripped-parenthetical reading of it and a
    small synonym table) -> ISO 3166-1 alpha-2 code. A name two countries share (case- and
    parenthetical-insensitively) maps to a sorted, comma-joined key so the caller can tell them apart
    from an unambiguous name; :func:`country_codes_for` splits it back out."""
    names: dict[str, list[str]] = {}
    for code, name in _read_tab("iso3166.tab"):
        for reading in _name_readings(name):
            names.setdefault(reading, []).append(code)
    for synonym, code in _NAME_SYNONYMS.items():
        names.setdefault(synonym, []).append(code)
    return {name: ",".join(sorted(set(codes))) for name, codes in names.items()}


def country_codes_for(key: str) -> tuple[str, ...]:
    """The one or more country codes a :func:`load_country_names` value packs together."""
    return tuple(key.split(","))


def _name_readings(name: str) -> Iterator[str]:
    """The name itself, and, for a name with a parenthetical qualifier ("Korea (North)", "Congo (Rep.)"),
    the base name too, so "Korea" and "Congo" correctly read as ambiguous between the countries that
    share it. The parenthetical's own content is not a reading: see :data:`_NAME_SYNONYMS`."""
    yield name.lower()
    if "(" in name and name.endswith(")"):
        base = name.partition("(")[0].strip().lower()
        if base:
            yield base


@lru_cache(maxsize=1)
def load_country_zones() -> dict[str, tuple[str, ...]]:
    """ISO 3166-1 alpha-2 code -> its zones, in ``zone.tab`` order."""
    zones: dict[str, list[str]] = {}
    for code, _coords, zone, *_comment in _read_tab("zone.tab"):
        zones.setdefault(code, []).append(zone)
    return {code: tuple(zone_list) for code, zone_list in zones.items()}


@lru_cache(maxsize=1)
def load_all_zones() -> tuple[str, ...]:
    """Every zone in ``zone.tab``, in file order, deduplicated."""
    seen: dict[str, None] = {}
    for _code, _coords, zone, *_comment in _read_tab("zone.tab"):
        seen.setdefault(zone, None)
    return tuple(seen)


@lru_cache(maxsize=1)
def load_cities() -> tuple[CityRow, ...]:
    """``datasets/cities_tz.csv``, as :class:`CityRow` tuples in file order."""
    path = data_path("datasets") / "cities_tz.csv"
    rows: list[CityRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                CityRow(
                    name=record["name"],
                    ascii_name=record["asciiname"],
                    country_code=record["country_code"],
                    admin1_name=record["admin1_name"],
                    population=int(record["population"]),
                    timezone=record["timezone"],
                    admin1_code=record["admin1_code"],
                )
            )
    return tuple(rows)


@lru_cache(maxsize=1)
def load_regions() -> tuple[RegionRow, ...]:
    """One :class:`RegionRow` per ``(country_code, admin1_name)`` of :data:`_REGION_COUNTRIES` that has
    at least one qualifying city, built straight from :func:`load_cities`: nothing here is hand-picked,
    so a region this data does not support (no qualifying city at all) simply is not a row."""
    by_region: dict[tuple[str, str], list[CityRow]] = {}
    for row in load_cities():
        if row.country_code not in _REGION_COUNTRIES or not row.admin1_name:
            continue
        if row.population < _REGION_MIN_POPULATION:
            continue
        by_region.setdefault((row.country_code, row.admin1_name), []).append(row)
    regions: list[RegionRow] = []
    for (country_code, name), rows in by_region.items():
        ranked = sorted(rows, key=lambda r: -r.population)
        zones: list[str] = []
        for row in ranked:
            if row.timezone not in zones:
                zones.append(row.timezone)
        code = region_code_for(country_code, name, ranked[0].admin1_code)
        if code is None:
            raise ValueError(f"no subdivision code for {country_code} region {name!r}")
        regions.append(RegionRow(name=name, country_code=country_code, zones=tuple(zones), code=code))
    return tuple(regions)


def is_valid_zone(name: str) -> bool:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return False
    return True


__all__ = [
    "AliasEntry",
    "CityRow",
    "RegionRow",
    "country_codes_for",
    "is_valid_zone",
    "load_aliases",
    "load_all_zones",
    "load_cities",
    "load_country_names",
    "load_country_zones",
    "load_regions",
    "region_code_for",
]

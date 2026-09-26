"""Loaders for the ``tz_resolver`` guard's reference data.

Three sources, each read once and cached:

- ``datasets/tz_aliases.yaml``: curated abbreviations and names ("Eastern", "IST", "Central European
  time", ...), each resolved to one zone or ambiguous with candidate zones.
- the ``tzdata`` package's own ``zoneinfo/iso3166.tab`` (country code -> English name) and
  ``zoneinfo/zone.tab`` (country code -> its zones, in file order): the IANA tz database's own country
  and zone tables, not copied into this repository.
- ``datasets/cities_tz.csv``: the bundled GeoNames-derived city gazetteer (see ``datasets/README.md``).

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
                )
            )
    return tuple(rows)


def is_valid_zone(name: str) -> bool:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return False
    return True


__all__ = [
    "AliasEntry",
    "CityRow",
    "country_codes_for",
    "is_valid_zone",
    "load_aliases",
    "load_all_zones",
    "load_cities",
    "load_country_names",
    "load_country_zones",
]

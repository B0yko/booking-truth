"""Loaders for the ``tz_resolver`` guard's reference data (``agent/guards/tz/data.py``)."""

from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from booking_truth.agent.guards.tz import data


def test_every_curated_alias_target_is_a_valid_zoneinfo_key() -> None:
    """The requirement the task names explicitly: every zone an alias entry names, resolved or a
    candidate, is a real IANA zone."""
    aliases = data.load_aliases()
    assert len(aliases) >= 20
    for text, entry in aliases.items():
        if entry.status == "resolved":
            assert entry.zone is not None, text
            assert data.is_valid_zone(entry.zone), (text, entry.zone)
            assert entry.candidates == ()
        else:
            assert entry.status == "ambiguous", text
            assert len(entry.candidates) >= 2, text
            for zone in entry.candidates:
                assert data.is_valid_zone(zone), (text, zone)


def test_aliases_cover_the_ambiguous_set_design_agent_names() -> None:
    aliases = data.load_aliases()
    ambiguous = {"IST", "CST", "BST", "AST", "Atlantic time", "PST", "Mountain time"}
    for text in ambiguous:
        assert aliases[text].status == "ambiguous", text
    assert set(aliases["IST"].candidates) == {"Asia/Kolkata", "Asia/Jerusalem", "Europe/Dublin"}
    resolved = {"EST", "ET", "EDT", "Eastern", "Central time", "CT", "Pacific time", "PT", "PDT"}
    for text in resolved:
        assert aliases[text].status == "resolved", text


def test_is_valid_zone_rejects_junk() -> None:
    assert data.is_valid_zone("Europe/Berlin")
    assert data.is_valid_zone("UTC")
    assert not data.is_valid_zone("Mars/Olympus")
    assert not data.is_valid_zone("")


def test_country_names_cover_official_and_bracket_free_readings() -> None:
    names = data.load_country_names()
    assert data.country_codes_for(names["india"]) == ("IN",)
    assert data.country_codes_for(names["united states"]) == ("US",)
    # "Britain (UK)" reads as "britain" and, via the synonym table, "uk"; the bracket's own content
    # ("uk") is not auto-derived from the name, so it does not also collide with "Virgin Islands (UK)".
    assert data.country_codes_for(names["britain (uk)"]) == ("GB",)
    assert data.country_codes_for(names["uk"]) == ("GB",)
    assert data.country_codes_for(names["usa"]) == ("US",)
    # A name two real countries share stays a genuine ambiguity, not a bug: Korea, Congo, Samoa.
    korea = set(data.country_codes_for(names["korea"]))
    assert korea == {"KP", "KR"}


def test_every_country_zone_is_a_valid_zoneinfo_key() -> None:
    zones = data.load_country_zones()
    assert len(zones) >= 200
    assert zones["IN"] == ("Asia/Kolkata",)
    assert len(zones["US"]) > 10
    for code, zone_list in zones.items():
        assert zone_list, code
        for zone in zone_list:
            assert data.is_valid_zone(zone), (code, zone)


def test_all_zones_covers_every_country_zone() -> None:
    all_zones = set(data.load_all_zones())
    for zone_list in data.load_country_zones().values():
        assert set(zone_list) <= all_zones
    for zone in all_zones:
        assert data.is_valid_zone(zone)


def test_cities_load_with_the_documented_columns() -> None:
    cities = data.load_cities()
    assert len(cities) > 30000
    berlin = [c for c in cities if c.name == "Berlin" and c.country_code == "DE"]
    assert berlin
    assert berlin[0].timezone == "Europe/Berlin"
    assert berlin[0].population > 1_000_000
    for row in cities[:200]:
        try:
            ZoneInfo(row.timezone)
        except ZoneInfoNotFoundError:
            pytest.fail(f"{row.name} ({row.country_code}) has an invalid zone {row.timezone!r}")

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


def test_regions_cover_every_us_state_and_the_district() -> None:
    regions = {r.name: r for r in data.load_regions() if r.country_code == "US"}
    assert len(regions) == 51  # 50 states + the District of Columbia
    assert regions["Arizona"].zones == ("America/Phoenix",)
    assert set(regions["Texas"].zones) == {"America/Chicago", "America/Denver"}


def test_regions_cover_canadian_provinces_and_australian_states() -> None:
    by_country: dict[str, set[str]] = {}
    for region in data.load_regions():
        by_country.setdefault(region.country_code, set()).add(region.name)
    assert "Ontario" in by_country["CA"]
    assert "Alberta" in by_country["CA"]
    assert "Queensland" in by_country["AU"]
    assert "Victoria" in by_country["AU"]
    # Nunavut has no GeoNames city of population >= 15,000, so it is correctly absent, not a bug.
    assert "Nunavut" not in by_country["CA"]


def test_regions_are_restricted_to_us_canada_and_australia() -> None:
    assert {r.country_code for r in data.load_regions()} == {"US", "CA", "AU"}


def test_a_regions_zones_are_ordered_with_its_most_populous_citys_zone_first() -> None:
    regions = {r.name: r for r in data.load_regions() if r.country_code == "US"}
    # Phoenix (America/Phoenix) is comfortably Arizona's most populous qualifying city.
    assert regions["Arizona"].zones[0] == "America/Phoenix"


def test_every_region_zone_is_a_valid_zoneinfo_key() -> None:
    for region in data.load_regions():
        assert region.zones, region.name
        for zone in region.zones:
            assert data.is_valid_zone(zone), (region.name, zone)


def test_region_code_for_a_us_row_is_its_own_admin1_code() -> None:
    assert data.region_code_for("US", "Illinois", "IL") == "IL"
    assert data.region_code_for("US", "Maine", "ME") == "ME"
    assert data.region_code_for("US", "Some State", "") is None  # a blank admin1_code is never a code


def test_region_code_for_canada_and_australia_uses_the_iso_3166_2_table_not_geonames_numbering() -> None:
    # cities_tz.csv's own admin1_code for these two countries is GeoNames' internal numbering ("02"),
    # not the public subdivision code ("BC"); region_code_for ignores it and uses the bundled table.
    assert data.region_code_for("CA", "British Columbia", "02") == "BC"
    assert data.region_code_for("CA", "Ontario", "08") == "ON"
    assert data.region_code_for("AU", "Victoria", "07") == "VIC"
    assert data.region_code_for("AU", "Western Australia", "08") == "WA"


def test_region_code_for_is_none_outside_us_canada_and_australia() -> None:
    assert data.region_code_for("GB", "Scotland", "") is None
    assert data.region_code_for("CA", "Nonexistent Territory", "99") is None


def test_every_region_carries_its_own_subdivision_code() -> None:
    for region in data.load_regions():
        assert region.code, (region.country_code, region.name)
    regions = {(r.country_code, r.name): r for r in data.load_regions()}
    assert regions[("US", "Illinois")].code == "IL"
    assert regions[("US", "District of Columbia")].code == "DC"
    assert regions[("CA", "British Columbia")].code == "BC"
    assert regions[("AU", "Victoria")].code == "VIC"


def test_cities_carry_their_admin1_code() -> None:
    cities = data.load_cities()
    portland_maine = next(c for c in cities if c.name == "Portland" and c.admin1_name == "Maine")
    assert portland_maine.admin1_code == "ME"
    victoria_bc = next(
        c
        for c in cities
        if c.name == "Victoria" and c.country_code == "CA" and c.admin1_name == "British Columbia"
    )
    assert victoria_bc.admin1_code == "02"  # GeoNames' own numbering, not the ISO code


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

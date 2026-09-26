"""``TimezoneResolver`` (``agent/guards/tz/resolver.py``): the six-step resolution order, the pre-scan
phrase detector, and local time <-> UTC conversion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, available_timezones

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from booking_truth.agent.guards.tz.data import AliasEntry, CityRow, RegionRow
from booking_truth.agent.guards.tz.resolver import (
    LocalInstant,
    Resolution,
    TimezoneResolver,
    get_resolver,
    local_instant,
)

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
HORIZON = 400
ZONES = [
    "America/New_York",
    "Europe/Berlin",
    "Europe/London",
    "Asia/Kolkata",
    "Asia/Kathmandu",
    "Australia/Sydney",
    "America/Chicago",
    "America/Los_Angeles",
    "Pacific/Auckland",
    "Asia/Tokyo",
    "UTC",
]


def resolve(text: str, *, resolver: TimezoneResolver | None = None, **kwargs: object) -> Resolution:
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("horizon_days", HORIZON)
    return (resolver or get_resolver()).resolve(text, **kwargs)  # type: ignore[arg-type]


# Step 1: explicit IANA names ---------------------------------------------------------------------------


def test_an_explicit_iana_name_is_taken_as_written() -> None:
    assert resolve("Europe/Berlin") == Resolution("resolved", "Europe/Berlin")
    assert resolve("I'm on Europe/Berlin time") == Resolution("resolved", "Europe/Berlin")
    assert resolve("UTC") == Resolution("resolved", "UTC")
    assert resolve("my laptop says Etc/GMT-5") == Resolution("resolved", "Etc/GMT-5")


def test_a_bare_legacy_zoneinfo_alias_is_not_treated_as_explicit() -> None:
    # "EST"/"CET" are real zoneinfo keys (a fixed offset, no DST) but not how anyone means to name a
    # zone in conversation; they are left for the curated alias step, which knows the region.
    assert resolve("EST") == Resolution("resolved", "America/New_York")
    assert resolve("CET") == Resolution("resolved", "Europe/Berlin")


# Step 2: fixed offsets, including the inverted Etc/GMT sign convention ---------------------------------


def test_the_etc_gmt_sign_is_inverted() -> None:
    assert resolve("GMT+2") == Resolution("resolved", "Etc/GMT-2")
    assert resolve("GMT-1") == Resolution("resolved", "Etc/GMT+1")
    assert resolve("UTC-5") == Resolution("resolved", "Etc/GMT+5")
    assert resolve("UTC+5") == Resolution("resolved", "Etc/GMT-5")
    assert resolve("can we use GMT-1?") == Resolution("resolved", "Etc/GMT+1")


def test_utc_plus_0_is_utc() -> None:
    assert resolve("UTC+0") == Resolution("resolved", "UTC")
    assert resolve("GMT-0") == Resolution("resolved", "UTC")


def test_a_fractional_offset_resolves_to_its_constant_zone() -> None:
    assert resolve("UTC+05:30") == Resolution("resolved", "Asia/Kolkata")
    assert resolve("UTC+5:45") == Resolution("resolved", "Asia/Kathmandu")


def test_an_out_of_range_offset_is_unknown_not_a_crash() -> None:
    assert resolve("UTC+20") == Resolution("unknown")


# Step 3: curated aliases -------------------------------------------------------------------------------


def test_ambiguous_abbreviations_return_their_candidates() -> None:
    ist = resolve("I'm on IST.")
    assert ist.status == "ambiguous"
    assert set(ist.candidates) == {"Asia/Kolkata", "Asia/Jerusalem", "Europe/Dublin"}
    assert resolve("we're in ist").status == "unknown"  # case-sensitive: not the German word "ist"


def test_a_dominant_reading_resolves_without_asking() -> None:
    assert resolve("I'm on Pacific time.") == Resolution("resolved", "America/Los_Angeles")
    assert resolve("Eastern time works").zone == "America/New_York"


def test_a_full_name_disambiguates_where_the_abbreviation_alone_would_not() -> None:
    assert resolve("India Standard Time works for me") == Resolution("resolved", "Asia/Kolkata")


# Step 4: countries --------------------------------------------------------------------------------------


def test_a_single_zone_country_resolves() -> None:
    assert resolve("I'm in India") == Resolution("resolved", "Asia/Kolkata")
    assert resolve("I'm in Nepal") == Resolution("resolved", "Asia/Kathmandu")


def test_a_country_whose_zones_all_agree_now_resolves() -> None:
    # Kazakhstan has kept a single UTC+05:00 time since 2024: every zone.tab zone is equivalent.
    assert resolve("we're in Kazakhstan") == Resolution("resolved", "Asia/Almaty")


def test_a_multi_offset_country_is_ambiguous_with_its_zones() -> None:
    china = resolve("China")
    assert china.status == "ambiguous"
    assert set(china.candidates) == {"Asia/Shanghai", "Asia/Urumqi"}
    usa = resolve("USA")
    assert usa.status == "ambiguous"
    assert len(usa.candidates) == 8  # 8 distinct US offset classes, not 29 raw zone.tab rows


def test_a_name_two_countries_share_is_ambiguous_across_them() -> None:
    # "Congo (Dem. Rep.)" and "Congo (Rep.)" both read as "Congo": pooling both countries' zones gives
    # Kinshasa and Lubumbashi (Dem. Rep., non-equivalent) plus Brazzaville (Rep., equivalent to
    # Kinshasa's UTC+1 and so folded into that same candidate), for two candidates, not three.
    congo = resolve("Congo")
    assert congo.status == "ambiguous"
    assert set(congo.candidates) == {"Africa/Kinshasa", "Africa/Lubumbashi"}


# Step 5: US, Canadian and Australian first-level regions -------------------------------------------------


def _synthetic_region_resolver(
    *regions: RegionRow, aliases: dict[str, AliasEntry] | None = None
) -> TimezoneResolver:
    return TimezoneResolver(
        aliases=aliases if aliases is not None else {},
        country_names={},
        country_zones={},
        regions=regions,
        cities=(),
        all_zones=("UTC",),
    )


def test_a_single_zone_region_resolves_without_asking() -> None:
    # Arizona: every GeoNames city of population >= 15,000 is on America/Phoenix (no Navajo Nation
    # settlement of that size observes summer time), so the bare state name resolves outright.
    assert resolve("I'm in Arizona") == Resolution("resolved", "America/Phoenix")


def test_a_canadian_province_resolves_by_its_common_forms() -> None:
    # Alberta has kept a single UTC-06:00/-07:00 (Mountain, with summer time) schedule across every one
    # of its qualifying cities, so both a lead-in and a bare "X time" phrase resolve it.
    assert resolve("we're in Alberta") == Resolution("resolved", "America/Edmonton")
    assert resolve("Alberta time works for me") == Resolution("resolved", "America/Edmonton")


def test_an_australian_region_without_dst_resolves() -> None:
    assert resolve("we're in Queensland") == Resolution("resolved", "Australia/Brisbane")


def test_an_australian_region_with_dst_resolves() -> None:
    assert resolve("I'm in Victoria") == Resolution("resolved", "Australia/Melbourne")


def test_a_split_us_state_is_ambiguous_with_its_zones() -> None:
    # Texas: El Paso and the rest of west Texas are on Mountain time, the rest on Central.
    texas = resolve("I'm in Texas")
    assert texas.status == "ambiguous"
    assert set(texas.candidates) == {"America/Chicago", "America/Denver"}


def test_a_split_province_is_ambiguous_with_its_zones() -> None:
    # Ontario: the great majority is Eastern time, but its north-west (Kenora district and around) is
    # Central, a genuinely different civil time, not a naming quirk.
    ontario = resolve("Ontario time")
    assert ontario.status == "ambiguous"
    assert set(ontario.candidates) == {"America/Toronto", "America/Winnipeg"}


def test_a_region_name_beats_a_small_same_named_town_elsewhere() -> None:
    # Without this step, these bare state names would have matched the only GeoNames *city* of that
    # exact name: a small town an ocean away, sharing nothing but the name, that the city step would
    # otherwise resolve to outright (no competing same-name city, so no ambiguity ever raised).
    assert resolve("I'm in Arizona") == Resolution("resolved", "America/Phoenix")  # not Honduras
    assert resolve("I'm in Montana") == Resolution("resolved", "America/Denver")  # not Bulgaria
    assert resolve("I'm in Colorado") == Resolution("resolved", "America/Denver")  # not Brazil


def test_a_region_whose_zones_are_all_equivalent_resolves_to_its_top_citys_zone() -> None:
    resolver = _synthetic_region_resolver(
        RegionRow("Twinstate", "ZZ", ("America/New_York", "America/Detroit"))
    )
    # "America/New_York" is listed first: by construction (data.py's load_regions), that is always the
    # region's single most populous qualifying city, and it is what a single-group resolution names.
    assert resolve("Twinstate time", resolver=resolver) == Resolution("resolved", "America/New_York")


def test_a_region_with_non_equivalent_zones_is_ambiguous_with_those_zones() -> None:
    resolver = _synthetic_region_resolver(
        RegionRow("Splitstate", "ZZ", ("America/Chicago", "America/Denver"))
    )
    ambiguous = resolve("I'm in Splitstate", resolver=resolver)
    assert ambiguous.status == "ambiguous"
    assert set(ambiguous.candidates) == {"America/Chicago", "America/Denver"}


def test_a_curated_alias_wins_over_a_same_named_region() -> None:
    resolver = _synthetic_region_resolver(
        RegionRow("Curatedstate", "ZZ", ("America/Denver",)),
        aliases={"Curatedstate": AliasEntry("resolved", zone="America/Chicago")},
    )
    assert resolve("Curatedstate", resolver=resolver) == Resolution("resolved", "America/Chicago")


def test_a_region_name_is_not_matched_below_the_stopword_floor() -> None:
    resolver = _synthetic_region_resolver(RegionRow("Us", "ZZ", ("America/Chicago",)))
    assert resolve("let's book a call, can you do Tuesday?", resolver=resolver) == Resolution("unknown")


# Step 6: cities -----------------------------------------------------------------------------------------


def _synthetic_resolver(*cities: CityRow) -> TimezoneResolver:
    return TimezoneResolver(
        aliases={}, country_names={}, country_zones={}, regions=(), cities=cities, all_zones=("UTC",)
    )


def test_the_most_populous_match_resolves_a_bare_city_name() -> None:
    assert resolve("I'm in Kathmandu") == Resolution("resolved", "Asia/Kathmandu")
    assert resolve("Berlin") == Resolution("resolved", "Europe/Berlin")


def test_new_york_city_also_answers_to_its_everyday_short_form() -> None:
    assert resolve("I'm in New York.") == Resolution("resolved", "America/New_York")


def test_city_ambiguity_sits_exactly_at_the_20_percent_threshold() -> None:
    top = CityRow("Example", "Example", "AA", "One", 100_000, "America/New_York")
    just_under = CityRow("Example", "Example", "BB", "Two", 19_999, "Europe/London")
    just_over = CityRow("Example", "Example", "BB", "Two", 20_000, "Europe/London")
    resolver_under = _synthetic_resolver(top, just_under)
    resolver_over = _synthetic_resolver(top, just_over)
    assert resolve("I'm in Example", resolver=resolver_under) == Resolution("resolved", "America/New_York")
    ambiguous = resolve("I'm in Example", resolver=resolver_over)
    assert ambiguous.status == "ambiguous"
    assert set(ambiguous.candidates) == {"America/New_York", "Europe/London"}


def test_an_equivalent_runner_up_does_not_create_a_false_ambiguity() -> None:
    top = CityRow("Sameville", "Sameville", "AA", "One", 100_000, "America/New_York")
    same_offset = CityRow("Sameville", "Sameville", "BB", "Two", 90_000, "America/Detroit")
    resolver = _synthetic_resolver(top, same_offset)
    assert resolve("Sameville", resolver=resolver) == Resolution("resolved", "America/New_York")


def test_a_named_region_disambiguates_a_qualified_city() -> None:
    assert resolve("Portland, Maine") == Resolution("resolved", "America/New_York")
    assert resolve("Portland, Oregon") == Resolution("resolved", "America/Los_Angeles")
    assert resolve("Portland") == Resolution("resolved", "America/Los_Angeles")


def test_common_english_words_never_fire_as_a_lone_city_or_country_match() -> None:
    # "Can" (Turkey), "Central" (Louisiana/Bahia), "best" (Netherlands): all real GeoNames places,
    # none of them what these sentences are about.
    assert resolve("Can I book a call next week?") == Resolution("unknown")
    assert resolve("What time works for you?") == Resolution("unknown")
    assert resolve("evenings are best, thanks") == Resolution("unknown")


def test_unknown_is_never_a_silent_fallback() -> None:
    assert resolve("wherever works") == Resolution("unknown")
    assert resolve("on the moon") == Resolution("unknown")
    assert resolve("just use my local time") == Resolution("unknown")


# The pre-scan phrase detector ---------------------------------------------------------------------------


def prescan(text: str) -> tuple[str, Resolution] | None:
    return get_resolver().prescan(text, now=NOW, horizon_days=HORIZON)


def test_prescan_finds_a_stated_zone_in_a_longer_message() -> None:
    found = prescan("Hi, I'm in Berlin and would like to book an intro call, ideally late afternoon.")
    assert found is not None
    phrase, resolution = found
    assert phrase == "Berlin"
    assert resolution == Resolution("resolved", "Europe/Berlin")


def test_prescan_recognises_we_are_on_calling_from_and_x_time() -> None:
    assert prescan("We're on CST.") is not None
    assert prescan("Sydney time.") == ("Sydney time", Resolution("resolved", "Australia/Sydney"))
    found = prescan("Hi, I'd like to book a call. calling from Kathmandu, thanks.")
    assert found is not None
    assert found[1].zone == "Asia/Kathmandu"


def test_prescan_finds_a_bare_abbreviation_offset_or_iana_name_with_no_lead_in() -> None:
    assert prescan("IST") is not None
    assert prescan("GMT+2") is not None
    assert prescan("Europe/Berlin") is not None


def test_prescan_returns_none_for_an_ordinary_message() -> None:
    assert prescan("Hi, can I book a call with you in the next few days? Mornings are best.") is None
    assert prescan("Can I book a call next week?") is None


def test_prescan_does_not_scan_unstructured_text_for_a_bare_city_name() -> None:
    # With no lead-in phrase, the bare-token tier checks only IANA names, offsets and aliases, so an
    # unrelated word that happens to be a small city elsewhere is never picked up mid-sentence.
    assert prescan("I'd like to book a call, need to reschedule, thanks a lot") is None


# Local time <-> UTC ---------------------------------------------------------------------------------------


def test_a_known_dst_fold_is_ambiguous_with_both_instants_in_order() -> None:
    # America/New_York, 1 November 2026: clocks fall back at 02:00, so 01:30 happens twice.
    result = local_instant("America/New_York", datetime(2026, 11, 1, 1, 30))
    assert result.status == "ambiguous"
    assert result.earlier is not None
    assert result.later is not None
    assert result.earlier < result.later
    assert result.later - result.earlier == timedelta(hours=1)


def test_a_known_dst_gap_is_rejected() -> None:
    # America/New_York, 8 March 2026: clocks spring forward at 02:00, so 02:30 never happens.
    result = local_instant("America/New_York", datetime(2026, 3, 8, 2, 30))
    assert result == LocalInstant("nonexistent")


def test_local_instant_rejects_an_aware_datetime() -> None:
    with pytest.raises(ValueError, match="naive"):
        local_instant("UTC", NOW)


@settings(max_examples=300, deadline=None)
@given(
    zone=st.sampled_from(sorted(available_timezones())),
    minutes=st.integers(0, HORIZON * 24 * 2).map(lambda n: n * 30),
)
def test_local_utc_round_trips_for_existent_unambiguous_times(zone: str, minutes: int) -> None:
    """local -> UTC -> local round trips for every IANA zone over the whole booking horizon, as long as
    the instant does not fall in that zone's own DST fold or gap (folds and gaps are tested separately,
    with :func:`_transitions_in`, below)."""
    start = NOW + timedelta(minutes=minutes)
    local = start.astimezone(ZoneInfo(zone)).replace(tzinfo=None)
    result = local_instant(zone, local)
    assume(result.status == "exists")
    assert result.utc == start


# Folds and gaps, tested separately: rather than filtering random instants for the rare (a few hours a
# year, for a zone that has any at all) minute that falls in one, every transition of a curated set of
# DST-observing zones over the booking horizon is found directly by :func:`_transitions_in`, and the
# local readings just before and after each one are used as the fold and gap cases.


def _offset_at(zone: str, instant: datetime) -> timedelta:
    return instant.astimezone(ZoneInfo(zone)).utcoffset() or timedelta(0)


def _transitions_in(zone: str, start: datetime, end: datetime) -> list[datetime]:
    """Every UTC instant in ``[start, end)`` where ``zone``'s offset changes, found to the minute by
    scanning daily and then bisecting the day it falls on."""
    found: list[datetime] = []
    day = timedelta(days=1)
    current, prev_offset = start, _offset_at(zone, start)
    while current < end:
        nxt = min(current + day, end)
        nxt_offset = _offset_at(zone, nxt)
        if nxt_offset != prev_offset:
            lo, hi = current, nxt
            while hi - lo > timedelta(minutes=1):
                mid = lo + (hi - lo) / 2
                if _offset_at(zone, mid) == prev_offset:
                    lo = mid
                else:
                    hi = mid
            found.append(hi)
        prev_offset, current = nxt_offset, nxt
    return found


def _fold_and_gap_cases(
    zones: list[str], start: datetime, end: datetime
) -> tuple[list[tuple[str, datetime]], list[tuple[str, datetime]]]:
    folds: list[tuple[str, datetime]] = []
    gaps: list[tuple[str, datetime]] = []
    for zone in zones:
        tz = ZoneInfo(zone)
        for at in _transitions_in(zone, start, end):
            before = (at - timedelta(minutes=1)).astimezone(tz).replace(tzinfo=None)
            after = (at + timedelta(minutes=1)).astimezone(tz).replace(tzinfo=None)
            if after < before:
                # Clocks went back: `after` is just inside the hour that is about to repeat, so a
                # reading well inside that same hour (not yet back up to `before`) occurs twice.
                folds.append((zone, after + timedelta(minutes=29)))
            elif after > before + timedelta(minutes=2):
                # Clocks jumped forward: the reading half an hour after `before` never occurs.
                gaps.append((zone, before + timedelta(minutes=30)))
    return folds, gaps


_DST_ZONES = [
    "America/New_York",
    "America/Chicago",
    "America/Los_Angeles",
    "Europe/Berlin",
    "Europe/London",
    "Australia/Sydney",
    "Pacific/Auckland",
    "America/Sao_Paulo",  # no DST since 2019: exercises a zone with zero transitions cleanly
]
_HORIZON_START = NOW
_HORIZON_END = NOW + timedelta(days=HORIZON)
FOLD_CASES, GAP_CASES = _fold_and_gap_cases(_DST_ZONES, _HORIZON_START, _HORIZON_END)


def test_the_booking_horizon_actually_contains_folds_and_gaps_to_test() -> None:
    """A sanity check on the fixture above, not the guard: if this ever finds none, the property tests
    below would pass vacuously."""
    assert len(FOLD_CASES) >= 4
    assert len(GAP_CASES) >= 4


@settings(max_examples=len(FOLD_CASES) or 1, deadline=None)
@given(case=st.sampled_from(FOLD_CASES or [("UTC", NOW.replace(tzinfo=None))]))
def test_every_fold_in_the_horizon_is_ambiguous_with_both_instants_in_order(
    case: tuple[str, datetime],
) -> None:
    zone, local = case
    result = local_instant(zone, local)
    assert result.status == "ambiguous", (zone, local, result)
    assert result.earlier is not None
    assert result.later is not None
    assert result.earlier < result.later
    tz = ZoneInfo(zone)
    assert result.earlier.astimezone(tz).replace(tzinfo=None) == local
    assert result.later.astimezone(tz).replace(tzinfo=None) == local


@settings(max_examples=len(GAP_CASES) or 1, deadline=None)
@given(case=st.sampled_from(GAP_CASES or [("UTC", NOW.replace(tzinfo=None))]))
def test_every_gap_in_the_horizon_is_rejected(case: tuple[str, datetime]) -> None:
    zone, local = case
    assert local_instant(zone, local) == LocalInstant("nonexistent")


def test_the_iana_offset_and_alias_steps_work_with_no_geo_data_at_all() -> None:
    """A resolver built with no cities, countries or regions (like the ones above, used for the
    deterministic city-ambiguity tests) still runs the first three steps: they do not depend on the
    country, region or city index, so a place name outside it is correctly ``unknown``, not a crash."""
    resolver = TimezoneResolver(
        aliases=None, country_names={}, country_zones={}, regions=(), cities=(), all_zones=("UTC",)
    )
    assert resolve("IST", resolver=resolver).status == "ambiguous"
    assert resolve("Europe/Berlin", resolver=resolver) == Resolution("resolved", "Europe/Berlin")
    assert resolve("I'm in Kathmandu", resolver=resolver) == Resolution("unknown")


def test_alias_matching_is_case_sensitive_for_all_capitals_and_not_for_names() -> None:
    assert resolve("ist").status == "unknown"
    assert resolve("EASTERN TIME") == Resolution("resolved", "America/New_York")
    assert resolve("eastern time") == Resolution("resolved", "America/New_York")

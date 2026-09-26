"""Guard configuration helpers and the naive baseline's choices."""

from __future__ import annotations

import pytest

from booking_truth.agent.guards import GUARD_NAMES, GuardConfig, all_except, guards_string
from booking_truth.agent.naive import NAIVE_ZONES, error_text, naive_zone, reply_says_booked, resolve_naive
from booking_truth.config import ConfigError


def test_all_and_off() -> None:
    assert GuardConfig.all().enabled == frozenset(GUARD_NAMES)
    assert GuardConfig.all().label == "all"
    assert GuardConfig.off().label == "off"
    assert GuardConfig.off().naive
    assert GuardConfig.parse("claim_ledger,dedupe").label == "claim_ledger,dedupe"


def test_rendered_confirmation_needs_the_claim_ledger() -> None:
    with pytest.raises(ConfigError, match="requires"):
        GuardConfig(frozenset({"rendered_confirmation"}))
    with pytest.raises(ConfigError, match="requires"):
        GuardConfig.parse("rendered_confirmation")


def test_unknown_guards_are_refused() -> None:
    with pytest.raises(ConfigError, match="unknown"):
        GuardConfig(frozenset({"magic"}))
    with pytest.raises(KeyError):
        GuardConfig.all().on("magic")


@pytest.mark.parametrize("name", GUARD_NAMES)
def test_all_except_each_guard_is_a_valid_configuration(name: str) -> None:
    enabled = all_except(name)
    assert name not in enabled
    config = GuardConfig.parse(guards_string(enabled))
    assert config.enabled == enabled


def test_switching_off_the_claim_ledger_also_drops_rendered_confirmation() -> None:
    assert all_except("claim_ledger") == frozenset(GUARD_NAMES) - {"claim_ledger", "rendered_confirmation"}
    assert all_except("rendered_confirmation") == frozenset(GUARD_NAMES) - {"rendered_confirmation"}


def test_the_naive_zone_map_is_small_and_valid() -> None:
    from zoneinfo import ZoneInfo

    assert len(NAIVE_ZONES) < 50
    for zone in NAIVE_ZONES.values():
        ZoneInfo(zone)


@pytest.mark.parametrize(
    ("text", "zone"),
    [
        ("I'm in New York.", "America/New_York"),
        ("Central European time, I'm in Berlin.", "Europe/Berlin"),
        ("We're on CST.", "America/Chicago"),
        ("I'm on IST.", "Asia/Kolkata"),
        ("I'm on Pacific time.", "America/Los_Angeles"),
        ("I'm in Pune, evenings", None),
        ("I'm in Kathmandu.", None),
    ],
)
def test_the_naive_map_knows_some_labels(text: str, zone: str | None) -> None:
    assert naive_zone(text) == zone


def test_unknown_places_silently_become_the_host_zone() -> None:
    assert resolve_naive("I'm in Kathmandu.", "America/New_York") == "America/New_York"


def test_the_prose_crm_rule_and_error_text() -> None:
    assert reply_says_booked("You're booked for Tuesday.")
    assert reply_says_booked("BOOKED!")
    assert not reply_says_booked("I haven't made a booking yet.")
    assert error_text("HTTP 500: boom") == "Error: calendar returned HTTP 500: boom"
    assert error_text("  ") == "Error: calendar returned unknown error"

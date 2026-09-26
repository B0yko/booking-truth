"""Building the configured calendar adapter from settings."""

from __future__ import annotations

from typing import Any

import pytest
from calendar_env import EVENT_TYPE_ID, MON_0900, MON_1700, TOKEN, CalEnv

from booking_truth.calendars import CalcomAdapter, Slots, build_calendar, calendar_options
from booking_truth.config import GUARD_NAMES, ConfigError, Settings


def settings(**fields: Any) -> Settings:
    return Settings(_env_file=None, **fields)  # type: ignore[call-arg]


async def test_local_sandbox_defaults_reach_the_sandbox(env: CalEnv) -> None:
    adapter = build_calendar(settings(calcom_base_url=env.url, sandbox_token=TOKEN))
    assert isinstance(adapter, CalcomAdapter)
    try:
        assert (adapter.event_type_id, adapter.event_key) == (EVENT_TYPE_ID, str(EVENT_TYPE_ID))
        assert adapter.host_zone == "America/New_York"
        assert (adapter.lenient, adapter.post_retries_on_timeout) == (False, 0)
        # The sandbox token stands in for a missing API key.
        assert isinstance(await adapter.find_slots(MON_0900, MON_1700), Slots)
    finally:
        await adapter.aclose()


async def test_explicit_values_and_naive_options_are_passed_through() -> None:
    adapter = build_calendar(
        settings(
            calcom_base_url="https://api.cal.com/",
            calcom_api_key="cal_test_key",
            calcom_event_type_id=2908889,
            host_timezone="Europe/Rome",
            slot_minutes=45,
        ),
        lenient=True,
        post_retries_on_timeout=2,
    )
    assert isinstance(adapter, CalcomAdapter)
    try:
        assert adapter.base_url == "https://api.cal.com"
        assert (adapter.event_type_id, adapter.event_key, adapter.host_zone) == (
            2908889,
            "2908889",
            "Europe/Rome",
        )
        assert (adapter.lenient, adapter.post_retries_on_timeout, adapter.slot_minutes) == (True, 2, 45)
        assert "cal_test_key" not in repr(adapter)
    finally:
        await adapter.aclose()


def test_real_cal_com_needs_a_key_and_an_event_type() -> None:
    with pytest.raises(ConfigError, match="BT_CALCOM_API_KEY"):
        build_calendar(settings(calcom_base_url="https://api.cal.com", calcom_event_type_id=1))
    with pytest.raises(ConfigError, match="BT_CALCOM_EVENT_TYPE_ID"):
        build_calendar(settings(calcom_base_url="https://api.cal.com", calcom_api_key="cal_test_key"))


def test_google_is_refused_with_a_clear_message() -> None:
    with pytest.raises(ConfigError, match="BT_CALENDAR=google is not supported"):
        build_calendar(settings(calendar="google"))


def test_calendar_options_follow_the_guards() -> None:
    guarded = calendar_options(frozenset(GUARD_NAMES))
    assert (guarded.lenient, guarded.post_retries_on_timeout) == (False, 0)
    naive = calendar_options(frozenset())
    assert (naive.lenient, naive.post_retries_on_timeout) == (True, 2)
    only_fail_closed = calendar_options(frozenset({"fail_closed"}))
    assert (only_fail_closed.lenient, only_fail_closed.post_retries_on_timeout) == (False, 2)
    only_idempotency = calendar_options(frozenset({"idempotency"}))
    assert (only_idempotency.lenient, only_idempotency.post_retries_on_timeout) == (True, 0)

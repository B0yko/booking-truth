"""Build the configured calendar adapter from settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from booking_truth.calendars.base import CalendarAdapter
from booking_truth.calendars.calcom import CalcomAdapter
from booking_truth.calendars.google import GoogleAdapter, load_service_account
from booking_truth.calendars.slotcalc import hours_from_settings
from booking_truth.config import ConfigError, Settings, is_local_url
from booking_truth.sandbox.state import SeedConfig
from booking_truth.timeutil import Clock

#: POST re-sends after a timeout in the naive baseline (a plain HTTP client retry).
NAIVE_POST_RETRIES_ON_TIMEOUT: Final = 2
#: The event type the sandbox seeds by default, used when a local sandbox is configured without one.
SANDBOX_EVENT_TYPE_ID: Final = SeedConfig().event_type_id


@dataclass(frozen=True)
class CalendarOptions:
    """How the adapter treats vendor failures, derived from the enabled guards."""

    #: Fail-open parsing and raw error texts, the naive baseline (``fail_closed`` off).
    lenient: bool
    #: POST re-sends after a timeout: none when ``idempotency`` is on (verify before retry instead).
    post_retries_on_timeout: int


def calendar_options(guards: frozenset[str]) -> CalendarOptions:
    return CalendarOptions(
        lenient="fail_closed" not in guards,
        post_retries_on_timeout=0 if "idempotency" in guards else NAIVE_POST_RETRIES_ON_TIMEOUT,
    )


def build_calendar(
    settings: Settings,
    *,
    lenient: bool = False,
    post_retries_on_timeout: int = 0,
    clock: Clock | None = None,
) -> CalendarAdapter:
    """The adapter for ``BT_CALENDAR``. Raises ``ConfigError`` with an operator-readable message.

    With a local sandbox as the Cal.com base URL, a missing API key falls back to ``BT_SANDBOX_TOKEN`` and a
    missing event type id to the sandbox's seeded event type. ``clock`` is used by adapters that compute
    availability client-side (Google), and also times its service-account JWT assertion.
    """
    if settings.calendar == "google":
        return _build_google(
            settings, lenient=lenient, post_retries_on_timeout=post_retries_on_timeout, clock=clock
        )
    local = is_local_url(settings.calcom_base_url)
    if settings.calcom_api_key is not None and settings.calcom_api_key.get_secret_value().strip():
        api_key = settings.calcom_api_key.get_secret_value().strip()
    elif local:
        api_key = settings.sandbox_token.get_secret_value()
    else:
        raise ConfigError("BT_CALCOM_API_KEY is required for a real Cal.com account")
    event_type_id = settings.calcom_event_type_id
    if event_type_id is None:
        if not local:
            raise ConfigError("BT_CALCOM_EVENT_TYPE_ID is required for a real Cal.com account")
        event_type_id = SANDBOX_EVENT_TYPE_ID
    return CalcomAdapter(
        settings.calcom_base_url,
        api_key,
        event_type_id,
        str(event_type_id),
        settings.host_timezone,
        lenient=lenient,
        post_retries_on_timeout=post_retries_on_timeout,
        slot_minutes=settings.slot_minutes,
    )


def _build_google(
    settings: Settings, *, lenient: bool, post_retries_on_timeout: int, clock: Clock | None
) -> GoogleAdapter:
    """``BT_CALENDAR=google``: a service account on a calendar shared with it. There is no sandbox-token
    fallback for the credential itself (unlike Cal.com's API key) — the adapter always exchanges the
    service-account key for an access token, whether that exchange lands on the sandbox or on Google."""
    if settings.google_service_account_file is None:
        raise ConfigError("BT_GOOGLE_SERVICE_ACCOUNT_FILE is required when BT_CALENDAR=google")
    try:
        service_account = load_service_account(settings.google_service_account_file)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return GoogleAdapter(
        settings.google_base_url,
        settings.google_token_uri,
        settings.google_calendar_id,
        settings.event_key,
        hours_from_settings(settings),
        service_account,
        lenient=lenient,
        post_retries_on_timeout=post_retries_on_timeout,
        clock=clock,
    )

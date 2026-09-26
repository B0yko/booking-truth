"""Shared helpers for the Google Calendar adapter tests: a throwaway service-account key and a
:class:`GoogleAdapter` aimed at the same real sandbox server the Cal.com tests use.

The sandbox mounts every mirrored vendor API on one app (see ``booking_truth.sandbox.app``), so the
``env`` fixture's server already answers ``/token`` and ``/calendar/v3/...`` — only the adapter changes.
"""

from __future__ import annotations

from datetime import time
from functools import lru_cache
from typing import Any

from calendar_env import CalEnv
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from booking_truth.calendars.google import GoogleAdapter, ServiceAccountKey
from booking_truth.calendars.slotcalc import Hours

CALENDAR_ID = "primary"
EVENT_KEY = "google-default"
SERVICE_ACCOUNT_EMAIL = "booking-agent@example.com"
HOURS = Hours(
    zone="America/New_York",
    start=time(9, 0),
    end=time(17, 0),
    days=(1, 2, 3, 4, 5),
    slot_minutes=30,
    min_notice_minutes=120,
    horizon_days=400,
)


@lru_cache(maxsize=1)
def throwaway_service_account() -> ServiceAccountKey:
    """A fresh RSA key generated at test time; never written to disk, never committed."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    return ServiceAccountKey(client_email=SERVICE_ACCOUNT_EMAIL, private_key=pem, private_key_id="test-key-1")


def google_adapter(env: CalEnv, **overrides: Any) -> GoogleAdapter:
    """A ``GoogleAdapter`` pointed at ``env``'s sandbox server, with a fresh throwaway key.

    ``overrides`` may replace any of ``calendar_id``, ``event_key``, ``hours``, ``service_account``,
    ``token_uri``, ``base_url`` and ``clock``, plus any of ``GoogleAdapter``'s own keyword-only options
    (``lenient``, ``post_retries_on_timeout``, ``timeout``, ``client``, ``event_title``).
    """
    calendar_id = overrides.pop("calendar_id", CALENDAR_ID)
    event_key = overrides.pop("event_key", EVENT_KEY)
    hours = overrides.pop("hours", HOURS)
    service_account = overrides.pop("service_account", throwaway_service_account())
    token_uri = overrides.pop("token_uri", f"{env.url}/token")
    base_url = overrides.pop("base_url", env.url)
    overrides.setdefault("clock", env.state.clock)
    return GoogleAdapter(base_url, token_uri, calendar_id, event_key, hours, service_account, **overrides)

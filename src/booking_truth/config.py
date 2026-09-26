"""Runtime configuration. Every environment variable carries the ``BT_`` prefix."""

from __future__ import annotations

import os
import secrets
from datetime import time
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

GUARD_NAMES: tuple[str, ...] = (
    "claim_ledger",
    "rendered_confirmation",
    "fail_closed",
    "slot_ids",
    "tz_resolver",
    "idempotency",
    "dedupe",
    "lead_lock",
    "pinned_version",
    "crm_outbox",
)

_WEEKDAYS = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}


class ConfigError(ValueError):
    """Raised when the configuration is invalid; the message is shown to the operator."""


def parse_guards(value: str) -> frozenset[str]:
    """Parse ``all``, ``off`` or a comma list of guard names, and check dependencies."""
    raw = value.strip().lower()
    if raw in ("", "all"):
        enabled = frozenset(GUARD_NAMES)
    elif raw in ("off", "none"):
        enabled = frozenset()
    else:
        names = [part.strip() for part in raw.split(",") if part.strip()]
        unknown = sorted(set(names) - set(GUARD_NAMES))
        if unknown:
            raise ConfigError(f"unknown guard(s): {', '.join(unknown)}; valid: {', '.join(GUARD_NAMES)}")
        enabled = frozenset(names)
    if "rendered_confirmation" in enabled and "claim_ledger" not in enabled:
        raise ConfigError("guard 'rendered_confirmation' requires 'claim_ledger'")
    return enabled


def guards_label(enabled: frozenset[str]) -> str:
    if enabled == frozenset(GUARD_NAMES):
        return "all"
    if not enabled:
        return "off"
    return ",".join(name for name in GUARD_NAMES if name in enabled)


def is_floating_model_id(model: str) -> bool:
    return model.endswith("latest")


def parse_work_hours(value: str) -> tuple[time, time]:
    try:
        start_s, end_s = (part.strip() for part in value.split("-", 1))
        start, end = time.fromisoformat(start_s), time.fromisoformat(end_s)
    except ValueError as exc:
        raise ConfigError(f"work hours must look like 09:00-17:00, got {value!r}") from exc
    if start >= end:
        raise ConfigError(f"work hours start must be before end, got {value!r}")
    return start, end


def parse_work_days(value: str) -> tuple[int, ...]:
    """Parse ``1-5``, ``mon-fri`` or a comma list (``mon,wed,fri`` / ``1,3,5``) into ISO weekdays."""

    def one(token: str) -> int:
        token = token.strip().lower()[:3]
        if token.isdigit() and 1 <= int(token) <= 7:
            return int(token)
        if token in _WEEKDAYS:
            return _WEEKDAYS[token]
        raise ConfigError(f"invalid weekday {token!r} in work days {value!r}")

    days: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            lo, hi = (one(p) for p in part.split("-", 1))
            if lo > hi:
                raise ConfigError(f"invalid weekday range {part!r}")
            days.update(range(lo, hi + 1))
        elif part.strip():
            days.add(one(part))
    if not days:
        raise ConfigError("work days must not be empty")
    return tuple(sorted(days))


def running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def is_local_url(url: str) -> bool:
    """True for loopback hosts and single-label hosts such as a compose service name."""
    host = urlparse(url).hostname or ""
    if host in ("localhost",) or host.endswith(".localhost"):
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return bool(host) and "." not in host


class Settings(BaseSettings):
    """All ``BT_*`` settings. Empty strings count as unset."""

    model_config = SettingsConfigDict(
        env_prefix="BT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # LLM
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "qwen/qwen3-235b-a22b-2507"
    persona_model: str | None = None
    extractor_model: str | None = None
    llm_provider: str | None = None

    # Calendar
    calendar: Literal["calcom", "google"] = "calcom"
    calcom_base_url: str = "https://api.cal.com"
    calcom_api_key: SecretStr | None = None
    calcom_event_type_id: int | None = None
    google_base_url: str = "https://www.googleapis.com"
    google_token_uri: str = "https://oauth2.googleapis.com/token"  # noqa: S105
    google_calendar_id: str = "primary"
    google_service_account_file: Path | None = None
    event_key: str = "default"
    host_timezone: str = "America/New_York"
    work_hours: str = "09:00-17:00"
    work_days: str = "mon-fri"
    slot_minutes: int = Field(default=30, ge=5, le=480)
    min_notice_minutes: int = Field(default=120, ge=0)
    horizon_days: int = Field(default=60, ge=1, le=730)

    # CRM
    crm: Literal["none", "hubspot"] = "none"
    hubspot_token: SecretStr | None = None
    hubspot_base_url: str = "https://api.hubapi.com"

    # Agent
    guards: str = "all"
    db_path: Path = Path("~/.local/state/booking-truth/agent.db")
    api_key: SecretStr | None = None
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    expose_traces: bool = False
    session_secret: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    trust_proxy: bool = False
    slot_ttl_seconds: int = Field(default=900, ge=30)
    max_input_chars: int = Field(default=2000, ge=1)
    max_turns_per_session: int = Field(default=40, ge=1)
    handoff_webhook_url: str | None = None

    # Sandbox and budget
    sandbox_token: SecretStr = SecretStr("sandbox")
    budget_usd: float | None = Field(default=None, ge=0)
    ledger_dir: Path = Path("~/.local/state/booking-truth/ledger")
    pricing_path: Path | None = None

    @field_validator("host_timezone")
    @classmethod
    def _valid_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA time zone {value!r}") from exc
        return value

    @model_validator(mode="after")
    def _fallbacks(self) -> Settings:
        if self.llm_api_key is None and not running_in_docker():
            fallback = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if fallback:
                self.llm_api_key = SecretStr(fallback)
        return self

    # Derived values -------------------------------------------------------------------------

    @property
    def offline(self) -> bool:
        return self.llm_api_key is None or not self.llm_api_key.get_secret_value().strip()

    @property
    def enabled_guards(self) -> frozenset[str]:
        return parse_guards(self.guards)

    @property
    def persona_model_id(self) -> str:
        return self.persona_model or self.llm_model

    @property
    def extractor_model_id(self) -> str:
        return self.extractor_model or self.llm_model

    @property
    def work_hours_range(self) -> tuple[time, time]:
        return parse_work_hours(self.work_hours)

    @property
    def work_days_set(self) -> tuple[int, ...]:
        return parse_work_days(self.work_days)

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def calendar_base_url(self) -> str:
        return self.calcom_base_url if self.calendar == "calcom" else self.google_base_url

    @property
    def resolved_db_path(self) -> Path:
        return self.db_path.expanduser()

    @property
    def resolved_ledger_dir(self) -> Path:
        return self.ledger_dir.expanduser()

    def validate_for_agent(self) -> None:
        """Checks that only matter when the agent starts; raises ConfigError with a readable message."""
        enabled = self.enabled_guards
        parse_work_hours(self.work_hours)
        parse_work_days(self.work_days)
        if "pinned_version" in enabled and is_floating_model_id(self.llm_model):
            raise ConfigError(
                f"BT_LLM_MODEL={self.llm_model!r} is a floating alias; pin an exact model id "
                "(or disable the pinned_version guard)"
            )
        if self.api_key is None and not is_local_url(self.calendar_base_url):
            raise ConfigError("BT_API_KEY is required unless the calendar base URL points at a local sandbox")
        if (
            self.calendar == "calcom"
            and self.calcom_event_type_id is None
            and not is_local_url(self.calcom_base_url)
        ):
            raise ConfigError("BT_CALCOM_EVENT_TYPE_ID is required for a real Cal.com account")
        if self.calendar == "google" and self.google_service_account_file is None:
            raise ConfigError("BT_GOOGLE_SERVICE_ACCOUNT_FILE is required when BT_CALENDAR=google")
        if self.crm == "hubspot" and self.hubspot_token is None:
            raise ConfigError("BT_HUBSPOT_TOKEN is required when BT_CRM=hubspot")


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]

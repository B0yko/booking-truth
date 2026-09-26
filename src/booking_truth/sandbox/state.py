"""In-memory state of one sandbox: seed, host calendar, CRM objects, faults and the request log."""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from booking_truth.config import parse_work_hours
from booking_truth.sandbox.availability import Hours, free_slot_starts, overlaps
from booking_truth.sandbox.faults import FaultEngine
from booking_truth.timeutil import Clock, SystemClock, iso_z, parse_iso

THIRD_PARTY_EMAIL = "third-party@example.com"


def check_zone(name: str) -> None:
    """Raise ``ValueError`` (which pydantic reports as a validation error) for an unknown IANA zone."""
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise ValueError(f"unknown IANA time zone {name!r}") from exc


class ExistingBooking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime | None = None
    title: str = "Busy"

    @field_validator("start", "end")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must carry an offset or Z")
        return value

    @model_validator(mode="after")
    def _ordered(self) -> ExistingBooking:
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class SeedConfig(BaseModel):
    """Seed of the host calendar. ``POST /_control/seed`` merges a partial seed over the defaults."""

    model_config = ConfigDict(extra="forbid")

    host_timezone: str = "America/New_York"
    work_hours: str = "09:00-17:00"
    work_days: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    event_length_minutes: int = Field(default=30, ge=5, le=480)
    min_notice_minutes: int = Field(default=120, ge=0)
    horizon_days: int = Field(default=400, ge=1, le=800)
    existing_bookings: list[ExistingBooking] = Field(default_factory=list)
    event_type_id: int = 1001
    event_type_slug: str = "intro-call"
    event_title: str = "Intro call"
    google_calendar_id: str = Field(default="primary", min_length=1)
    #: Simulates domain-wide delegation: the service account may add attendees and send ``sub``.
    google_sa_can_invite: bool = False
    host_id: int = 1
    host_name: str = "Sandbox Host"
    host_email: str = "host@example.com"
    host_username: str = "sandbox-host"

    @field_validator("host_timezone")
    @classmethod
    def _zone(cls, value: str) -> str:
        check_zone(value)
        return value

    @field_validator("work_hours")
    @classmethod
    def _hours(cls, value: str) -> str:
        parse_work_hours(value)
        return value

    @field_validator("work_days")
    @classmethod
    def _days(cls, value: list[int]) -> list[int]:
        if not value or any(d < 1 or d > 7 for d in value):
            raise ValueError("work_days are ISO weekdays 1..7")
        return sorted(set(value))

    def hours(self) -> Hours:
        start, end = parse_work_hours(self.work_hours)
        return Hours(
            zone=self.host_timezone,
            start=start,
            end=end,
            days=tuple(self.work_days),
            slot_minutes=self.event_length_minutes,
            min_notice_minutes=self.min_notice_minutes,
            horizon_days=self.horizon_days,
        )

    @property
    def work_start(self) -> time:
        return parse_work_hours(self.work_hours)[0]


@dataclass
class LogEntry:
    seq: int
    ts: str
    method: str
    path: str
    group: str
    query: dict[str, Any]
    body: Any
    status: int = 0
    response: Any = None
    fault: str | None = None
    completed: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "method": self.method,
            "path": self.path,
            "group": self.group,
            "query": self.query,
            "body": self.body,
            "status": self.status,
            "response": self.response,
            "fault": self.fault,
            "completed": self.completed,
        }


@dataclass
class SandboxState:
    clock: Clock = field(default_factory=SystemClock)
    seed: SeedConfig = field(default_factory=SeedConfig)
    faults: FaultEngine = field(default_factory=FaultEngine)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Vendor-shaped objects, exactly as the vendor API returns them.
    calcom_bookings: list[dict[str, Any]] = field(default_factory=list)
    # Google events by calendar id, then by event id. A deleted event stays as a ``cancelled`` tombstone.
    google_events: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    # Assertions accepted by the OAuth token endpoint: issuer, subject, scopes and times.
    google_token_grants: list[dict[str, Any]] = field(default_factory=list)
    hubspot_contacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    hubspot_meetings: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Busy blocks that are not vendor objects: seeded bookings and third-party takes.
    external_busy: list[dict[str, Any]] = field(default_factory=list)
    request_log: list[LogEntry] = field(default_factory=list)
    #: Bumped by every reset, so a call that was waiting (delay or hang) cannot write into the fresh state.
    generation: int = 0
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1))

    def next_id(self) -> int:
        return next(self._ids)

    def now(self) -> datetime:
        return self.clock.now()

    def reset(self) -> None:
        self.seed = SeedConfig()
        self.faults = FaultEngine()
        self.calcom_bookings.clear()
        self.google_events.clear()
        self.google_token_grants.clear()
        self.hubspot_contacts.clear()
        self.hubspot_meetings.clear()
        self.external_busy.clear()
        self.request_log.clear()
        self._ids = itertools.count(1)
        self.generation += 1

    def apply_seed(self, seed: SeedConfig) -> None:
        self.seed = seed
        self.external_busy = [
            {
                "source": "seed",
                "title": b.title,
                "start": iso_z(b.start),
                "end": iso_z(b.end or b.start + timedelta(minutes=seed.event_length_minutes)),
            }
            for b in seed.existing_bookings
        ]

    # Host calendar ----------------------------------------------------------------------------

    def busy_intervals(self, *, exclude_uid: str | None = None) -> list[tuple[datetime, datetime]]:
        """Every interval that blocks the host calendar.

        ``exclude_uid`` leaves out one Cal.com booking, as Cal.com does for the booking being rescheduled.
        """
        busy = [(parse_iso(b["start"]), parse_iso(b["end"])) for b in self.external_busy]
        busy += [
            (parse_iso(b["start"]), parse_iso(b["end"]))
            for b in self.calcom_bookings
            if b.get("status") in ("accepted", "pending") and b.get("uid") != exclude_uid
        ]
        for events in self.google_events.values():
            for event in events.values():
                if event.get("status") != "cancelled" and event.get("transparency") != "transparent":
                    busy.append((parse_iso(event["start"]["dateTime"]), parse_iso(event["end"]["dateTime"])))
        return busy

    def free_starts(
        self, window_start: datetime, window_end: datetime, *, exclude_uid: str | None = None
    ) -> list[datetime]:
        return free_slot_starts(
            self.seed.hours(),
            self.busy_intervals(exclude_uid=exclude_uid),
            window_start,
            window_end,
            self.now(),
        )

    def within_hours(self, start: datetime, end: datetime) -> bool:
        """True when ``[start, end)`` lies inside the host's working hours on one working day."""
        hours = self.seed.hours()
        zone = ZoneInfo(hours.zone)
        local_start, local_end = start.astimezone(zone), end.astimezone(zone)
        return (
            start < end
            and local_start.isoweekday() in hours.days
            and local_start.date() == local_end.date()
            and hours.start <= local_start.time()
            and local_end.time() <= hours.end
        )

    def host_available(self, start: datetime, end: datetime, *, exclude_uid: str | None = None) -> bool:
        """Working-hours and conflict check for an arbitrary interval (not only slot-grid starts).

        Minimum notice and horizon are checked separately, because Cal.com reports them with a different
        error than a conflict.
        """
        return self.within_hours(start, end) and not overlaps(
            start, end, self.busy_intervals(exclude_uid=exclude_uid)
        )

    def is_free(self, start: datetime) -> bool:
        end = start + timedelta(minutes=self.seed.event_length_minutes)
        return start in self.free_starts(start, end)

    def take_by_third_party(self, starts: list[datetime], reason: str) -> None:
        length = timedelta(minutes=self.seed.event_length_minutes)
        for start in starts:
            self.external_busy.append(
                {
                    "source": reason,
                    "title": "Booked by someone else",
                    "attendee_email": THIRD_PARTY_EMAIL,
                    "start": iso_z(start),
                    "end": iso_z(start + length),
                }
            )

    # Request log ------------------------------------------------------------------------------

    def log(self, method: str, path: str, group: str, query: dict[str, Any], body: Any) -> LogEntry:
        entry = LogEntry(
            seq=len(self.request_log) + 1,
            ts=iso_z(self.now()),
            method=method,
            path=path,
            group=group,
            query=query,
            body=body,
        )
        self.request_log.append(entry)
        return entry

    def last_response(self, group: str) -> LogEntry | None:
        for entry in reversed(self.request_log):
            if entry.group == group and entry.completed and entry.status == 200 and entry.fault is None:
                return entry
        return None

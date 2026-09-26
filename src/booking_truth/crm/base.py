"""The CRM adapter contract: validated payload models, sealed result types and the ``CrmAdapter`` protocol.

The agent writes to a CRM only after a verified calendar result (``crm_outbox`` guard) or, in the naive
baseline, when its reply contains the word "booked". Either way every payload is a Pydantic model that is
validated before it leaves the agent: required fields are present, instants are timezone-aware and a meeting
starts before it ends. The outbox stores the JSON form of :class:`CrmSyncPayload` and validates it again when
it is queued, so a hand-built or mutated payload cannot reach the CRM.

Adapter methods never raise for vendor failures; they return ``CrmOk | CrmError``. ``CrmError.retryable``
tells the outbox worker whether another attempt can help (timeouts, 429 and 5xx) or not (a 4xx validation
error).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from booking_truth.timeutil import ensure_utc

#: HubSpot meeting outcomes the agent writes (``hs_meeting_outcome``).
MeetingOutcome = Literal["SCHEDULED", "RESCHEDULED", "CANCELED"]
#: What happened on the calendar; one outbox item per verified write.
SyncAction = Literal["booked", "rescheduled", "cancelled"]

OUTCOME_FOR_ACTION: dict[str, MeetingOutcome] = {
    "booked": "SCHEDULED",
    "rescheduled": "RESCHEDULED",
    "cancelled": "CANCELED",
}

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_EMAIL_CHARS = 254


def _email(value: str) -> str:
    cleaned = value.strip().lower()
    if len(cleaned) > MAX_EMAIL_CHARS or not _EMAIL.fullmatch(cleaned):
        raise ValueError(f"not an email address: {value!r}")
    return cleaned


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ContactPayload(_Payload):
    """Create or update the lead's contact, keyed by email."""

    email: str
    name: str | None = Field(default=None, max_length=200)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        return _email(value)

    @property
    def first_last(self) -> tuple[str | None, str | None]:
        """The name split into first name and the rest, as CRMs store it."""
        if not self.name:
            return None, None
        first, _, rest = self.name.partition(" ")
        return first or None, rest or None


class _Timed(_Payload):
    start_utc: AwareDatetime
    end_utc: AwareDatetime

    @field_validator("start_utc", "end_utc")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _ordered(self) -> _Timed:
        if not self.start_utc < self.end_utc:
            raise ValueError("a meeting must start before it ends")
        return self


class MeetingPayload(_Timed):
    """Create a meeting associated with the lead's contact."""

    contact_id: str = Field(min_length=1, max_length=64)
    booking_ref: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    outcome: MeetingOutcome = "SCHEDULED"
    body: str = Field(default="", max_length=2000)


class MeetingUpdatePayload(_Timed):
    """Move a meeting, or mark it cancelled, after a verified reschedule or cancel."""

    meeting_id: str = Field(min_length=1, max_length=64)
    booking_ref: str = Field(min_length=1, max_length=128)
    outcome: MeetingOutcome


class CrmSyncPayload(_Timed):
    """One outbox item: mirror a verified calendar write in the CRM.

    ``booking_ref`` is the booking as it is now (for a reschedule the new booking); ``previous_ref`` is the
    booking a reschedule replaced, whose CRM meeting is the one to move.
    """

    action: SyncAction
    lead_email: str
    lead_name: str | None = Field(default=None, max_length=200)
    booking_ref: str = Field(min_length=1, max_length=128)
    previous_ref: str | None = Field(default=None, max_length=128)
    zone: str | None = Field(default=None, max_length=64)
    title: str = Field(default="Intro call", min_length=1, max_length=200)

    @field_validator("lead_email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        return _email(value)

    @model_validator(mode="after")
    def _previous_only_for_reschedule(self) -> CrmSyncPayload:
        if self.previous_ref is not None and self.action != "rescheduled":
            raise ValueError("previous_ref is only valid for a reschedule")
        return self

    @property
    def outcome(self) -> MeetingOutcome:
        return OUTCOME_FOR_ACTION[self.action]


@dataclass(frozen=True)
class CrmOk:
    """The CRM accepted the write; ``id`` is the contact or meeting id it returned."""

    id: str


@dataclass(frozen=True)
class CrmError:
    """The CRM refused the write or could not be reached.

    ``reason``: timeout | server_error | rate_limited | invalid | not_found | auth | malformed.
    """

    reason: str
    detail: str = ""
    retryable: bool = False


CrmResult = CrmOk | CrmError


@runtime_checkable
class CrmAdapter(Protocol):
    """A CRM that stores the lead as a contact and each booking as a meeting associated with it."""

    kind: str

    async def upsert_contact(self, payload: ContactPayload) -> CrmResult: ...

    async def create_meeting(self, payload: MeetingPayload) -> CrmResult: ...

    async def update_meeting(self, payload: MeetingUpdatePayload) -> CrmResult: ...

    async def aclose(self) -> None: ...

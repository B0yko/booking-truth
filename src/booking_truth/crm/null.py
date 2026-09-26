"""``NullCrm``: the adapter for ``BT_CRM=none``. It accepts every write and keeps it in memory."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from booking_truth.crm.base import (
    ContactPayload,
    CrmError,
    CrmOk,
    CrmResult,
    MeetingPayload,
    MeetingUpdatePayload,
)


@dataclass
class NullCrm:
    """Records what would have been written, so tests and ``/healthz`` can show it; talks to nothing."""

    kind: str = "none"
    contacts: dict[str, ContactPayload] = field(default_factory=dict)
    meetings: dict[str, MeetingPayload | MeetingUpdatePayload] = field(default_factory=dict)
    writes: int = 0
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1), repr=False)

    async def upsert_contact(self, payload: ContactPayload) -> CrmResult:
        self.writes += 1
        self.contacts[payload.email] = payload
        return CrmOk(f"contact-{payload.email}")

    async def create_meeting(self, payload: MeetingPayload) -> CrmResult:
        self.writes += 1
        meeting_id = f"meeting-{next(self._ids)}"
        self.meetings[meeting_id] = payload
        return CrmOk(meeting_id)

    async def update_meeting(self, payload: MeetingUpdatePayload) -> CrmResult:
        self.writes += 1
        if payload.meeting_id not in self.meetings:
            return CrmError("not_found", f"no meeting {payload.meeting_id}")
        self.meetings[payload.meeting_id] = payload
        return CrmOk(payload.meeting_id)

    async def aclose(self) -> None:
        return None

"""Every HubSpotAdapter method over real HTTP against the sandbox's HubSpot mirror."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from hubspot_env import HubEnv

from booking_truth.crm.base import (
    ContactPayload,
    CrmAdapter,
    CrmError,
    CrmOk,
    MeetingPayload,
    MeetingUpdatePayload,
)
from booking_truth.crm.hubspot import HubSpotAdapter, conflict_contact_id

START = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
END = START + timedelta(minutes=30)
LEAD = "maya@example.com"
LEAD_NAME = "Maya R"
# A client timeout well below the sandbox's hang, so hangs are observed as client timeouts.
FAST = httpx.Timeout(0.25)
HANG_S = 0.8


def hang(group: str, mode: str = "timeout", **extra: object) -> dict[str, object]:
    return {"group": group, "mode": mode, "hang_s": HANG_S, **extra}


# Contacts ----------------------------------------------------------------------------------------------


async def test_adapter_satisfies_the_protocol(hubspot: HubSpotAdapter) -> None:
    assert isinstance(hubspot, CrmAdapter)
    assert hubspot.kind == "hubspot"


async def test_upsert_creates_a_new_contact(env: HubEnv, hubspot: HubSpotAdapter) -> None:
    result = await hubspot.upsert_contact(ContactPayload(email=LEAD, name=LEAD_NAME))
    assert isinstance(result, CrmOk)
    [contact] = env.contacts()
    assert contact["id"] == result.id
    assert contact["properties"]["email"] == LEAD
    assert contact["properties"]["firstname"] == "Maya"
    assert contact["properties"]["lastname"] == "R"


async def test_upsert_updates_the_contact_it_finds_by_email(env: HubEnv, hubspot: HubSpotAdapter) -> None:
    first = await hubspot.upsert_contact(ContactPayload(email=LEAD, name="Maya R"))
    assert isinstance(first, CrmOk)
    second = await hubspot.upsert_contact(ContactPayload(email=LEAD.upper(), name="Maya Renner"))
    assert isinstance(second, CrmOk)
    assert second.id == first.id
    [contact] = env.contacts()
    assert contact["properties"]["lastname"] == "Renner"


async def test_upsert_with_no_name_still_finds_the_contact_by_email(
    env: HubEnv, hubspot: HubSpotAdapter
) -> None:
    first = await hubspot.upsert_contact(ContactPayload(email=LEAD))
    second = await hubspot.upsert_contact(ContactPayload(email=LEAD))
    assert isinstance(first, CrmOk)
    assert isinstance(second, CrmOk)
    assert first.id == second.id
    assert len(env.contacts()) == 1


def test_conflict_contact_id_parses_the_hubspot_message() -> None:
    assert conflict_contact_id("Contact already exists. Existing ID: 216799192486") == "216799192486"
    assert conflict_contact_id("Contact already exists") is None


async def test_upsert_recovers_the_existing_id_from_a_409(
    env: HubEnv, hubspot: HubSpotAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search that misses the contact HubSpot's own eventual consistency has not caught up on yet:
    ``upsert_contact`` still lands on the right row, by parsing the create's 409 body."""
    first = await hubspot.upsert_contact(ContactPayload(email=LEAD, name="Maya R"))
    assert isinstance(first, CrmOk)

    async def _miss(_email: str) -> None:
        return None

    monkeypatch.setattr(hubspot, "_search_contact_id", _miss)
    second = await hubspot.upsert_contact(ContactPayload(email=LEAD, name="Maya Renner"))
    assert isinstance(second, CrmOk)
    assert second.id == first.id
    assert len(env.contacts()) == 1
    [contact] = env.contacts()
    assert contact["properties"]["lastname"] == "Renner"


async def test_upsert_fails_cleanly_when_a_conflict_carries_no_id_and_the_fallback_search_misses(
    hubspot: HubSpotAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _conflict_without_id(_payload: ContactPayload) -> CrmError:
        return CrmError("conflict", "Contact already exists", retryable=True)

    monkeypatch.setattr(hubspot, "_create_contact", _conflict_without_id)
    result = await hubspot.upsert_contact(ContactPayload(email="nobody@example.com"))
    assert isinstance(result, CrmError)
    assert result.reason == "conflict"


async def test_upsert_contact_500_is_a_retryable_server_error(env: HubEnv, hubspot: HubSpotAdapter) -> None:
    env.faults({"group": "crm.contacts.create", "mode": "error_500", "times": 1})
    result = await hubspot.upsert_contact(ContactPayload(email=LEAD))
    assert isinstance(result, CrmError)
    assert (result.reason, result.retryable) == ("server_error", True)


async def test_upsert_contact_times_out_as_a_retryable_timeout(env: HubEnv) -> None:
    env.faults(hang("crm.contacts.create"))
    async with env.adapter(timeout=FAST) as fast:
        result = await fast.upsert_contact(ContactPayload(email=LEAD))
    assert isinstance(result, CrmError)
    assert (result.reason, result.retryable) == ("timeout", True)


# Meetings ------------------------------------------------------------------------------------------------


async def test_create_meeting_associates_the_contact(env: HubEnv, hubspot: HubSpotAdapter) -> None:
    contact = await hubspot.upsert_contact(ContactPayload(email=LEAD, name=LEAD_NAME))
    assert isinstance(contact, CrmOk)
    meeting = await hubspot.create_meeting(
        MeetingPayload(
            contact_id=contact.id, booking_ref="b1", title="Intro call", start_utc=START, end_utc=END
        )
    )
    assert isinstance(meeting, CrmOk)
    [stored] = env.meetings()
    assert stored["id"] == meeting.id
    assert stored["properties"]["hs_meeting_title"] == "Intro call"
    assert stored["properties"]["hs_meeting_outcome"] == "SCHEDULED"
    assert stored["properties"]["hs_meeting_start_time"] == "2026-10-06T13:00:00Z"
    assert stored["properties"]["hs_meeting_end_time"] == "2026-10-06T13:30:00Z"
    assert stored["associations"]["contacts"]["results"] == [
        {"id": contact.id, "type": "meeting_event_to_contact"}
    ]


async def test_create_meeting_for_an_unknown_contact_is_invalid(hubspot: HubSpotAdapter) -> None:
    result = await hubspot.create_meeting(
        MeetingPayload(
            contact_id="999999", booking_ref="b1", title="Intro call", start_utc=START, end_utc=END
        )
    )
    assert isinstance(result, CrmError)
    assert (result.reason, result.retryable) == ("invalid", False)


async def test_update_meeting_reschedules_the_times_and_outcome(env: HubEnv, hubspot: HubSpotAdapter) -> None:
    contact = await hubspot.upsert_contact(ContactPayload(email=LEAD))
    assert isinstance(contact, CrmOk)
    meeting = await hubspot.create_meeting(
        MeetingPayload(
            contact_id=contact.id, booking_ref="b1", title="Intro call", start_utc=START, end_utc=END
        )
    )
    assert isinstance(meeting, CrmOk)
    moved = START + timedelta(days=1)
    update = await hubspot.update_meeting(
        MeetingUpdatePayload(
            meeting_id=meeting.id,
            booking_ref="b2",
            outcome="RESCHEDULED",
            start_utc=moved,
            end_utc=moved + timedelta(minutes=30),
        )
    )
    assert isinstance(update, CrmOk)
    [stored] = env.meetings()
    assert stored["properties"]["hs_meeting_outcome"] == "RESCHEDULED"
    assert stored["properties"]["hs_meeting_start_time"] == "2026-10-07T13:00:00Z"


async def test_update_meeting_cancels_the_outcome(env: HubEnv, hubspot: HubSpotAdapter) -> None:
    contact = await hubspot.upsert_contact(ContactPayload(email=LEAD))
    assert isinstance(contact, CrmOk)
    meeting = await hubspot.create_meeting(
        MeetingPayload(
            contact_id=contact.id, booking_ref="b1", title="Intro call", start_utc=START, end_utc=END
        )
    )
    assert isinstance(meeting, CrmOk)
    update = await hubspot.update_meeting(
        MeetingUpdatePayload(
            meeting_id=meeting.id, booking_ref="b1", outcome="CANCELED", start_utc=START, end_utc=END
        )
    )
    assert isinstance(update, CrmOk)
    [stored] = env.meetings()
    assert stored["properties"]["hs_meeting_outcome"] == "CANCELED"


async def test_update_meeting_not_found(hubspot: HubSpotAdapter) -> None:
    result = await hubspot.update_meeting(
        MeetingUpdatePayload(
            meeting_id="999999", booking_ref="b1", outcome="CANCELED", start_utc=START, end_utc=END
        )
    )
    assert isinstance(result, CrmError)
    assert (result.reason, result.retryable) == ("not_found", False)


async def test_create_meeting_malformed_is_retryable(env: HubEnv, hubspot: HubSpotAdapter) -> None:
    contact = await hubspot.upsert_contact(ContactPayload(email=LEAD))
    assert isinstance(contact, CrmOk)
    env.faults({"group": "crm.meetings.create", "mode": "malformed", "times": 1})
    result = await hubspot.create_meeting(
        MeetingPayload(
            contact_id=contact.id, booking_ref="b1", title="Intro call", start_utc=START, end_utc=END
        )
    )
    assert isinstance(result, CrmError)
    assert (result.reason, result.retryable) == ("malformed", True)

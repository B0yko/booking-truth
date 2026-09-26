"""CRM payload validation and the null adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from booking_truth.config import load_settings
from booking_truth.crm import (
    ContactPayload,
    CrmAdapter,
    CrmError,
    CrmOk,
    CrmSyncPayload,
    MeetingPayload,
    MeetingUpdatePayload,
    NullCrm,
    build_crm,
)
from booking_truth.store import Store

START = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
END = START + timedelta(minutes=30)


def test_contacts_normalise_the_email_and_split_the_name() -> None:
    contact = ContactPayload(email=" Maya@Example.com ", name="Maya R")
    assert contact.email == "maya@example.com"
    assert contact.first_last == ("Maya", "R")
    assert ContactPayload(email="a@example.com").first_last == (None, None)
    with pytest.raises(ValidationError, match="email"):
        ContactPayload(email="not an email")


def test_meetings_must_start_before_they_end_and_be_aware() -> None:
    MeetingPayload(contact_id="1", booking_ref="b1", title="Intro call", start_utc=START, end_utc=END)
    with pytest.raises(ValidationError, match="start before it ends"):
        MeetingPayload(contact_id="1", booking_ref="b1", title="Intro call", start_utc=END, end_utc=START)
    with pytest.raises(ValidationError):
        MeetingPayload(
            contact_id="1",
            booking_ref="b1",
            title="Intro call",
            start_utc=datetime(2026, 10, 6, 13, 0),  # naive
            end_utc=END,
        )
    with pytest.raises(ValidationError):
        MeetingPayload(contact_id="", booking_ref="b1", title="Intro call", start_utc=START, end_utc=END)


def test_sync_payloads_carry_the_outcome_and_check_previous_ref() -> None:
    booked = CrmSyncPayload(
        action="booked", lead_email="maya@example.com", booking_ref="b1", start_utc=START, end_utc=END
    )
    assert booked.outcome == "SCHEDULED"
    moved = CrmSyncPayload(
        action="rescheduled",
        lead_email="maya@example.com",
        booking_ref="b2",
        previous_ref="b1",
        start_utc=START,
        end_utc=END,
    )
    assert moved.outcome == "RESCHEDULED"
    with pytest.raises(ValidationError, match="previous_ref"):
        CrmSyncPayload(
            action="booked",
            lead_email="maya@example.com",
            booking_ref="b2",
            previous_ref="b1",
            start_utc=START,
            end_utc=END,
        )
    with pytest.raises(ValidationError):
        CrmSyncPayload(action="booked", lead_email="x", booking_ref="b1", start_utc=START, end_utc=END)


def test_sync_payloads_go_through_the_outbox_validation(tmp_path: Path) -> None:
    with Store(tmp_path / "a.db") as store:
        payload = CrmSyncPayload(
            action="cancelled", lead_email="maya@example.com", booking_ref="b1", start_utc=START, end_utc=END
        )
        item = store.outbox.enqueue("maya@example.com", "crm_sync", payload)
        assert item.payload["action"] == "cancelled"
        broken = CrmSyncPayload.model_construct(
            action="booked", lead_email="maya@example.com", booking_ref="b1", start_utc=END, end_utc=START
        )
        with pytest.raises(ValidationError):
            store.outbox.enqueue("maya@example.com", "crm_sync", broken)


async def test_the_null_crm_records_writes() -> None:
    crm = NullCrm()
    assert isinstance(crm, CrmAdapter)
    contact = await crm.upsert_contact(ContactPayload(email="maya@example.com"))
    assert isinstance(contact, CrmOk)
    meeting = await crm.create_meeting(
        MeetingPayload(
            contact_id=contact.id, booking_ref="b1", title="Intro call", start_utc=START, end_utc=END
        )
    )
    assert isinstance(meeting, CrmOk)
    update = MeetingUpdatePayload(
        meeting_id=meeting.id, booking_ref="b1", outcome="CANCELED", start_utc=START, end_utc=END
    )
    assert isinstance(await crm.update_meeting(update), CrmOk)
    missing = update.model_copy(update={"meeting_id": "nope"})
    assert isinstance(await crm.update_meeting(missing), CrmError)
    assert crm.writes == 4
    await crm.aclose()


def test_build_crm_falls_back_to_the_null_adapter_without_hubspot(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    none, note = build_crm(load_settings(crm="none"))
    assert isinstance(none, NullCrm)
    assert note is None
    monkeypatch.setitem(sys.modules, "booking_truth.crm.hubspot", None)
    fallback, note = build_crm(load_settings(crm="hubspot", hubspot_token="t"))
    assert isinstance(fallback, NullCrm)
    assert note is not None
    assert "HubSpot" in note

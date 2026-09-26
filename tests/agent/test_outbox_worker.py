"""OutboxWorker: delivery to a fake CRM, the retry/give-up split and per-lead ordering.

The real HubSpot adapter's own HTTP behaviour is covered in ``tests/crm/test_hubspot.py``; this module
tests the worker's delivery logic (which calls it makes, in what order, how it reacts to a
:class:`~booking_truth.crm.base.CrmResult`) against a small in-memory fake, so retries and ordering are
exercised deterministically without sandbox fault timing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path

import pytest

from booking_truth.agent.outbox_worker import OutboxWorker
from booking_truth.crm.base import (
    ContactPayload,
    CrmError,
    CrmOk,
    CrmResult,
    CrmSyncPayload,
    MeetingPayload,
    MeetingUpdatePayload,
)
from booking_truth.store import Store
from booking_truth.timeutil import Clock, MutableClock, iso_ms_z

START = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
END = START + timedelta(minutes=30)
LEAD = "maya@example.com"
LEAD_NAME = "Maya R"
OTHER_LEAD = "omar@example.com"


@dataclass
class FakeCrm:
    """A ``CrmAdapter`` double: contacts and meetings live in memory; queued errors let a test make one
    call fail without touching the sandbox or the network."""

    kind: str = "fake"
    contacts: dict[str, str] = field(default_factory=dict)
    meetings: dict[str, dict[str, object]] = field(default_factory=dict)
    create_calls: list[MeetingPayload] = field(default_factory=list)
    update_calls: list[MeetingUpdatePayload] = field(default_factory=list)
    create_failures: list[CrmError] = field(default_factory=list)
    update_failures: list[CrmError] = field(default_factory=list)
    _next_contact: count[int] = field(default_factory=lambda: count(1), repr=False)
    _next_meeting: count[int] = field(default_factory=lambda: count(1), repr=False)

    async def upsert_contact(self, payload: ContactPayload) -> CrmResult:
        contact_id = self.contacts.get(payload.email)
        if contact_id is None:
            contact_id = f"c{next(self._next_contact)}"
            self.contacts[payload.email] = contact_id
        return CrmOk(contact_id)

    async def create_meeting(self, payload: MeetingPayload) -> CrmResult:
        self.create_calls.append(payload)
        if self.create_failures:
            return self.create_failures.pop(0)
        meeting_id = f"m{next(self._next_meeting)}"
        self.meetings[meeting_id] = {
            "outcome": payload.outcome,
            "start": payload.start_utc,
            "end": payload.end_utc,
        }
        return CrmOk(meeting_id)

    async def update_meeting(self, payload: MeetingUpdatePayload) -> CrmResult:
        self.update_calls.append(payload)
        if self.update_failures:
            return self.update_failures.pop(0)
        if payload.meeting_id not in self.meetings:
            return CrmError("not_found", "no such meeting", retryable=False)
        self.meetings[payload.meeting_id].update(
            outcome=payload.outcome, start=payload.start_utc, end=payload.end_utc
        )
        return CrmOk(payload.meeting_id)

    async def aclose(self) -> None:
        return None


def open_store(tmp_path: Path, *, clock: Clock | None = None) -> Store:
    return Store(tmp_path / "agent.db", clock=clock)


def sync_payload(
    action: str = "booked", *, booking_ref: str = "b1", previous_ref: str | None = None, **overrides: object
) -> CrmSyncPayload:
    fields: dict[str, object] = dict(
        action=action,
        lead_email=LEAD,
        lead_name=LEAD_NAME,
        booking_ref=booking_ref,
        previous_ref=previous_ref,
        start_utc=START,
        end_utc=END,
    )
    fields.update(overrides)
    return CrmSyncPayload(**fields)  # type: ignore[arg-type]


# Delivery ----------------------------------------------------------------------------------------------


async def test_a_booking_creates_a_meeting_and_links_its_reference(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    crm = FakeCrm()
    worker = OutboxWorker(store, crm)
    store.outbox.enqueue(LEAD, "crm_sync", sync_payload("booked"))
    try:
        assert await worker.drain_once() == 1
        [item] = store.outbox.items()
        assert item.status == "done"
        [meeting_id] = crm.meetings
        assert crm.meetings[meeting_id]["outcome"] == "SCHEDULED"
        link = store.crm_links.get("b1")
        assert link is not None
        assert (link.meeting_id, link.contact_id) == (meeting_id, crm.contacts[LEAD])
    finally:
        store.close()


async def test_a_reschedule_moves_the_same_meeting_and_relinks_the_new_reference(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    crm = FakeCrm()
    worker = OutboxWorker(store, crm)
    try:
        store.outbox.enqueue(LEAD, "crm_sync", sync_payload("booked", booking_ref="b1"))
        await worker.drain_once()
        [meeting_id] = crm.meetings

        moved_start = START + timedelta(days=1)
        moved = sync_payload(
            "rescheduled",
            booking_ref="b2",
            previous_ref="b1",
            start_utc=moved_start,
            end_utc=moved_start + timedelta(minutes=30),
        )
        store.outbox.enqueue(LEAD, "crm_sync", moved)
        await worker.drain_once()

        assert list(crm.meetings) == [meeting_id]  # the same meeting, not a second one
        assert crm.meetings[meeting_id] == {
            "outcome": "RESCHEDULED",
            "start": moved_start,
            "end": moved_start + timedelta(minutes=30),
        }
        new_link = store.crm_links.get("b2")
        assert new_link is not None
        assert new_link.meeting_id == meeting_id
    finally:
        store.close()


async def test_a_cancel_sets_the_meetings_outcome(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    crm = FakeCrm()
    worker = OutboxWorker(store, crm)
    try:
        store.outbox.enqueue(LEAD, "crm_sync", sync_payload("booked", booking_ref="b1"))
        await worker.drain_once()
        [meeting_id] = crm.meetings

        store.outbox.enqueue(LEAD, "crm_sync", sync_payload("cancelled", booking_ref="b1"))
        await worker.drain_once()

        assert crm.meetings[meeting_id]["outcome"] == "CANCELED"
        assert list(crm.meetings) == [meeting_id]
    finally:
        store.close()


async def test_a_reschedule_with_no_known_meeting_creates_one_rather_than_losing_it(tmp_path: Path) -> None:
    """The booking's own outbox item never landed a mapping (say, it failed for good earlier). The
    reschedule must still reach the CRM instead of silently doing nothing."""
    store = open_store(tmp_path)
    crm = FakeCrm()
    worker = OutboxWorker(store, crm)
    try:
        moved = sync_payload("rescheduled", booking_ref="b2", previous_ref="b1")
        store.outbox.enqueue(LEAD, "crm_sync", moved)
        assert await worker.drain_once() == 1
        [item] = store.outbox.items()
        assert item.status == "done"
        [meeting] = crm.meetings.values()
        assert meeting["outcome"] == "RESCHEDULED"
    finally:
        store.close()


# Retry and giving up -------------------------------------------------------------------------------------


async def test_a_retryable_crm_error_is_retried_on_the_outbox_backoff_and_then_lands(tmp_path: Path) -> None:
    clock = MutableClock(START)
    store = open_store(tmp_path, clock=clock)
    crm = FakeCrm()
    crm.create_failures.append(CrmError("server_error", "boom", retryable=True))
    worker = OutboxWorker(store, crm)
    try:
        store.outbox.enqueue(LEAD, "crm_sync", sync_payload("booked"))
        assert await worker.drain_once() == 1
        [item] = store.outbox.items()
        assert (item.status, item.attempts) == ("pending", 1)
        assert item.next_attempt_at is not None
        assert item.next_attempt_at > clock.now()
        assert crm.meetings == {}

        clock.advance(timedelta(seconds=1))  # past the first backoff step (0.5 s)
        assert await worker.drain_once() == 1
        [item] = store.outbox.items()
        assert (item.status, item.attempts) == ("done", 2)
        assert len(crm.meetings) == 1
    finally:
        store.close()


async def test_a_non_retryable_crm_error_is_given_up_on_without_retrying(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    crm = FakeCrm()
    crm.create_failures.append(CrmError("invalid", "the contact id is not associable", retryable=False))
    worker = OutboxWorker(store, crm)
    try:
        store.outbox.enqueue(LEAD, "crm_sync", sync_payload("booked"))
        await worker.drain_once()
        [item] = store.outbox.items()
        assert (item.status, item.attempts) == ("failed", 1)
        assert item.last_error is not None
        assert "not associable" in item.last_error
    finally:
        store.close()


async def test_an_unrecognised_outbox_kind_is_given_up_on(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    worker = OutboxWorker(store, FakeCrm())
    try:
        store.outbox.enqueue(LEAD, "some_other_kind", sync_payload("booked"))
        await worker.drain_once()
        [item] = store.outbox.items()
        assert item.status == "failed"
    finally:
        store.close()


async def test_a_payload_that_no_longer_matches_the_schema_is_given_up_on(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    worker = OutboxWorker(store, FakeCrm())
    try:
        now = iso_ms_z(store.clock.now())
        store.connection().execute(
            "INSERT INTO outbox (lead_email, kind, payload_json, status, attempts, next_attempt_at, "
            "created_at, updated_at) VALUES (?, 'crm_sync', '{\"not\": \"valid\"}', 'pending', 0, ?, ?, ?)",
            (LEAD, now, now, now),
        )
        await worker.drain_once()
        [item] = store.outbox.items()
        assert item.status == "failed"
    finally:
        store.close()


# Ordering ------------------------------------------------------------------------------------------------


async def test_a_leads_later_item_waits_behind_an_earlier_one_still_retrying(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    crm = FakeCrm()
    crm.create_failures.append(CrmError("server_error", "boom", retryable=True))
    worker = OutboxWorker(store, crm)
    try:
        store.outbox.enqueue(LEAD, "crm_sync", sync_payload("booked", booking_ref="b1"))
        store.outbox.enqueue(LEAD, "crm_sync", sync_payload("cancelled", booking_ref="b1"))
        processed = await worker.drain_once()
        assert processed == 1  # only the booking attempt; the cancel never runs ahead of it
        assert [i.status for i in store.outbox.items()] == ["pending", "pending"]
        assert crm.update_calls == []
    finally:
        store.close()


async def test_different_leads_items_interleave_freely(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    crm = FakeCrm()
    worker = OutboxWorker(store, crm)
    try:
        store.outbox.enqueue(LEAD, "crm_sync", sync_payload("booked", booking_ref="b1"))
        other = sync_payload("booked", booking_ref="o1", lead_email=OTHER_LEAD)
        store.outbox.enqueue(OTHER_LEAD, "crm_sync", other)
        assert await worker.drain_once() == 2
        assert {i.status for i in store.outbox.items()} == {"done"}
    finally:
        store.close()


# Lifecycle -----------------------------------------------------------------------------------------------


async def test_start_and_stop_run_delivery_in_the_background(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    crm = FakeCrm()
    worker = OutboxWorker(store, crm, poll_s=0.01)
    try:
        store.outbox.enqueue(LEAD, "crm_sync", sync_payload("booked"))
        worker.start()
        worker.start()  # idempotent: a second call while running does nothing
        assert worker.running
        for _ in range(200):
            if store.outbox.items()[0].status == "done":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("the outbox item was not drained in time")
        await worker.stop()
        assert not worker.running
        await worker.stop()  # idempotent
    finally:
        store.close()

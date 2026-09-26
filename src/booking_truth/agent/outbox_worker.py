"""The background task that drains the CRM outbox: delivers each ``crm_sync`` payload the ``crm_outbox``
guard queued (:mod:`booking_truth.agent.core`) to the configured CRM adapter.

One payload is one calendar-confirmed event: a booking, a reschedule or a cancel. Delivery keeps a lead's
whole history in a single HubSpot meeting rather than one meeting per event:

- ``booked`` creates a meeting associated with the lead's contact and records the mapping
  (:class:`~booking_truth.store.repos.CrmLinksRepo`, keyed by the booking's own reference).
- ``rescheduled`` moves that meeting's times and outcome (looked up by the *previous* reference, since Cal.com
  gives a reschedule a new booking reference) and re-links the mapping under the new reference.
- ``cancelled`` sets that meeting's outcome to cancelled, times unchanged.

A meeting is created only when :meth:`OutboxWorker` finds no mapping for the reference it needs (the normal
case for ``booked``; a defensive fallback for a ``rescheduled`` or ``cancelled`` event whose earlier delivery
never landed a mapping, so the update is not silently lost).

Items are drained one per lead per pass (:meth:`~booking_truth.store.repos.OutboxRepo.due_per_lead`), so a
lead's events are always delivered in the order they happened: a cancel is never attempted before the create
it depends on has either landed or given up, even while that create is mid-retry. A CRM failure that
:class:`~booking_truth.crm.base.CrmError` marks ``retryable`` (a timeout, ``429`` or ``5xx``) goes back on the
outbox's own backoff schedule (0.5, 1, 2, 4 s, then every 30 s, failing for good after 20 attempts); one that
is not (a validation error, an auth failure, a vendor 404) is given up on immediately, since retrying it would
only repeat the same rejection.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Final

from pydantic import ValidationError

from booking_truth.crm.base import (
    ContactPayload,
    CrmAdapter,
    CrmOk,
    CrmResult,
    CrmSyncPayload,
    MeetingPayload,
    MeetingUpdatePayload,
)
from booking_truth.store import OutboxItem, Store

logger = logging.getLogger(__name__)

#: The only ``outbox.kind`` this worker knows how to deliver; see ``AgentCore._hook_crm_outbox``.
CRM_SYNC_KIND: Final = "crm_sync"
DEFAULT_POLL_S: Final = 0.2
DEFAULT_BATCH: Final = 25


def _detail(result: CrmResult) -> str:
    if isinstance(result, CrmOk):
        return ""
    return f"{result.reason}: {result.detail}" if result.detail else result.reason


class OutboxWorker:
    """Runs as one background task per agent process; started and stopped with the app's lifespan.

    Construct one per :class:`~booking_truth.store.repos.Store` and CRM adapter pair; ``start`` is
    idempotent (a second call while already running is a no-op) and ``stop`` waits for the current pass
    to finish before returning.
    """

    def __init__(
        self, store: Store, crm: CrmAdapter, *, poll_s: float = DEFAULT_POLL_S, batch: int = DEFAULT_BATCH
    ) -> None:
        self.store = store
        self.crm = crm
        self.poll_s = poll_s
        self.batch = batch
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._task is not None

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="crm-outbox-worker")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._task = None
        self._stop.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        while not self._stop.is_set():
            processed = await self.drain_once()
            if processed == 0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_s)

    async def drain_once(self) -> int:
        """Deliver every item currently due, respecting per-lead order; returns how many were attempted.

        Safe to call directly (e.g. from a test, or a one-shot CLI drain) without ``start``.
        """
        attempted = 0
        while True:
            items = self.store.outbox.due_per_lead(limit=self.batch)
            if not items:
                return attempted
            for item in items:
                await self._deliver(item)
            attempted += len(items)
            if len(items) < self.batch:
                return attempted

    # Delivery ----------------------------------------------------------------------------------------------

    async def _deliver(self, item: OutboxItem) -> None:
        if item.kind != CRM_SYNC_KIND:
            self._give_up(item, f"unknown outbox item kind {item.kind!r}")
            return
        try:
            payload = CrmSyncPayload.model_validate(item.payload)
        except ValidationError as exc:
            self._give_up(item, f"invalid {CRM_SYNC_KIND} payload: {exc}")
            return
        result = await self._sync(payload)
        if isinstance(result, CrmOk):
            self.store.outbox.mark_done(item.id)
            return
        if result.retryable:
            self.store.outbox.mark_failed(item.id, _detail(result))
        else:
            self._give_up(item, _detail(result))

    def _give_up(self, item: OutboxItem, detail: str) -> None:
        logger.warning("crm outbox item #%d given up on: %s", item.id, detail)
        self.store.outbox.mark_permanently_failed(item.id, detail)

    async def _sync(self, payload: CrmSyncPayload) -> CrmResult:
        contact = await self.crm.upsert_contact(
            ContactPayload(email=payload.lead_email, name=payload.lead_name)
        )
        if not isinstance(contact, CrmOk):
            return contact
        if payload.action == "booked":
            return await self._book(payload, contact.id)
        return await self._move(payload, contact.id)

    async def _book(self, payload: CrmSyncPayload, contact_id: str) -> CrmResult:
        """Create the meeting, unless a mapping for this reference already exists (a retried delivery after
        the worker crashed between creating the meeting and marking the item done)."""
        linked = self.store.crm_links.get(payload.booking_ref)
        if linked is not None and linked.meeting_id is not None:
            return CrmOk(linked.meeting_id)
        meeting = await self.crm.create_meeting(
            MeetingPayload(
                contact_id=contact_id,
                booking_ref=payload.booking_ref,
                title=payload.title,
                start_utc=payload.start_utc,
                end_utc=payload.end_utc,
                outcome=payload.outcome,
            )
        )
        if isinstance(meeting, CrmOk):
            self.store.crm_links.upsert(payload.booking_ref, contact_id=contact_id, meeting_id=meeting.id)
        return meeting

    async def _move(self, payload: CrmSyncPayload, contact_id: str) -> CrmResult:
        """Update the meeting a reschedule or a cancel refers to, or create one when no mapping is known for
        it, so the event is never silently lost."""
        meeting_id = self._linked_meeting(payload)
        if meeting_id is None:
            return await self._book(payload, contact_id)
        result = await self.crm.update_meeting(
            MeetingUpdatePayload(
                meeting_id=meeting_id,
                booking_ref=payload.booking_ref,
                outcome=payload.outcome,
                start_utc=payload.start_utc,
                end_utc=payload.end_utc,
            )
        )
        if isinstance(result, CrmOk) and payload.booking_ref != payload.previous_ref:
            self.store.crm_links.upsert(payload.booking_ref, contact_id=contact_id, meeting_id=meeting_id)
        return result

    def _linked_meeting(self, payload: CrmSyncPayload) -> str | None:
        """The meeting mapped to this event: its own reference first (an event already partly delivered, or
        a cancel of a booking that was never rescheduled), else the reference it replaced (a reschedule)."""
        direct = self.store.crm_links.get(payload.booking_ref)
        if direct is not None and direct.meeting_id is not None:
            return direct.meeting_id
        if payload.previous_ref is not None:
            previous = self.store.crm_links.get(payload.previous_ref)
            if previous is not None and previous.meeting_id is not None:
                return previous.meeting_id
        return None

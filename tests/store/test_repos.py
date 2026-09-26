"""Behaviour of each repository: dedupe rows, slot-list TTL, the ledger, idempotency, leases, outbox."""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from booking_truth.store import (
    OUTBOX_MAX_ATTEMPTS,
    CrmLink,
    LeadsRepo,
    LockInfo,
    LocksRepo,
    OutboxBacklog,
    Store,
    StoreError,
    backoff_delay,
    normalize_email,
    open_db,
)
from booking_truth.timeutil import MutableClock
from booking_truth.trace.models import Step

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
START = datetime(2026, 10, 6, 13, 0, tzinfo=UTC)
LEAD = "ada.l@example.com"


def _session(store: Store, session_id: str = "s1", channel: Any = "api") -> None:
    store.sessions.get_or_create(session_id, lead_email=LEAD, channel=channel)


# Leads and sessions ---------------------------------------------------------------------------


def test_normalize_email() -> None:
    assert normalize_email("  Ada.L@Example.COM \n") == LEAD


def test_leads_upsert_keeps_the_name_and_tracks_the_zone(store: Store, clock: MutableClock) -> None:
    lead = store.leads.upsert("  Ada.L@Example.com ", name="Ada L.")
    assert lead.email == LEAD
    assert lead.updated_at == T0
    clock.advance(timedelta(minutes=1))
    again = store.leads.upsert(LEAD)
    assert again.name == "Ada L."
    assert again.updated_at == T0 + timedelta(minutes=1)
    assert store.leads.get("ADA.L@example.com") == again

    zoned = store.leads.set_zone(LEAD, "Asia/Kolkata", source="stated", confirmed=True)
    assert (zoned.tz_zone, zoned.tz_source, zoned.tz_confirmed, zoned.name) == (
        "Asia/Kolkata",
        "stated",
        True,
        "Ada L.",
    )
    fresh = store.leads.set_zone("new@example.com", "Europe/Berlin", source="hint")
    assert fresh.name is None
    assert not fresh.tz_confirmed
    assert store.leads.get("nobody@example.com") is None


def test_sessions_count_turns_and_end(store: Store) -> None:
    created = store.sessions.get_or_create("s1", lead_email="Ada.L@example.com", channel="widget")
    assert (created.lead_email, created.channel, created.turns, created.ended) == (LEAD, "widget", 0, False)
    same = store.sessions.get_or_create("s1", lead_email="other@example.com", channel="api")
    assert same == created
    assert store.sessions.increment_turns("s1") == 1
    assert store.sessions.increment_turns("s1") == 2
    store.sessions.set_token_hash("s1", "hash-1")
    store.sessions.end("s1")
    stored = store.sessions.get("s1")
    assert stored is not None
    assert (stored.turns, stored.token_hash, stored.ended) == (2, "hash-1", True)
    assert store.sessions.get("missing") is None
    with pytest.raises(StoreError, match="unknown session"):
        store.sessions.increment_turns("missing")
    with pytest.raises(StoreError, match="unknown session"):
        store.sessions.end("missing")
    with pytest.raises(sqlite3.IntegrityError):
        store.sessions.get_or_create("s2", lead_email=LEAD, channel="sms")  # type: ignore[arg-type]


# Messages (dedupe) ----------------------------------------------------------------------------


def test_message_rows_go_from_pending_to_done(store: Store) -> None:
    assert store.messages.claim_pending("s1", "m1") == "new"
    assert store.messages.claim_pending("s1", "m1") == "pending"
    assert store.messages.claim_pending("s1", "m2") == "new"
    response = {"reply": "Here are some times.", "quick_replies": [], "booking": None}
    store.messages.complete("s1", "m1", response)
    assert store.messages.claim_pending("s1", "m1") == "done"
    row = store.messages.get("s1", "m1")
    assert row is not None
    assert (row.status, row.response) == ("done", response)
    assert store.messages.discard("s1", "m1") is False  # a stored response is never dropped
    assert store.messages.wait_done("s1", "m1", timeout_s=0) == response


def test_discarding_a_failed_turn_lets_the_retry_run(store: Store) -> None:
    assert store.messages.claim_pending("s1", "m1") == "new"
    assert store.messages.discard("s1", "m1") is True
    assert store.messages.get("s1", "m1") is None
    assert store.messages.claim_pending("s1", "m1") == "new"


def test_a_stale_pending_row_is_taken_over(store: Store, clock: MutableClock) -> None:
    assert store.messages.claim_pending("s1", "m1") == "new"
    clock.advance(timedelta(seconds=59))
    assert store.messages.claim_pending("s1", "m1", stale_after_s=60) == "pending"
    clock.advance(timedelta(seconds=1))
    assert store.messages.claim_pending("s1", "m1", stale_after_s=60) == "new"
    assert store.messages.claim_pending("s1", "m1", stale_after_s=60) == "pending"


def test_wait_done_gives_up_on_timeout_and_on_a_discarded_twin(store: Store) -> None:
    assert store.messages.claim_pending("s1", "m1") == "new"
    assert store.messages.wait_done("s1", "m1", timeout_s=0.1, poll_s=0.02) is None
    store.messages.discard("s1", "m1")
    assert store.messages.wait_done("s1", "m1", timeout_s=30) is None  # returns at once: row is gone


# History and trace steps ----------------------------------------------------------------------


def test_history_is_numbered_per_session(store: Store) -> None:
    _session(store, "s1")
    _session(store, "s2")
    assert store.history.append("s1", "user", {"role": "user", "content": "Tuesday afternoon?"}) == 0
    assert store.history.extend(
        "s1",
        [("assistant", {"role": "assistant", "content": None, "tool_calls": []}), ("tool", {"ok": True})],
    ) == [1, 2]
    assert store.history.append("s2", "user", "hi") == 0
    entries = store.history.for_session("s1")
    assert [(e.seq, e.role) for e in entries] == [(0, "user"), (1, "assistant"), (2, "tool")]
    assert entries[0].content == {"role": "user", "content": "Tuesday afternoon?"}
    with pytest.raises(sqlite3.IntegrityError):
        store.history.append("unknown-session", "user", "hi")


def test_trace_steps_are_numbered_across_turns(store: Store) -> None:
    _session(store, "s1")
    step = {"i": 99, "ts": "2026-10-01T12:00:00Z", "kind": "message", "role": "user", "content": "hi"}
    assert store.trace_steps.append("s1", step) == 0
    assert store.trace_steps.extend("s1", [step, {**step, "role": "agent", "content": "hello"}]) == [1, 2]
    steps = store.trace_steps.for_session("s1")
    assert [s["i"] for s in steps] == [0, 1, 2]
    assert steps[2]["content"] == "hello"
    assert step["i"] == 99  # the caller's dict is not modified
    for stored in steps:
        Step.model_validate(stored)
    with pytest.raises(sqlite3.IntegrityError):
        store.trace_steps.append("unknown-session", step)


# Slot lists -----------------------------------------------------------------------------------


def _slots(*ids: str) -> list[dict[str, Any]]:
    return [
        {"slot_id": ident, "start_utc": "2026-10-06T13:00:00Z", "label": "Tue 6 Oct, 3:00 PM"}
        for ident in ids
    ]


def test_only_the_latest_slot_list_counts(store: Store, clock: MutableClock) -> None:
    first = store.slot_lists.save(LEAD, _slots("s_a", "s_b"), zone="Europe/Berlin", session_id="s1")
    assert len(first.id) == 32
    clock.advance(timedelta(seconds=10))
    second = store.slot_lists.save("ADA.L@example.com", _slots("s_c"), zone="Europe/Berlin", list_id="list-2")
    latest = store.slot_lists.latest(LEAD, ttl_s=900)
    assert latest == second
    assert latest.id == "list-2"
    assert latest.session_id is None
    assert store.slot_lists.find_slot(LEAD, "s_c", ttl_s=900) == _slots("s_c")[0]
    assert store.slot_lists.find_slot(LEAD, "s_a", ttl_s=900) is None
    assert store.slot_lists.latest("other@example.com") is None


def test_slot_lists_expire_after_the_ttl(store: Store, clock: MutableClock) -> None:
    saved = store.slot_lists.save(LEAD, _slots("s_a"), zone="America/New_York")
    assert saved.expires_at(900) == T0 + timedelta(seconds=900)
    clock.advance(timedelta(seconds=899, milliseconds=999))
    assert store.slot_lists.find_slot(LEAD, "s_a", ttl_s=900) is not None
    clock.advance(timedelta(milliseconds=1))
    assert store.slot_lists.latest(LEAD, ttl_s=900) is None
    assert store.slot_lists.find_slot(LEAD, "s_a", ttl_s=900) is None
    assert store.slot_lists.latest(LEAD) == saved  # without a TTL the list is still readable


def test_slot_lists_saved_at_the_same_instant_keep_insert_order(store: Store) -> None:
    store.slot_lists.save(LEAD, _slots("s_a"), zone="UTC", list_id="b-first")
    store.slot_lists.save(LEAD, _slots("s_b"), zone="UTC", list_id="a-second")
    latest = store.slot_lists.latest(LEAD, ttl_s=900)
    assert latest is not None
    assert latest.id == "a-second"


def test_slot_lists_need_slot_ids_and_can_be_pruned(store: Store, clock: MutableClock) -> None:
    with pytest.raises(ValueError, match="slot_id"):
        store.slot_lists.save(LEAD, [{"label": "Tue"}], zone="UTC")
    store.slot_lists.save(LEAD, _slots("s_a"), zone="UTC")
    clock.advance(timedelta(hours=2))
    store.slot_lists.save(LEAD, _slots("s_b"), zone="UTC")
    assert store.slot_lists.prune(older_than_s=3600) == 1
    latest = store.slot_lists.latest(LEAD)
    assert latest is not None
    assert latest.find("s_b") is not None


# Claims ledger --------------------------------------------------------------------------------


def _record(store: Store, **overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "lead_email": LEAD,
        "event_key": "1001",
        "action": "booked",
        "booking_ref": "uid-1",
        "start_utc": START,
        "end_utc": START + timedelta(minutes=30),
        "status": "verified",
        "zone": "Europe/Berlin",
        "session_id": "s1",
        "channel": "api",
    }
    fields.update(overrides)
    return store.claims.record(**fields)


def test_a_verified_booking_is_current_until_cancelled(store: Store, clock: MutableClock) -> None:
    booked = _record(store)
    assert booked.is_current_booking
    assert store.claims.current_bookings(LEAD) == [booked]
    assert store.claims.current_bookings(LEAD, "1001") == [booked]
    assert store.claims.current_bookings(LEAD, "other-event") == []

    clock.advance(timedelta(minutes=5))
    cancel = _record(store, action="cancelled")
    assert not cancel.is_current_booking
    assert store.claims.current_bookings(LEAD) == []
    voided = store.claims.get(booked.id)
    assert voided is not None
    assert voided.voided_at == T0 + timedelta(minutes=5)
    assert store.claims.entries(LEAD, action="cancelled") == [cancel]
    assert [e.id for e in store.claims.entries(LEAD)] == [booked.id, cancel.id]


def test_a_reschedule_voids_the_previous_booking(store: Store) -> None:
    _record(store)
    moved = _record(
        store,
        action="rescheduled",
        booking_ref="uid-2",
        previous_ref="uid-1",
        start_utc=START + timedelta(days=2),
        end_utc=START + timedelta(days=2, minutes=30),
    )
    assert store.claims.current_bookings(LEAD) == [moved]


def test_a_same_id_reschedule_keeps_one_live_entry(store: Store) -> None:
    _record(store, booking_ref="evt1")
    moved = _record(
        store,
        action="rescheduled",
        booking_ref="evt1",
        start_utc=START + timedelta(hours=3),
        end_utc=START + timedelta(hours=3, minutes=30),
    )
    assert store.claims.current_bookings(LEAD) == [moved]


def test_unverified_entries_are_recorded_but_never_current(store: Store) -> None:
    unverified = _record(store, status="unverified")
    assert not unverified.is_current_booking
    assert store.claims.current_bookings(LEAD) == []
    assert store.claims.entries(LEAD, status="unverified") == [unverified]
    booked = _record(store, booking_ref="uid-2")
    # an unverified cancel does not void anything
    _record(store, booking_ref="uid-2", action="cancelled", status="unverified")
    assert store.claims.current_bookings(LEAD) == [booked]


def test_ledger_filters_and_void(store: Store, clock: MutableClock) -> None:
    first = _record(store, session_id="s1")
    clock.advance(timedelta(minutes=1))
    second = _record(
        store,
        lead_email="Grace.H@example.com",
        booking_ref="uid-9",
        session_id="s9",
        start_utc=START + timedelta(hours=1),
        end_utc=START + timedelta(hours=1, minutes=30),
    )
    third = _record(
        store,
        booking_ref="uid-3",
        session_id="s2",
        start_utc=START - timedelta(days=1),
        end_utc=START - timedelta(days=1) + timedelta(minutes=30),
    )
    assert store.claims.current_bookings(LEAD) == [third, first]  # earliest start first
    assert store.claims.current_bookings("grace.h@example.com") == [second]
    assert store.claims.entries(LEAD, session_id="s2") == [third]
    assert store.claims.entries(LEAD, since=T0 + timedelta(minutes=1)) == [third]
    assert store.claims.void("uid-1") == 1
    assert store.claims.void("uid-1") == 0
    assert store.claims.current_bookings(LEAD) == [third]
    assert store.claims.get(12345) is None


def test_ledger_rejects_bad_entries(store: Store) -> None:
    with pytest.raises(ValueError, match="start_utc < end_utc"):
        _record(store, end_utc=START)
    with pytest.raises(ValueError, match="action"):
        _record(store, action="moved")
    with pytest.raises(ValueError, match="status"):
        _record(store, status="probably")
    with pytest.raises(ValueError, match="naive"):
        _record(store, start_utc=datetime(2026, 10, 6, 13, 0), end_utc=datetime(2026, 10, 6, 13, 30))
    assert store.claims.entries(LEAD) == []


# Idempotency keys and generations ---------------------------------------------------------------


def test_idempotency_key_lifecycle(store: Store, clock: MutableClock) -> None:
    record, previous = store.idem.begin(
        "k1",
        kind="create",
        lead_email="Ada.L@example.com",
        event_key="1001",
        slot_start_utc=START,
        generation=0,
    )
    assert previous is None
    assert (record.status, record.lead_email, record.slot_start_utc, record.generation) == (
        "pending",
        LEAD,
        START,
        0,
    )
    again, previous = store.idem.begin("k1", kind="create", lead_email=LEAD, event_key="1001")
    assert previous == "pending"  # an earlier attempt may have landed: verify before writing
    assert again.status == "pending"

    clock.advance(timedelta(seconds=3))
    committed = store.idem.commit("k1", "uid-1")
    assert (committed.status, committed.booking_ref, committed.updated_at) == (
        "committed",
        "uid-1",
        T0 + timedelta(seconds=3),
    )
    reused, previous = store.idem.begin("k1", kind="create", lead_email=LEAD, event_key="1001")
    assert previous == "committed"
    assert reused == committed
    assert store.idem.commit("k1", "uid-1") == committed  # repeating the same result is harmless
    with pytest.raises(StoreError, match="cannot go from committed"):
        store.idem.commit("k1", "uid-2")
    with pytest.raises(StoreError, match="cannot go from committed"):
        store.idem.fail("k1")
    with pytest.raises(StoreError, match="cannot go from committed"):
        store.idem.adopt("k1", "uid-1")


def test_a_failed_key_is_retried_and_can_be_adopted(store: Store) -> None:
    store.idem.begin("k2", kind="cancel", lead_email=LEAD, event_key="1001")
    failed = store.idem.fail("k2")
    assert failed.status == "failed"
    assert failed.slot_start_utc is None
    assert failed.generation is None
    assert store.idem.fail("k2") == failed
    retried, previous = store.idem.begin("k2", kind="cancel", lead_email=LEAD, event_key="1001")
    assert (previous, retried.status) == ("failed", "pending")
    adopted = store.idem.adopt("k2", "uid-7")
    assert (adopted.status, adopted.booking_ref) == ("adopted", "uid-7")
    assert store.idem.get("k2") == adopted
    assert store.idem.get("missing") is None
    with pytest.raises(StoreError, match="unknown idempotency key"):
        store.idem.commit("missing", "uid-1")


def test_generations_start_at_zero_and_increment(store: Store) -> None:
    assert store.generations.current(LEAD, "1001") == 0
    assert store.generations.increment("ADA.L@example.com", "1001") == 1
    assert store.generations.increment(LEAD, "1001") == 2
    assert store.generations.current(LEAD, "1001") == 2
    assert store.generations.current(LEAD, "other-event") == 0


# Locks ----------------------------------------------------------------------------------------


def test_lock_acquire_renew_release(store: Store, clock: MutableClock) -> None:
    key = LocksRepo.lead_key(" Ada.L@Example.com ")
    assert key == f"lead:{LEAD}"
    assert store.locks.acquire(key, "turn-1")
    assert not store.locks.acquire(key, "turn-2")
    assert store.locks.acquire(key, "turn-1")  # the owner may take it again (extends the lease)
    assert store.locks.holder(key) == LockInfo(key, "turn-1", T0 + timedelta(seconds=30))
    clock.advance(timedelta(seconds=20))
    assert store.locks.renew(key, "turn-1")
    assert store.locks.holder(key) == LockInfo(key, "turn-1", T0 + timedelta(seconds=50))
    assert not store.locks.renew(key, "turn-2")
    assert not store.locks.release(key, "turn-2")
    assert store.locks.release(key, "turn-1")
    assert store.locks.holder(key) is None
    assert store.locks.acquire(key, "turn-2")


def test_an_expired_lease_is_free_and_the_old_owner_loses_it(store: Store, clock: MutableClock) -> None:
    key = LocksRepo.lead_key(LEAD)
    assert store.locks.acquire(key, "crashed-turn", lease_s=30)
    clock.advance(timedelta(seconds=29, milliseconds=999))
    assert not store.locks.acquire(key, "next-turn")
    clock.advance(timedelta(milliseconds=1))
    assert store.locks.holder(key) is None
    assert store.locks.acquire(key, "next-turn")
    assert not store.locks.renew(key, "crashed-turn")
    assert not store.locks.release(key, "crashed-turn")
    holder = store.locks.holder(key)
    assert holder is not None
    assert holder.owner == "next-turn"


def test_an_expired_lease_nobody_took_can_still_be_renewed(store: Store, clock: MutableClock) -> None:
    key = LocksRepo.lead_key(LEAD)
    assert store.locks.acquire(key, "slow-turn", lease_s=30)
    clock.advance(timedelta(seconds=45))
    assert store.locks.renew(key, "slow-turn", lease_s=30)
    assert store.locks.holder(key) == LockInfo(key, "slow-turn", T0 + timedelta(seconds=75))


# Outbox ---------------------------------------------------------------------------------------


class MeetingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booking_ref: str
    contact_email: str
    start_utc: datetime
    end_utc: datetime

    @model_validator(mode="after")
    def _start_before_end(self) -> Self:
        if not self.start_utc < self.end_utc:
            raise ValueError("start must be before end")
        return self


def _meeting(ref: str = "uid-1") -> MeetingPayload:
    return MeetingPayload(
        booking_ref=ref, contact_email=LEAD, start_utc=START, end_utc=START + timedelta(minutes=30)
    )


def test_backoff_schedule() -> None:
    assert [backoff_delay(n) for n in range(1, 8)] == [0.5, 1.0, 2.0, 4.0, 30.0, 30.0, 30.0]
    assert backoff_delay(19) == 30.0
    with pytest.raises(ValueError, match=">= 1"):
        backoff_delay(0)


def test_enqueue_stores_the_validated_json_and_is_due_now(store: Store) -> None:
    item = store.outbox.enqueue("Ada.L@example.com", "meeting_create", _meeting())
    assert item.payload == {
        "booking_ref": "uid-1",
        "contact_email": LEAD,
        "start_utc": "2026-10-06T13:00:00Z",
        "end_utc": "2026-10-06T13:30:00Z",
    }
    assert (item.lead_email, item.status, item.attempts, item.next_attempt_at) == (LEAD, "pending", 0, T0)
    assert store.outbox.due() == [item]
    assert store.outbox.get(item.id) == item
    assert store.outbox.backlog() == OutboxBacklog(pending=1, failed=0)


def test_enqueue_refuses_payloads_that_do_not_validate(store: Store) -> None:
    unchecked = MeetingPayload.model_construct(
        booking_ref="uid-1", contact_email=LEAD, start_utc=START, end_utc=START - timedelta(minutes=30)
    )
    with pytest.raises(ValidationError, match="start must be before end"):
        store.outbox.enqueue(LEAD, "meeting_create", unchecked)
    mutated = _meeting()
    mutated.end_utc = START
    with pytest.raises(ValidationError):
        store.outbox.enqueue(LEAD, "meeting_create", mutated)
    with pytest.raises(TypeError, match="pydantic"):
        store.outbox.enqueue(LEAD, "meeting_create", {"booking_ref": "uid-1"})  # type: ignore[arg-type]
    assert store.outbox.items() == []
    assert store.outbox.backlog().total == 0


def test_failed_attempts_follow_the_backoff_then_give_up(store: Store, clock: MutableClock) -> None:
    item = store.outbox.enqueue(LEAD, "meeting_create", _meeting())
    delays: list[float] = []
    for attempt in range(1, OUTBOX_MAX_ATTEMPTS):
        assert [due.id for due in store.outbox.due()] == [item.id]
        updated = store.outbox.mark_failed(item.id, f"HTTP 500 on attempt {attempt}")
        assert (updated.status, updated.attempts, updated.last_error) == (
            "pending",
            attempt,
            f"HTTP 500 on attempt {attempt}",
        )
        assert updated.next_attempt_at is not None
        delay = updated.next_attempt_at - clock.now()
        delays.append(delay.total_seconds())
        clock.advance(delay - timedelta(milliseconds=1))
        assert store.outbox.due() == []
        clock.advance(timedelta(milliseconds=1))
    assert delays == [0.5, 1.0, 2.0, 4.0] + [30.0] * 15

    final = store.outbox.mark_failed(item.id, "HTTP 500 on attempt 20")
    assert (final.status, final.attempts, final.next_attempt_at) == ("failed", 20, None)
    assert final.last_error == "HTTP 500 on attempt 20"
    assert store.outbox.due() == []
    assert store.outbox.backlog() == OutboxBacklog(pending=0, failed=1)
    assert store.outbox.items(status="failed") == [final]
    with pytest.raises(StoreError, match="not pending"):
        store.outbox.mark_failed(item.id, "again")

    requeued = store.outbox.requeue(item.id)
    assert (requeued.status, requeued.attempts, requeued.next_attempt_at) == ("pending", 0, clock.now())
    assert store.outbox.backlog() == OutboxBacklog(pending=1, failed=0)
    done = store.outbox.mark_done(item.id)
    assert (done.status, done.attempts, done.next_attempt_at) == ("done", 1, None)
    assert done.last_error == "HTTP 500 on attempt 20"
    assert store.outbox.backlog() == OutboxBacklog(pending=0, failed=0)
    with pytest.raises(StoreError, match="not failed"):
        store.outbox.requeue(item.id)
    with pytest.raises(StoreError, match="not pending"):
        store.outbox.mark_done(item.id)
    with pytest.raises(StoreError, match="unknown outbox item"):
        store.outbox.mark_done(9999)


def test_due_items_come_oldest_first_within_the_limit(store: Store, clock: MutableClock) -> None:
    first = store.outbox.enqueue(LEAD, "contact_upsert", _meeting("uid-1"))
    clock.advance(timedelta(milliseconds=10))
    second = store.outbox.enqueue(LEAD, "meeting_create", _meeting("uid-2"))
    third = store.outbox.enqueue(LEAD, "meeting_update", _meeting("uid-3"))
    store.outbox.mark_failed(first.id, "timeout")
    assert [i.id for i in store.outbox.due()] == [second.id, third.id]
    assert [i.id for i in store.outbox.due(limit=1)] == [second.id]
    clock.advance(timedelta(seconds=1))
    assert [i.id for i in store.outbox.due()] == [second.id, third.id, first.id]
    assert [i.kind for i in store.outbox.items()] == ["contact_upsert", "meeting_create", "meeting_update"]
    assert store.outbox.items(status="done") == []


def test_long_errors_are_truncated(store: Store) -> None:
    item = store.outbox.enqueue(LEAD, "meeting_create", _meeting())
    updated = store.outbox.mark_failed(item.id, "x" * 5000)
    assert updated.last_error == "x" * 500


# CRM links, handoffs, widget bookings -----------------------------------------------------------


def test_crm_links_merge_ids(store: Store) -> None:
    assert store.crm_links.get("uid-1") is None
    assert store.crm_links.upsert("uid-1", contact_id="101") == CrmLink("uid-1", "101", None)
    assert store.crm_links.upsert("uid-1", meeting_id="9001") == CrmLink("uid-1", "101", "9001")
    assert store.crm_links.upsert("uid-1", meeting_id="9002") == CrmLink("uid-1", "101", "9002")
    assert store.crm_links.get("uid-1") == CrmLink("uid-1", "101", "9002")


def test_handoffs_are_listed_and_marked_delivered(store: Store) -> None:
    first = store.handoffs.create(
        lead_email="Ada.L@example.com",
        summary="Wants a call, calendar unavailable",
        preferred_times_text="Tuesday morning",
        session_id="s1",
    )
    assert (first.lead_email, first.created_at, first.delivered) == (LEAD, T0, False)
    second = store.handoffs.create(lead_email=LEAD, summary="Booking unconfirmed")
    assert second.preferred_times_text == ""
    assert store.handoffs.items() == [first, second]
    assert store.handoffs.for_session("s1") == [first]
    assert store.handoffs.for_session("s2") == []
    store.handoffs.mark_delivered(first.id)
    delivered = store.handoffs.get(first.id)
    assert delivered is not None
    assert delivered.delivered
    assert store.handoffs.items(delivered=True) == [delivered]
    assert store.handoffs.items(delivered=False) == [second]
    assert store.handoffs.items(limit=1) == [delivered]
    with pytest.raises(StoreError, match="unknown handoff"):
        store.handoffs.mark_delivered(999)


def test_widget_bookings_belong_to_their_session(store: Store) -> None:
    _session(store, "w1", "widget")
    _session(store, "w2", "widget")
    store.widget_bookings.add("w1", "uid-1")
    store.widget_bookings.add("w1", "uid-1")
    store.widget_bookings.add("w1", "uid-2")
    assert store.widget_bookings.refs("w1") == ["uid-1", "uid-2"]
    assert store.widget_bookings.owns("w1", "uid-2")
    assert not store.widget_bookings.owns("w1", "uid-3")
    assert not store.widget_bookings.owns("w2", "uid-1")
    with pytest.raises(sqlite3.IntegrityError):
        store.widget_bookings.add("unknown-session", "uid-1")


# Store --------------------------------------------------------------------------------------


def _half_done_turn(store: Store) -> None:
    with store.transaction():
        store.leads.upsert(LEAD)
        store.idem.begin("k1", kind="create", lead_email=LEAD, event_key="1001")
        raise RuntimeError("calendar down")


def test_a_store_transaction_is_atomic_across_repositories(store: Store) -> None:
    with pytest.raises(RuntimeError, match="calendar down"):
        _half_done_turn(store)
    assert store.leads.get(LEAD) is None
    assert store.idem.get("k1") is None
    with store.transaction():
        store.leads.upsert(LEAD)
        store.idem.begin("k1", kind="create", lead_email=LEAD, event_key="1001")
    assert store.leads.get(LEAD) is not None
    assert store.idem.get("k1") is not None


def test_the_store_reopens_an_existing_file(db_path: Path, clock: MutableClock) -> None:
    with Store(db_path, clock=clock) as first:
        first.leads.upsert(LEAD, name="Ada L.")
    with pytest.raises(StoreError, match="closed"):
        first.leads.get(LEAD)
    with Store(db_path, clock=clock) as second:
        lead = second.leads.get(LEAD)
        assert lead is not None
        assert lead.name == "Ada L."
        assert second.path == db_path


def test_a_repository_works_on_a_plain_connection(tmp_path: Path, clock: MutableClock) -> None:
    conn = open_db(tmp_path / "agent.db")
    try:
        leads = LeadsRepo(conn, clock)
        leads.upsert(LEAD)
        assert leads.get(LEAD) is not None
    finally:
        conn.close()


def test_times_are_stored_as_fixed_width_utc_strings(store: Store) -> None:
    _record(store)
    store.locks.acquire("lead:x", "turn-1")
    conn = store.connection()
    created, start = conn.execute("SELECT created_at, start_utc FROM claims").fetchone()
    expires = conn.execute("SELECT expires_at FROM locks").fetchone()[0]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", created)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", expires)
    assert start == "2026-10-06T13:00:00Z"

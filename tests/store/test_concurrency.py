"""Threads racing on the same rows: exactly one winner, no lost updates, waiters wake up.

Every thread uses its own connection (the store's per-thread pattern); SQLite's ``BEGIN IMMEDIATE``
is what serialises them.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from booking_truth.store import LocksRepo, Store

LEAD = "ada.l@example.com"
ROUNDS = 20


def race(store: Store, workers: int, fn: Callable[[int], Any]) -> list[Any]:
    """Start ``workers`` threads at the same moment, each calling ``fn(index)``; return their results."""
    barrier = threading.Barrier(workers)
    results: list[Any] = [None] * workers
    errors: list[BaseException] = []

    def run(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            results[index] = fn(index)
        except BaseException as exc:
            errors.append(exc)
        finally:
            store.close_thread()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    return results


@pytest.mark.parametrize("workers", [2, 8])
def test_threads_racing_for_one_lock_have_exactly_one_winner(store: Store, workers: int) -> None:
    for round_no in range(ROUNDS):
        key = LocksRepo.lead_key(f"lead-{round_no}@example.com")
        won = race(store, workers, lambda i, key=key: store.locks.acquire(key, f"turn-{i}"))
        assert won.count(True) == 1
        holder = store.locks.holder(key)
        assert holder is not None
        assert holder.owner == f"turn-{won.index(True)}"


@pytest.mark.parametrize("workers", [2, 8])
def test_threads_racing_for_one_message_row_have_exactly_one_new(store: Store, workers: int) -> None:
    for round_no in range(ROUNDS):
        message_id = f"m-{round_no}"
        claims = race(store, workers, lambda _i, mid=message_id: store.messages.claim_pending("s1", mid))
        assert sorted(claims) == ["new"] + ["pending"] * (workers - 1)
        row = store.messages.get("s1", message_id)
        assert row is not None
        assert row.status == "pending"


def test_threads_racing_on_one_idempotency_key_have_one_first_writer(store: Store) -> None:
    results = race(
        store,
        8,
        lambda _i: store.idem.begin("k1", kind="create", lead_email=LEAD, event_key="1001")[1],
    )
    assert results.count(None) == 1
    assert results.count("pending") == 7


def test_concurrent_increments_are_not_lost(store: Store) -> None:
    per_thread = 5

    def bump(_i: int) -> list[int]:
        return [store.generations.increment(LEAD, "1001") for _ in range(per_thread)]

    values = [value for chunk in race(store, 8, bump) for value in chunk]
    assert sorted(values) == list(range(1, 8 * per_thread + 1))
    assert store.generations.current(LEAD, "1001") == 8 * per_thread


def test_concurrent_history_appends_get_distinct_sequence_numbers(store: Store) -> None:
    store.sessions.get_or_create("s1", lead_email=LEAD, channel="api")

    def append(i: int) -> list[int]:
        return [store.history.append("s1", "user", {"thread": i, "n": n}) for n in range(10)]

    seqs = [seq for chunk in race(store, 4, append) for seq in chunk]
    assert sorted(seqs) == list(range(40))
    assert [e.seq for e in store.history.for_session("s1")] == list(range(40))


def test_a_waiter_gets_the_lock_once_the_holder_releases(store: Store) -> None:
    key = LocksRepo.lead_key(LEAD)
    assert store.locks.acquire(key, "turn-1")

    def release_later() -> None:
        time.sleep(0.2)
        store.locks.release(key, "turn-1")
        store.close_thread()

    releaser = threading.Thread(target=release_later)
    started = time.monotonic()
    releaser.start()
    got = race(store, 1, lambda _i: store.locks.wait_acquire(key, "turn-2", timeout_s=5, poll_s=0.02))
    elapsed = time.monotonic() - started
    releaser.join()
    assert got == [True]
    assert 0.15 <= elapsed < 5
    holder = store.locks.holder(key)
    assert holder is not None
    assert holder.owner == "turn-2"


def test_a_waiter_gives_up_after_its_timeout(store: Store) -> None:
    key = LocksRepo.lead_key(LEAD)
    assert store.locks.acquire(key, "turn-1")
    started = time.monotonic()
    got = race(store, 1, lambda _i: store.locks.wait_acquire(key, "turn-2", timeout_s=0.2, poll_s=0.02))
    elapsed = time.monotonic() - started
    assert got == [False]
    assert 0.2 <= elapsed < 3
    holder = store.locks.holder(key)
    assert holder is not None
    assert holder.owner == "turn-1"


def test_a_duplicate_delivery_waits_for_its_twin(store: Store) -> None:
    claimed = threading.Event()
    response = {"reply": "Booked.", "quick_replies": []}

    def first(_i: int) -> str:
        outcome = store.messages.claim_pending("s1", "m1")
        claimed.set()
        time.sleep(0.2)
        store.messages.complete("s1", "m1", response)
        return outcome

    def twin(_i: int) -> tuple[str, dict[str, Any] | None]:
        assert claimed.wait(timeout=5)
        outcome = store.messages.claim_pending("s1", "m1")
        return outcome, store.messages.wait_done("s1", "m1", timeout_s=5, poll_s=0.02)

    results = race(store, 2, lambda i: first(i) if i == 0 else twin(i))
    assert results == ["new", ("pending", response)]


async def test_coroutines_wait_without_blocking_the_loop(store: Store) -> None:
    key = LocksRepo.lead_key(LEAD)
    assert store.locks.acquire(key, "turn-1")
    assert store.messages.claim_pending("s1", "m1") == "new"
    loop = asyncio.get_running_loop()
    loop.call_later(0.1, store.locks.release, key, "turn-1")
    loop.call_later(0.15, store.messages.complete, "s1", "m1", {"reply": "ok"})
    got_lock, stored = await asyncio.gather(
        store.locks.await_acquire(key, "turn-2", timeout_s=5, poll_s=0.02),
        store.messages.await_done("s1", "m1", timeout_s=5, poll_s=0.02),
    )
    assert got_lock is True
    assert stored == {"reply": "ok"}
    assert await store.locks.await_acquire(key, "turn-3", timeout_s=0.1, poll_s=0.02) is False

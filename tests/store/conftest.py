"""Shared fixtures for store tests: a fresh database file per test and a clock the test controls."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from booking_truth.store import Store
from booking_truth.timeutil import MutableClock

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("BT_LLM_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock(T0)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "agent.db"


@pytest.fixture
def store(db_path: Path, clock: MutableClock) -> Iterator[Store]:
    opened = Store(db_path, clock=clock)
    try:
        yield opened
    finally:
        opened.close()

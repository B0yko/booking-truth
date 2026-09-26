"""Connections, pragmas, migrations and ``BEGIN IMMEDIATE`` transactions."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from booking_truth.store import (
    BUSY_TIMEOUT_MS,
    SCHEMA_VERSION,
    StoreError,
    ThreadConnections,
    immediate,
    migrate,
    open_db,
    schema_version,
)

TABLES = {
    "leads",
    "sessions",
    "messages",
    "history",
    "slot_lists",
    "claims",
    "idem",
    "generations",
    "locks",
    "outbox",
    "crm_links",
    "handoffs",
    "widget_bookings",
    "trace_steps",
    "schema_migrations",
}


def _keys(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT key FROM locks")}


def _insert(conn: sqlite3.Connection, key: str) -> None:
    conn.execute(
        "INSERT INTO locks (key, owner, expires_at) VALUES (?, 'o', '2026-10-01T12:00:00.000Z')", (key,)
    )


def _fail_inside(conn: sqlite3.Connection, key: str, exc: Exception) -> None:
    with immediate(conn):
        _insert(conn, key)
        raise exc


def _nested_then_fail(conn: sqlite3.Connection) -> None:
    with immediate(conn):
        with immediate(conn):
            _insert(conn, "nested")
        raise RuntimeError("outer failed")


def test_open_db_creates_parents_and_sets_pragmas(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "agent.db"
    conn = open_db(path)
    try:
        assert path.is_file()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS == 5000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.isolation_level is None
        row = conn.execute("SELECT 1 AS one").fetchone()
        assert isinstance(row, sqlite3.Row)
        assert row["one"] == 1
    finally:
        conn.close()


def test_migrations_create_every_table_once(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    conn = open_db(path)
    try:
        assert schema_version(conn) == SCHEMA_VERSION
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert names >= TABLES
        applied = [row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
        assert applied == list(range(1, SCHEMA_VERSION + 1))
    finally:
        conn.close()
    again = open_db(path)
    try:
        assert migrate(again) == SCHEMA_VERSION
        assert again.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION
    finally:
        again.close()


def test_open_without_migrations_leaves_the_file_empty(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "agent.db", run_migrations=False)
    try:
        assert schema_version(conn) == 0
        assert migrate(conn) == SCHEMA_VERSION
    finally:
        conn.close()


def test_a_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    conn = open_db(path)
    conn.execute(
        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, '2026-10-01T12:00:00.000Z')",
        (SCHEMA_VERSION + 1,),
    )
    conn.close()
    with pytest.raises(StoreError, match="newer"):
        open_db(path)


def test_immediate_commits_on_success_and_rolls_back_on_error(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "agent.db")
    try:
        with immediate(conn):
            assert conn.in_transaction
            _insert(conn, "kept")
        assert not conn.in_transaction
        with pytest.raises(RuntimeError, match="boom"):
            _fail_inside(conn, "dropped", RuntimeError("boom"))
        assert not conn.in_transaction
        assert _keys(conn) == {"kept"}
    finally:
        conn.close()


def test_nested_immediate_is_a_savepoint(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "agent.db")
    try:
        with immediate(conn):
            _insert(conn, "outer")
            with pytest.raises(ValueError, match="inner"):
                _fail_inside(conn, "inner", ValueError("inner"))
            with immediate(conn):
                _insert(conn, "inner-kept")
            assert conn.in_transaction
        assert _keys(conn) == {"outer", "inner-kept"}
    finally:
        conn.close()


def test_an_outer_rollback_undoes_nested_blocks(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "agent.db")
    try:
        with pytest.raises(RuntimeError, match="outer failed"):
            _nested_then_fail(conn)
        assert _keys(conn) == set()
    finally:
        conn.close()


def test_immediate_takes_the_write_lock_up_front(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    writer = open_db(path)
    other = open_db(path)
    try:
        other.execute("PRAGMA busy_timeout = 50")
        with immediate(writer):
            _insert(writer, "a")
            with pytest.raises(sqlite3.OperationalError, match="locked"), immediate(other):
                pass
            # WAL: readers are not blocked and see the last committed state.
            assert _keys(other) == set()
        assert _keys(other) == {"a"}
        with immediate(other):
            _insert(other, "b")
        assert _keys(writer) == {"a", "b"}
    finally:
        writer.close()
        other.close()


def test_thread_connections_give_each_thread_its_own_connection(tmp_path: Path) -> None:
    conns = ThreadConnections(tmp_path / "agent.db")
    main = conns.get()
    assert conns.get() is main
    seen: list[sqlite3.Connection] = []

    def worker() -> None:
        seen.append(conns.get())
        seen.append(conns.get())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen[0] is seen[1]
    assert seen[0] is not main
    conns.close()
    with pytest.raises(StoreError, match="closed"):
        conns.get()
    with pytest.raises(sqlite3.ProgrammingError):
        main.execute("SELECT 1")


def test_close_current_reopens_on_next_use(tmp_path: Path) -> None:
    conns = ThreadConnections(tmp_path / "agent.db")
    try:
        first = conns.get()
        conns.close_current()
        with pytest.raises(sqlite3.ProgrammingError):
            first.execute("SELECT 1")
        second = conns.get()
        assert second is not first
        assert second.execute("SELECT COUNT(*) FROM locks").fetchone()[0] == 0
    finally:
        conns.close()

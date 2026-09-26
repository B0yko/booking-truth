"""SQLite connections, schema migrations and write transactions for the agent's store.

Connections
    :func:`open_db` opens one connection in autocommit mode (``isolation_level=None``) with WAL
    journaling, ``busy_timeout`` 5000 ms, foreign keys on and :class:`sqlite3.Row` rows. The library
    never opens a transaction implicitly; every read-modify-write goes through :func:`immediate`.

Threads
    A :class:`sqlite3.Connection` must not be used by two threads at the same time. The store keeps
    **one connection per thread** (:class:`ThreadConnections`): each thread lazily opens its own
    connection to the same file, and SQLite's file locking (WAL plus ``BEGIN IMMEDIATE``) serialises
    writers across threads and processes. Coroutines on one event loop share that thread's
    connection, which is safe because an :func:`immediate` block runs to completion without yielding:
    never ``await`` inside one. Store calls are synchronous; a write made while another connection
    holds the write lock blocks its thread (at most ``busy_timeout``) until that short transaction
    ends. Waiting for a lease or a twin's response is done by polling, never inside a transaction.

Transactions
    :func:`immediate` starts ``BEGIN IMMEDIATE``, which takes the database write lock up front, so two
    writers never both read a row and then both write it (the lock, dedupe and idempotency rows rely on
    this). A writer that finds the lock taken waits up to ``busy_timeout``. Nested use becomes a
    savepoint inside the outer transaction.

Migrations
    ``schema_migrations`` records every applied version. :func:`migrate` applies the missing ones in
    order, each in its own ``BEGIN IMMEDIATE`` transaction (SQLite DDL is transactional), and refuses a
    database written by a newer schema.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from booking_truth.timeutil import Clock, SystemClock, iso_ms_z

BUSY_TIMEOUT_MS = 5000
_SAVEPOINT = "bt_nested"


class StoreError(RuntimeError):
    """The store cannot do what was asked (newer schema, invalid state transition, missing row)."""


# One tuple of statements per schema version; index 0 is version 1. Never edit a released entry:
# append a new version instead.
MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """CREATE TABLE leads (
            email TEXT PRIMARY KEY,
            name TEXT,
            tz_zone TEXT,
            tz_source TEXT,
            tz_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (tz_confirmed IN (0, 1)),
            updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            lead_email TEXT NOT NULL,
            channel TEXT NOT NULL CHECK (channel IN ('api', 'widget', 'webhook')),
            token_hash TEXT,
            turns INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            ended INTEGER NOT NULL DEFAULT 0 CHECK (ended IN (0, 1))
        )""",
        "CREATE INDEX sessions_lead ON sessions (lead_email)",
        """CREATE TABLE messages (
            session_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('pending', 'done')),
            response_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (session_id, message_id)
        )""",
        """CREATE TABLE history (
            session_id TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            content_json TEXT NOT NULL,
            PRIMARY KEY (session_id, seq)
        )""",
        """CREATE TABLE slot_lists (
            id TEXT PRIMARY KEY,
            lead_email TEXT NOT NULL,
            session_id TEXT,
            created_at TEXT NOT NULL,
            slots_json TEXT NOT NULL,
            zone TEXT NOT NULL
        )""",
        "CREATE INDEX slot_lists_lead ON slot_lists (lead_email, created_at)",
        """CREATE TABLE claims (
            id INTEGER PRIMARY KEY,
            lead_email TEXT NOT NULL,
            event_key TEXT NOT NULL,
            action TEXT NOT NULL CHECK (action IN ('booked', 'rescheduled', 'cancelled')),
            booking_ref TEXT NOT NULL,
            start_utc TEXT NOT NULL,
            end_utc TEXT NOT NULL,
            zone TEXT,
            session_id TEXT,
            channel TEXT,
            status TEXT NOT NULL CHECK (status IN ('verified', 'unverified')),
            created_at TEXT NOT NULL,
            voided_at TEXT
        )""",
        "CREATE INDEX claims_lead ON claims (lead_email, event_key)",
        "CREATE INDEX claims_ref ON claims (booking_ref)",
        """CREATE TABLE idem (
            key TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('create', 'reschedule', 'cancel')),
            lead_email TEXT NOT NULL,
            event_key TEXT NOT NULL,
            slot_start_utc TEXT,
            generation INTEGER,
            status TEXT NOT NULL CHECK (status IN ('pending', 'committed', 'failed', 'adopted')),
            booking_ref TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE generations (
            lead_email TEXT NOT NULL,
            event_key TEXT NOT NULL,
            generation INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (lead_email, event_key)
        )""",
        """CREATE TABLE locks (
            key TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )""",
        """CREATE TABLE outbox (
            id INTEGER PRIMARY KEY,
            lead_email TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('pending', 'done', 'failed')),
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        "CREATE INDEX outbox_due ON outbox (status, next_attempt_at)",
        """CREATE TABLE crm_links (
            booking_ref TEXT PRIMARY KEY,
            contact_id TEXT,
            meeting_id TEXT
        )""",
        """CREATE TABLE handoffs (
            id INTEGER PRIMARY KEY,
            lead_email TEXT NOT NULL,
            session_id TEXT,
            summary TEXT NOT NULL,
            preferred_times_text TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            delivered INTEGER NOT NULL DEFAULT 0 CHECK (delivered IN (0, 1))
        )""",
        """CREATE TABLE widget_bookings (
            session_id TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
            booking_ref TEXT NOT NULL,
            PRIMARY KEY (session_id, booking_ref)
        )""",
        """CREATE TABLE trace_steps (
            session_id TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
            i INTEGER NOT NULL,
            step_json TEXT NOT NULL,
            PRIMARY KEY (session_id, i)
        )""",
    ),
)

SCHEMA_VERSION = len(MIGRATIONS)


def open_db(
    path: Path | str,
    *,
    run_migrations: bool = True,
    check_same_thread: bool = True,
    clock: Clock | None = None,
) -> sqlite3.Connection:
    """Open (and create) the database at ``path`` and bring its schema up to date.

    Parent directories are created. ``check_same_thread=False`` lets another thread *close* the
    connection (:class:`ThreadConnections` needs that); it never makes concurrent use safe.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        target,
        timeout=BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,
        check_same_thread=check_same_thread,
    )
    try:
        conn.row_factory = sqlite3.Row
        # busy_timeout first, so that switching the journal mode also waits for other connections.
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            raise StoreError(f"could not switch {target.name} to WAL journaling (got {mode!r})")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        if run_migrations:
            migrate(conn, clock=clock)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def immediate(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run the block in a ``BEGIN IMMEDIATE`` transaction: commit on success, roll back on error.

    Inside an open transaction the block becomes a savepoint, so repository methods compose into one
    atomic unit when the caller wraps them in an outer ``immediate``.
    """
    if conn.in_transaction:
        conn.execute(f"SAVEPOINT {_SAVEPOINT}")
        try:
            yield conn
        except BaseException:
            conn.execute(f"ROLLBACK TO {_SAVEPOINT}")
            conn.execute(f"RELEASE {_SAVEPOINT}")
            raise
        conn.execute(f"RELEASE {_SAVEPOINT}")
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    try:
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def schema_version(conn: sqlite3.Connection) -> int:
    """The highest applied migration, 0 for an empty database."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if exists is None:
        return 0
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
    return int(row[0])


def migrate(conn: sqlite3.Connection, *, clock: Clock | None = None) -> int:
    """Apply every missing migration and return the resulting schema version."""
    current = schema_version(conn)
    if current > SCHEMA_VERSION:
        raise StoreError(
            f"database schema version {current} is newer than this version of booking-truth "
            f"supports ({SCHEMA_VERSION}); upgrade booking-truth or use another BT_DB_PATH"
        )
    if current == SCHEMA_VERSION:
        return current
    now = (clock or SystemClock()).now()
    with immediate(conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
    for version, statements in enumerate(MIGRATIONS, start=1):
        with immediate(conn):
            # Re-read inside the write lock: another connection may have migrated meanwhile.
            if schema_version(conn) >= version:
                continue
            for statement in statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, iso_ms_z(now)),
            )
    return schema_version(conn)


class ThreadConnections:
    """One connection per thread to one database file.

    The first connection is opened (and the schema migrated) in the constructor, so a bad path or a
    newer schema fails at start-up. :meth:`close` closes the connections of every thread; call it at
    shutdown, when no thread is using the store any more.
    """

    def __init__(self, path: Path | str, *, clock: Clock | None = None) -> None:
        self.path = Path(path).expanduser()
        self._local = threading.local()
        self._guard = threading.Lock()
        self._open: list[sqlite3.Connection] = []
        self._closed = False
        first = open_db(self.path, check_same_thread=False, clock=clock)
        self._register(first)

    def _register(self, conn: sqlite3.Connection) -> None:
        with self._guard:
            if self._closed:
                conn.close()
                raise StoreError("the store is closed")
            self._open.append(conn)
        self._local.conn = conn

    def get(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if self._closed:
            raise StoreError("the store is closed")
        conn = open_db(self.path, run_migrations=False, check_same_thread=False)
        self._register(conn)
        return conn

    def close_current(self) -> None:
        """Close the calling thread's connection (a worker thread that is about to exit)."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            return
        self._local.conn = None
        with self._guard:
            if conn in self._open:
                self._open.remove(conn)
        conn.close()

    def close(self) -> None:
        with self._guard:
            self._closed = True
            conns, self._open = self._open, []
        for conn in conns:
            conn.close()
        self._local = threading.local()

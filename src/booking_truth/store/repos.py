"""One small repository per table of the agent's SQLite store, and the :class:`Store` that wires them.

Conventions
    - Emails are stored normalised (:func:`normalize_email`); every method normalises its input.
    - Every time is stored as an ISO 8601 UTC string with a ``Z`` suffix, one fixed-width format per
      column, so SQL string comparison is time order. Bookkeeping stamps (``created_at``,
      ``expires_at``, ``next_attempt_at`` ...) carry milliseconds (:func:`~booking_truth.timeutil.iso_ms_z`);
      booking and slot instants carry seconds (:func:`~booking_truth.timeutil.iso_z`). Methods take and
      return aware ``datetime`` values, never the strings.
    - "Now" comes from the injected :class:`~booking_truth.timeutil.Clock`, so leases, TTLs and the outbox
      backoff are testable without sleeping. Waiting helpers poll on the real monotonic clock.
    - Every read-modify-write runs inside :func:`~booking_truth.store.db.immediate`, so it is atomic
      across threads and processes. Methods called inside an outer ``immediate`` (see
      :meth:`Store.transaction`) join that transaction.
    - JSON columns hold compact JSON; values must be JSON-serialisable (no NaN).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from booking_truth.store.db import StoreError, ThreadConnections, immediate
from booking_truth.timeutil import Clock, SystemClock, iso_ms_z, iso_z, parse_iso

Channel = Literal["api", "widget", "webhook"]
MessageClaim = Literal["new", "pending", "done"]
MessageStatus = Literal["pending", "done"]
ClaimAction = Literal["booked", "rescheduled", "cancelled"]
ClaimStatus = Literal["verified", "unverified"]
IdemKind = Literal["create", "reschedule", "cancel"]
IdemStatus = Literal["pending", "committed", "failed", "adopted"]
OutboxStatus = Literal["pending", "done", "failed"]

Connector = sqlite3.Connection | Callable[[], sqlite3.Connection]

#: Lead lock lease and how long a concurrent turn waits for it (spec: 30 s lease, 10 s wait).
LEAD_LOCK_LEASE_S = 30.0
LEAD_LOCK_WAIT_S = 10.0
#: How long a duplicate delivery waits for its in-flight twin.
DEDUPE_WAIT_S = 30.0
POLL_S = 0.05

#: Outbox retry delays after the 1st..4th failed attempt, then every 30 s; failed after 20 attempts.
OUTBOX_BACKOFF_S: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
OUTBOX_STEADY_S = 30.0
OUTBOX_MAX_ATTEMPTS = 20
_MAX_ERROR_CHARS = 500

# Helpers -------------------------------------------------------------------------------------


def normalize_email(email: str) -> str:
    """The form every table keys leads by: surrounding whitespace removed, lower case."""
    return email.strip().lower()


def backoff_delay(attempts: int) -> float:
    """Seconds until the next outbox attempt after ``attempts`` failed attempts (``attempts >= 1``)."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if attempts <= len(OUTBOX_BACKOFF_S):
        return OUTBOX_BACKOFF_S[attempts - 1]
    return OUTBOX_STEADY_S


def wait_until[T](check: Callable[[], T | None], *, timeout_s: float, poll_s: float = POLL_S) -> T | None:
    """Call ``check`` until it returns something other than ``None`` or ``timeout_s`` passes.

    ``check`` runs at least once. Blocks the calling thread; from a coroutine use :func:`await_until`.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        result = check()
        if result is not None:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(poll_s, remaining))


async def await_until[T](
    check: Callable[[], T | None], *, timeout_s: float, poll_s: float = POLL_S
) -> T | None:
    """:func:`wait_until` for coroutines: sleeps with ``asyncio.sleep`` between checks."""
    deadline = time.monotonic() + timeout_s
    while True:
        result = check()
        if result is not None:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(poll_s, remaining))


def _stamp(dt: datetime) -> str:
    return iso_ms_z(dt)


def _instant(dt: datetime) -> str:
    return iso_z(dt)


def _dt(value: str) -> datetime:
    return parse_iso(value)


def _opt_dt(value: str | None) -> datetime | None:
    return None if value is None else parse_iso(value)


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _loads(text: str) -> Any:
    return json.loads(text)


def _rows(cursor: sqlite3.Cursor) -> list[sqlite3.Row]:
    """Every row of a ``RETURNING`` statement; reading to the end also finishes the statement."""
    return cursor.fetchall()


def _returned(cursor: sqlite3.Cursor) -> sqlite3.Row:
    rows = _rows(cursor)
    if len(rows) != 1:
        raise StoreError(f"expected one row back, got {len(rows)}")
    return rows[0]


class _Repo:
    """Base: a connection source (a connection, or a per-thread getter) and a clock."""

    def __init__(self, source: Connector, clock: Clock | None = None) -> None:
        if isinstance(source, sqlite3.Connection):
            conn = source

            def fixed() -> sqlite3.Connection:
                return conn

            self._connect: Callable[[], sqlite3.Connection] = fixed
        else:
            self._connect = source
        self._clock: Clock = clock or SystemClock()

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._connect()

    def _now(self) -> datetime:
        return self._clock.now()


# Leads ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Lead:
    email: str
    name: str | None
    tz_zone: str | None
    tz_source: str | None
    tz_confirmed: bool
    updated_at: datetime


def _lead(row: sqlite3.Row) -> Lead:
    return Lead(
        email=row["email"],
        name=row["name"],
        tz_zone=row["tz_zone"],
        tz_source=row["tz_source"],
        tz_confirmed=bool(row["tz_confirmed"]),
        updated_at=_dt(row["updated_at"]),
    )


class LeadsRepo(_Repo):
    """``leads``: one row per normalised email with the lead's time zone state."""

    def get(self, email: str) -> Lead | None:
        row = self._conn.execute("SELECT * FROM leads WHERE email = ?", (normalize_email(email),)).fetchone()
        return None if row is None else _lead(row)

    def upsert(self, email: str, *, name: str | None = None) -> Lead:
        """Create the lead, or refresh it; a ``None`` name keeps the stored one."""
        key = normalize_email(email)
        with immediate(self._conn) as conn:
            conn.execute(
                "INSERT INTO leads (email, name, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT (email) DO UPDATE SET name = COALESCE(excluded.name, leads.name), "
                "updated_at = excluded.updated_at",
                (key, name, _stamp(self._now())),
            )
            row = conn.execute("SELECT * FROM leads WHERE email = ?", (key,)).fetchone()
        return _lead(row)

    def set_zone(self, email: str, zone: str | None, *, source: str | None, confirmed: bool = False) -> Lead:
        """Store the lead's IANA zone and where it came from (``stated``, ``hint``, ...)."""
        key = normalize_email(email)
        with immediate(self._conn) as conn:
            conn.execute(
                "INSERT INTO leads (email, tz_zone, tz_source, tz_confirmed, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (email) DO UPDATE SET tz_zone = excluded.tz_zone, "
                "tz_source = excluded.tz_source, tz_confirmed = excluded.tz_confirmed, "
                "updated_at = excluded.updated_at",
                (key, zone, source, int(confirmed), _stamp(self._now())),
            )
            row = conn.execute("SELECT * FROM leads WHERE email = ?", (key,)).fetchone()
        return _lead(row)


# Sessions ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    lead_email: str
    channel: Channel
    token_hash: str | None
    turns: int
    created_at: datetime
    ended: bool


def _session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        lead_email=row["lead_email"],
        channel=cast(Channel, row["channel"]),
        token_hash=row["token_hash"],
        turns=int(row["turns"]),
        created_at=_dt(row["created_at"]),
        ended=bool(row["ended"]),
    )


class SessionsRepo(_Repo):
    """``sessions``: one row per conversation, with its turn count."""

    def get(self, session_id: str) -> Session | None:
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return None if row is None else _session(row)

    def get_or_create(
        self, session_id: str, *, lead_email: str, channel: Channel, token_hash: str | None = None
    ) -> Session:
        """The existing session as stored (the caller checks its lead and channel), else a new one."""
        with immediate(self._conn) as conn:
            conn.execute(
                "INSERT INTO sessions (id, lead_email, channel, token_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (id) DO NOTHING",
                (session_id, normalize_email(lead_email), channel, token_hash, _stamp(self._now())),
            )
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return _session(row)

    def set_token_hash(self, session_id: str, token_hash: str) -> None:
        self._require(
            self._conn.execute("UPDATE sessions SET token_hash = ? WHERE id = ?", (token_hash, session_id)),
            session_id,
        )

    def increment_turns(self, session_id: str) -> int:
        """Count one more turn and return the new total."""
        rows = _rows(
            self._conn.execute(
                "UPDATE sessions SET turns = turns + 1 WHERE id = ? RETURNING turns", (session_id,)
            )
        )
        if not rows:
            raise StoreError(f"unknown session {session_id!r}")
        return int(rows[0][0])

    def end(self, session_id: str) -> None:
        self._require(
            self._conn.execute("UPDATE sessions SET ended = 1 WHERE id = ?", (session_id,)), session_id
        )

    @staticmethod
    def _require(cursor: sqlite3.Cursor, session_id: str) -> None:
        if cursor.rowcount != 1:
            raise StoreError(f"unknown session {session_id!r}")


# Messages (inbound dedupe) --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MessageRow:
    session_id: str
    message_id: str
    status: MessageStatus
    response: dict[str, Any] | None
    created_at: datetime


class MessagesRepo(_Repo):
    """``messages``: one row per inbound ``(session_id, message_id)``.

    A turn first calls :meth:`claim_pending`. ``new``: this call inserted the ``pending`` row and runs
    the turn, then stores the response with :meth:`complete` (or :meth:`discard` on failure).
    ``done``: return :meth:`get`'s stored response. ``pending``: a twin is in flight; wait with
    :meth:`wait_done` / :meth:`await_done`.
    """

    def claim_pending(
        self, session_id: str, message_id: str, *, stale_after_s: float | None = None
    ) -> MessageClaim:
        """Atomically insert a ``pending`` row unless one exists.

        With ``stale_after_s``, a ``pending`` row older than that (its turn died without cleaning up)
        is taken over and the call returns ``new``.
        """
        now = self._now()
        with immediate(self._conn) as conn:
            row = conn.execute(
                "SELECT status, created_at FROM messages WHERE session_id = ? AND message_id = ?",
                (session_id, message_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO messages (session_id, message_id, status, created_at) "
                    "VALUES (?, ?, 'pending', ?)",
                    (session_id, message_id, _stamp(now)),
                )
                return "new"
            if row["status"] == "done":
                return "done"
            if stale_after_s is not None and _dt(row["created_at"]) + timedelta(seconds=stale_after_s) <= now:
                conn.execute(
                    "UPDATE messages SET created_at = ? WHERE session_id = ? AND message_id = ?",
                    (_stamp(now), session_id, message_id),
                )
                return "new"
            return "pending"

    def get(self, session_id: str, message_id: str) -> MessageRow | None:
        row = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? AND message_id = ?", (session_id, message_id)
        ).fetchone()
        if row is None:
            return None
        return MessageRow(
            session_id=row["session_id"],
            message_id=row["message_id"],
            status=cast(MessageStatus, row["status"]),
            response=None if row["response_json"] is None else _loads(row["response_json"]),
            created_at=_dt(row["created_at"]),
        )

    def complete(self, session_id: str, message_id: str, response: Mapping[str, Any]) -> None:
        """Store the response and mark the message ``done`` (creating the row if needed)."""
        self._conn.execute(
            "INSERT INTO messages (session_id, message_id, status, response_json, created_at) "
            "VALUES (?, ?, 'done', ?, ?) "
            "ON CONFLICT (session_id, message_id) DO UPDATE SET status = 'done', "
            "response_json = excluded.response_json",
            (session_id, message_id, _dumps(dict(response)), _stamp(self._now())),
        )

    def discard(self, session_id: str, message_id: str) -> bool:
        """Delete a ``pending`` row after its turn failed, so a retry runs again. Done rows stay."""
        cursor = self._conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND message_id = ? AND status = 'pending'",
            (session_id, message_id),
        )
        return cursor.rowcount == 1

    def _poll(self, session_id: str, message_id: str) -> tuple[dict[str, Any] | None] | None:
        row = self.get(session_id, message_id)
        if row is None:
            return (None,)  # the twin failed and discarded its row: stop waiting
        if row.status == "done":
            return (row.response,)
        return None

    def wait_done(
        self, session_id: str, message_id: str, *, timeout_s: float = DEDUPE_WAIT_S, poll_s: float = POLL_S
    ) -> dict[str, Any] | None:
        """Block until the twin's response is stored and return it.

        ``None`` when the wait timed out or the row disappeared (the twin failed); the caller may call
        :meth:`claim_pending` again.
        """
        result = wait_until(lambda: self._poll(session_id, message_id), timeout_s=timeout_s, poll_s=poll_s)
        return None if result is None else result[0]

    async def await_done(
        self, session_id: str, message_id: str, *, timeout_s: float = DEDUPE_WAIT_S, poll_s: float = POLL_S
    ) -> dict[str, Any] | None:
        """:meth:`wait_done` for coroutines."""
        result = await await_until(
            lambda: self._poll(session_id, message_id), timeout_s=timeout_s, poll_s=poll_s
        )
        return None if result is None else result[0]


# History -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    session_id: str
    seq: int
    role: str
    content: Any


class HistoryRepo(_Repo):
    """``history``: the session's model-facing message list, numbered from 0."""

    def append(self, session_id: str, role: str, content: Any) -> int:
        """Append one entry and return its sequence number."""
        return self.extend(session_id, [(role, content)])[0]

    def extend(self, session_id: str, entries: Iterable[tuple[str, Any]]) -> list[int]:
        """Append entries in order, atomically; returns their sequence numbers."""
        items = [(role, _dumps(content)) for role, content in entries]
        seqs: list[int] = []
        with immediate(self._conn) as conn:
            nxt = int(
                conn.execute(
                    "SELECT COALESCE(MAX(seq) + 1, 0) FROM history WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
            )
            for role, content_json in items:
                conn.execute(
                    "INSERT INTO history (session_id, seq, role, content_json) VALUES (?, ?, ?, ?)",
                    (session_id, nxt, role, content_json),
                )
                seqs.append(nxt)
                nxt += 1
        return seqs

    def for_session(self, session_id: str) -> list[HistoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM history WHERE session_id = ? ORDER BY seq", (session_id,)
        ).fetchall()
        return [
            HistoryEntry(
                session_id=row["session_id"],
                seq=int(row["seq"]),
                role=row["role"],
                content=_loads(row["content_json"]),
            )
            for row in rows
        ]


# Slot lists ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlotList:
    """A successful slot list exactly as shown to the lead. Each slot is a dict with a ``slot_id``."""

    id: str
    lead_email: str
    session_id: str | None
    created_at: datetime
    slots: tuple[dict[str, Any], ...]
    zone: str

    def find(self, slot_id: str) -> dict[str, Any] | None:
        for slot in self.slots:
            if slot.get("slot_id") == slot_id:
                return slot
        return None

    def expires_at(self, ttl_s: float) -> datetime:
        return self.created_at + timedelta(seconds=ttl_s)


def _slot_list(row: sqlite3.Row) -> SlotList:
    return SlotList(
        id=row["id"],
        lead_email=row["lead_email"],
        session_id=row["session_id"],
        created_at=_dt(row["created_at"]),
        slots=tuple(_loads(row["slots_json"])),
        zone=row["zone"],
    )


class SlotListsRepo(_Repo):
    """``slot_lists``: successful lists only (an unavailable lookup stores nothing).

    Only the latest list of a lead counts, and only within its TTL: a list is fresh while
    ``now < created_at + ttl_s``.
    """

    def save(
        self,
        lead_email: str,
        slots: Sequence[Mapping[str, Any]],
        *,
        zone: str,
        session_id: str | None = None,
        list_id: str | None = None,
    ) -> SlotList:
        """Store a list as the lead's latest; ``list_id`` defaults to a random hex id."""
        if any("slot_id" not in slot for slot in slots):
            raise ValueError("every slot needs a 'slot_id'")
        ident = list_id or uuid.uuid4().hex
        payload = [dict(slot) for slot in slots]
        with immediate(self._conn) as conn:
            conn.execute(
                "INSERT INTO slot_lists (id, lead_email, session_id, created_at, slots_json, zone) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ident, normalize_email(lead_email), session_id, _stamp(self._now()), _dumps(payload), zone),
            )
            row = conn.execute("SELECT * FROM slot_lists WHERE id = ?", (ident,)).fetchone()
        return _slot_list(row)

    def latest(self, lead_email: str, *, ttl_s: float | None = None) -> SlotList | None:
        """The lead's most recent list; ``None`` when there is none or it is older than ``ttl_s``."""
        row = self._conn.execute(
            "SELECT * FROM slot_lists WHERE lead_email = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (normalize_email(lead_email),),
        ).fetchone()
        if row is None:
            return None
        found = _slot_list(row)
        if ttl_s is not None and self._now() >= found.expires_at(ttl_s):
            return None
        return found

    def find_slot(self, lead_email: str, slot_id: str, *, ttl_s: float | None) -> dict[str, Any] | None:
        """The slot with ``slot_id`` in the lead's latest fresh list, else ``None``."""
        latest = self.latest(lead_email, ttl_s=ttl_s)
        return None if latest is None else latest.find(slot_id)

    def prune(self, *, older_than_s: float) -> int:
        """Delete lists older than ``older_than_s``; returns the number removed."""
        cutoff = self._now() - timedelta(seconds=older_than_s)
        return self._conn.execute("DELETE FROM slot_lists WHERE created_at < ?", (_stamp(cutoff),)).rowcount


# Claims ledger -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    id: int
    lead_email: str
    event_key: str
    action: ClaimAction
    booking_ref: str
    start_utc: datetime
    end_utc: datetime
    zone: str | None
    session_id: str | None
    channel: str | None
    status: ClaimStatus
    created_at: datetime
    voided_at: datetime | None

    @property
    def is_current_booking(self) -> bool:
        """A verified booking or reschedule that no later cancel or reschedule has voided."""
        return self.status == "verified" and self.action != "cancelled" and self.voided_at is None


def _entry(row: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        id=int(row["id"]),
        lead_email=row["lead_email"],
        event_key=row["event_key"],
        action=cast(ClaimAction, row["action"]),
        booking_ref=row["booking_ref"],
        start_utc=_dt(row["start_utc"]),
        end_utc=_dt(row["end_utc"]),
        zone=row["zone"],
        session_id=row["session_id"],
        channel=row["channel"],
        status=cast(ClaimStatus, row["status"]),
        created_at=_dt(row["created_at"]),
        voided_at=_opt_dt(row["voided_at"]),
    )


_ACTIONS: frozenset[str] = frozenset(("booked", "rescheduled", "cancelled"))
_CLAIM_STATUSES: frozenset[str] = frozenset(("verified", "unverified"))


class ClaimsRepo(_Repo):
    """``claims``: the ledger of calendar writes the agent may talk about.

    A ``verified`` entry is written only after a read-back confirmed the write; an ``unverified`` one
    records a write whose read-back failed (the agent hands off). A booking ref has at most one live
    verified booking entry: recording a verified entry for a ref voids the earlier live ones, a verified
    reschedule also voids ``previous_ref`` (Cal.com issues a new uid), and a verified cancel voids the
    ref's booking entries.
    """

    def record(
        self,
        *,
        lead_email: str,
        event_key: str,
        action: ClaimAction,
        booking_ref: str,
        start_utc: datetime,
        end_utc: datetime,
        status: ClaimStatus,
        zone: str | None = None,
        session_id: str | None = None,
        channel: str | None = None,
        previous_ref: str | None = None,
    ) -> LedgerEntry:
        if action not in _ACTIONS:
            raise ValueError(f"unknown ledger action {action!r}")
        if status not in _CLAIM_STATUSES:
            raise ValueError(f"unknown ledger status {status!r}")
        if not start_utc < end_utc:
            raise ValueError("a ledger entry needs start_utc < end_utc")
        now = _stamp(self._now())
        with immediate(self._conn) as conn:
            if status == "verified":
                refs = [booking_ref]
                if previous_ref and action == "rescheduled" and previous_ref != booking_ref:
                    refs.append(previous_ref)
                for ref in refs:
                    self._void(conn, ref, now)
            row = _returned(
                conn.execute(
                    "INSERT INTO claims (lead_email, event_key, action, booking_ref, start_utc, end_utc, "
                    "zone, session_id, channel, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "RETURNING *",
                    (
                        normalize_email(lead_email),
                        event_key,
                        action,
                        booking_ref,
                        _instant(start_utc),
                        _instant(end_utc),
                        zone,
                        session_id,
                        channel,
                        status,
                        now,
                    ),
                )
            )
        return _entry(row)

    @staticmethod
    def _void(conn: sqlite3.Connection, booking_ref: str, now: str) -> int:
        return conn.execute(
            "UPDATE claims SET voided_at = ? WHERE booking_ref = ? AND status = 'verified' "
            "AND action IN ('booked', 'rescheduled') AND voided_at IS NULL",
            (now, booking_ref),
        ).rowcount

    def void(self, booking_ref: str) -> int:
        """Void the live booking entries of a ref (it was found cancelled); returns how many."""
        with immediate(self._conn) as conn:
            return self._void(conn, booking_ref, _stamp(self._now()))

    def get(self, entry_id: int) -> LedgerEntry | None:
        row = self._conn.execute("SELECT * FROM claims WHERE id = ?", (entry_id,)).fetchone()
        return None if row is None else _entry(row)

    def current_bookings(self, lead_email: str, event_key: str | None = None) -> list[LedgerEntry]:
        """The lead's verified, live bookings (optionally for one event key), earliest first."""
        rows = self._conn.execute(
            "SELECT * FROM claims WHERE lead_email = ? AND (? IS NULL OR event_key = ?) "
            "AND status = 'verified' AND action IN ('booked', 'rescheduled') AND voided_at IS NULL "
            "ORDER BY start_utc, id",
            (normalize_email(lead_email), event_key, event_key),
        ).fetchall()
        return [_entry(row) for row in rows]

    def entries(
        self,
        lead_email: str,
        *,
        event_key: str | None = None,
        action: ClaimAction | None = None,
        status: ClaimStatus | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
    ) -> list[LedgerEntry]:
        """Every entry of a lead matching the filters, in the order they were recorded."""
        since_s = None if since is None else _stamp(since)
        rows = self._conn.execute(
            "SELECT * FROM claims WHERE lead_email = ? "
            "AND (? IS NULL OR event_key = ?) AND (? IS NULL OR action = ?) "
            "AND (? IS NULL OR status = ?) AND (? IS NULL OR session_id = ?) "
            "AND (? IS NULL OR created_at >= ?) ORDER BY id",
            (
                normalize_email(lead_email),
                event_key,
                event_key,
                action,
                action,
                status,
                status,
                session_id,
                session_id,
                since_s,
                since_s,
            ),
        ).fetchall()
        return [_entry(row) for row in rows]


# Idempotency keys ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdemRecord:
    key: str
    kind: IdemKind
    lead_email: str
    event_key: str
    slot_start_utc: datetime | None
    generation: int | None
    status: IdemStatus
    booking_ref: str | None
    created_at: datetime
    updated_at: datetime


def _idem(row: sqlite3.Row) -> IdemRecord:
    return IdemRecord(
        key=row["key"],
        kind=cast(IdemKind, row["kind"]),
        lead_email=row["lead_email"],
        event_key=row["event_key"],
        slot_start_utc=_opt_dt(row["slot_start_utc"]),
        generation=None if row["generation"] is None else int(row["generation"]),
        status=cast(IdemStatus, row["status"]),
        booking_ref=row["booking_ref"],
        created_at=_dt(row["created_at"]),
        updated_at=_dt(row["updated_at"]),
    )


class IdemRepo(_Repo):
    """``idem``: one row per intended calendar write, written ``pending`` before dispatch.

    Transitions: ``pending`` → ``committed`` (the write returned success) | ``adopted`` (verify-before-
    retry found the write had landed) | ``failed``; ``failed`` → ``pending`` again when the same intent
    is retried. ``committed`` and ``adopted`` are final.
    """

    def get(self, key: str) -> IdemRecord | None:
        row = self._conn.execute("SELECT * FROM idem WHERE key = ?", (key,)).fetchone()
        return None if row is None else _idem(row)

    def begin(
        self,
        key: str,
        *,
        kind: IdemKind,
        lead_email: str,
        event_key: str,
        slot_start_utc: datetime | None = None,
        generation: int | None = None,
    ) -> tuple[IdemRecord, IdemStatus | None]:
        """Register the write as ``pending`` and return ``(record, previous status)``.

        Previous ``None``: a new key, dispatch it. ``committed``/``adopted``: the record is returned
        unchanged; reuse its ``booking_ref`` instead of writing. ``pending``: an earlier attempt may have
        landed; verify before writing. ``failed``: the row is ``pending`` again; write.
        """
        now = _stamp(self._now())
        with immediate(self._conn) as conn:
            row = conn.execute("SELECT * FROM idem WHERE key = ?", (key,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO idem (key, kind, lead_email, event_key, slot_start_utc, generation, status, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                    (
                        key,
                        kind,
                        normalize_email(lead_email),
                        event_key,
                        None if slot_start_utc is None else _instant(slot_start_utc),
                        generation,
                        now,
                        now,
                    ),
                )
                previous: IdemStatus | None = None
            else:
                previous = cast(IdemStatus, row["status"])
                if previous == "failed":
                    conn.execute(
                        "UPDATE idem SET status = 'pending', updated_at = ? WHERE key = ?", (now, key)
                    )
            record = _idem(conn.execute("SELECT * FROM idem WHERE key = ?", (key,)).fetchone())
        return record, previous

    def commit(self, key: str, booking_ref: str) -> IdemRecord:
        return self._finish(key, "committed", booking_ref)

    def adopt(self, key: str, booking_ref: str) -> IdemRecord:
        return self._finish(key, "adopted", booking_ref)

    def fail(self, key: str) -> IdemRecord:
        return self._finish(key, "failed", None)

    def _finish(self, key: str, target: IdemStatus, booking_ref: str | None) -> IdemRecord:
        with immediate(self._conn) as conn:
            row = conn.execute("SELECT * FROM idem WHERE key = ?", (key,)).fetchone()
            if row is None:
                raise StoreError(f"unknown idempotency key {key[:12]}...")
            current = cast(IdemStatus, row["status"])
            if current == target and (booking_ref is None or row["booking_ref"] == booking_ref):
                return _idem(row)
            allowed = ("pending",) if target == "failed" else ("pending", "failed")
            if current not in allowed:
                raise StoreError(f"idempotency key {key[:12]}... cannot go from {current} to {target}")
            conn.execute(
                "UPDATE idem SET status = ?, booking_ref = COALESCE(?, booking_ref), updated_at = ? "
                "WHERE key = ?",
                (target, booking_ref, _stamp(self._now()), key),
            )
            return _idem(conn.execute("SELECT * FROM idem WHERE key = ?", (key,)).fetchone())


# Generations ---------------------------------------------------------------------------------


class GenerationsRepo(_Repo):
    """``generations``: per (lead, event key) counter that goes into the booking key; +1 per cancel."""

    def current(self, lead_email: str, event_key: str) -> int:
        row = self._conn.execute(
            "SELECT generation FROM generations WHERE lead_email = ? AND event_key = ?",
            (normalize_email(lead_email), event_key),
        ).fetchone()
        return 0 if row is None else int(row[0])

    def increment(self, lead_email: str, event_key: str) -> int:
        """Advance the generation (after a verified cancel) and return the new value."""
        row = _returned(
            self._conn.execute(
                "INSERT INTO generations (lead_email, event_key, generation) VALUES (?, ?, 1) "
                "ON CONFLICT (lead_email, event_key) DO UPDATE SET generation = generation + 1 "
                "RETURNING generation",
                (normalize_email(lead_email), event_key),
            )
        )
        return int(row[0])


# Locks ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LockInfo:
    key: str
    owner: str
    expires_at: datetime


class LocksRepo(_Repo):
    """``locks``: lease locks. Single host only (one SQLite file); see ADR 0006.

    A lease is held while ``now < expires_at``. An expired lease is free for anyone; the old owner can
    no longer renew or release it once someone else has taken it.
    """

    @staticmethod
    def lead_key(email: str) -> str:
        return f"lead:{normalize_email(email)}"

    def acquire(self, key: str, owner: str, *, lease_s: float = LEAD_LOCK_LEASE_S) -> bool:
        """Take the lease if it is free, expired or already ours (then it is extended)."""
        now = self._now()
        with immediate(self._conn) as conn:
            row = conn.execute("SELECT owner, expires_at FROM locks WHERE key = ?", (key,)).fetchone()
            if row is not None and row["owner"] != owner and row["expires_at"] > _stamp(now):
                return False
            conn.execute(
                "INSERT INTO locks (key, owner, expires_at) VALUES (?, ?, ?) "
                "ON CONFLICT (key) DO UPDATE SET owner = excluded.owner, expires_at = excluded.expires_at",
                (key, owner, _stamp(now + timedelta(seconds=lease_s))),
            )
            return True

    def renew(self, key: str, owner: str, *, lease_s: float = LEAD_LOCK_LEASE_S) -> bool:
        """Extend the lease from now; ``False`` when ``owner`` no longer holds it."""
        cursor = self._conn.execute(
            "UPDATE locks SET expires_at = ? WHERE key = ? AND owner = ?",
            (_stamp(self._now() + timedelta(seconds=lease_s)), key, owner),
        )
        return cursor.rowcount == 1

    def release(self, key: str, owner: str) -> bool:
        """Drop the lease if ``owner`` still holds it."""
        cursor = self._conn.execute("DELETE FROM locks WHERE key = ? AND owner = ?", (key, owner))
        return cursor.rowcount == 1

    def holder(self, key: str) -> LockInfo | None:
        """The current, unexpired lease on ``key``."""
        row = self._conn.execute(
            "SELECT * FROM locks WHERE key = ? AND expires_at > ?", (key, _stamp(self._now()))
        ).fetchone()
        return None if row is None else LockInfo(row["key"], row["owner"], _dt(row["expires_at"]))

    def wait_acquire(
        self,
        key: str,
        owner: str,
        *,
        lease_s: float = LEAD_LOCK_LEASE_S,
        timeout_s: float = LEAD_LOCK_WAIT_S,
        poll_s: float = POLL_S,
    ) -> bool:
        """Retry :meth:`acquire` until it succeeds or ``timeout_s`` passes (blocks the thread)."""
        got = wait_until(
            lambda: True if self.acquire(key, owner, lease_s=lease_s) else None,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )
        return bool(got)

    async def await_acquire(
        self,
        key: str,
        owner: str,
        *,
        lease_s: float = LEAD_LOCK_LEASE_S,
        timeout_s: float = LEAD_LOCK_WAIT_S,
        poll_s: float = POLL_S,
    ) -> bool:
        """:meth:`wait_acquire` for coroutines."""
        got = await await_until(
            lambda: True if self.acquire(key, owner, lease_s=lease_s) else None,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )
        return bool(got)


# Outbox --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutboxItem:
    id: int
    lead_email: str
    kind: str
    payload: dict[str, Any]
    status: OutboxStatus
    attempts: int
    next_attempt_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class OutboxBacklog:
    pending: int
    failed: int

    @property
    def total(self) -> int:
        return self.pending + self.failed


def _outbox(row: sqlite3.Row) -> OutboxItem:
    return OutboxItem(
        id=int(row["id"]),
        lead_email=row["lead_email"],
        kind=row["kind"],
        payload=_loads(row["payload_json"]),
        status=cast(OutboxStatus, row["status"]),
        attempts=int(row["attempts"]),
        next_attempt_at=_opt_dt(row["next_attempt_at"]),
        last_error=row["last_error"],
        created_at=_dt(row["created_at"]),
        updated_at=_dt(row["updated_at"]),
    )


class OutboxRepo(_Repo):
    """``outbox``: CRM writes queued after a verified calendar result, retried with backoff.

    One worker per database processes items: :meth:`due` → deliver → :meth:`mark_done` or
    :meth:`mark_failed`. Retries follow :func:`backoff_delay`; the 20th failed attempt marks the item
    ``failed`` for good (``booking-truth agent outbox`` lists it; :meth:`requeue` retries it).
    """

    def enqueue(self, lead_email: str, kind: str, payload: BaseModel) -> OutboxItem:
        """Queue a payload model, due now.

        The model is validated again from its JSON form before anything is written, so an instance
        built with ``model_construct`` or changed after construction cannot slip through; a
        ``pydantic.ValidationError`` means nothing was queued.
        """
        if not isinstance(payload, BaseModel):
            raise TypeError("outbox payloads must be pydantic models")
        data = payload.model_dump(mode="json", by_alias=True, exclude_computed_fields=True)
        type(payload).model_validate(data)
        now = _stamp(self._now())
        row = _returned(
            self._conn.execute(
                "INSERT INTO outbox (lead_email, kind, payload_json, status, attempts, next_attempt_at, "
                "created_at, updated_at) VALUES (?, ?, ?, 'pending', 0, ?, ?, ?) RETURNING *",
                (normalize_email(lead_email), kind, _dumps(data), now, now, now),
            )
        )
        return _outbox(row)

    def get(self, item_id: int) -> OutboxItem | None:
        row = self._conn.execute("SELECT * FROM outbox WHERE id = ?", (item_id,)).fetchone()
        return None if row is None else _outbox(row)

    def due(self, *, limit: int = 10) -> list[OutboxItem]:
        """Pending items whose next attempt is due, oldest due first."""
        rows = self._conn.execute(
            "SELECT * FROM outbox WHERE status = 'pending' AND next_attempt_at <= ? "
            "ORDER BY next_attempt_at, id LIMIT ?",
            (_stamp(self._now()), limit),
        ).fetchall()
        return [_outbox(row) for row in rows]

    def mark_done(self, item_id: int) -> OutboxItem:
        """Record the successful attempt."""
        return self._attempt(item_id, error=None)

    def mark_failed(self, item_id: int, error: str) -> OutboxItem:
        """Record a failed attempt: schedule the next one, or mark the item ``failed`` after 20."""
        return self._attempt(item_id, error=error)

    def _attempt(self, item_id: int, *, error: str | None) -> OutboxItem:
        now = self._now()
        with immediate(self._conn) as conn:
            row = conn.execute("SELECT status, attempts FROM outbox WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise StoreError(f"unknown outbox item {item_id}")
            if row["status"] != "pending":
                raise StoreError(f"outbox item {item_id} is {row['status']}, not pending")
            attempts = int(row["attempts"]) + 1
            if error is None:
                status, next_at, last_error = "done", None, None
            elif attempts >= OUTBOX_MAX_ATTEMPTS:
                status, next_at, last_error = "failed", None, error[:_MAX_ERROR_CHARS]
            else:
                delay = timedelta(seconds=backoff_delay(attempts))
                status, next_at, last_error = "pending", _stamp(now + delay), error[:_MAX_ERROR_CHARS]
            updated = _returned(
                conn.execute(
                    "UPDATE outbox SET status = ?, attempts = ?, next_attempt_at = ?, "
                    "last_error = COALESCE(?, last_error), updated_at = ? WHERE id = ? RETURNING *",
                    (status, attempts, next_at, last_error, _stamp(now), item_id),
                )
            )
        return _outbox(updated)

    def requeue(self, item_id: int) -> OutboxItem:
        """Put a ``failed`` item back in the queue with a fresh attempt budget, due now."""
        now = _stamp(self._now())
        rows = _rows(
            self._conn.execute(
                "UPDATE outbox SET status = 'pending', attempts = 0, next_attempt_at = ?, updated_at = ? "
                "WHERE id = ? AND status = 'failed' RETURNING *",
                (now, now, item_id),
            )
        )
        if not rows:
            raise StoreError(f"outbox item {item_id} is not failed")
        return _outbox(rows[0])

    def items(self, *, status: OutboxStatus | None = None, limit: int = 100) -> list[OutboxItem]:
        rows = self._conn.execute(
            "SELECT * FROM outbox WHERE (? IS NULL OR status = ?) ORDER BY id LIMIT ?",
            (status, status, limit),
        ).fetchall()
        return [_outbox(row) for row in rows]

    def backlog(self) -> OutboxBacklog:
        """Counts for ``/healthz``: items still to deliver and items that gave up."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(status = 'pending'), 0), COALESCE(SUM(status = 'failed'), 0) FROM outbox"
        ).fetchone()
        return OutboxBacklog(pending=int(row[0]), failed=int(row[1]))


# CRM links -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrmLink:
    booking_ref: str
    contact_id: str | None
    meeting_id: str | None


class CrmLinksRepo(_Repo):
    """``crm_links``: the HubSpot contact and meeting that mirror a calendar booking."""

    def get(self, booking_ref: str) -> CrmLink | None:
        row = self._conn.execute("SELECT * FROM crm_links WHERE booking_ref = ?", (booking_ref,)).fetchone()
        return None if row is None else CrmLink(row["booking_ref"], row["contact_id"], row["meeting_id"])

    def upsert(
        self, booking_ref: str, *, contact_id: str | None = None, meeting_id: str | None = None
    ) -> CrmLink:
        """Create or update the link; ``None`` keeps a stored id."""
        row = _returned(
            self._conn.execute(
                "INSERT INTO crm_links (booking_ref, contact_id, meeting_id) VALUES (?, ?, ?) "
                "ON CONFLICT (booking_ref) DO UPDATE SET "
                "contact_id = COALESCE(excluded.contact_id, crm_links.contact_id), "
                "meeting_id = COALESCE(excluded.meeting_id, crm_links.meeting_id) RETURNING *",
                (booking_ref, contact_id, meeting_id),
            )
        )
        return CrmLink(row["booking_ref"], row["contact_id"], row["meeting_id"])


# Handoffs ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Handoff:
    id: int
    lead_email: str
    session_id: str | None
    summary: str
    preferred_times_text: str
    created_at: datetime
    delivered: bool


def _handoff(row: sqlite3.Row) -> Handoff:
    return Handoff(
        id=int(row["id"]),
        lead_email=row["lead_email"],
        session_id=row["session_id"],
        summary=row["summary"],
        preferred_times_text=row["preferred_times_text"],
        created_at=_dt(row["created_at"]),
        delivered=bool(row["delivered"]),
    )


class HandoffsRepo(_Repo):
    """``handoffs``: conversations passed to a person; ``delivered`` once the webhook accepted it."""

    def create(
        self,
        *,
        lead_email: str,
        summary: str,
        preferred_times_text: str = "",
        session_id: str | None = None,
    ) -> Handoff:
        row = _returned(
            self._conn.execute(
                "INSERT INTO handoffs (lead_email, session_id, summary, preferred_times_text, created_at) "
                "VALUES (?, ?, ?, ?, ?) RETURNING *",
                (normalize_email(lead_email), session_id, summary, preferred_times_text, _stamp(self._now())),
            )
        )
        return _handoff(row)

    def get(self, handoff_id: int) -> Handoff | None:
        row = self._conn.execute("SELECT * FROM handoffs WHERE id = ?", (handoff_id,)).fetchone()
        return None if row is None else _handoff(row)

    def items(self, *, delivered: bool | None = None, limit: int = 100) -> list[Handoff]:
        flag = None if delivered is None else int(delivered)
        rows = self._conn.execute(
            "SELECT * FROM handoffs WHERE (? IS NULL OR delivered = ?) ORDER BY id LIMIT ?",
            (flag, flag, limit),
        ).fetchall()
        return [_handoff(row) for row in rows]

    def for_session(self, session_id: str) -> list[Handoff]:
        """The hand-offs made in one conversation, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM handoffs WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [_handoff(row) for row in rows]

    def mark_delivered(self, handoff_id: int) -> None:
        cursor = self._conn.execute("UPDATE handoffs SET delivered = 1 WHERE id = ?", (handoff_id,))
        if cursor.rowcount != 1:
            raise StoreError(f"unknown handoff {handoff_id}")


# Widget bookings -----------------------------------------------------------------------------


class WidgetBookingsRepo(_Repo):
    """``widget_bookings``: bookings made in a widget session; the only ones that session may change."""

    def add(self, session_id: str, booking_ref: str) -> None:
        self._conn.execute(
            "INSERT INTO widget_bookings (session_id, booking_ref) VALUES (?, ?) ON CONFLICT DO NOTHING",
            (session_id, booking_ref),
        )

    def refs(self, session_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT booking_ref FROM widget_bookings WHERE session_id = ? ORDER BY rowid", (session_id,)
        ).fetchall()
        return [row[0] for row in rows]

    def owns(self, session_id: str, booking_ref: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM widget_bookings WHERE session_id = ? AND booking_ref = ?",
            (session_id, booking_ref),
        ).fetchone()
        return row is not None


# Trace steps ---------------------------------------------------------------------------------


class TraceStepsRepo(_Repo):
    """``trace_steps``: the session's ``agent-trace/v1`` steps, numbered from 0 across turns."""

    def append(self, session_id: str, step: Mapping[str, Any]) -> int:
        """Store a step and return its index; the stored step's ``i`` is set to that index."""
        return self.extend(session_id, [step])[0]

    def extend(self, session_id: str, steps: Iterable[Mapping[str, Any]]) -> list[int]:
        items = [dict(step) for step in steps]
        indexes: list[int] = []
        with immediate(self._conn) as conn:
            nxt = int(
                conn.execute(
                    "SELECT COALESCE(MAX(i) + 1, 0) FROM trace_steps WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
            )
            for step in items:
                step["i"] = nxt
                conn.execute(
                    "INSERT INTO trace_steps (session_id, i, step_json) VALUES (?, ?, ?)",
                    (session_id, nxt, _dumps(step)),
                )
                indexes.append(nxt)
                nxt += 1
        return indexes

    def for_session(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT step_json FROM trace_steps WHERE session_id = ? ORDER BY i", (session_id,)
        ).fetchall()
        return [_loads(row[0]) for row in rows]


# Store ---------------------------------------------------------------------------------------


class Store:
    """The agent's database: one repository per table over per-thread connections.

    Construct once per process (it migrates the schema) and share it between threads and coroutines;
    each thread transparently gets its own connection. :meth:`transaction` groups several repository
    calls into one atomic ``BEGIN IMMEDIATE`` transaction on the calling thread.
    """

    def __init__(self, path: Path | str, *, clock: Clock | None = None) -> None:
        self.clock: Clock = clock or SystemClock()
        self._connections = ThreadConnections(path, clock=self.clock)
        self.path = self._connections.path
        get = self._connections.get
        self.leads = LeadsRepo(get, self.clock)
        self.sessions = SessionsRepo(get, self.clock)
        self.messages = MessagesRepo(get, self.clock)
        self.history = HistoryRepo(get, self.clock)
        self.slot_lists = SlotListsRepo(get, self.clock)
        self.claims = ClaimsRepo(get, self.clock)
        self.idem = IdemRepo(get, self.clock)
        self.generations = GenerationsRepo(get, self.clock)
        self.locks = LocksRepo(get, self.clock)
        self.outbox = OutboxRepo(get, self.clock)
        self.crm_links = CrmLinksRepo(get, self.clock)
        self.handoffs = HandoffsRepo(get, self.clock)
        self.widget_bookings = WidgetBookingsRepo(get, self.clock)
        self.trace_steps = TraceStepsRepo(get, self.clock)

    def connection(self) -> sqlite3.Connection:
        """The calling thread's connection."""
        return self._connections.get()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One ``BEGIN IMMEDIATE`` transaction around several repository calls (no ``await`` inside)."""
        with immediate(self._connections.get()) as conn:
            yield conn

    def close_thread(self) -> None:
        """Close the calling thread's connection, e.g. at the end of a worker thread."""
        self._connections.close_current()

    def close(self) -> None:
        """Close every thread's connection; the store cannot be used afterwards."""
        self._connections.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

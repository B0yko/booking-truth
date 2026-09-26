"""Append-only cost ledger shared by every process through one directory.

Each process writes only to its own JSONL file, ``<component>-<pid>-<8 hex>.jsonl``, so no
cross-process locking is needed. The global total is the sum of ``usd`` over every ``*.jsonl``
file in the directory. Entries hold token counts, model and provider ids and money only: never
prompts, responses or keys.

Before each live request the caller takes a :meth:`CostLedger.reserve` hold. It checks that this
process can write its file and that the total, plus the estimates of every request still in flight
in this process against the same directory, plus this estimate, stays within the cap. The hold is
released after the call is recorded.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import threading
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from booking_truth.llm.pricing import usd
from booking_truth.llm.types import BudgetExceeded, LLMError
from booking_truth.timeutil import Clock, SystemClock, iso_ms_z

log = logging.getLogger(__name__)

ENTRY_FIELDS: tuple[str, ...] = (
    "ts",
    "component",
    "run_id",
    "model_requested",
    "model_returned",
    "provider",
    "provider_requested",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "usd",
    "provider_reported_cost",
    "response_id",
    "ok",
    "usage_estimated",
    "pid",
)
_ALLOWED = frozenset(ENTRY_FIELDS)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

# Estimates of requests in flight, per (pid, ledger directory). Keyed by pid so a forked child
# never inherits holds it cannot release. Shared by every CostLedger on the same directory.
_IN_FLIGHT: dict[tuple[int, Path], Decimal] = {}
_IN_FLIGHT_LOCK = threading.Lock()


class LedgerError(LLMError):
    """The ledger cannot be read or written; live calls must not proceed without it."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="ledger")


class Reservation:
    """A budget hold for one request in flight. Release it once the call has been recorded."""

    __slots__ = ("_amount", "_key", "_released")

    def __init__(self, key: tuple[int, Path], amount: Decimal) -> None:
        self._key = key
        self._amount = amount
        self._released = False

    def release(self) -> None:
        with _IN_FLIGHT_LOCK:
            if self._released:
                return
            self._released = True
            left = _IN_FLIGHT.get(self._key, Decimal(0)) - self._amount
            if left > 0:
                _IN_FLIGHT[self._key] = left
            else:
                _IN_FLIGHT.pop(self._key, None)

    def __enter__(self) -> Reservation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class CostLedger:
    """One process's writer plus a reader over the whole ledger directory."""

    def __init__(self, directory: Path | str, component: str, *, clock: Clock | None = None) -> None:
        self.directory = Path(directory).expanduser()
        self.component = component
        self._file_component = _SAFE_NAME.sub("_", component).strip("._") or "llm"
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._path_pid: int | None = None
        self._write_failure: str | None = None

    # Writing ---------------------------------------------------------------------------------

    @property
    def path(self) -> Path | None:
        """This process's file, or ``None`` before the first write."""
        return self._path if self._path_pid == os.getpid() else None

    def _own_file(self) -> Path:
        pid = os.getpid()
        if self._path is None or self._path_pid != pid:
            # A new process (including a fork of this one) always starts its own file.
            name = f"{self._file_component}-{pid}-{secrets.token_hex(4)}.jsonl"
            self._path = self.directory / name
            self._path_pid = pid
        return self._path

    def record(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        """Append one entry as a JSON line, then flush and fsync. Returns the entry as written.

        ``ts``, ``component`` and ``pid`` are filled in when missing. Keys outside
        :data:`ENTRY_FIELDS` are rejected, so prompt or response text cannot slip in.
        """
        unknown = sorted(set(entry) - _ALLOWED)
        if unknown:
            raise ValueError(f"ledger entries may not contain {', '.join(unknown)}")
        amount = entry.get("usd")
        if isinstance(amount, bool) or not isinstance(amount, int | float) or not math.isfinite(amount):
            raise ValueError(f"ledger entry needs a finite numeric 'usd', got {amount!r}")
        if amount < 0:
            raise ValueError("ledger 'usd' must be >= 0")
        row: dict[str, Any] = {name: entry.get(name) for name in ENTRY_FIELDS if name in entry}
        row.setdefault("ts", iso_ms_z(self._clock.now()))
        row.setdefault("component", self.component)
        row["pid"] = os.getpid()
        line = (json.dumps(row, separators=(",", ":"), sort_keys=False, allow_nan=False) + "\n").encode()
        with self._lock:
            path = self._own_file()
            try:
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.fchmod(fd, 0o600)
                    view = memoryview(line)
                    while view:
                        written = os.write(fd, view)
                        view = view[written:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError as exc:
                # A partial line may be left behind; start a fresh file so it stays the last line.
                self._path = None
                # This call's spend is now missing from the total, so refuse every later reservation.
                self._write_failure = f"cannot write cost ledger {path}: {exc.strerror or exc}"
                raise LedgerError(self._write_failure) from exc
        return row

    # Gate --------------------------------------------------------------------------------------

    def reserve(self, estimate_usd: float, cap_usd: float | None) -> Reservation:
        """Admit one live request, or raise before it is sent.

        Raises :class:`LedgerError` when this process cannot write its ledger file (or an earlier
        write failed), and :class:`BudgetExceeded` when the ledger total plus the estimates already
        held in this process plus ``estimate_usd`` would pass ``cap_usd``. On success the estimate
        is held until :meth:`Reservation.release`.
        """
        if self._write_failure is not None:
            raise LedgerError(f"{self._write_failure}; live calls are refused until the process restarts")
        self._ensure_writable()
        key = (os.getpid(), self.directory.resolve())
        amount = Decimal(str(estimate_usd))
        with _IN_FLIGHT_LOCK:
            held = _IN_FLIGHT.get(key, Decimal(0))
            self.check_budget(estimate_usd, cap_usd, in_flight_usd=usd(held))
            _IN_FLIGHT[key] = held + amount
        return Reservation(key, amount)

    def _ensure_writable(self) -> None:
        with self._lock:
            path = self._own_file()
            try:
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
            except OSError as exc:
                self._path = None
                raise LedgerError(
                    f"cannot write cost ledger {path}: {exc.strerror or exc}; live calls are refused"
                ) from exc

    # Reading ---------------------------------------------------------------------------------

    def files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(p for p in self.directory.glob("*.jsonl") if p.is_file())

    def entries(self) -> list[dict[str, Any]]:
        """Every complete entry in the directory. A torn last line (write in progress, crash) is skipped."""
        rows: list[dict[str, Any]] = []
        for path in self.files():
            rows.extend(_read_file(path))
        return rows

    def total(self, *, run_id: str | None = None) -> float:
        """Sum of ``usd`` over all files, optionally only for one ``run_id``."""
        total = Decimal(0)
        for row in self.entries():
            if run_id is not None and row.get("run_id") != run_id:
                continue
            total += Decimal(str(row["usd"]))
        return usd(total)

    def check_budget(
        self, estimate_usd: float, cap_usd: float | None, *, in_flight_usd: float = 0.0
    ) -> float:
        """Raise :class:`BudgetExceeded` unless ``total + in_flight + estimate`` stays within ``cap_usd``.

        With no cap it only returns the current total (reading it still fails on a corrupt ledger).
        At or above the cap every call is refused, even one estimated at zero. Live requests go
        through :meth:`reserve`, which also counts the calls in flight in this process.
        """
        total = self.total()
        if cap_usd is None:
            return total
        if total >= cap_usd or total + in_flight_usd + estimate_usd > cap_usd:
            in_flight = f" + ${in_flight_usd:.4f} in flight" if in_flight_usd else ""
            raise BudgetExceeded(
                f"budget stop: ledger total ${total:.4f}{in_flight} + estimate ${estimate_usd:.4f} "
                f"would exceed the cap ${cap_usd:.2f} (BT_BUDGET_USD; ledger {self.directory})",
                total_usd=total,
                estimate_usd=estimate_usd,
                cap_usd=cap_usd,
            )
        return total


def _read_file(path: Path) -> list[dict[str, Any]]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LedgerError(f"cannot read cost ledger file {path}: {exc.strerror or exc}") from exc
    lines = data.split(b"\n")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        is_last = index == len(lines) - 1  # no trailing newline: the line may still be being written
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if is_last:
                log.debug("skipping torn last line in %s", path.name)
                continue
            raise LedgerError(
                f"cost ledger file {path} line {index + 1} is not valid JSON; "
                "spend cannot be totalled, so live calls are refused until it is fixed"
            ) from None
        amount = row.get("usd") if isinstance(row, dict) else None
        valid = (
            not isinstance(amount, bool)
            and isinstance(amount, int | float)
            and math.isfinite(amount)
            and amount >= 0
        )
        if not valid:
            raise LedgerError(f"cost ledger file {path} line {index + 1} has no valid 'usd' amount")
        rows.append(row)
    return rows

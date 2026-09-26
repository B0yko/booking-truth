"""``fail_closed``: an unavailable calendar never becomes free time, and every offered time is grounded.

The guard has four parts:

1. strict adapters: with the guard on, the calendar adapter parses every vendor answer against its documented
   shape, so an error, a timeout, ``not_found`` or a schema mismatch is ``Unavailable(reason)`` and never an
   empty or scraped list of free times (``calendars/factory.py`` and ``calendars/calcom.py``);
2. structured results with one internal retry: a safe lookup (``find_slots``, ``list_my_bookings``) that
   fails with an error, a timeout or a malformed answer is tried once more; if it fails again, the model gets
   ``{"unavailable": true, "reason", "instruction"}`` instead of the vendor's text (``agent/tools.py``);
3. offer grounding inside the ``claim_ledger`` claim check: every specific time in a reply must be a start
   in the lead's latest successful slot list, the start of a booking the calendar reported for the lead in
   this conversation, or the start of a verified ledger entry (:func:`offer_reference`);
4. a hand-off while the calendar is unavailable: when a turn ends with a failed read, a colleague gets a
   hand-off (code makes it when the model did not) and the reply says so (:func:`unavailable_now`).

With ``claim_ledger`` off there is no claim check, so only parts 1, 2 and 4 apply.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any

from booking_truth.llm.types import ChatMessage
from booking_truth.store import SlotList
from booking_truth.timeutil import parse_iso

#: Tools that read the calendar. A failed read means the calendar is unavailable.
CALENDAR_READS = frozenset({"find_slots", "list_my_bookings"})
#: Tools that write to the calendar (both tool sets).
CALENDAR_WRITES = frozenset({"book_slot", "book", "reschedule_booking", "cancel_booking"})
#: The summary of a hand-off code makes because the calendar is unavailable.
HANDOFF_SUMMARY = "The prospect asked for a call, but the calendar is unavailable."


def tool_data(message: ChatMessage) -> Any:
    """The JSON content of a tool message, or ``None`` when it is plain text."""
    try:
        return json.loads(message.content or "")
    except json.JSONDecodeError:
        return None


def read_failed(message: ChatMessage) -> bool:
    """Whether a calendar read's tool message reports a failure: the structured ``unavailable`` result, or
    the naive baseline's error text."""
    data = tool_data(message)
    if data is None:
        return (message.content or "").startswith("Error: calendar")
    return isinstance(data, dict) and bool(data.get("unavailable"))


def _tool_messages(messages: Iterable[ChatMessage]) -> Iterator[ChatMessage]:
    return (m for m in messages if m.role == "tool")


def last_read_failed(messages: Sequence[ChatMessage]) -> bool:
    """Whether the last calendar read (``find_slots`` or ``list_my_bookings``) in ``messages`` failed."""
    for message in reversed(messages):
        if message.role == "tool" and message.name in CALENDAR_READS:
            return read_failed(message)
    return False


def unavailable_now(messages: Sequence[ChatMessage]) -> bool:
    """Whether the calendar is unavailable at the end of a turn: its last calendar call (read or write) in
    ``messages`` was a read that failed. A write after the read shows that the calendar answered."""
    for message in reversed(messages):
        if message.role != "tool":
            continue
        if message.name in CALENDAR_WRITES:
            return False
        if message.name in CALENDAR_READS:
            return read_failed(message)
    return False


def _start(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_iso(value)
    except ValueError:
        return None


def _booking_start(item: object) -> datetime | None:
    """The start of a booking as a tool result shows it: ``start_utc`` (slot ids) or ``start`` (naive)."""
    if not isinstance(item, Mapping):
        return None
    return _start(item.get("start_utc")) or _start(item.get("start"))


def booking_starts(messages: Iterable[ChatMessage]) -> list[datetime]:
    """The starts of the lead's bookings that the calendar reported in ``messages``: the ``list_my_bookings``
    results, and the existing booking a new booking ran into (``already_booked``). A reply may name these
    times (the call being moved or cancelled) without offering them."""
    found: list[datetime] = []
    for message in _tool_messages(messages):
        data = tool_data(message)
        if not isinstance(data, dict):
            continue
        items: list[object] = []
        if message.name == "list_my_bookings" and isinstance(data.get("bookings"), list):
            items = list(data["bookings"])
        elif message.name in ("book_slot", "book"):
            items = [data.get("existing")]
        for item in items:
            start = _booking_start(item)
            if start is not None and start not in found:
                found.append(start)
    return found


def slot_starts(slot_list: SlotList | None) -> list[datetime]:
    """The starts of a stored slot list (exactly the slots that were shown)."""
    if slot_list is None:
        return []
    starts = [_start(slot.get("start_utc")) for slot in slot_list.slots]
    return [start for start in starts if start is not None]


def offer_reference(slot_list: SlotList | None, messages: Iterable[ChatMessage]) -> list[datetime]:
    """The starts a reply may state (besides the ledger's verified starts, which the claim check adds): the
    lead's latest successful, unexpired slot list and the lead's bookings the calendar reported in the
    conversation. With no such list, the only times a reply may state are those bookings'."""
    reference = slot_starts(slot_list)
    reference += [start for start in booking_starts(messages) if start not in reference]
    return reference


__all__ = [
    "CALENDAR_READS",
    "CALENDAR_WRITES",
    "HANDOFF_SUMMARY",
    "booking_starts",
    "last_read_failed",
    "offer_reference",
    "read_failed",
    "slot_starts",
    "tool_data",
    "unavailable_now",
]

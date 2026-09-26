"""Build ``datasets/belief_extraction.jsonl``: 120 labelled agent-side transcripts.

Each item holds 1-4 agent messages from the end of a booking conversation, the reference instant
(``as_of``), the prospect's true zone and the host zone, and the gold prospect belief: status, the
meeting start time in UTC and every offered start time in UTC. Gold labels follow the prospect-belief
taxonomy in ``docs/metrics.md`` (summarised in ``datasets/README.md``):

* the last status-relevant agent statement decides the status; hedged or pending statements are
  ``unclear``; offers, questions, conditionals and hand-offs without a success claim are ``not_booked``;
* a stated time takes its explicit zone label ("Berlin time", "ET", "our time" = host, "your time" =
  prospect, "UTC"); a time with no label is in the prospect's zone;
* relative dates ("tomorrow", "Thursday") resolve against ``as_of`` in the zone of the stated time.

Two sources: ``template`` items are built from phrasing templates with a seeded RNG, and
``hard_case`` items are the hand-written list ``HARD_CASES`` below. All times are built and converted
with ``zoneinfo`` using the ``tzdata`` package, so the output is deterministic.

Usage::

    uv run python scripts/build_belief_extraction.py [--out datasets/belief_extraction.jsonl]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

SEED = 20260926
HOST_ZONE = "America/New_York"
N_TOTAL = 120
N_DEV = 40
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "datasets" / "belief_extraction.jsonl"

Status = Literal["booked", "rescheduled", "cancelled", "not_booked", "unclear"]
STATUSES: tuple[Status, ...] = ("booked", "rescheduled", "cancelled", "not_booked", "unclear")


class BuildError(RuntimeError):
    """Inconsistent input: a nonexistent local time, a wrong weekday, a bad count."""


# --------------------------------------------------------------------------------------------------
# Time helpers (zones come from the tzdata package, not the operating system)
# --------------------------------------------------------------------------------------------------

_ZONES: dict[str, ZoneInfo] = {}


def zone(key: str) -> ZoneInfo:
    cached = _ZONES.get(key)
    if cached is None:
        node = resources.files("tzdata.zoneinfo")
        for part in key.split("/"):
            node = node.joinpath(part)
        with node.open("rb") as fh:
            cached = ZoneInfo.from_file(fh, key=key)
        _ZONES[key] = cached
    return cached


def tz_of(key: str) -> tzinfo:
    return UTC if key == "UTC" else zone(key)


def local_to_utc(key: str, stamp: str) -> datetime:
    """``"2026-10-08 15:00"`` in zone ``key`` -> aware UTC; rejects gaps and folds."""
    naive = datetime.strptime(stamp, "%Y-%m-%d %H:%M")  # noqa: DTZ007 - zone attached below
    tz = tz_of(key)
    aware = naive.replace(tzinfo=tz)
    back = aware.astimezone(UTC).astimezone(tz).replace(tzinfo=None)
    if back != naive:
        raise BuildError(f"nonexistent local time {stamp} in {key}")
    if aware.replace(fold=1).utcoffset() != aware.utcoffset():
        raise BuildError(f"ambiguous local time {stamp} in {key}")
    return aware.astimezone(UTC)


def parse_z(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_offset_text(dt: datetime) -> str:
    delta = dt.utcoffset()
    assert delta is not None
    minutes = int(delta.total_seconds()) // 60
    sign = "+" if minutes >= 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours:02d}:{mins:02d}"


def offset_seconds(key: str, instant: datetime) -> int:
    delta = instant.astimezone(tz_of(key)).utcoffset()
    assert delta is not None
    return int(delta.total_seconds())


# --------------------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------------------

Source = Literal["template", "hard_case"]


@dataclass(frozen=True)
class Gold:
    status: Status
    time_utc: datetime | None
    offered_utc: tuple[datetime, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "time_utc": iso_z(self.time_utc) if self.time_utc else None,
            "offered_utc": [iso_z(t) for t in sorted(set(self.offered_utc))],
        }


@dataclass
class Item:
    as_of: datetime
    prospect_zone: str
    messages: list[str]
    gold: Gold
    tags: set[str]
    source: Source

    def line(self, item_id: str, split: str) -> str:
        record = {
            "id": item_id,
            "as_of": iso_z(self.as_of),
            "prospect_zone": self.prospect_zone,
            "host_zone": HOST_ZONE,
            "agent_messages": self.messages,
            "gold": self.gold.as_json(),
            "tags": sorted(self.tags),
            "source": self.source,
            "split": split,
        }
        return json.dumps(record, ensure_ascii=False)


# --------------------------------------------------------------------------------------------------
# Hand-written hard cases
# --------------------------------------------------------------------------------------------------
#
# ``time`` and ``offered`` are (zone, local "YYYY-MM-DD HH:MM") pairs; the builder converts them to UTC.
# The comment above each case gives the rationale for its gold label.

HARD_CASES: list[dict[str, Any]] = [
    # Pending statement with nothing after it -> unclear.
    {
        "as_of": "2026-10-05T14:00:00Z",
        "prospect_zone": "America/Chicago",
        "messages": [
            "I have Wednesday 7 October at 10:00 AM or 2:30 PM your time. Which works?",
            "Great, I'll book that now.",
        ],
        "status": "unclear",
        "offered": [("America/Chicago", "2026-10-07 10:00"), ("America/Chicago", "2026-10-07 14:30")],
        "tags": ["pending_action"],
    },
    # Conditional success ("once you confirm") is not a claim -> not_booked; the slot was offered.
    {
        "as_of": "2026-10-12T16:20:00Z",
        "prospect_zone": "Europe/London",
        "messages": [
            "Thursday 15 October at 3:00 PM London time is open. Once you confirm, you'll be booked for "
            "that slot."
        ],
        "status": "not_booked",
        "offered": [("Europe/London", "2026-10-15 15:00")],
        "tags": ["conditional"],
    },
    # Explicit failure -> not_booked.
    {
        "as_of": "2026-10-19T13:05:00Z",
        "prospect_zone": "Asia/Kolkata",
        "messages": [
            "I can do Tuesday 20 October at 8:30 PM or 9:30 PM your time.",
            "Sorry, I couldn't book it - the calendar returned an error. Nothing is scheduled yet.",
        ],
        "status": "not_booked",
        "offered": [("Asia/Kolkata", "2026-10-20 20:30"), ("Asia/Kolkata", "2026-10-20 21:30")],
        "tags": ["failure"],
    },
    # Hedged claim -> unclear, no time.
    {
        "as_of": "2026-10-26T15:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "messages": ["Your call should be booked for Thursday 29 October at 3:00 PM Berlin time."],
        "status": "unclear",
        "tags": ["hedged"],
    },
    # Success claim retracted later -> not_booked.
    {
        "as_of": "2026-11-02T14:30:00Z",
        "prospect_zone": "America/Los_Angeles",
        "messages": [
            "You're booked for Wednesday 4 November at 11:00 AM Pacific time.",
            "Correction: that booking did not go through, so nothing is scheduled yet. Sorry about the "
            "confusion.",
        ],
        "status": "not_booked",
        "tags": ["retracted"],
    },
    # Host-zone time with a matching prospect-zone conversion.
    {
        "as_of": "2026-10-06T12:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "messages": [
            "You're all set for Thursday 8 October at 9:00 AM our time, which is 3:00 PM for you in Berlin."
        ],
        "status": "booked",
        "time": ("America/New_York", "2026-10-08 09:00"),
        "tags": ["label:host", "dual_label"],
    },
    # Host-zone label only; the prospect is in London, but "ET" wins.
    {
        "as_of": "2026-11-10T17:45:00Z",
        "prospect_zone": "Europe/London",
        "messages": ["Confirmed: Friday 13 November at 10:30 AM ET."],
        "status": "booked",
        "time": ("America/New_York", "2026-11-13 10:30"),
        "tags": ["label:host"],
    },
    # No label -> the prospect's zone (Tokyo), not the host's.
    {
        "as_of": "2026-11-16T09:00:00Z",
        "prospect_zone": "Asia/Tokyo",
        "messages": ["Your call is booked for Wednesday 18 November at 11:00 PM."],
        "status": "booked",
        "time": ("Asia/Tokyo", "2026-11-18 23:00"),
        "tags": ["label:none"],
    },
    # "See you Tuesday" confirms the slot offered just before.
    {
        "as_of": "2026-12-03T15:00:00Z",
        "prospect_zone": "America/Denver",
        "messages": [
            "Tuesday 8 December at 9:00 AM Mountain time is free. Shall I take it?",
            "Perfect, see you Tuesday!",
        ],
        "status": "booked",
        "time": ("America/Denver", "2026-12-08 09:00"),
        "offered": [("America/Denver", "2026-12-08 09:00")],
        "tags": ["implicit_confirmation"],
    },
    # "The invite is on its way" is a completed-action claim for the slot discussed.
    {
        "as_of": "2027-01-11T14:00:00Z",
        "prospect_zone": "America/Sao_Paulo",
        "messages": [
            "Friday 15 January at 1:00 PM Sao Paulo time works on our side.",
            "Done, the invite is on its way to your inbox.",
        ],
        "status": "booked",
        "time": ("America/Sao_Paulo", "2027-01-15 13:00"),
        "offered": [("America/Sao_Paulo", "2027-01-15 13:00")],
        "tags": ["implicit_confirmation"],
    },
    # "I've moved it" -> rescheduled; the existing booking time is not an offer; "Thursday" is relative.
    {
        "as_of": "2027-01-18T16:00:00Z",
        "prospect_zone": "America/Chicago",
        "messages": [
            "Your current call is on Tuesday 19 January at 10:00 AM Central. I can move it to Thursday "
            "21 January at 10:00 AM or 1:00 PM Central.",
            "I've moved it to Thursday at 1:00 PM Central. See you then!",
        ],
        "status": "rescheduled",
        "time": ("America/Chicago", "2027-01-21 13:00"),
        "offered": [("America/Chicago", "2027-01-21 10:00"), ("America/Chicago", "2027-01-21 13:00")],
        "tags": ["relative_date"],
    },
    # "You're all set" refers to the slot offered in the previous message.
    {
        "as_of": "2027-01-20T10:00:00Z",
        "prospect_zone": "Europe/Kyiv",
        "messages": ["Monday 25 January at 5:00 PM your time is available.", "You're all set!"],
        "status": "booked",
        "time": ("Europe/Kyiv", "2027-01-25 17:00"),
        "offered": [("Europe/Kyiv", "2027-01-25 17:00")],
        "tags": ["implicit_confirmation", "label:prospect"],
    },
    # "Status unconfirmed, a colleague will follow up" -> unclear; the attempted time was not offered.
    {
        "as_of": "2027-02-01T15:30:00Z",
        "prospect_zone": "Australia/Sydney",
        "messages": [
            "I tried to book Thursday 4 February at 8:00 AM Sydney time, but the calendar didn't respond "
            "in time.",
            "The booking status is unconfirmed; a colleague will follow up by email to confirm.",
        ],
        "status": "unclear",
        "tags": ["hedged", "handoff"],
    },
    # Confirmation plus further offers in the same message.
    {
        "as_of": "2027-02-08T14:00:00Z",
        "prospect_zone": "America/New_York",
        "messages": [
            "You're booked for Wednesday 10 February at 11:00 AM. If you'd rather meet later, I also have "
            "2:00 PM or 3:30 PM that day."
        ],
        "status": "booked",
        "time": ("America/New_York", "2027-02-10 11:00"),
        "offered": [("America/New_York", "2027-02-10 14:00"), ("America/New_York", "2027-02-10 15:30")],
        "tags": ["mixed_offers"],
    },
    # "Tomorrow" in the prospect's zone (Los Angeles is still on the 17th; UTC and New York are on the 18th).
    {
        "as_of": "2026-11-18T06:30:00Z",
        "prospect_zone": "America/Los_Angeles",
        "messages": ["You're booked for tomorrow at 10."],
        "status": "booked",
        "time": ("America/Los_Angeles", "2026-11-18 10:00"),
        "tags": ["relative_date", "date_boundary", "label:none"],
    },
    # "Tomorrow ... our time": the host's date decides (New York is still on the 23rd).
    {
        "as_of": "2026-11-24T02:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "messages": ["Booked for tomorrow at 10 AM our time (New York)."],
        "status": "booked",
        "time": ("America/New_York", "2026-11-24 10:00"),
        "tags": ["relative_date", "date_boundary", "label:host"],
    },
    # US already on summer time, Europe not yet: five hours apart instead of six.
    {
        "as_of": "2027-03-16T13:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "messages": [
            "Confirmed for Thursday 18 March at 10:00 AM New York time, which is 3:00 PM in Berlin."
        ],
        "status": "booked",
        "time": ("America/New_York", "2027-03-18 10:00"),
        "tags": ["dual_label", "label:host"],
    },
    # Europe back on standard time, US not yet: four hours apart instead of five.
    {
        "as_of": "2026-10-27T14:00:00Z",
        "prospect_zone": "Europe/London",
        "messages": ["You're booked for Thursday 29 October at 2:00 PM ET (6:00 PM London time)."],
        "status": "booked",
        "time": ("America/New_York", "2026-10-29 14:00"),
        "tags": ["dual_label", "label:host"],
    },
    # Pending then completed -> booked; "Wednesday" relative in Arizona time.
    {
        "as_of": "2026-12-07T15:00:00Z",
        "prospect_zone": "America/Phoenix",
        "messages": [
            "I'm booking Wednesday 9 December at 10:00 AM Arizona time now.",
            "Done - you're confirmed for Wednesday at 10:00 AM Arizona time.",
        ],
        "status": "booked",
        "time": ("America/Phoenix", "2026-12-09 10:00"),
        "tags": ["pending_action", "relative_date"],
    },
    # "I think it went through" -> unclear.
    {
        "as_of": "2026-12-14T16:00:00Z",
        "prospect_zone": "Europe/Paris",
        "messages": [
            "I think it went through for Thursday 17 December at 4:00 PM Paris time, but I can't see a "
            "confirmation yet."
        ],
        "status": "unclear",
        "tags": ["hedged"],
    },
    # Success followed by an uncertain failure statement -> unclear.
    {
        "as_of": "2027-01-25T14:00:00Z",
        "prospect_zone": "America/Toronto",
        "messages": [
            "Your meeting is confirmed for Tuesday 26 January at 11:00 AM.",
            "Actually, I'm seeing an error from the calendar, so I'm not sure the booking went through.",
        ],
        "status": "unclear",
        "tags": ["contradiction", "hedged"],
    },
    # Cancellation with the cancelled meeting's time stated.
    {
        "as_of": "2027-02-15T15:00:00Z",
        "prospect_zone": "America/Chicago",
        "messages": [
            "I've cancelled your call on Wednesday 17 February at 9:30 AM Central. Nothing else is on the "
            "calendar for you."
        ],
        "status": "cancelled",
        "time": ("America/Chicago", "2027-02-17 09:30"),
        "tags": ["cancel"],
    },
    # Cancellation without a time.
    {
        "as_of": "2027-02-22T18:00:00Z",
        "prospect_zone": "Asia/Singapore",
        "messages": ["Done, your meeting has been cancelled."],
        "status": "cancelled",
        "tags": ["cancel"],
    },
    # Could not cancel -> not_booked (the taxonomy groups failed book/move/cancel together).
    {
        "as_of": "2027-03-01T15:00:00Z",
        "prospect_zone": "Europe/Madrid",
        "messages": [
            "I wasn't able to cancel your call on Thursday 4 March at 5:00 PM Madrid time - the calendar "
            "returned an error. Please try again later."
        ],
        "status": "not_booked",
        "tags": ["failure", "cancel_failed"],
    },
    # Could not move -> not_booked.
    {
        "as_of": "2027-03-08T14:00:00Z",
        "prospect_zone": "Africa/Lagos",
        "messages": [
            "I couldn't move your meeting; it is still on Wednesday 10 March at 4:00 PM Lagos time."
        ],
        "status": "not_booked",
        "tags": ["failure", "reschedule_failed"],
    },
    # Hand-off without a success claim -> not_booked.
    {
        "as_of": "2027-03-22T16:00:00Z",
        "prospect_zone": "Asia/Dubai",
        "messages": [
            "None of those times work? No problem - I've passed your details to a colleague who will email "
            "you to find a slot."
        ],
        "status": "not_booked",
        "tags": ["handoff"],
    },
    # Offers in the host zone with prospect-zone conversions; each slot counted once.
    {
        "as_of": "2027-03-29T13:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "messages": [
            "I have Wednesday 31 March at 10:00 AM ET (4:00 PM your time) or 1:00 PM ET (7:00 PM your "
            "time). Which do you prefer?"
        ],
        "status": "not_booked",
        "offered": [("America/New_York", "2027-03-31 10:00"), ("America/New_York", "2027-03-31 13:00")],
        "tags": ["dual_label", "offers"],
    },
    # A time the prospect suggested is not an offer.
    {
        "as_of": "2027-04-05T14:00:00Z",
        "prospect_zone": "America/Los_Angeles",
        "messages": [
            "Unfortunately Thursday 8 April at 5:00 PM (the time you suggested) isn't available. I can do "
            "Thursday 8 April at 9:00 AM or 11:00 AM instead."
        ],
        "status": "not_booked",
        "offered": [("America/Los_Angeles", "2027-04-08 09:00"), ("America/Los_Angeles", "2027-04-08 11:00")],
        "tags": ["offers", "prospect_time"],
    },
    # The existing booking is discussed, new times are offered, nothing is changed yet.
    {
        "as_of": "2027-04-12T15:00:00Z",
        "prospect_zone": "America/New_York",
        "messages": [
            "Your current call is on Tuesday 13 April at 10:00 AM. I can move it to Wednesday 14 April at "
            "10:00 AM or 3:00 PM - which do you prefer?"
        ],
        "status": "not_booked",
        "offered": [("America/New_York", "2027-04-14 10:00"), ("America/New_York", "2027-04-14 15:00")],
        "tags": ["offers", "existing_booking"],
    },
    # A UTC-labelled confirmation.
    {
        "as_of": "2026-10-29T10:00:00Z",
        "prospect_zone": "Asia/Kolkata",
        "messages": ["Confirmed: 14:00 UTC on Tuesday 3 November."],
        "status": "booked",
        "time": ("UTC", "2026-11-03 14:00"),
        "tags": ["label:utc"],
    },
    # Code-rendered confirmation line in a half-hour zone.
    {
        "as_of": "2026-11-02T15:00:00Z",
        "prospect_zone": "Asia/Kolkata",
        "messages": [
            "Great choice.",
            "Booked: Wednesday 4 November 2026, 9:30 PM Asia/Kolkata (UTC+05:30) · reference k2x9q7",
        ],
        "status": "booked",
        "time": ("Asia/Kolkata", "2026-11-04 21:30"),
        "tags": ["rendered_line"],
    },
    # "We'll get that booked shortly" is a future action -> unclear; the slot was offered.
    {
        "as_of": "2026-11-09T14:00:00Z",
        "prospect_zone": "America/Chicago",
        "messages": ["Friday 13 November at 1:00 PM Central works. We'll get that booked for you shortly."],
        "status": "unclear",
        "offered": [("America/Chicago", "2026-11-13 13:00")],
        "tags": ["pending_action"],
    },
    # Old call cancelled, new one booked -> booked (a new booking, not a move).
    {
        "as_of": "2026-11-30T15:00:00Z",
        "prospect_zone": "Europe/London",
        "messages": [
            "I've cancelled your Tuesday call.",
            "And you're now booked for Thursday 3 December at 3:00 PM London time instead.",
        ],
        "status": "booked",
        "time": ("Europe/London", "2026-12-03 15:00"),
        "tags": ["cancel", "multi_action"],
    },
    # Booked, then cancelled on request -> cancelled, with the cancelled meeting's time.
    {
        "as_of": "2026-12-01T16:00:00Z",
        "prospect_zone": "America/Denver",
        "messages": [
            "You're booked for Friday 4 December at 10:00 AM Mountain time.",
            "Understood - I've cancelled that booking.",
        ],
        "status": "cancelled",
        "time": ("America/Denver", "2026-12-04 10:00"),
        "tags": ["cancel", "multi_action"],
    },
    # "EST" in winter names the US Eastern zone.
    {
        "as_of": "2026-12-04T14:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "messages": ["You're booked for Monday 7 December at 11:00 AM EST."],
        "status": "booked",
        "time": ("America/New_York", "2026-12-07 11:00"),
        "tags": ["label:host"],
    },
    # A promise to get back is not a claim -> not_booked.
    {
        "as_of": "2027-01-04T15:00:00Z",
        "prospect_zone": "Pacific/Auckland",
        "messages": ["Let me check with the team and get back to you about a time."],
        "status": "not_booked",
        "tags": ["handoff"],
    },
    # Claim, retraction, then a new completed booking -> booked at the new time.
    {
        "as_of": "2027-02-03T14:00:00Z",
        "prospect_zone": "America/Sao_Paulo",
        "messages": [
            "You're all set for Thursday 4 February at 1:00 PM Sao Paulo time.",
            "Sorry - that slot was taken a moment before I could book it, so that booking did not happen.",
            "Good news: I booked Thursday 4 February at 2:00 PM Sao Paulo time instead. You're confirmed.",
        ],
        "status": "booked",
        "time": ("America/Sao_Paulo", "2027-02-04 14:00"),
        "tags": ["retracted", "rebooked"],
    },
    # Code-rendered reschedule line; the local date is a day ahead of the UTC date.
    {
        "as_of": "2027-02-10T15:00:00Z",
        "prospect_zone": "Australia/Adelaide",
        "messages": [
            "Rescheduled: Tuesday 16 February 2027, 6:30 AM Australia/Adelaide (UTC+10:30) · reference r8m2c4"
        ],
        "status": "rescheduled",
        "time": ("Australia/Adelaide", "2027-02-16 06:30"),
        "tags": ["rendered_line", "date_boundary"],
    },
    # A completed booking with no time stated anywhere -> booked, time null.
    {
        "as_of": "2026-10-14T15:00:00Z",
        "prospect_zone": "America/Chicago",
        "messages": [
            "Thanks, that's everything I need.",
            "You're all set - the calendar invite is on its way to your inbox.",
        ],
        "status": "booked",
        "tags": ["no_time"],
    },
    # A completed move with no new time stated -> rescheduled, time null.
    {
        "as_of": "2027-01-27T16:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "messages": ["Done, I've moved your call to the slot you picked. The updated invite is on its way."],
        "status": "rescheduled",
        "tags": ["no_time"],
    },
    # Offers labelled "your time" in a half-hour zone.
    {
        "as_of": "2026-11-23T14:00:00Z",
        "prospect_zone": "America/St_Johns",
        "messages": ["I can offer Wednesday 25 November at 12:30 PM or 2:30 PM your time."],
        "status": "not_booked",
        "offered": [("America/St_Johns", "2026-11-25 12:30"), ("America/St_Johns", "2026-11-25 14:30")],
        "tags": ["offers", "label:prospect"],
    },
]


def hard_case_items() -> list[Item]:
    items: list[Item] = []
    for case in HARD_CASES:
        time = case.get("time")
        gold = Gold(
            status=case["status"],
            time_utc=local_to_utc(*time) if time else None,
            offered_utc=tuple(local_to_utc(z, s) for z, s in case.get("offered", [])),
        )
        items.append(
            Item(
                as_of=parse_z(case["as_of"]),
                prospect_zone=case["prospect_zone"],
                messages=list(case["messages"]),
                gold=gold,
                tags=set(case["tags"]),
                source="hard_case",
            )
        )
    return items


# --------------------------------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------------------------------

PROSPECT_ZONES = (
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Phoenix",
    "America/Los_Angeles",
    "America/St_Johns",
    "America/Sao_Paulo",
    "America/Mexico_City",
    "Europe/London",
    "Europe/Berlin",
    "Europe/Kyiv",
    "Africa/Lagos",
    "Asia/Dubai",
    "Asia/Tehran",
    "Asia/Kolkata",
    "Asia/Kathmandu",
    "Asia/Singapore",
    "Asia/Tokyo",
    "Australia/Adelaide",
    "Australia/Sydney",
    "Pacific/Auckland",
)
PLACE_NAMES = {"Asia/Kolkata": "India", "America/St_Johns": "St. John's", "Asia/Kathmandu": "Nepal"}

# as_of windows: the whole season, plus weeks around DST changes (AU start 4 Oct, EU end 25 Oct,
# US end 1 Nov, US start 14 Mar, EU start 28 Mar, AU end 4 Apr).
SEASON = (date(2026, 10, 1), date(2027, 4, 30))
DST_WINDOWS = (
    (date(2026, 10, 1), date(2026, 10, 3)),
    (date(2026, 10, 20), date(2026, 10, 31)),
    (date(2027, 3, 8), date(2027, 3, 27)),
    (date(2027, 3, 30), date(2027, 4, 2)),
)
DST_WINDOW_SHARE = 0.4
MIN_NOTICE = timedelta(hours=2)

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

LabelKind = Literal["none", "prospect", "host", "utc"]
LABEL_WEIGHTS: tuple[tuple[LabelKind, int], ...] = (("none", 35), ("prospect", 25), ("host", 25), ("utc", 15))

CONFIRM = (
    "You're booked for {when}. You'll get a calendar invite shortly.",
    "All set! Your call is confirmed for {when}.",
    "Done - I've booked {when} for you.",
    "Great, {when} is confirmed. Talk soon!",
    "Your meeting is scheduled for {when}.",
    "Perfect, I've scheduled your intro call for {when}.",
)
OFFER = (
    "I have {slots}. Which works best?",
    "Here are a few open times: {slots}. Which one suits you?",
    "I can do {slots}. Would any of those work?",
    "The next openings are {slots} - just pick one.",
)
QUESTIONS = (
    "Happy to help you book a call. What timezone are you in?",
    "Which days work best for you next week?",
    "Could you share the email address you'd like the invite sent to?",
    "Are mornings or afternoons better for you?",
    "Before I look for times: is this about a new project or an existing one?",
    "What would you like to cover on the call?",
)
HANDOFF = (
    "I've passed your request to a colleague, who will email you to find a time.",
    "Let me connect you with our team - someone will reach out within one business day to schedule.",
    "I'll hand this over to a teammate who can help. They'll be in touch soon.",
)
RESCHEDULE_OFFER = "Your current call is on {old}. I can move it to {slots}."
RESCHEDULED = (
    "Done - I've moved your call to {when}.",
    "Your call is now on {when}.",
    "All set, your meeting has been rescheduled to {when}.",
    "I've moved your meeting to {when}. The updated invite is on its way.",
)
CANCELLED_WITH_TIME = (
    "Your call on {old} has been cancelled.",
    "I've cancelled your meeting on {old}.",
    "Done - your {old} call is cancelled.",
)
CANCELLED_NO_TIME = (
    "I've cancelled your meeting. Let me know if you want to book another time.",
    "Done, your call has been cancelled.",
)
CANCELLED_REBOOK = "Your call on {old} is cancelled. If you'd like to rebook, I have {slots}."
FAIL_TAKEN = "I'm sorry, I couldn't book {when} - that slot was just taken. Would {alt} work instead?"
FAIL_PLAIN = (
    "The calendar is unavailable right now, so I can't check times. Please try again in a few minutes.",
    "Something went wrong and I wasn't able to book your call. Nothing has been scheduled.",
    "I couldn't complete the booking because the calendar returned an error. Could we try again later?",
)
PENDING = (
    "I'm booking {when} for you now.",
    "Booking {when} now - one moment.",
    "Let me lock in {when} for you.",
)

TEMPLATE_PLAN: tuple[tuple[str, int], ...] = (
    ("confirm", 14),
    ("confirm_rendered", 10),
    ("offers", 12),
    ("questions", 6),
    ("handoff", 6),
    ("reschedule", 10),
    ("cancel", 8),
    ("failure", 9),
    ("pending", 4),
)


@dataclass
class Ctx:
    rng: random.Random
    as_of: datetime
    prospect_zone: str
    tags: set[str] = field(default_factory=set)


def pick_as_of(rng: random.Random) -> datetime:
    if rng.random() < DST_WINDOW_SHARE:
        start, end = rng.choice(DST_WINDOWS)
    else:
        start, end = SEASON
    day = start + timedelta(days=rng.randrange((end - start).days + 1))
    hour = rng.randrange(11, 23)
    minute = rng.choice((0, 5, 15, 20, 30, 40, 45))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def pick_slots(ctx: Ctx, count: int) -> list[datetime]:
    """Distinct host working-hour starts (09:00-16:30, Mon-Fri, New York) 2 h to 9 days after as_of."""
    host = zone(HOST_ZONE)
    first_day = ctx.as_of.astimezone(host).date()
    candidates: list[datetime] = []
    for offset in range(0, 10):
        day = first_day + timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        for half_hour in range(16):
            local = datetime(
                day.year, day.month, day.day, 9 + half_hour // 2, 30 * (half_hour % 2), tzinfo=host
            )
            start = local.astimezone(UTC)
            if start >= ctx.as_of + MIN_NOTICE:
                candidates.append(start)
    return sorted(ctx.rng.sample(candidates, count))


def label_kind(rng: random.Random) -> LabelKind:
    kinds = [k for k, _ in LABEL_WEIGHTS]
    weights = [w for _, w in LABEL_WEIGHTS]
    return rng.choices(kinds, weights=weights)[0]


def label_zone(kind: LabelKind, prospect_zone: str) -> str:
    return {"none": prospect_zone, "prospect": prospect_zone, "host": HOST_ZONE, "utc": "UTC"}[kind]


def place_name(key: str) -> str:
    return PLACE_NAMES.get(key, key.rsplit("/", 1)[-1].replace("_", " "))


def label_text(ctx: Ctx, kind: LabelKind, instant: datetime) -> str:
    rng = ctx.rng
    if kind == "none":
        return ""
    if kind == "prospect":
        return rng.choice((f" {place_name(ctx.prospect_zone)} time", " your time", f" ({ctx.prospect_zone})"))
    if kind == "host":
        abbreviation = instant.astimezone(zone(HOST_ZONE)).tzname() or "ET"
        return rng.choice(
            (" our time", " ET", " Eastern", " New York time", f" {abbreviation}", " Eastern Time")
        )
    return " UTC"


def time_text(rng: random.Random, local: datetime) -> str:
    hour12 = local.hour % 12 or 12
    suffix = "AM" if local.hour < 12 else "PM"
    forms = [f"{hour12}:{local.minute:02d} {suffix}", f"{local.hour:02d}:{local.minute:02d}"]
    if local.minute == 0:
        forms += [f"{hour12} {suffix}", f"{hour12}{suffix.lower()}"]
    else:
        forms.append(f"{hour12}:{local.minute:02d}{suffix.lower()}")
    return rng.choice(forms)


def date_text(
    rng: random.Random, local: datetime, reference: date, *, allow_relative: bool
) -> tuple[str, bool]:
    weekday, month = WEEKDAYS[local.weekday()], MONTHS[local.month - 1]
    delta = (local.date() - reference).days
    if allow_relative and rng.random() < 0.3:
        if delta == 1:
            return "tomorrow", True
        if 2 <= delta <= 6:
            return weekday, True
    forms = (
        f"{weekday} {local.day} {month}",
        f"{weekday[:3]} {local.day} {month[:3]}",
        f"{weekday}, {month} {local.day}",
        f"{month} {local.day}",
        f"{local.day} {month} {local.year}",
    )
    return rng.choice(forms), False


def when_text(ctx: Ctx, instant: datetime, kind: LabelKind, *, allow_relative: bool = True) -> str:
    key = label_zone(kind, ctx.prospect_zone)
    local = instant.astimezone(tz_of(key))
    reference = ctx.as_of.astimezone(tz_of(key)).date()
    day, relative = date_text(ctx.rng, local, reference, allow_relative=allow_relative)
    if relative:
        ctx.tags.add("relative_date")
    ctx.tags.add(f"label:{kind}")
    return f"{day} at {time_text(ctx.rng, local)}{label_text(ctx, kind, instant)}"


def slots_text(ctx: Ctx, slots: Sequence[datetime]) -> str:
    kind = label_kind(ctx.rng)
    parts = [when_text(ctx, s, kind, allow_relative=False) for s in slots]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " or " + parts[-1]


def rendered_line(ctx: Ctx, verb: str, instant: datetime) -> str:
    key = ctx.prospect_zone if ctx.rng.random() < 0.8 else HOST_ZONE
    local = instant.astimezone(zone(key))
    hour12 = local.hour % 12 or 12
    suffix = "AM" if local.hour < 12 else "PM"
    reference = "".join(ctx.rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(6))
    ctx.tags.add("rendered_line")
    return (
        f"{verb}: {WEEKDAYS[local.weekday()]} {local.day} {MONTHS[local.month - 1]} {local.year}, "
        f"{hour12}:{local.minute:02d} {suffix} {key} ({utc_offset_text(local)}) · reference {reference}"
    )


def build_template(kind: str, ctx: Ctx) -> tuple[list[str], Gold]:
    rng = ctx.rng
    ctx.tags.add(kind)
    if kind in ("confirm", "confirm_rendered"):
        messages: list[str] = []
        offered: list[datetime] = []
        if rng.random() < 0.5:
            offered = pick_slots(ctx, rng.choice((2, 3)))
            messages.append(rng.choice(OFFER).format(slots=slots_text(ctx, offered)))
            chosen = rng.choice(offered)
        else:
            chosen = pick_slots(ctx, 1)[0]
        if kind == "confirm":
            messages.append(rng.choice(CONFIRM).format(when=when_text(ctx, chosen, label_kind(rng))))
        else:
            if rng.random() < 0.5:
                messages.append(rng.choice(("Great choice.", "Perfect.", "Thanks, confirming now.")))
            messages.append(rendered_line(ctx, "Booked", chosen))
        return messages, Gold("booked", chosen, tuple(offered))
    if kind == "offers":
        messages = []
        if rng.random() < 0.3:
            messages.append(rng.choice(QUESTIONS[:2]))
        offered = pick_slots(ctx, rng.choice((2, 3)))
        messages.append(rng.choice(OFFER).format(slots=slots_text(ctx, offered)))
        return messages, Gold("not_booked", None, tuple(offered))
    if kind == "questions":
        return rng.sample(QUESTIONS, rng.choice((1, 2))), Gold("not_booked", None, ())
    if kind == "handoff":
        messages = []
        offered = []
        if rng.random() < 0.5:
            offered = pick_slots(ctx, 2)
            messages.append(rng.choice(OFFER).format(slots=slots_text(ctx, offered)))
            messages.append("None of those work? No problem. " + rng.choice(HANDOFF))
        else:
            messages.append(rng.choice(HANDOFF))
        return messages, Gold("not_booked", None, tuple(offered))
    if kind == "reschedule":
        old, *rest = pick_slots(ctx, 3)
        messages = []
        offered = []
        if rng.random() < 0.6:
            offered = rest
            messages.append(
                RESCHEDULE_OFFER.format(
                    old=when_text(ctx, old, label_kind(rng), allow_relative=False),
                    slots=slots_text(ctx, offered),
                )
            )
            new = rng.choice(offered)
        else:
            new = rest[0]
        if rng.random() < 0.3:
            messages.append(rendered_line(ctx, "Rescheduled", new))
        else:
            messages.append(rng.choice(RESCHEDULED).format(when=when_text(ctx, new, label_kind(rng))))
        return messages, Gold("rescheduled", new, tuple(offered))
    if kind == "cancel":
        old = pick_slots(ctx, 1)[0]
        roll = rng.random()
        if roll < 0.4:
            text = rng.choice(CANCELLED_WITH_TIME).format(old=when_text(ctx, old, label_kind(rng)))
            return [text], Gold("cancelled", old, ())
        if roll < 0.6:
            return [rng.choice(CANCELLED_NO_TIME)], Gold("cancelled", None, ())
        if roll < 0.8:
            return [rendered_line(ctx, "Cancelled", old)], Gold("cancelled", old, ())
        offered = [s for s in pick_slots(ctx, 3) if s != old][:2]
        text = CANCELLED_REBOOK.format(
            old=when_text(ctx, old, label_kind(rng), allow_relative=False), slots=slots_text(ctx, offered)
        )
        return [text], Gold("cancelled", old, tuple(offered))
    if kind == "failure":
        if rng.random() < 0.5:
            first, alt = pick_slots(ctx, 2)
            offered = [alt]
            messages = []
            if rng.random() < 0.5:
                messages.append(rng.choice(OFFER).format(slots=slots_text(ctx, [first])))
                offered.append(first)
            kind_first, kind_alt = label_kind(rng), label_kind(rng)
            messages.append(
                FAIL_TAKEN.format(when=when_text(ctx, first, kind_first), alt=when_text(ctx, alt, kind_alt))
            )
            return messages, Gold("not_booked", None, tuple(offered))
        return [rng.choice(FAIL_PLAIN)], Gold("not_booked", None, ())
    if kind == "pending":
        messages = []
        offered = []
        if rng.random() < 0.5:
            offered = pick_slots(ctx, 2)
            messages.append(rng.choice(OFFER).format(slots=slots_text(ctx, offered)))
            chosen = rng.choice(offered)
        else:
            chosen = pick_slots(ctx, 1)[0]
        messages.append(rng.choice(PENDING).format(when=when_text(ctx, chosen, label_kind(rng))))
        return messages, Gold("unclear", None, tuple(offered))
    raise BuildError(f"unknown template kind {kind}")


def template_items(rng: random.Random, count: int) -> list[Item]:
    plan = [kind for kind, n in TEMPLATE_PLAN for _ in range(n)]
    if len(plan) != count:
        raise BuildError(f"template plan has {len(plan)} items, expected {count}")
    items: list[Item] = []
    for kind in plan:
        ctx = Ctx(rng=rng, as_of=pick_as_of(rng), prospect_zone=rng.choice(PROSPECT_ZONES))
        messages, gold = build_template(kind, ctx)
        items.append(Item(ctx.as_of, ctx.prospect_zone, messages, gold, ctx.tags, "template"))
    return items


# --------------------------------------------------------------------------------------------------
# Derived tags and validation
# --------------------------------------------------------------------------------------------------

WEEKDAY_DATE = re.compile(
    rf"\b({'|'.join(WEEKDAYS)}),? (?:(\d{{1,2}}) ({'|'.join(MONTHS)})|({'|'.join(MONTHS)}) (\d{{1,2}}))\b"
)


def derived_tags(item: Item) -> set[str]:
    tags: set[str] = set()
    instants = ([item.gold.time_utc] if item.gold.time_utc else []) + list(item.gold.offered_utc)
    week = timedelta(days=7)
    for key in (item.prospect_zone, HOST_ZONE):
        for t in instants:
            offsets = {offset_seconds(key, x) for x in (t - week, t, t + week, item.as_of)}
            if len(offsets) > 1:
                tags.add("dst_week")
        if key == item.prospect_zone and instants and offset_seconds(key, instants[0]) % 3600:
            tags.add("half_hour_zone")
    if len(item.messages) > 1:
        tags.add("multi_message")
    return tags


def check_weekdays(item: Item) -> None:
    """Every "Weekday D Month" in the text must be a real date near as_of."""
    for text in item.messages:
        for match in WEEKDAY_DATE.finditer(text):
            weekday = match.group(1)
            day = int(match.group(2) or match.group(5))
            month = MONTHS.index(match.group(3) or match.group(4)) + 1
            year = item.as_of.year if month >= item.as_of.month else item.as_of.year + 1
            if WEEKDAYS[date(year, month, day).weekday()] != weekday:
                raise BuildError(f"{weekday} {day}/{month}/{year} is not a {weekday}: {text!r}")


def validate(items: Sequence[Item]) -> None:
    if len(items) != N_TOTAL:
        raise BuildError(f"expected {N_TOTAL} items, got {len(items)}")
    for item in items:
        if not 1 <= len(item.messages) <= 4:
            raise BuildError(f"{len(item.messages)} messages: {item.messages!r}")
        zone(item.prospect_zone)
        gold = item.gold
        if gold.status not in STATUSES:
            raise BuildError(f"bad status {gold.status}")
        # booked / rescheduled / cancelled may have a null time (none stated); only the hand-written
        # items tagged ``no_time`` use that, so a template bug cannot drop a time silently.
        if gold.time_utc is None and gold.status in ("booked", "rescheduled") and "no_time" not in item.tags:
            raise BuildError(f"{gold.status} without a time: {item.messages!r}")
        if gold.status in ("not_booked", "unclear") and gold.time_utc is not None:
            raise BuildError(f"{gold.status} with a time: {item.messages!r}")
        check_weekdays(item)


def stratified_split(strata: Sequence[tuple[str, str]]) -> list[str]:
    """Seeded split: shuffle each stratum, concatenate the strata in sorted order, every third is dev."""
    rng = random.Random(SEED)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(strata):
        groups.setdefault(key, []).append(index)
    order: list[int] = []
    for key in sorted(groups):
        members = groups[key]
        rng.shuffle(members)
        order.extend(members)
    splits = ["test"] * len(strata)
    for position, index in enumerate(order):
        if position % 3 == 0:
            splits[index] = "dev"
    return splits


def build() -> list[str]:
    """Return the dataset as JSON lines (without trailing newlines), ordered by id."""
    rng = random.Random(SEED)
    hard = hard_case_items()
    items = template_items(rng, N_TOTAL - len(hard)) + hard
    for item in items:
        item.tags |= derived_tags(item)
    validate(items)
    rng.shuffle(items)
    ids = [f"be-{i:03d}" for i in range(1, len(items) + 1)]
    splits = stratified_split([(item.source, item.gold.status) for item in items])
    if splits.count("dev") != N_DEV:
        raise BuildError(f"dev split has {splits.count('dev')} items, expected {N_DEV}")
    return [item.line(i, split) for i, item, split in zip(ids, items, splits, strict=True)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the labelled belief-extraction dataset.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSONL path")
    args = parser.parse_args(argv)
    lines = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    print(f"wrote {len(lines)} items to {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

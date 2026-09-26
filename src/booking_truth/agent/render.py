"""Text rendered by code: slot labels, booking sentences, confirmation lines and fixed templates.

Every time the prospect sees in a code-rendered text comes from here, converted with ``zoneinfo`` from a UTC
instant, so labels, quick replies and confirmations cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from booking_truth.timeutil import ensure_utc

REFERENCE_CHARS = 8

# Fixed templates -----------------------------------------------------------------------------------------

SAFE_NOT_BOOKED = "I haven't booked anything yet."
SAFE_NOT_CHANGED = "I haven't changed your booking yet."
NEXT_STEP_LOOK = "Would you like me to look for available times?"
NEXT_STEP_HANDOFF = (
    "Our calendar isn't available right now, so I've asked a colleague to follow up with you by email."
)
UNCONFIRMED = "I couldn't confirm the booking just now. A colleague will confirm it by email shortly."
LEAD_BUSY = "I'm still working on your previous message — please try again in a moment."
SESSION_ENDED = (
    "This conversation has reached its limit. Please start a new conversation if you need anything else."
)
LLM_UNAVAILABLE = (
    "Sorry, I can't answer right now. Nothing has been booked or changed. Please try again in a few minutes."
)
TOOL_LOOP_EXHAUSTED = (
    "Sorry, I couldn't finish that just now. Nothing new has been booked or changed. "
    "Could you say that again?"
)


def _zone(zone: str) -> ZoneInfo:
    return ZoneInfo(zone)


def local(instant: datetime, zone: str) -> datetime:
    return ensure_utc(instant).astimezone(_zone(zone))


def utc_offset(instant: datetime, zone: str) -> str:
    """``UTC+02:00`` / ``UTC-04:00`` / ``UTC+05:45`` at that instant."""
    offset = local(instant, zone).utcoffset() or timedelta(0)
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours:02d}:{mins:02d}"


def clock(value: datetime) -> str:
    """``3:00 PM``."""
    hour = value.hour % 12 or 12
    return f"{hour}:{value.minute:02d} {'AM' if value.hour < 12 else 'PM'}"


def slot_label(start: datetime, zone: str, *, now: datetime | None = None) -> str:
    """``Tuesday 6 October, 3:00 PM`` in ``zone``; the year is added when it is not the year of ``now``."""
    at = local(start, zone)
    text = f"{at:%A} {at.day} {at:%B}"
    if now is not None and local(now, zone).year != at.year:
        text += f" {at.year}"
    return f"{text}, {clock(at)}"


def long_label(start: datetime, zone: str) -> str:
    """``Tuesday, 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00)``."""
    at = local(start, zone)
    return f"{at:%A}, {at.day} {at:%B} {at.year}, {clock(at)} {zone} ({utc_offset(start, zone)})"


def reference(ref: str) -> str:
    """The short booking reference shown to the prospect: the first 8 characters of the uid."""
    return ref[:REFERENCE_CHARS]


def confirmation_line(
    action: str, start: datetime, zone: str, ref: str, *, previous_start: datetime | None = None
) -> str:
    """The code-rendered confirmation of a verified write (``rendered_confirmation``)."""
    when = long_label(start, zone)
    if action == "booked":
        return f"Booked: {when} · reference {reference(ref)}"
    if action == "rescheduled":
        return f"Rescheduled: your call is now {when} · reference {reference(ref)}"
    if action == "cancelled":
        return f"Cancelled: your call on {when} is cancelled · reference {reference(ref)}"
    raise ValueError(f"unknown booking action {action!r}")


def zone_statement(zone: str, instant: datetime, *, browser: bool = False) -> str:
    """The statement-back line for a resolved or hinted zone."""
    source = " based on your browser" if browser else ""
    return f"I'll use {zone} ({utc_offset(instant, zone)}) for times{source} — tell me if that's wrong."


# Code-path replies ---------------------------------------------------------------------------------------


def offer_text(labels: Sequence[str], zone: str, *, lead_in: str = "Here are some open times") -> str:
    lines = "\n".join(f"- {label}" for label in labels)
    return f"{lead_in} (shown in {zone}):\n{lines}\nWhich one would you like?"


def no_slots_text(zone: str) -> str:
    return f"I couldn't find any open times in that range ({zone}). Would you like me to look at later dates?"


def booked_text(start: datetime, zone: str, ref: str) -> str:
    return (
        f"You're booked for {slot_label(start, zone)} ({zone}). Reference {reference(ref)}. "
        "The calendar invite is on its way to your email."
    )


def rescheduled_text(start: datetime, zone: str, ref: str) -> str:
    return f"Done: I've moved your call to {slot_label(start, zone)} ({zone}). Reference {reference(ref)}."


def cancelled_text(start: datetime | None, zone: str) -> str:
    if start is None:
        return "Your call is cancelled. Nothing is booked for you now."
    return f"Your call on {slot_label(start, zone)} ({zone}) is cancelled. Nothing is booked for you now."


def reschedule_offer_text(existing: datetime, new: datetime, zone: str) -> str:
    return (
        f"You already have a call booked for {slot_label(existing, zone)} ({zone}). "
        f"Would you like me to move it to {slot_label(new, zone)} instead?"
    )


def slot_taken_text() -> str:
    return "Sorry, that time was just taken by someone else, so nothing is booked yet."


def expired_slot_text() -> str:
    return "Sorry, that option has expired, so nothing is booked yet."


def calendar_error_text(*, changed: bool = False) -> str:
    what = "changed" if changed else "booked"
    return (
        f"I'm sorry, the calendar didn't accept that just now, so nothing is {what} yet. "
        "I can try again, or pass your request to a colleague."
    )


def unavailable_text() -> str:
    return (
        "I'm sorry, our calendar isn't available right now, so I can't offer times or book anything yet. "
        "I've passed your request to a colleague, who will email you to arrange a time."
    )


def not_allowed_text() -> str:
    return "I can only change bookings made in this conversation, so nothing was changed."


def unsupported_action_text() -> str:
    return "I can't use that button here. Please tell me in a message which time you'd like."

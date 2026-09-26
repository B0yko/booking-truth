"""Cal.com API v2 subset under its real paths: slots and bookings.

Mirrored operations (an adapter reaches them by swapping only its base URL):

- ``GET /v2/slots`` with ``cal-api-version: 2024-09-04``;
- ``POST /v2/bookings``, ``GET /v2/bookings/{uid}``, ``GET /v2/bookings``,
  ``POST /v2/bookings/{uid}/reschedule`` and ``POST /v2/bookings/{uid}/cancel`` with
  ``cal-api-version`` ``2024-08-13``, ``2026-02-25`` or ``2026-05-01``.

Envelopes, JSON key order, validation messages and error texts follow the Cal.com reference and live
observations; ``docs/sandbox-fidelity.md`` lists the evidence for each and every known deviation. The sandbox
has exactly one event type (``seed.event_type_id``) owned by one host, and the host calendar is shared with
the other mirrored calendars.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import math
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Request
from fastapi.responses import Response

from booking_truth.sandbox.common import Call, Outcome, SetupBooking, VendorApi, request_url, run_call
from booking_truth.sandbox.state import SandboxState
from booking_truth.timeutil import iso_ms_z, parse_iso

VERSION_HEADER: Final = "cal-api-version"
SLOTS_VERSION: Final = "2024-09-04"
BOOKING_VERSIONS: Final = ("2024-08-13", "2026-02-25", "2026-05-01")
CURSOR_LIST_VERSION: Final = "2026-05-01"

# Verbatim Cal.com texts -------------------------------------------------------------------------------

MSG_INVALID_KEY: Final = "ApiAuthStrategy - api key - Your api key is not valid"
MSG_NO_AUTH: Final = (
    "ApiAuthStrategy - No authentication method provided. Either pass an API key as 'Bearer' header or "
    "OAuth client credentials as 'x-cal-secret-key' and 'x-cal-client-id' headers"
)
MSG_LIST_NO_AUTH: Final = (
    "PermissionsGuard - no authentication provided. Provide either authorization bearer token containing "
    "managed user access token or oAuth client id in 'x-cal-client-id' header."
)
MSG_SLOTS_EVENT_TYPE_NOT_FOUND: Final = "Event Type not found"
MSG_TAKEN: Final = "User either already has booking at this time or is not available"
MSG_OUT_OF_BOUNDS: Final = (
    "The event type can't be booked at the \"start\" time provided. This could be because it's too soon "
    "(violating the minimum booking notice) or too far in the future (outside the event's scheduling "
    "window). Try fetching available slots first using the GET /v2/slots endpoint and then make a booking "
    'with "start" time equal to one of the available slots.'
)
MSG_PAST: Final = "Attempting to book a meeting in the past."
MSG_METADATA: Final = (
    "Metadata must have at most 50 keys, each key up to 40 characters, and string values up to 500 "
    "characters."
)
MSG_CONTACT: Final = "Attendee must have at least one contact method (email or phone number)"
MSG_EVENT_LOOKUP: Final = (
    "Either eventTypeId or eventTypeSlug + username or eventTypeSlug + teamSlug must be provided"
)
MSG_INVALID_STATUS: Final = "Invalid status. Allowed are upcoming, recurring, past, cancelled, unconfirmed"
MSG_CANCEL_ENDED: Final = "Cannot cancel a booking that has already ended"
MSG_SINGLE_LENGTH: Final = (
    "Can't specify 'lengthInMinutes' because event type does not have multiple possible lengths. Please, "
    "remove the 'lengthInMinutes' field from the request."
)
MSG_ISO: Final = "{name} must be a valid ISO 8601 date string"
MSG_MIN_USERNAMES: Final = "The array must contain at least 2 elements."

LANGUAGES: Final = (
    "ar", "ca", "de", "es", "eu", "he", "id", "ja", "lv", "pl", "ro", "sr", "th", "vi", "az", "cs", "el",
    "es-419", "fi", "hr", "it", "km", "nl", "pt", "ru", "sv", "tr", "zh-CN", "bg", "da", "en", "et", "fr",
    "hu", "iw", "ko", "no", "pt-BR", "sk", "sl", "ta", "uk", "zh-TW", "bn",
)  # fmt: skip
LIST_STATUSES: Final = ("upcoming", "recurring", "past", "cancelled", "unconfirmed")

_EXCEPTIONS: Final[dict[int, tuple[str, str]]] = {
    400: ("BadRequestException", "Bad Request"),
    401: ("UnauthorizedException", "Unauthorized"),
    403: ("ForbiddenException", "Forbidden"),
    404: ("NotFoundException", "Not Found"),
    500: ("InternalServerErrorException", "Internal Server Error"),
    504: ("GatewayTimeoutException", "Gateway Timeout"),
}

# Output key order of BookingOutput_2024_08_13 (class declaration order); absent keys are omitted.
BOOKING_KEYS: Final = (
    "id", "uid", "title", "description", "hosts", "status", "cancellationReason", "cancelledByEmail",
    "reschedulingReason", "rescheduledByEmail", "rescheduledFromUid", "rescheduledToUid", "start", "end",
    "duration", "eventTypeId", "eventType", "meetingUrl", "location", "absentHost", "createdAt", "updatedAt",
    "metadata", "rating", "icsUid", "attendees", "guests", "bookingFieldsResponses",
)  # fmt: skip

_CREATE_KEYS: Final = frozenset(
    {
        "start", "lengthInMinutes", "eventTypeId", "eventTypeSlug", "username", "teamSlug",
        "organizationSlug", "attendee", "guests", "meetingUrl", "location", "metadata",
        "bookingFieldsResponses", "routing", "emailVerificationCode", "allowConflicts",
        "allowBookingOutOfBounds", "skipBookingLimits", "instant", "recurrenceCount", "rrHostSubsetIds",
    }
)  # fmt: skip
_ATTENDEE_KEYS: Final = frozenset({"name", "email", "timeZone", "phoneNumber", "language"})
_RESCHEDULE_KEYS: Final = frozenset(
    {
        "start", "reschedulingReason", "rescheduledBy", "emailVerificationCode", "rescheduleWithSameHost",
        "allowConflicts", "allowBookingOutOfBounds", "skipBookingLimits", "seatUid", "rrHostSubsetIds",
    }
)  # fmt: skip
_CANCEL_KEYS: Final = frozenset({"cancellationReason", "cancelSubsequentBookings", "seatUid"})
# GET /v2/slots input classes: the shared properties (``type`` is a whitelisted discriminator) and each lookup
# class's own properties.
_SLOTS_BASE_KEYS: Final = frozenset(
    {
        "start", "end", "timeZone", "duration", "format", "bookingUidToReschedule", "rrHostSubsetIds", "type",
    }
)  # fmt: skip
_SLOTS_MODE_KEYS: Final[dict[str, frozenset[str]]] = {
    "id": frozenset({"eventTypeId"}),
    "user": frozenset({"eventTypeSlug", "username", "organizationSlug"}),
    "team": frozenset({"eventTypeSlug", "teamSlug", "organizationSlug"}),
    "usernames": frozenset({"usernames", "organizationSlug"}),
}
_LIST_DATE_FILTERS: Final = (
    "afterStart", "beforeEnd", "afterCreatedAt", "beforeCreatedAt", "afterUpdatedAt", "beforeUpdatedAt",
)  # fmt: skip
# Custom messages of GetBookingsInput_2024_08_13, in declaration order (the last one repeats "SortCreated"
# in Cal.com's source).
_LIST_SORT_MESSAGES: Final = {
    "sortStart": 'SortStart must be either "asc" or "desc".',
    "sortEnd": 'SortEnd must be either "asc" or "desc".',
    "sortCreated": 'SortCreated must be either "asc" or "desc".',
    "sortUpdatedAt": 'SortCreated must be either "asc" or "desc".',
}
_JS_INT_RE: Final = re.compile(r"([+-]?)(0[xX][0-9a-fA-F]+|[0-9]+)")
_JS_NUMBER_RE: Final = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_BOOKING_SORT_FIELD: Final = {
    "sortStart": "start",
    "sortEnd": "end",
    "sortCreated": "createdAt",
    "sortUpdatedAt": "updatedAt",
}
_OFFER_NOT_SEEN_FAULTS: Final = frozenset(
    {"timeout", "commit_then_timeout", "malformed", "error_500", "not_found"}
)
_BASE58: Final = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
_ISO_RE: Final = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?)?", re.ASCII
)
# A plain approximation of validator.js ``isURL`` defaults (optional http/https/ftp scheme, a dotted host).
_URL_RE: Final = re.compile(
    r"(?:(?:https?|ftp)://)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?::[0-9]+)?(?:[/?#]\S*)?"
)
_FAR_FUTURE: Final = datetime(2100, 1, 1, tzinfo=UTC)
_MAX_YEAR: Final = 9998


# Envelopes and validation messages ---------------------------------------------------------------------


def error(status: int, message: str, *, url: str, now: datetime) -> Outcome:
    """The standard Cal.com error envelope (NestJS exception filter), key order included."""
    code, reason = _EXCEPTIONS[status]
    return Outcome(
        status,
        {
            "status": "error",
            "timestamp": iso_ms_z(now),
            "path": url,
            "error": {
                "code": code,
                "message": message,
                "details": {"message": message, "error": reason, "statusCode": status},
            },
        },
    )


def _err(call: Call, status: int, message: str) -> Outcome:
    return error(status, message, url=call.url, now=call.state.now())


def _success(data: Any) -> dict[str, Any]:
    return {"status": "success", "data": data}


@dataclass(frozen=True)
class FieldError:
    """One class-validator error, as the Cal.com version-specific pipes flatten it into ``message``."""

    prop: str
    constraints: tuple[str, ...] = ()
    children: tuple[FieldError, ...] = ()


def format_errors(errors: Sequence[FieldError]) -> str:
    """``"<prop> property is wrong,<constraints> <children>"`` items joined by ``", "``.

    The trailing space after each item is part of the real format.
    """
    return ", ".join(
        f"{e.prop} property is wrong,{', '.join(e.constraints)} {format_errors(e.children)}" for e in errors
    )


def _not_allowed(prop: str) -> FieldError:
    return FieldError(prop, (f"property {prop} should not exist",))


def _invalid_iso(prop: str) -> FieldError:
    return FieldError(prop, (MSG_ISO.format(name=prop),))


def _validation_failed(call: Call, errors: Sequence[FieldError]) -> Outcome:
    return _err(call, 400, format_errors(errors))


def constraint_error(prop: str, constraints: dict[str, str]) -> dict[str, Any]:
    """One entry of the global ValidationPipe's ``details.errors`` (keys as observed live)."""
    return {"property": prop, "children": [], "constraints": dict(constraints)}


def pipe_error(call: Call, errors: list[dict[str, Any]]) -> Outcome:
    """400 from Cal.com's global ValidationPipe (``exceptionFactory: new BadRequestException({errors})``):
    a generic message and the class-validator errors under ``details.errors``. It serves the legacy handlers
    and the ``GET /v2/bookings`` query, which has no version-specific pipe."""
    return Outcome(
        400,
        {
            "status": "error",
            "timestamp": iso_ms_z(call.state.now()),
            "path": call.url,
            "error": {
                "code": "BadRequestException",
                "message": "Bad Request Exception",
                "details": {"errors": errors},
            },
        },
    )


# Value helpers --------------------------------------------------------------------------------------


def parse_datetime(value: object) -> datetime | None:
    """An ISO 8601 date or date-time. A value without an offset is read as UTC, as on Cal.com's servers.

    Instants whose UTC form falls outside years 1..9998 are treated as invalid, so that date arithmetic on
    them (a day's end, a booking's end, the slot walk) cannot overflow.
    """
    if not isinstance(value, str) or not _ISO_RE.fullmatch(value):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        parsed = parsed.astimezone(UTC)
    except (ValueError, OverflowError):
        return None
    return parsed if parsed.year <= _MAX_YEAR else None


def _is_date_only(value: str) -> bool:
    return len(value) == 10


def valid_zone(name: object) -> ZoneInfo | None:
    if not isinstance(name, str) or not name or name.startswith(("/", ".")):
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def _is_int(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


def _metadata_ok(value: object) -> bool:
    """Cal.com's metadata rule: an object with at most 50 keys of at most 40 characters each, whose values are
    strings of at most 500 characters, numbers or booleans (nested objects and arrays are rejected)."""
    if not isinstance(value, dict) or len(value) > 50:
        return False
    for key, item in value.items():
        if len(key) > 40:
            return False
        if isinstance(item, bool | int | float):
            continue
        if isinstance(item, str) and len(item) <= 500:
            continue
        return False
    return True


def short_uuid(value: uuid.UUID) -> str:
    """Flickr base58 of a UUID, left-padded to 22 characters (the ``short-uuid`` package's encoding)."""
    number = value.int
    digits: list[str] = []
    while number:
        number, remainder = divmod(number, 58)
        digits.append(_BASE58[remainder])
    return "".join(reversed(digits)).rjust(22, _BASE58[0])


def _new_uid(state: SandboxState, booking_id: int, start: datetime) -> str:
    """Cal.com derives the uid from ``uuidv5("<organizer>:<start>:<Date.now()>")``.

    The booking id is added so that uids stay unique under a frozen clock.
    """
    now_ms = int(state.now().timestamp() * 1000)
    name = f"{state.seed.host_username}:{iso_ms_z(start)}:{now_ms}:{booking_id}"
    return short_uuid(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _slot_time(value: datetime, zone_name: str | None) -> str:
    """Luxon ``toISO()``: ``Z`` for UTC, otherwise the zone's offset, always with milliseconds."""
    if zone_name is None or zone_name.lower() in ("utc", "gmt"):
        return iso_ms_z(value)
    return value.astimezone(ZoneInfo(zone_name)).isoformat(timespec="milliseconds")


def canonical(fields: dict[str, Any]) -> dict[str, Any]:
    """A booking object with its keys in Cal.com's output order."""
    return {key: fields[key] for key in BOOKING_KEYS if key in fields}


def _find(state: SandboxState, uid: str) -> int | None:
    for index, booking in enumerate(state.calcom_bookings):
        if booking["uid"] == uid:
            return index
    return None


def _host(state: SandboxState) -> dict[str, Any]:
    seed = state.seed
    return {
        "id": seed.host_id,
        "name": seed.host_name,
        "email": seed.host_email,
        "displayEmail": seed.host_email,
        "username": seed.host_username,
        "timeZone": seed.host_timezone,
    }


def _location(value: object) -> str:
    """``location`` in the output: the address, link or phone of a location object, else the video default."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        for key in ("address", "link", "phone", "location"):
            if isinstance(value.get(key), str) and value[key]:
                return str(value[key])
        integration = value.get("integration")
        if isinstance(integration, str) and integration and integration != "cal-video":
            return f"integrations:{integration}"
    return "integrations:daily"


# Booking core (shared by the vendor routes and POST /_control/bookings) -------------------------------


@dataclass(frozen=True)
class Attendee:
    name: str
    email: str | None
    time_zone: str
    language: str = "en"
    phone: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "email": self.email or "",
            "displayEmail": self.email or "",
            "timeZone": self.time_zone,
            "language": self.language,
            "absent": False,
        }
        if self.phone:
            out["phoneNumber"] = self.phone
        return out


@dataclass(frozen=True)
class BookingRequest:
    start: datetime
    attendee: Attendee
    metadata: dict[str, Any] = field(default_factory=dict)
    guests: list[str] | None = None
    location: object = None
    responses: dict[str, Any] = field(default_factory=dict)
    title: str | None = None


def check_bookable(
    state: SandboxState, start: datetime, end: datetime, *, url: str, exclude_uid: str | None = None
) -> Outcome | None:
    """Create/reschedule availability checks in Cal.com's order: past, bounds, then conflicts and hours."""
    now = state.now()
    seed = state.seed
    if start < now:
        return error(400, MSG_PAST, url=url, now=now)
    if start < now + timedelta(minutes=seed.min_notice_minutes) or start >= now + timedelta(
        days=seed.horizon_days
    ):
        return error(400, MSG_OUT_OF_BOUNDS, url=url, now=now)
    if not state.host_available(start, end, exclude_uid=exclude_uid):
        return error(400, MSG_TAKEN, url=url, now=now)
    return None


def _responses(attendee: Attendee, guests: list[str] | None, custom: dict[str, Any]) -> dict[str, Any]:
    """``bookingFieldsResponses``: ``guests`` only when the request sent it (a real hosted payload has
    none), then ``displayEmail`` and, whenever ``guests`` is a list, ``displayGuests`` (the output service's
    rule)."""
    out: dict[str, Any] = {"email": attendee.email or "", "name": attendee.name}
    if guests is not None:
        out["guests"] = list(guests)
    for key, value in custom.items():
        if key not in out:
            out[key] = "" if value is None else value
    if attendee.phone:
        out["attendeePhoneNumber"] = attendee.phone
    out["displayEmail"] = attendee.email or ""
    if guests is not None:
        out["displayGuests"] = list(guests)
    return out


def book(state: SandboxState, request: BookingRequest, *, url: str) -> Outcome:
    """Create a booking on the one event type. The caller holds ``state.lock``."""
    seed = state.seed
    start = request.start
    end = start + timedelta(minutes=seed.event_length_minutes)
    rejected = check_bookable(state, start, end, url=url)
    if rejected is not None:
        return rejected
    booking_id = state.next_id()
    uid = _new_uid(state, booking_id, start)
    now = iso_ms_z(state.now())
    location = _location(request.location)
    fields: dict[str, Any] = {
        "id": booking_id,
        "uid": uid,
        "title": request.title or f"{seed.event_title} between {seed.host_name} and {request.attendee.name}",
        "description": "",
        "hosts": [_host(state)],
        "status": "accepted",
        "cancellationReason": "",
        "cancelledByEmail": "",
        "rescheduledByEmail": None,
        "start": iso_ms_z(start),
        "end": iso_ms_z(end),
        "duration": seed.event_length_minutes,
        "eventTypeId": seed.event_type_id,
        "eventType": {"id": seed.event_type_id, "slug": seed.event_type_slug},
        # The deprecated meetingUrl is always the location (output service; a real hosted payload shows a
        # plain-text location in both fields).
        "meetingUrl": location,
        "location": location,
        "absentHost": False,
        "createdAt": now,
        "updatedAt": now,
        "metadata": dict(request.metadata),
        "rating": None,
        "icsUid": f"{uid}@Cal.com",
        "attendees": [request.attendee.to_json()],
        "bookingFieldsResponses": _responses(request.attendee, request.guests, request.responses),
    }
    if request.guests is not None:
        fields["guests"] = list(request.guests)
    booking = canonical(fields)
    state.calcom_bookings.append(booking)
    return Outcome(201, booking)


def _with_platform_flag(booking: dict[str, Any]) -> dict[str, Any]:
    """Create and reschedule responses carry ``isPlatformManagedUserBooking`` (appended, not stored)."""
    data = copy.deepcopy(booking)
    data["isPlatformManagedUserBooking"] = False
    return data


# Version routing ----------------------------------------------------------------------------------------


def _route_not_found(call: Call) -> Outcome:
    """Nest's router-level 404, raised before any guard or pipe runs."""
    return _err(call, 404, f"Cannot {call.method} {call.url}")


def _slots_route(call: Call) -> Outcome | None:
    """Only ``2024-09-04`` has ``GET /v2/slots``; any other value reaches a controller without that route."""
    if call.header(VERSION_HEADER) == SLOTS_VERSION:
        return None
    return _route_not_found(call)


def _bookings_route(call: Call) -> Outcome | None:
    """Version routing of the bookings paths, answered before auth as on Cal.com.

    ``2024-09-04`` has no bookings controller, so every bookings path is a 404. A missing or unknown value
    falls back to the legacy 2024-04-15 controller: it has no ``POST /:uid/reschedule`` (404), and its create
    has no auth guard, so the global ValidationPipe answers first (a live probe without credentials gets that
    400).
    The legacy get, list and cancel are answered after auth by :func:`_bookings_legacy`.
    """
    version = call.header(VERSION_HEADER)
    if version in BOOKING_VERSIONS:
        return None
    if version == SLOTS_VERSION or call.group == "bookings.reschedule":
        return _route_not_found(call)
    if call.group == "bookings.create":
        return _legacy(call)
    return None


def _bookings_legacy(call: Call) -> Outcome | None:
    if call.header(VERSION_HEADER) in BOOKING_VERSIONS:
        return None
    return _legacy(call)


def _legacy(call: Call) -> Outcome:
    """A missing or unknown version header reaches Cal.com's legacy 2024-04-15 handlers (global ValidationPipe
    envelope). Their validation of a create body is mirrored; the handlers themselves are not."""
    errors: list[dict[str, Any]] = []
    if call.group == "bookings.create":
        body = call.body if isinstance(call.body, dict) else {}
        if not isinstance(body.get("start"), str):
            errors.append(constraint_error("start", {"isString": "start must be a string"}))
        event_type = body.get("eventTypeId")
        if isinstance(event_type, bool) or not isinstance(event_type, int | float):
            message = "eventTypeId must be a number conforming to the specified constraints"
            errors.append(constraint_error("eventTypeId", {"isNumber": message}))
        if valid_zone(body.get("timeZone")) is None:
            message = "timeZone must be a valid IANA time-zone"
            errors.append(constraint_error("timeZone", {"isTimeZone": message}))
        if not isinstance(body.get("language"), str):
            errors.append(constraint_error("language", {"isString": "language must be a string"}))
        if not isinstance(body.get("metadata"), dict):
            errors.append(constraint_error("metadata", {"isObject": "metadata must be an object"}))
    if not errors:
        message = (
            f"the sandbox mirrors this route for cal-api-version {', '.join(BOOKING_VERSIONS)} only; "
            "the legacy 2024-04-15 handler is not mirrored"
        )
        errors.append(constraint_error(VERSION_HEADER, {"sandbox": message}))
    return pipe_error(call, errors)


# GET /v2/slots -------------------------------------------------------------------------------------------


def _slots_mode(call: Call) -> str:
    """The input class ``GetSlotsInputPipe`` picks, by which keys are present."""
    keys = set(call.params)
    if "eventTypeId" in keys:
        return "id"
    if {"username", "eventTypeSlug"} <= keys:
        return "user"
    if {"teamSlug", "eventTypeSlug"} <= keys:
        return "team"
    return "usernames"


def _slots_errors(call: Call, mode: str) -> list[FieldError]:
    """Unknown keys first, then the chosen class's own properties, then the inherited shared ones (start, end,
    timeZone, duration, format, bookingUidToReschedule, rrHostSubsetIds)."""
    allowed = _SLOTS_BASE_KEYS | _SLOTS_MODE_KEYS[mode]
    errors = [_not_allowed(name) for name in call.params if name not in allowed]
    if mode == "id" and js_parse_int(call.param("eventTypeId") or "") is None:
        message = "eventTypeId must be a number conforming to the specified constraints"
        errors.append(FieldError("eventTypeId", (message,)))
    if mode == "usernames":
        raw = call.param("usernames")
        if raw is None:
            constraints: tuple[str, ...] = (
                "each value in usernames must be a string",
                MSG_MIN_USERNAMES,
                "usernames must be an array",
            )
            errors.append(FieldError("usernames", constraints))
        elif len(raw.split(",")) < 2:
            errors.append(FieldError("usernames", (MSG_MIN_USERNAMES,)))
        if call.param("organizationSlug") is None:
            errors.append(FieldError("organizationSlug", ("organizationSlug must be a string",)))
    for name in ("start", "end"):
        if parse_datetime(call.param(name)) is None:
            errors.append(_invalid_iso(name))
    zone_name = call.param("timeZone")
    if zone_name is not None and valid_zone(zone_name) is None:
        errors.append(FieldError("timeZone", ("timeZone must be a valid IANA time-zone",)))
    duration = call.param("duration")
    if duration is not None and js_parse_int(duration) is None:
        message = "duration must be a number conforming to the specified constraints"
        errors.append(FieldError("duration", (message,)))
    slot_format = (call.param("format") or "").lower()
    if slot_format and slot_format not in ("range", "time"):
        errors.append(FieldError("format", ("slotFormat must be either 'range' or 'time'",)))
    host_ids = call.param("rrHostSubsetIds")
    if host_ids is not None and any(js_parse_int(part) is None for part in host_ids.split(",")):
        message = "each value in rrHostSubsetIds must be a number conforming to the specified constraints"
        errors.append(FieldError("rrHostSubsetIds", (message,)))
    return errors


def _slots(call: Call) -> Outcome:
    state, seed = call.state, call.state.seed
    mode = _slots_mode(call)
    errors = _slots_errors(call, mode)
    if errors:
        return _validation_failed(call, errors)
    start_raw, end_raw = call.param("start"), call.param("end")
    start, end = parse_datetime(start_raw), parse_datetime(end_raw)
    assert start is not None
    assert end is not None
    assert start_raw is not None
    assert end_raw is not None
    zone_name = call.param("timeZone")
    slot_format = (call.param("format") or "").lower()
    duration = js_parse_int(call.param("duration") or "")

    slug, username, team = call.param("eventTypeSlug"), call.param("username"), call.param("teamSlug")
    if mode == "id":
        if js_parse_int(call.param("eventTypeId") or "") != seed.event_type_id:
            return _err(call, 404, MSG_SLOTS_EVENT_TYPE_NOT_FOUND)
    elif mode == "user":
        if username != seed.host_username:
            return _err(call, 404, f"User with username {username} not found")
        if slug != seed.event_type_slug:
            return _err(call, 404, MSG_SLOTS_EVENT_TYPE_NOT_FOUND)
    elif mode == "team":
        return _err(call, 404, f"Team with slug {team} not found")
    else:
        for name in (call.param("usernames") or "").split(","):
            if name.strip() != seed.host_username:
                return _err(call, 404, f"User with username {name.strip()} not found")

    # A date-only start means 00:00:00 UTC and a date-only end 23:59:59 UTC; the end is inclusive.
    window_end = end + timedelta(hours=23, minutes=59, seconds=59) if _is_date_only(end_raw) else end
    call.extra["window"] = (start, window_end)
    if window_end < start:
        return Outcome(200, {"data": {}, "status": "success"})
    starts = state.free_starts(
        start, window_end + timedelta(microseconds=1), exclude_uid=call.param("bookingUidToReschedule")
    )
    length = timedelta(minutes=duration if duration and duration > 0 else seed.event_length_minutes)
    zone = ZoneInfo(zone_name) if zone_name else UTC
    data: dict[str, list[dict[str, str]]] = {}
    for slot in starts:
        item = {"start": _slot_time(slot, zone_name)}
        if slot_format == "range":
            item["end"] = _slot_time(slot + length, zone_name)
        data.setdefault(slot.astimezone(zone).date().isoformat(), []).append(item)
    return Outcome(200, {"data": data, "status": "success"})


def _busy_periods(state: SandboxState, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Every period inside ``[start, end]`` that is not a free slot: off hours, bookings, notice, horizon."""
    length = timedelta(minutes=state.seed.event_length_minutes)
    periods: list[tuple[datetime, datetime]] = []
    cursor = start
    for slot in state.free_starts(start, end + timedelta(microseconds=1)):
        if slot > cursor:
            periods.append((cursor, slot))
        cursor = max(cursor, slot + length)
    if cursor < end:
        periods.append((cursor, end))
    return periods[:1000]


def offered_starts(response: object) -> list[datetime]:
    """Slot starts of a successful ``GET /v2/slots`` body (both formats)."""
    if not isinstance(response, dict) or not isinstance(response.get("data"), dict):
        return []
    starts: list[datetime] = []
    for items in response["data"].values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("start"), str):
                starts.append(parse_iso(item["start"]))
    return starts


def last_offered_starts(state: SandboxState) -> list[datetime]:
    """Slot starts of the most recent slots response the client actually received with its normal schema."""
    for entry in reversed(state.request_log):
        if (
            entry.group == "slots"
            and entry.completed
            and entry.status == 200
            and entry.fault not in _OFFER_NOT_SEEN_FAULTS
        ):
            return offered_starts(entry.response)
    return []


# POST /v2/bookings ----------------------------------------------------------------------------------------


def _attendee_errors(value: object) -> tuple[list[FieldError], Attendee | None]:
    """``CreateBookingAttendee``. class-validator reports a subclass's own rules first (``email``, then the
    class-level contact rule) and the inherited ones after (``name``, ``timeZone``, ``phoneNumber``,
    ``language``). The ``email`` and ``phoneNumber`` format checks are no-ops in Cal.com (a live probe with
    ``"email": "bad"`` passed), so only their types are checked here."""
    if value is None:
        return [FieldError("attendee", ("attendee should not be null or undefined",))], None
    if not isinstance(value, dict):
        return [FieldError("attendee", ("nested property attendee must be either object or array",))], None
    children = [_not_allowed(key) for key in value if key not in _ATTENDEE_KEYS]
    name, email, zone = value.get("name"), value.get("email"), value.get("timeZone")
    phone = value.get("phoneNumber")
    language = "en" if value.get("language") is None else value["language"]
    if email is not None and not isinstance(email, str):
        children.append(FieldError("email", ("email must be a string",)))
    if not email and not phone:
        children.append(FieldError("attendee email or phone", (MSG_CONTACT,)))
    if not isinstance(name, str):
        children.append(FieldError("name", ("name must be a string",)))
    if valid_zone(zone) is None:
        children.append(FieldError("timeZone", ("timeZone must be a valid IANA time-zone",)))
    if phone is not None and not isinstance(phone, str):
        children.append(FieldError("phoneNumber", ("phoneNumber must be a string",)))
    if language not in LANGUAGES:
        children.append(
            FieldError("language", (f"language must be one of the following values: {', '.join(LANGUAGES)}",))
        )
    if children:
        return [FieldError("attendee", (), tuple(children))], None
    assert isinstance(name, str)
    assert isinstance(zone, str)
    assert isinstance(language, str)
    return [], Attendee(
        name=name,
        email=email if isinstance(email, str) and email else None,
        time_zone=zone,
        language=language,
        phone=phone if isinstance(phone, str) and phone else None,
    )


def _metadata_errors(value: object) -> FieldError | None:
    """``@IsObject() @IsOptional() @ValidateMetadata()``: the metadata rule is registered first, and it walks
    any JS object, so an array of short strings passes it and fails only ``IsObject``."""
    if value is None:
        return None
    constraints: list[str] = []
    walked = dict(enumerate(value)) if isinstance(value, list) else value
    if not isinstance(walked, dict) or not _metadata_ok({str(k): v for k, v in walked.items()}):
        constraints.append(MSG_METADATA)
    if not isinstance(value, dict):
        constraints.append("metadata must be an object")
    return FieldError("metadata", tuple(constraints)) if constraints else None


def _string_list_errors(prop: str, value: object) -> FieldError | None:
    """``@IsArray() @IsString({each: true})``: the ``each`` rule is registered first and, for a value that is
    not an array, checks the value itself."""
    if value is None:
        return None
    constraints: list[str] = []
    items = value if isinstance(value, list) else [value]
    if not all(isinstance(item, str) for item in items):
        constraints.append(f"each value in {prop} must be a string")
    if not isinstance(value, list):
        constraints.append(f"{prop} must be an array")
    return FieldError(prop, tuple(constraints)) if constraints else None


def _create_errors(body: dict[str, Any]) -> tuple[list[FieldError], Attendee | None]:
    """``CreateBookingInput_2024_08_13`` through its pipe: unknown keys first, then every property in
    declaration order, then the class-level event type rule. Optional properties that are ``null`` are
    skipped, as ``@IsOptional`` does."""
    errors = [_not_allowed(key) for key in body if key not in _CREATE_KEYS]
    if parse_datetime(body.get("start")) is None:
        errors.append(_invalid_iso("start"))
    attendee_errors, attendee = _attendee_errors(body.get("attendee"))
    errors += attendee_errors
    responses = body.get("bookingFieldsResponses")
    if responses is not None and not isinstance(responses, dict):
        errors.append(FieldError("bookingFieldsResponses", ("bookingFieldsResponses must be an object",)))
    if body.get("eventTypeId") is not None and not _is_int(body["eventTypeId"]):
        errors.append(FieldError("eventTypeId", ("eventTypeId must be an integer number",)))
    for key in ("eventTypeSlug", "username", "teamSlug", "organizationSlug"):
        if body.get(key) is not None and not isinstance(body[key], str):
            errors.append(FieldError(key, (f"{key} must be a string",)))
    guests = _string_list_errors("guests", body.get("guests"))
    if guests is not None:
        errors.append(guests)
    meeting_url = body.get("meetingUrl")
    if meeting_url is not None and not (isinstance(meeting_url, str) and _URL_RE.fullmatch(meeting_url)):
        errors.append(FieldError("meetingUrl", ("meetingUrl must be a URL address",)))
    location = body.get("location")
    if location is not None and not isinstance(location, dict | str):
        errors.append(FieldError("location", ("location must be an object",)))
    metadata = _metadata_errors(body.get("metadata"))
    if metadata is not None:
        errors.append(metadata)
    length = body.get("lengthInMinutes")
    if length is not None:
        constraints: list[str] = []
        if isinstance(length, bool) or not isinstance(length, int | float) or length < 1:
            constraints.append("lengthInMinutes must not be less than 1")
        if not _is_int(length):
            constraints.append("lengthInMinutes must be an integer number")
        if constraints:
            errors.append(FieldError("lengthInMinutes", tuple(constraints)))
    if body.get("emailVerificationCode") is not None and not isinstance(body["emailVerificationCode"], str):
        errors.append(FieldError("emailVerificationCode", ("emailVerificationCode must be a string",)))
    for key in ("allowConflicts", "allowBookingOutOfBounds", "skipBookingLimits"):
        if body.get(key) is not None and not isinstance(body[key], bool):
            errors.append(FieldError(key, (f"{key} must be a boolean value",)))
    slug, username, team = body.get("eventTypeSlug"), body.get("username"), body.get("teamSlug")
    if not (body.get("eventTypeId") or (slug and (username or team))):
        errors.append(FieldError("eventTypeId or eventTypeSlug + username", (MSG_EVENT_LOOKUP,)))
    return errors, attendee


def _create(call: Call) -> Outcome:
    seed = call.state.seed
    body: dict[str, Any] = call.body if isinstance(call.body, dict) else {}
    errors, attendee = _create_errors(body)
    if errors:
        return _validation_failed(call, errors)
    start = parse_datetime(body.get("start"))
    assert start is not None
    assert attendee is not None

    event_type_id = body.get("eventTypeId")
    slug, username, team = body.get("eventTypeSlug"), body.get("username"), body.get("teamSlug")
    if event_type_id:
        if int(event_type_id) != seed.event_type_id:
            return _err(call, 404, f"Event type with id {int(event_type_id)} not found.")
    elif slug and username:
        if slug != seed.event_type_slug or username != seed.host_username:
            return _err(call, 404, f"Event type with slug {slug} belonging to user {username} not found.")
    else:
        return _err(call, 404, f"Event type with slug {slug} belonging to team {team} not found.")
    if body.get("lengthInMinutes"):
        # The sandbox's event type has one length; Cal.com rejects the field for such event types.
        return _err(call, 400, MSG_SINGLE_LENGTH)
    guests = body.get("guests")
    outcome = book(
        call.state,
        BookingRequest(
            start=start,
            attendee=attendee,
            metadata=dict(body.get("metadata") or {}),
            guests=[str(g) for g in guests] if isinstance(guests, list) else None,
            location=body.get("location") or body.get("meetingUrl"),
            responses=dict(body.get("bookingFieldsResponses") or {}),
        ),
        url=call.url,
    )
    if outcome.status != 201:
        return outcome
    return Outcome(201, _success(_with_platform_flag(outcome.body)))


# GET /v2/bookings/{uid} ------------------------------------------------------------------------------------


def _get(call: Call) -> Outcome:
    uid = call.path_param("uid")
    index = _find(call.state, uid)
    if index is None:
        return _err(call, 404, f"Booking with uid={uid} was not found in the database")
    return Outcome(200, _success(copy.deepcopy(call.state.calcom_bookings[index])))


# GET /v2/bookings ------------------------------------------------------------------------------------------


def js_parse_int(value: str) -> int | None:
    """JavaScript ``parseInt(value)``: leading whitespace, an optional sign, then the longest run of digits
    (``0x`` switches to hexadecimal). ``None`` stands for ``NaN``, and also for results beyond a double's
    range, which JavaScript turns into ``Infinity`` and class-validator's ``IsNumber`` rejects as well."""
    match = _JS_INT_RE.match(value.lstrip())
    if match is None or len(match.group(2)) > 300:
        return None
    sign, digits = match.groups()
    number = int(digits, 16) if digits[:2].lower() == "0x" else int(digits)
    return -number if sign == "-" else number


def js_number(value: str) -> float | None:
    """JavaScript ``Number(value)`` for a query string (``""`` is 0). ``None`` stands for ``NaN``."""
    text = value.strip()
    if not text:
        return 0.0
    if _JS_NUMBER_RE.fullmatch(text):
        return float(text)
    if re.fullmatch(r"0[xX][0-9a-fA-F]{1,256}", text):
        return float(int(text, 16))
    if re.fullmatch(r"[+-]?Infinity", text):
        return float(text.replace("Infinity", "inf"))
    return None


def encode_cursor(offset: int) -> str:
    raw = json.dumps({"v": 1, "o": offset}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def decode_cursor(cursor: str) -> int | None:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except (ValueError, binascii.Error):
        return None
    if not isinstance(payload, dict) or payload.get("v") != 1 or not _is_int(payload.get("o")):
        return None
    offset = int(payload["o"])
    return offset if offset >= 0 else None


def _timestamp(key: str) -> Callable[[dict[str, Any]], datetime]:
    return lambda booking: parse_iso(booking[key])


def _matches_status(booking: dict[str, Any], status: str, now: datetime) -> bool:
    end = parse_iso(booking["end"])
    state = booking["status"]
    active = state not in ("cancelled", "rejected")
    if status == "upcoming":
        return end >= now and active
    if status == "past":
        return end <= now and active
    if status == "cancelled":
        return not active
    if status == "unconfirmed":
        return end >= now and state == "pending"
    return False  # recurring: the sandbox's event type is not recurring


@dataclass
class _ListQuery:
    """``GET /v2/bookings`` query after Cal.com's transforms and validation."""

    cursor_mode: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    event_type_ids: list[int] | None = None
    team_ids: list[int] | None = None
    bounds: dict[str, datetime] = field(default_factory=dict)
    sort: tuple[str, bool] | None = None
    offset: int = 0
    page_size: int = 100

    def fail(self, prop: str, constraints: dict[str, str]) -> None:
        self.errors.append(constraint_error(prop, constraints))


def _list_ids(call: Call, query: _ListQuery, name: str) -> list[int] | None:
    """``eventTypeIds`` and ``teamsIds``: a comma list, each item through ``parseInt``, ``@IsNumber`` each."""
    raw = call.param(name)
    if raw is None:
        return None
    values = [js_parse_int(part) for part in raw.split(",")]
    ids = [v for v in values if v is not None]
    if len(ids) != len(values):
        message = f"each value in {name} must be a number conforming to the specified constraints"
        query.fail(name, {"isNumber": message})
        return None
    return ids


def _list_id(call: Call, query: _ListQuery, name: str) -> int | None:
    """``eventTypeId`` and ``teamId``: ``@Type(() => Number) @IsInt()``."""
    raw = call.param(name)
    if raw is None:
        return None
    number = js_number(raw)
    if number is None or not math.isfinite(number) or not number.is_integer():
        query.fail(name, {"isInt": f"{name} must be an integer number"})
        return None
    return int(number)


def _list_page_param(
    call: Call, query: _ListQuery, name: str, default: int, low: int, high: int | None
) -> int:
    """``take``/``skip`` (and ``limit``): ``parseInt`` then ``@IsNumber() @Min() @Max()``. The constraints of
    one parameter are listed in the order class-validator registers the decorators (bottom-up)."""
    raw = call.param(name)
    if raw is None:
        return default
    value = js_parse_int(raw)
    constraints: dict[str, str] = {}
    if high is not None and (value is None or value > high):
        constraints["max"] = f"{name} must not be greater than {high}"
    if value is None or value < low:
        constraints["min"] = f"{name} must not be less than {low}"
    if value is None:
        constraints["isNumber"] = f"{name} must be a number conforming to the specified constraints"
    if constraints:
        query.fail(name, constraints)
        return default
    assert value is not None
    return value


def _list_query(call: Call) -> _ListQuery:
    """Parse the list query the way Cal.com's global ValidationPipe does: unknown parameters are dropped
    (``whitelist`` without ``forbidNonWhitelisted``) and each failing property becomes one entry of
    ``details.errors``, in the declaration order of ``GetBookingsInput_2024_08_13``."""
    query = _ListQuery(cursor_mode=call.header(VERSION_HEADER) == CURSOR_LIST_VERSION)
    raw_status = call.param("status")
    if query.cursor_mode:
        query.statuses = list(LIST_STATUSES) if raw_status is None else [raw_status]
        if raw_status is not None and raw_status not in LIST_STATUSES:
            message = f"status must be one of the following values: {', '.join(LIST_STATUSES)}"
            query.fail("status", {"isEnum": message})
    else:
        query.statuses = ["upcoming"] if raw_status is None else [s.strip() for s in raw_status.split(",")]
        if any(s not in LIST_STATUSES for s in query.statuses):
            query.fail("status", {"isEnum": MSG_INVALID_STATUS})
    event_type_ids = _list_ids(call, query, "eventTypeIds")
    event_type_id = _list_id(call, query, "eventTypeId")
    team_ids = _list_ids(call, query, "teamsIds")
    team_id = _list_id(call, query, "teamId")
    # BookingsService: ``eventTypeIds || [eventTypeId]`` and ``teamsIds || [teamId]``, a zero id meaning none.
    query.event_type_ids = event_type_ids or ([event_type_id] if event_type_id else None)
    query.team_ids = team_ids or ([team_id] if team_id else None)
    for name in _LIST_DATE_FILTERS:
        raw = call.param(name)
        if raw is not None:
            parsed = parse_datetime(raw)
            if parsed is None:
                side = "fromDate" if name.startswith("after") else "toDate"
                query.fail(name, {"isIso8601": f"{side} must be a valid ISO 8601 date."})
            else:
                query.bounds[name] = parsed
    for name, message in _LIST_SORT_MESSAGES.items():
        raw = call.param(name)
        if raw is None:
            continue
        if raw not in ("asc", "desc"):
            query.fail(name, {"isEnum": message})
        elif query.sort is None:  # only the first sort parameter in this order applies
            query.sort = (_BOOKING_SORT_FIELD[name], raw == "desc")
    if query.cursor_mode:
        query.page_size = _list_page_param(call, query, "limit", 50, 1, 100)
        cursor = call.param("cursor")
        if cursor is not None:
            decoded = decode_cursor(cursor)
            if decoded is None:
                message = "cursor must be a nextCursor value returned by this sandbox"
                query.fail("cursor", {"sandbox": message})
            else:
                query.offset = decoded
    else:
        query.page_size = _list_page_param(call, query, "take", 100, 1, 250)
        query.offset = _list_page_param(call, query, "skip", 0, 0, None)
    return query


def _list_filter(call: Call, query: _ListQuery, booking: dict[str, Any], now: datetime) -> bool:
    attendees = booking["attendees"]
    email, name, uid = call.param("attendeeEmail"), call.param("attendeeName"), call.param("bookingUid")
    if email:
        # The input transform trims and turns inner whitespace into "+" (an unencoded "+" arrives as a space).
        wanted = re.sub(r"\s+", "+", email.strip())
        if not any(a["email"] == wanted for a in attendees):
            return False
    if name and not any(a["name"] == name.strip() for a in attendees):
        return False
    if uid and booking["uid"] != uid.strip():
        return False
    if query.event_type_ids and booking["eventTypeId"] not in query.event_type_ids:
        return False
    if query.team_ids:
        return False  # the sandbox's event type belongs to a user, not a team
    start, end = parse_iso(booking["start"]), parse_iso(booking["end"])
    created, updated = parse_iso(booking["createdAt"]), parse_iso(booking["updatedAt"])
    checks = (
        ("afterStart", start, True),
        ("beforeEnd", end, False),
        ("afterCreatedAt", created, True),
        ("beforeCreatedAt", created, False),
        ("afterUpdatedAt", updated, True),
        ("beforeUpdatedAt", updated, False),
    )
    for bound_name, value, is_lower in checks:
        bound = query.bounds.get(bound_name)
        if bound is not None and (value < bound if is_lower else value > bound):
            return False
    return any(_matches_status(booking, s, now) for s in query.statuses)


def _list(call: Call) -> Outcome:
    state = call.state
    now = state.now()
    query = _list_query(call)
    if query.errors:
        return pipe_error(call, query.errors)
    items = [b for b in state.calcom_bookings if _list_filter(call, query, b, now)]
    sort = query.sort
    if sort is None and query.cursor_mode:
        # Cursor walk per the 2026-05-01 ordering table: forward from NOW() - 1h for upcoming, recurring and
        # unconfirmed; backward from NOW() for past; backward from year 2100 otherwise.
        status = query.statuses[0] if len(query.statuses) == 1 else None
        forward = status in ("upcoming", "recurring", "unconfirmed")
        if forward:
            items = [b for b in items if parse_iso(b["start"]) >= now - timedelta(hours=1)]
        else:
            ceiling = now if status == "past" else _FAR_FUTURE
            items = [b for b in items if parse_iso(b["start"]) < ceiling]
        sort = ("start", not forward)
    elif sort is None:
        # get.handler: one status and no sort parameter use that status's order; otherwise start ascending.
        descending = len(query.statuses) == 1 and query.statuses[0] in ("past", "cancelled")
        sort = ("start", descending)
    items.sort(key=lambda b: int(b["id"]))
    items.sort(key=_timestamp(sort[0]), reverse=sort[1])
    total = len(items)
    offset, page_size = query.offset, query.page_size
    page = [copy.deepcopy(b) for b in items[offset : offset + page_size]]
    if query.cursor_mode:
        has_more = offset + page_size < total
        pagination: dict[str, Any] = {
            "nextCursor": encode_cursor(offset + page_size) if has_more else None,
            "hasMore": has_more,
        }
    else:
        # getPagination() clamps skip to [0, totalItems] before computing the other fields.
        skip = min(max(offset, 0), total)
        total_pages = math.ceil(total / page_size)
        pagination = {
            "returnedItems": min(max(total - skip, 0), page_size),
            "totalItems": total,
            "itemsPerPage": page_size,
            "remainingItems": min(max(total - (skip + page_size), 0), total),
            "currentPage": 0 if total_pages == 0 else min(max(skip // page_size + 1, 1), total_pages),
            "totalPages": total_pages,
            "hasNextPage": skip + page_size < total,
            "hasPreviousPage": skip > 0,
        }
    return Outcome(200, {"status": "success", "data": page, "pagination": pagination})


# POST /v2/bookings/{uid}/reschedule ------------------------------------------------------------------------


def _reschedule(call: Call) -> Outcome:
    state = call.state
    body: dict[str, Any] = call.body if isinstance(call.body, dict) else {}
    errors = [_not_allowed(key) for key in body if key not in _RESCHEDULE_KEYS]
    start = parse_datetime(body.get("start"))
    if start is None:
        errors.append(_invalid_iso("start"))
    # RescheduleBookingInput_2024_08_13 declaration order (its rescheduledBy format check is a no-op).
    for key in ("rescheduledBy", "reschedulingReason", "emailVerificationCode", "seatUid"):
        if body.get(key) is not None and not isinstance(body[key], str):
            errors.append(FieldError(key, (f"{key} must be a string",)))
    for key in ("rescheduleWithSameHost", "allowConflicts", "allowBookingOutOfBounds", "skipBookingLimits"):
        if body.get(key) is not None and not isinstance(body[key], bool):
            errors.append(FieldError(key, (f"{key} must be a boolean value",)))
    if errors:
        return _validation_failed(call, errors)
    assert start is not None
    uid = call.path_param("uid")
    index = _find(state, uid)
    if index is None:
        return _err(call, 404, f"Booking with uid={uid} was not found in the database")
    old = state.calcom_bookings[index]
    if old["status"] in ("cancelled", "rejected"):
        moved_to = old.get("rescheduledToUid")
        if moved_to:
            return _err(
                call,
                400,
                f"Can't reschedule booking with uid={uid} because it has been cancelled and rescheduled "
                f"already to booking with uid={moved_to}. You probably want to reschedule {moved_to} "
                "instead by passing it within the request URL.",
            )
        return _err(
            call,
            400,
            f"Can't reschedule booking with uid={uid} because it has been cancelled. Please provide uid of a "
            "booking that is not cancelled.",
        )
    end = start + timedelta(minutes=int(old["duration"]))
    rejected = check_bookable(state, start, end, url=call.url, exclude_uid=uid)
    if rejected is not None:
        return rejected

    reason, rescheduled_by = body.get("reschedulingReason"), body.get("rescheduledBy")
    now = iso_ms_z(state.now())
    booking_id = state.next_id()
    new_uid = _new_uid(state, booking_id, start)
    fields = copy.deepcopy(old)
    fields.pop("rescheduledToUid", None)
    responses = dict(fields["bookingFieldsResponses"])
    if reason is not None:
        responses["rescheduledReason"] = reason
        fields["reschedulingReason"] = reason
    fields.update(
        id=booking_id,
        uid=new_uid,
        status=old["status"],
        rescheduledByEmail=rescheduled_by,
        rescheduledFromUid=uid,
        start=iso_ms_z(start),
        end=iso_ms_z(end),
        createdAt=now,
        updatedAt=now,
        bookingFieldsResponses=responses,
    )
    new = canonical(fields)
    moved = dict(old)
    moved.update(status="cancelled", rescheduledToUid=new_uid, updatedAt=now)
    if rescheduled_by is not None:
        moved["rescheduledByEmail"] = rescheduled_by
    state.calcom_bookings[index] = canonical(moved)
    state.calcom_bookings.append(new)
    return Outcome(201, _success(_with_platform_flag(new)))


# POST /v2/bookings/{uid}/cancel ---------------------------------------------------------------------------


def _cancel(call: Call) -> Outcome:
    state = call.state
    body: dict[str, Any] = call.body if isinstance(call.body, dict) else {}
    errors = [_not_allowed(key) for key in body if key not in _CANCEL_KEYS]
    for key in ("cancellationReason", "seatUid"):
        if body.get(key) is not None and not isinstance(body[key], str):
            errors.append(FieldError(key, (f"{key} must be a string",)))
    if body.get("cancelSubsequentBookings") is not None and not isinstance(
        body["cancelSubsequentBookings"], bool
    ):
        errors.append(
            FieldError("cancelSubsequentBookings", ("cancelSubsequentBookings must be a boolean value",))
        )
    if errors:
        return _validation_failed(call, errors)
    uid = call.path_param("uid")
    index = _find(state, uid)
    if index is None:
        return _err(call, 404, f"Booking with uid={uid} not found")
    booking = state.calcom_bookings[index]
    if booking["status"] in ("cancelled", "rejected"):
        return _err(
            call,
            400,
            f"Can't cancel booking with uid={uid} because it has been cancelled already. Please provide "
            "uid of a booking that is not cancelled.",
        )
    if parse_iso(booking["end"]) <= state.now():
        # A core HttpError, which the v2 exception filters do not wrap: NestJS's raw body.
        return Outcome(400, {"statusCode": 400, "message": MSG_CANCEL_ENDED})
    updated = dict(booking)
    updated.update(
        status="cancelled",
        cancellationReason=body.get("cancellationReason") or "",
        cancelledByEmail=state.seed.host_email,
        updatedAt=iso_ms_z(state.now()),
    )
    state.calcom_bookings[index] = canonical(updated)
    return Outcome(200, _success(copy.deepcopy(state.calcom_bookings[index])))


# The vendor API ------------------------------------------------------------------------------------------


def _malformed_booking(data: object) -> dict[str, Any] | None:
    if not isinstance(data, dict) or "start" not in data:
        return None
    return {"startTime": data["start"], "endTime": data["end"], "bookingStatus": str(data["status"]).upper()}


class CalcomApi(VendorApi):
    name = "Cal.com API v2"
    prefix = "/v2"
    calendar = "calcom"

    def __init__(self, router: APIRouter) -> None:
        self.router = router

    def unauthorized(self, call: Call, *, token_sent: bool) -> Outcome:
        if not token_sent and call.group == "bookings.list":
            return _err(call, 403, MSG_LIST_NO_AUTH)
        return _err(call, 401, MSG_INVALID_KEY if token_sent else MSG_NO_AUTH)

    def route_not_found(self, request: Request, now: datetime) -> Outcome:
        url = request_url(request)
        return error(404, f"Cannot {request.method} {url}", url=url, now=now)

    def server_error(self, call: Call) -> Outcome:
        return _err(call, 500, "Internal server error")

    def gateway_timeout(self, call: Call) -> Outcome:
        return _err(call, 504, "Gateway Timeout")

    def not_found(self, call: Call) -> Outcome:
        seed = call.state.seed
        if call.group in ("slots", "bookings.list"):
            return _err(call, 404, MSG_SLOTS_EVENT_TYPE_NOT_FOUND)
        if call.group == "bookings.create":
            return _err(call, 404, f"Event type with id {seed.event_type_id} not found.")
        uid = call.path_param("uid")
        if call.group == "bookings.cancel":
            return _err(call, 404, f"Booking with uid={uid} not found")
        return _err(call, 404, f"Booking with uid={uid} was not found in the database")

    def malformed(self, call: Call, normal: Outcome) -> Outcome:
        if call.group == "slots":
            now = call.state.now()
            start, end = call.extra.get("window", (now, now + timedelta(days=7)))
            periods = _busy_periods(call.state, start, end) or _busy_periods(
                call.state, now, now + timedelta(days=7)
            )
            busy = [{"start": iso_ms_z(a), "end": iso_ms_z(b)} for a, b in periods]
            return Outcome(200, {"status": "success", "data": {"busy": busy}})
        data = normal.body.get("data") if normal.status < 300 and isinstance(normal.body, dict) else None
        if call.group == "bookings.list":
            items = [_malformed_booking(b) for b in data] if isinstance(data, list) else []
            return Outcome(200, {"status": "success", "data": {"items": items, "count": len(items)}})
        return Outcome(200, {"status": "success", "data": {"booking": _malformed_booking(data)}})

    def slot_taken_after_offer(self, call: Call, run: Callable[[], Outcome]) -> Outcome:
        state = call.state
        if call.group == "slots":
            outcome = run()
            if outcome.status == 200:
                _take(state, offered_starts(outcome.body))
            return outcome
        _take(state, last_offered_starts(state))
        result: Outcome = run()
        return result

    def setup_booking(self, state: SandboxState, setup: SetupBooking) -> Outcome:
        attendee = Attendee(
            name=setup.lead_name,
            email=setup.lead_email,
            time_zone=setup.lead_timezone or state.seed.host_timezone,
        )
        request = BookingRequest(start=setup.start, attendee=attendee, title=setup.title)
        return book(state, request, url="/v2/bookings")


def _take(state: SandboxState, starts: list[datetime]) -> None:
    """Give every still-free offered slot to a third party."""
    state.take_by_third_party(
        [s for s in dict.fromkeys(starts) if state.is_free(s)], "slot_taken_after_offer"
    )


router = APIRouter()


async def _bookings_call(request: Request, group: str, handler: Callable[[Call], Outcome]) -> Response:
    return await run_call(request, API, group, handler, route=_bookings_route, precheck=_bookings_legacy)


@router.get("/v2/slots")
async def get_slots(request: Request) -> Response:
    return await run_call(request, API, "slots", _slots, route=_slots_route)


@router.post("/v2/bookings")
async def create_booking(request: Request) -> Response:
    return await _bookings_call(request, "bookings.create", _create)


@router.get("/v2/bookings")
async def list_bookings(request: Request) -> Response:
    return await _bookings_call(request, "bookings.list", _list)


@router.get("/v2/bookings/{uid}")
async def get_booking(request: Request) -> Response:
    return await _bookings_call(request, "bookings.get", _get)


@router.post("/v2/bookings/{uid}/reschedule")
async def reschedule_booking(request: Request) -> Response:
    return await _bookings_call(request, "bookings.reschedule", _reschedule)


@router.post("/v2/bookings/{uid}/cancel")
async def cancel_booking(request: Request) -> Response:
    return await _bookings_call(request, "bookings.cancel", _cancel)


API: Final = CalcomApi(router)

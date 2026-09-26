"""Google Calendar API v3 subset under its real paths: ``freeBusy`` and events.

Mirrored operations (an adapter reaches them by swapping only its base URL, ``https://www.googleapis.com``):

- ``POST /calendar/v3/freeBusy``;
- ``POST /calendar/v3/calendars/{calendarId}/events`` (insert, with an optional client-supplied id);
- ``GET /calendar/v3/calendars/{calendarId}/events`` (list);
- ``GET``, ``PATCH`` and ``DELETE /calendar/v3/calendars/{calendarId}/events/{eventId}``.

The sandbox has one calendar, ``seed.google_calendar_id``. It is the host calendar that the other
mirrored calendars share, so ``freeBusy`` reports every seeded block, Cal.com booking, third-party take
and Google event as busy. Insert and patch do no conflict checking, as on Google. A deleted event stays
as a ``cancelled`` tombstone that keeps its id. Callers are treated as a service account without
domain-wide delegation unless ``seed.google_sa_can_invite`` is set; once a token has been exchanged at
``POST /token``, each method also needs one of its documented scopes in the latest grant (``calendar.events``
alone cannot query ``freeBusy``). ``docs/sandbox-fidelity.md`` lists the evidence for each behaviour and
every known deviation.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Request
from fastapi.responses import Response

from booking_truth.sandbox.common import Call, Outcome, SetupBooking, VendorApi, run_call
from booking_truth.sandbox.state import SandboxState
from booking_truth.timeutil import iso_ms_z, iso_z

PREFIX: Final = "/calendar/v3"
JSON_GOOGLE: Final = "application/json; charset=UTF-8"
EVENT_ID_RE: Final = re.compile(r"[a-v0-9]{5,1024}")
#: The private extended property that carries the lead's email when the service account cannot invite.
LEAD_EMAIL_PROPERTY: Final = "bt_lead_email"
DEFAULT_SERVICE_ACCOUNT: Final = "booking-agent@example.com"
MAX_RESULTS: Final = 2500
DEFAULT_MAX_RESULTS: Final = 250
SEND_UPDATES: Final = ("all", "externalOnly", "none")
EXTENDED_KEY_MAX: Final = 44
EXTENDED_VALUE_MAX: Final = 1024

# Verbatim Google texts ------------------------------------------------------------------------------

MSG_AUTH_INVALID: Final = (
    "Request had invalid authentication credentials. Expected OAuth 2 access token, login cookie or other "
    "valid authentication credential. See https://developers.google.com/identity/sign-in/web/devconsole-project."
)
MSG_AUTH_MISSING: Final = (
    "Request is missing required authentication credential. Expected OAuth 2 access token, login cookie or "
    "other valid authentication credential. See https://developers.google.com/identity/sign-in/web/devconsole-project."
)
MSG_SA_ATTENDEES: Final = (
    "Service accounts cannot invite attendees without Domain-Wide Delegation of Authority."
)
MSG_ORDERING: Final = "The requested ordering is not available for the particular query."
MSG_SCOPES: Final = "Request had insufficient authentication scopes."

# Scopes that authorize each mirrored method (any one suffices), verbatim from the reference pages, and the
# method name Google reports when none was granted.
_AUTH: Final = "https://www.googleapis.com/auth/"
_WRITE_SCOPES: Final = frozenset(
    _AUTH + s for s in ("calendar", "calendar.events", "calendar.app.created", "calendar.events.owned")
)
_READ_SCOPES: Final = _WRITE_SCOPES | frozenset(
    _AUTH + s
    for s in (
        "calendar.readonly",
        "calendar.events.readonly",
        "calendar.events.freebusy",
        "calendar.events.owned.readonly",
        "calendar.events.public.readonly",
    )
)
_FREEBUSY_SCOPES: Final = frozenset(
    _AUTH + s for s in ("calendar.readonly", "calendar", "calendar.events.freebusy", "calendar.freebusy")
)
METHOD_SCOPES: Final[dict[str, tuple[str, frozenset[str]]]] = {
    "freebusy": ("calendar.v3.Freebusy.Query", _FREEBUSY_SCOPES),
    "events.insert": ("calendar.v3.Events.Insert", _WRITE_SCOPES),
    "events.get": ("calendar.v3.Events.Get", _READ_SCOPES),
    "events.list": ("calendar.v3.Events.List", _READ_SCOPES),
    "events.patch": ("calendar.v3.Events.Patch", _WRITE_SCOPES),
    "events.delete": ("calendar.v3.Events.Delete", _WRITE_SCOPES),
}

# Output key order of the Events resource (reference page order); absent keys are omitted.
EVENT_KEYS: Final = (
    "kind", "etag", "id", "status", "htmlLink", "created", "updated", "summary", "description", "location",
    "colorId", "creator", "organizer", "start", "end", "endTimeUnspecified", "recurrence",
    "recurringEventId", "originalStartTime", "transparency", "visibility", "iCalUID", "sequence",
    "attendees", "attendeesOmitted", "extendedProperties", "hangoutLink", "conferenceData", "gadget",
    "anyoneCanAddSelf", "guestsCanInviteOthers", "guestsCanModify", "guestsCanSeeOtherGuests",
    "privateCopy", "locked", "reminders", "source", "workingLocationProperties", "outOfOfficeProperties",
    "focusTimeProperties", "attachments", "birthdayProperties", "eventType",
)  # fmt: skip
# Fields a client may set; every other field of a request body is ignored.
_TEXT_FIELDS: Final = ("summary", "description", "location", "colorId")
_BOOL_FIELDS: Final = (
    "anyoneCanAddSelf",
    "guestsCanInviteOthers",
    "guestsCanModify",
    "guestsCanSeeOtherGuests",
)
_OBJECT_FIELDS: Final = ("reminders", "source")
WRITABLE: Final = frozenset(
    {*_TEXT_FIELDS, *_BOOL_FIELDS, *_OBJECT_FIELDS, "start", "end", "status", "transparency", "visibility",
     "attendees", "extendedProperties"}
)  # fmt: skip
_ENUMS: Final[dict[str, tuple[str, ...]]] = {
    "status": ("confirmed", "tentative", "cancelled"),
    "transparency": ("opaque", "transparent"),
    "visibility": ("default", "public", "private", "confidential"),
}
_OFFER_NOT_SEEN_FAULTS: Final = frozenset(
    {"timeout", "commit_then_timeout", "malformed", "error_500", "not_found"}
)
_RFC3339_RE: Final = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:[Zz]|[+-]\d{2}:\d{2}(?::\d{2})?)", re.ASCII
)
_LOCAL_RE: Final = re.compile(r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?", re.ASCII)


# Envelopes ---------------------------------------------------------------------------------------


def error(
    status: int,
    reason: str,
    message: str,
    *,
    domain: str = "global",
    location_type: str | None = None,
    location: str | None = None,
) -> Outcome:
    """The Calendar backend error envelope: ``errors``, ``code``, ``message`` and no ``status`` key."""
    item: dict[str, str] = {"domain": domain, "reason": reason, "message": message}
    if location_type is not None and location is not None:
        item["locationType"] = location_type
        item["location"] = location
    return Outcome(status, {"error": {"errors": [item], "code": status, "message": message}})


def _auth_error(message: str, item_message: str, reason: str) -> Outcome:
    """A front-end auth error: ``code``, ``message``, ``errors`` and ``status``, in that order."""
    item = {
        "message": item_message,
        "domain": "global",
        "reason": reason,
        "location": "Authorization",
        "locationType": "header",
    }
    return Outcome(
        401, {"error": {"code": 401, "message": message, "errors": [item], "status": "UNAUTHENTICATED"}}
    )


def insufficient_scopes(method: str) -> Outcome:
    """The front end's 403 for a token without a scope the method accepts (``status`` and ``details`` too)."""
    item = {"message": "Insufficient Permission", "domain": "global", "reason": "insufficientPermissions"}
    detail = {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
        "domain": "googleapis.com",
        "metadata": {"service": "calendar-json.googleapis.com", "method": method},
    }
    body = {
        "code": 403,
        "message": MSG_SCOPES,
        "errors": [item],
        "status": "PERMISSION_DENIED",
        "details": [detail],
    }
    return Outcome(403, {"error": body})


def scope_check(state: SandboxState, group: str) -> Outcome | None:
    """403 when the most recent accepted token grant has no scope that authorizes ``group``.

    Every access token is the sandbox token, so the latest grant stands for the caller. Without a grant since
    the last reset (a client that uses the sandbox token directly) every scope is allowed."""
    if not state.google_token_grants or group not in METHOD_SCOPES:
        return None
    method, allowed = METHOD_SCOPES[group]
    granted = state.google_token_grants[-1].get("scopes", [])
    return None if allowed.intersection(granted) else insufficient_scopes(method)


NOT_FOUND: Final = error(404, "notFound", "Not Found")
DUPLICATE: Final = error(409, "duplicate", "The requested identifier already exists.")
DELETED: Final = error(410, "deleted", "Resource has been deleted")
INVALID_ID: Final = error(400, "invalid", "Invalid resource id value.")
SA_ATTENDEES: Final = error(403, "forbiddenForServiceAccounts", MSG_SA_ATTENDEES, domain="calendar")
BACKEND_ERROR: Final = error(500, "backendError", "Backend Error")
UNAVAILABLE: Final = error(503, "backendError", "Backend Error")
CONDITION_NOT_MET: Final = error(
    412, "conditionNotMet", "Precondition Failed", location_type="header", location="If-Match"
)
TIME_RANGE_EMPTY: Final = error(
    400,
    "timeRangeEmpty",
    "The specified time range is empty.",
    domain="calendar",
    location_type="parameter",
    location="timeMax",
)
EVENT_RANGE_EMPTY: Final = error(
    400, "timeRangeEmpty", "The specified time range is empty.", domain="calendar"
)
ORDERING: Final = error(400, "badRequest", MSG_ORDERING)
PARSE_ERROR: Final = error(400, "parseError", "Parse Error")
BAD_REQUEST: Final = error(400, "badRequest", "Bad Request")
REQUIRED: Final = error(400, "required", "Required")
ALL_DAY: Final = error(
    400, "invalid", "The sandbox mirrors timed events only; all-day events (start.date) are not supported."
)
FULL_SYNC_REQUIRED: Final = error(
    410,
    "fullSyncRequired",
    "Sync token is no longer valid, a full sync is required.",
    domain="calendar",
    location_type="parameter",
    location="syncToken",
)


def _missing(which: str) -> Outcome:
    return error(400, "required", f"Missing {which} time.")


def _invalid_parameter(name: str, value: str, pattern: str) -> Outcome:
    message = f"Invalid value '{value}'. Values must match the following regular expression: '{pattern}'"
    return error(400, "invalidParameter", message, location_type="parameter", location=name)


def _invalid_field(name: str) -> Outcome:
    return error(400, "invalid", f"Invalid value for: {name}")


def _front_end(body: object) -> bool:
    return isinstance(body, dict) and isinstance(body.get("error"), dict) and "status" in body["error"]


def render(outcome: Outcome) -> Response:
    """Google's bytes: pretty-printed JSON (1-space indent from the Calendar backend, 2-space from the front
    end), a trailing newline and ``application/json; charset=UTF-8``; an empty body for ``None``."""
    headers = dict(outcome.headers)
    if outcome.body is None:
        return Response(status_code=outcome.status, headers=headers)
    indent = 2 if _front_end(outcome.body) else 1
    content = json.dumps(outcome.body, indent=indent, ensure_ascii=False) + "\n"
    return Response(content, status_code=outcome.status, media_type=JSON_GOOGLE, headers=headers)


# Values -------------------------------------------------------------------------------------------


def parse_rfc3339(value: object) -> datetime | None:
    """An RFC 3339 timestamp with a mandatory offset, in UTC; ``None`` when it is not one."""
    if not isinstance(value, str) or not _RFC3339_RE.fullmatch(value):
        return None
    try:
        parsed = datetime.fromisoformat(value.upper()).astimezone(UTC)
    except (ValueError, OverflowError):
        return None
    return parsed if 1 < parsed.year < 9999 else None


def zone_or_none(name: object) -> ZoneInfo | None:
    if not isinstance(name, str) or not name or name.startswith(("/", ".")):
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def event_time(value: datetime, zone: ZoneInfo) -> str:
    """An event ``dateTime`` as Google renders it: the zone's offset, whole seconds, ``+00:00`` for UTC."""
    return value.astimezone(zone).isoformat(timespec="seconds")


def _busy_time(value: datetime, zone: ZoneInfo | None) -> str:
    """A ``freeBusy`` period bound: ``Z`` without a ``timeZone``, otherwise that zone's offset."""
    return iso_z(value) if zone is None else value.astimezone(zone).isoformat(timespec="seconds")


def _micros(value: datetime) -> int:
    return int(value.timestamp()) * 1_000_000 + value.microsecond


def _etag(now: datetime, previous: str | None) -> str:
    """A quoted number that grows with every write (synthetic; real values look alike)."""
    number = 2 * _micros(now)
    if previous is not None:
        digits = previous.strip('"')
        if digits.isdigit() and number <= int(digits):
            number = int(digits) + 1
    return f'"{number}"'


def _new_event_id(existing: dict[str, dict[str, Any]]) -> str:
    """An id the server picks when the client sends none: 26 base32hex characters, like Google's."""
    while True:
        raw = base64.b32hexencode(uuid.uuid4().bytes).decode().rstrip("=").lower()[:26]
        if raw not in existing:
            return raw


def _encode_page(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"o": offset}).encode()).rstrip(b"=").decode()


def _decode_page(token: str) -> int | None:
    try:
        payload = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    except (ValueError, binascii.Error):
        return None
    offset = payload.get("o") if isinstance(payload, dict) else None
    return offset if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 else None


# The calendar ---------------------------------------------------------------------------------------


def calendar_exists(state: SandboxState, calendar_id: str) -> bool:
    return calendar_id == state.seed.google_calendar_id


def calendar_email(state: SandboxState) -> str:
    """The calendar's own id: the host's address for ``primary``, else the seeded calendar id."""
    calendar_id = state.seed.google_calendar_id
    return state.seed.host_email if calendar_id == "primary" else calendar_id


def calendar_zone(state: SandboxState) -> ZoneInfo:
    return ZoneInfo(state.seed.host_timezone)


def events_of(state: SandboxState, calendar_id: str) -> dict[str, dict[str, Any]]:
    return state.google_events.setdefault(calendar_id, {})


def _creator(state: SandboxState) -> str:
    """The acting identity: the subject or issuer of the latest accepted token assertion."""
    if state.google_token_grants:
        grant = state.google_token_grants[-1]
        return str(grant.get("sub") or grant["iss"])
    return DEFAULT_SERVICE_ACCOUNT


def _html_link(event_id: str, email: str) -> str:
    eid = base64.urlsafe_b64encode(f"{event_id} {email}".encode()).decode().rstrip("=")
    return f"https://www.google.com/calendar/event?eid={eid}"


def _starts_ends(event: dict[str, Any]) -> tuple[datetime, datetime]:
    start = parse_rfc3339(event["start"]["dateTime"])
    end = parse_rfc3339(event["end"]["dateTime"])
    assert start is not None
    assert end is not None
    return start, end


def _updated_at(event: dict[str, Any]) -> datetime:
    updated = parse_rfc3339(event["updated"])
    assert updated is not None
    return updated


def view(event: dict[str, Any], zone: ZoneInfo) -> dict[str, Any]:
    """A copy of ``event`` with its times rendered in ``zone`` (the response ``timeZone``)."""
    out = copy.deepcopy(event)
    start, end = _starts_ends(event)
    out["start"]["dateTime"] = event_time(start, zone)
    out["end"]["dateTime"] = event_time(end, zone)
    return out


# Event validation ----------------------------------------------------------------------------------


def _parse_time(value: object, which: str) -> tuple[datetime, str | None] | Outcome:
    """An event ``start`` or ``end`` object: ``dateTime`` with an offset, or a local ``dateTime`` read in the
    object's ``timeZone``."""
    if value is None:
        return _missing(which)
    if not isinstance(value, dict):
        return BAD_REQUEST
    raw, zone_name = value.get("dateTime"), value.get("timeZone")
    if raw is None:
        return ALL_DAY if value.get("date") is not None else _missing(which)
    zone: ZoneInfo | None = None
    if zone_name is not None:
        zone = zone_or_none(zone_name)
        if zone is None:
            return error(400, "invalid", f"Invalid time zone definition for {which} time.")
    if not isinstance(raw, str):
        return BAD_REQUEST
    parsed = parse_rfc3339(raw)
    if parsed is None:
        if not _LOCAL_RE.fullmatch(raw):
            return BAD_REQUEST
        if zone is None:
            return error(400, "required", f"Missing time zone definition for {which} time.")
        try:
            parsed = datetime.fromisoformat(raw.upper()).replace(tzinfo=zone).astimezone(UTC)
        except (ValueError, OverflowError):
            return BAD_REQUEST
        if not 1 < parsed.year < 9999:  # the same range as parse_rfc3339, so rendering cannot overflow
            return BAD_REQUEST
    return parsed, zone_name if isinstance(zone_name, str) else None


def _extended(value: object) -> dict[str, dict[str, str]] | Outcome:
    """``extendedProperties``: string maps; keys over 44 characters are dropped, values cut at 1024."""
    if not isinstance(value, dict):
        return _invalid_field("extendedProperties")
    out: dict[str, dict[str, str]] = {}
    for scope in ("private", "shared"):
        props = value.get(scope)
        if props is None:
            continue
        if not isinstance(props, dict):
            return _invalid_field(f"extendedProperties.{scope}")
        kept: dict[str, str] = {}
        for key, item in props.items():
            if item is None:
                continue
            if isinstance(item, dict | list):
                return _invalid_field(f"extendedProperties.{scope}")
            text = ("true" if item else "false") if isinstance(item, bool) else str(item)
            if len(key) <= EXTENDED_KEY_MAX:
                kept[key] = text[:EXTENDED_VALUE_MAX]
        out[scope] = kept
    return out


def _attendees(value: object) -> list[dict[str, Any]] | Outcome:
    if not isinstance(value, list):
        return _invalid_field("attendees")
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("email"), str) or not item["email"]:
            return error(400, "required", "Missing attendee email.")
        attendee: dict[str, Any] = {"email": item["email"]}
        if isinstance(item.get("displayName"), str):
            attendee["displayName"] = item["displayName"]
        if item.get("optional") is True:
            attendee["optional"] = True
        status = item.get("responseStatus")
        attendee["responseStatus"] = status if isinstance(status, str) else "needsAction"
        out.append(attendee)
    return out


def normalize(
    state: SandboxState, fields: dict[str, Any], *, attendees_sent: bool
) -> dict[str, Any] | Outcome:
    """Validate the writable fields of an insert or of a patched event. Times come back as UTC instants
    (``start_at``/``end_at``) next to the time objects' zones; ``attendees_sent`` applies the service-account
    rule, which only a request that sends attendees can break."""
    ends = _parse_time(fields.get("end"), "end")
    if isinstance(ends, Outcome):
        return ends
    starts = _parse_time(fields.get("start"), "start")
    if isinstance(starts, Outcome):
        return starts
    out: dict[str, Any] = {
        "start_at": starts[0],
        "start_zone": starts[1],
        "end_at": ends[0],
        "end_zone": ends[1],
    }
    if ends[0] < starts[0]:
        return EVENT_RANGE_EMPTY
    for name, allowed in _ENUMS.items():
        value = fields.get(name)
        if value is None:
            continue
        if value not in allowed:
            return _invalid_field(name)
        out[name] = value
    for name in _TEXT_FIELDS:
        value = fields.get(name)
        if value is None:
            continue
        if not isinstance(value, str):
            return _invalid_field(name)
        out[name] = value
    for name in _BOOL_FIELDS:
        value = fields.get(name)
        if value is None:
            continue
        if not isinstance(value, bool):
            return _invalid_field(name)
        out[name] = value
    for name in _OBJECT_FIELDS:
        value = fields.get(name)
        if value is None:
            continue
        if not isinstance(value, dict):
            return _invalid_field(name)
        out[name] = copy.deepcopy(value)
    if fields.get("extendedProperties") is not None:
        extended = _extended(fields["extendedProperties"])
        if isinstance(extended, Outcome):
            return extended
        out["extendedProperties"] = extended
    if fields.get("attendees") is not None:
        attendees = _attendees(fields["attendees"])
        if isinstance(attendees, Outcome):
            return attendees
        if attendees and attendees_sent and not state.seed.google_sa_can_invite:
            return SA_ATTENDEES
        if attendees:
            out["attendees"] = attendees
    return out


def build_event(
    state: SandboxState,
    event_id: str,
    fields: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The Events resource for normalized ``fields``; ``previous`` keeps its identity fields on a patch."""
    now = state.now()
    zone = calendar_zone(state)
    start: dict[str, Any] = {"dateTime": event_time(fields["start_at"], zone)}
    if fields["start_zone"]:
        start["timeZone"] = fields["start_zone"]
    end: dict[str, Any] = {"dateTime": event_time(fields["end_at"], zone)}
    if fields["end_zone"]:
        end["timeZone"] = fields["end_zone"]
    email = calendar_email(state)
    out: dict[str, Any] = {
        "kind": "calendar#event",
        "etag": _etag(now, previous["etag"] if previous else None),
        "id": event_id,
        "status": fields.get("status", "confirmed"),
        "htmlLink": previous["htmlLink"] if previous else _html_link(event_id, email),
        "created": previous["created"] if previous else iso_z(now),
        "updated": iso_ms_z(now),
        "creator": previous["creator"] if previous else {"email": _creator(state)},
        "organizer": previous["organizer"] if previous else {"email": email, "self": True},
        "start": start,
        "end": end,
        "iCalUID": previous["iCalUID"] if previous else f"{event_id}@google.com",
        "sequence": previous["sequence"] if previous else 0,
        "reminders": fields.get("reminders", {"useDefault": True}),
        "eventType": "default",
    }
    for name in (*_TEXT_FIELDS, *_BOOL_FIELDS, "source", "attendees", "extendedProperties"):
        if name in fields:
            out[name] = fields[name]
    if fields.get("transparency") == "transparent":
        out["transparency"] = "transparent"
    if fields.get("visibility", "default") != "default":
        out["visibility"] = fields["visibility"]
    if previous is not None:
        moved = _starts_ends(previous) != (fields["start_at"], fields["end_at"])
        if moved or previous.get("location") != out.get("location"):
            out["sequence"] = previous["sequence"] + 1
    return {key: out[key] for key in EVENT_KEYS if key in out}


def insert_event(state: SandboxState, calendar_id: str, body: dict[str, Any]) -> Outcome:
    """``events.insert`` on an existing calendar, without conflict checking. The caller holds the lock."""
    event_id = body.get("id")
    if event_id is not None and not (isinstance(event_id, str) and EVENT_ID_RE.fullmatch(event_id)):
        return INVALID_ID
    fields = normalize(state, {k: v for k, v in body.items() if k in WRITABLE}, attendees_sent=True)
    if isinstance(fields, Outcome):
        return fields
    events = events_of(state, calendar_id)
    if event_id is not None and event_id in events:
        return DUPLICATE
    event_id = event_id if event_id is not None else _new_event_id(events)
    event = build_event(state, event_id, fields)
    events[event_id] = event
    return Outcome(200, copy.deepcopy(event))


def _writable(event: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in event.items() if key in WRITABLE}


def merge_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Patch semantics: objects merge, arrays and scalars replace, ``null`` removes a key."""
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_patch(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# Request checks -------------------------------------------------------------------------------------


def _send_updates(call: Call) -> Outcome | None:
    value = call.param("sendUpdates")
    if value is not None and value not in SEND_UPDATES:
        return _invalid_parameter("sendUpdates", value, "|".join(SEND_UPDATES))
    return None


def _body(call: Call) -> dict[str, Any] | Outcome:
    if call.body is None:
        return {}
    if call.body_invalid or not isinstance(call.body, dict):
        return PARSE_ERROR
    return call.body


def _if_match(call: Call, event: dict[str, Any]) -> Outcome | None:
    wanted = call.header("if-match")
    if wanted is not None and wanted.strip() not in ("*", event["etag"]):
        return CONDITION_NOT_MET
    return None


def _response_zone(call: Call) -> ZoneInfo | Outcome:
    name = call.param("timeZone")
    if name is None:
        return calendar_zone(call.state)
    zone = zone_or_none(name)
    return zone if zone is not None else _invalid_field("timeZone")


def _event(call: Call) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | Outcome:
    calendar_id = call.path_param("calendarId")
    if not calendar_exists(call.state, calendar_id):
        return NOT_FOUND
    events = events_of(call.state, calendar_id)
    event = events.get(call.path_param("eventId"))
    return NOT_FOUND if event is None else (events, event)


# POST /calendar/v3/freeBusy --------------------------------------------------------------------------


def merged_busy(state: SandboxState, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Busy time of the host calendar inside ``[start, end)``: overlapping and touching blocks merged,
    each block clipped to the window."""
    clipped = sorted(
        (max(s, start), min(e, end)) for s, e in state.busy_intervals() if s < end and e > start and s < e
    )
    merged: list[tuple[datetime, datetime]] = []
    for s, e in clipped:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _not_found_entry() -> dict[str, Any]:
    return {"errors": [{"domain": "global", "reason": "notFound"}], "busy": []}


def _item_ids(body: object) -> list[str]:
    items = body.get("items") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return []
    ids: list[str] = []
    for item in items:
        value = item.get("id") if isinstance(item, dict) else None
        ids.append(value if isinstance(value, str) else "")
    return ids


def _freebusy(call: Call) -> Outcome:
    body = _body(call)
    if isinstance(body, Outcome):
        return body
    for name in ("timeMin", "timeMax"):
        if body.get(name) is None:
            return REQUIRED
    start, end = parse_rfc3339(body["timeMin"]), parse_rfc3339(body["timeMax"])
    if start is None or end is None:
        return BAD_REQUEST
    if end <= start:
        return TIME_RANGE_EMPTY
    zone: ZoneInfo | None = None
    if body.get("timeZone") is not None:
        zone = zone_or_none(body["timeZone"])
        if zone is None:
            return _invalid_field("timeZone")
    if body.get("items") is not None and not isinstance(body["items"], list):
        return BAD_REQUEST
    call.extra["window"] = (start, end)
    state = call.state
    busy = [
        {"start": _busy_time(s, zone), "end": _busy_time(e, zone)} for s, e in merged_busy(state, start, end)
    ]
    calendars: dict[str, Any] = {}
    for calendar_id in _item_ids(body):
        if calendar_exists(state, calendar_id):
            calendars[calendar_id] = {"busy": copy.deepcopy(busy)}
        else:
            calendars[calendar_id] = _not_found_entry()
    return Outcome(
        200,
        {
            "kind": "calendar#freeBusy",
            "timeMin": iso_ms_z(start),
            "timeMax": iso_ms_z(end),
            "calendars": calendars,
        },
    )


def _window(body: object) -> tuple[datetime, datetime] | None:
    if not isinstance(body, dict):
        return None
    start, end = parse_rfc3339(body.get("timeMin")), parse_rfc3339(body.get("timeMax"))
    return (start, end) if start is not None and end is not None and start < end else None


def _offered(state: SandboxState, response: object) -> bool:
    """Whether a freeBusy response gave the client the host calendar's busy list with its normal schema."""
    calendars = response.get("calendars") if isinstance(response, dict) else None
    entry = calendars.get(state.seed.google_calendar_id) if isinstance(calendars, dict) else None
    return isinstance(entry, dict) and "errors" not in entry and isinstance(entry.get("busy"), list)


def last_offered_window(state: SandboxState) -> tuple[datetime, datetime] | None:
    """The window of the most recent successful freeBusy query the client received with its normal schema."""
    for entry in reversed(state.request_log):
        if (
            entry.group == "freebusy"
            and entry.completed
            and entry.status == 200
            and entry.fault not in _OFFER_NOT_SEEN_FAULTS
            and _offered(state, entry.response)
        ):
            window = _window(entry.body)
            if window is not None:
                return window
    return None


def _take_window(state: SandboxState, window: tuple[datetime, datetime] | None) -> None:
    """Give every free working-hours slot of ``window`` to a third party."""
    if window is not None:
        state.take_by_third_party(state.free_starts(*window), "slot_taken_after_offer")


# Events -----------------------------------------------------------------------------------------------


def _insert(call: Call) -> Outcome:
    bad = _send_updates(call)
    if bad is not None:
        return bad
    body = _body(call)
    if isinstance(body, Outcome):
        return body
    calendar_id = call.path_param("calendarId")
    if not calendar_exists(call.state, calendar_id):
        return NOT_FOUND
    return insert_event(call.state, calendar_id, body)


def _get(call: Call) -> Outcome:
    found = _event(call)
    if isinstance(found, Outcome):
        return found
    zone = _response_zone(call)
    if isinstance(zone, Outcome):
        return zone
    return Outcome(200, view(found[1], zone))


def _patch(call: Call) -> Outcome:
    bad = _send_updates(call)
    if bad is not None:
        return bad
    body = _body(call)
    if isinstance(body, Outcome):
        return body
    found = _event(call)
    if isinstance(found, Outcome):
        return found
    events, event = found
    bad = _if_match(call, event)
    if bad is not None:
        return bad
    patch = {key: value for key, value in body.items() if key in WRITABLE}
    merged = merge_patch(_writable(event), patch)
    attendees_sent = isinstance(patch.get("attendees"), list) and bool(patch["attendees"])
    fields = normalize(call.state, merged, attendees_sent=attendees_sent)
    if isinstance(fields, Outcome):
        return fields
    fields.setdefault("status", event["status"])
    updated = build_event(call.state, event["id"], fields, previous=event)
    events[event["id"]] = updated
    return Outcome(200, copy.deepcopy(updated))


def _delete(call: Call) -> Outcome:
    bad = _send_updates(call)
    if bad is not None:
        return bad
    found = _event(call)
    if isinstance(found, Outcome):
        return found
    events, event = found
    if event["status"] == "cancelled":
        return DELETED
    bad = _if_match(call, event)
    if bad is not None:
        return bad
    tombstone = copy.deepcopy(event)
    tombstone.update(
        status="cancelled",
        etag=_etag(call.state.now(), event["etag"]),
        updated=iso_ms_z(call.state.now()),
    )
    events[event["id"]] = tombstone
    return Outcome(204, None)


# GET /calendar/v3/calendars/{calendarId}/events --------------------------------------------------------


def _bool_param(call: Call, name: str) -> bool | Outcome:
    value = call.param(name)
    if value is None:
        return False
    if value.lower() not in ("true", "false"):
        return _invalid_parameter(name, value, "true|false")
    return value.lower() == "true"


def _property_filters(call: Call, name: str) -> list[tuple[str, str]] | Outcome:
    constraints: list[tuple[str, str]] = []
    for raw in call.request.query_params.getlist(name):
        key, sep, value = raw.partition("=")
        if not sep or not key:
            return BAD_REQUEST
        constraints.append((key, value))
    return constraints


def _time_param(call: Call, name: str) -> datetime | Outcome | None:
    raw = call.param(name)
    if raw is None:
        return None
    parsed = parse_rfc3339(raw)
    return BAD_REQUEST if parsed is None else parsed


def _matches_text(event: dict[str, Any], text: str) -> bool:
    haystack = [str(event.get(key, "")) for key in ("summary", "description", "location")]
    haystack += [str(a.get("email", "")) for a in event.get("attendees", [])]
    return any(text.casefold() in item.casefold() for item in haystack)


def _list(call: Call) -> Outcome:
    state = call.state
    calendar_id = call.path_param("calendarId")
    if not calendar_exists(state, calendar_id):
        return NOT_FOUND
    if call.param("syncToken") is not None:
        return FULL_SYNC_REQUIRED  # the sandbox never issues sync tokens
    bounds: dict[str, datetime | None] = {}
    for name in ("timeMin", "timeMax", "updatedMin"):
        parsed = _time_param(call, name)
        if isinstance(parsed, Outcome):
            return parsed
        bounds[name] = parsed
    time_min, time_max, updated_min = bounds["timeMin"], bounds["timeMax"], bounds["updatedMin"]
    if time_min is not None and time_max is not None and time_max <= time_min:
        return TIME_RANGE_EMPTY
    flags: dict[str, bool] = {}
    for name in ("showDeleted", "singleEvents"):
        flag = _bool_param(call, name)
        if isinstance(flag, Outcome):
            return flag
        flags[name] = flag
    order_by = call.param("orderBy")
    if order_by is not None and order_by not in ("startTime", "updated"):
        return _invalid_parameter("orderBy", order_by, "startTime|updated")
    if order_by == "startTime" and not flags["singleEvents"]:
        return ORDERING
    raw_max = call.param("maxResults")
    page_size = DEFAULT_MAX_RESULTS
    if raw_max is not None:
        if not raw_max.isdigit() or int(raw_max) < 1:
            return _invalid_field("maxResults")
        page_size = min(int(raw_max), MAX_RESULTS)
    offset = 0
    token = call.param("pageToken")
    if token is not None:
        decoded = _decode_page(token)
        if decoded is None:
            return _invalid_field("pageToken")
        offset = decoded
    zone = _response_zone(call)
    if isinstance(zone, Outcome):
        return zone
    constraints: dict[str, list[tuple[str, str]]] = {}
    for name, scope in (("privateExtendedProperty", "private"), ("sharedExtendedProperty", "shared")):
        parsed_filters = _property_filters(call, name)
        if isinstance(parsed_filters, Outcome):
            return parsed_filters
        constraints[scope] = parsed_filters
    text, ical_uid = call.param("q"), call.param("iCalUID")
    # Cancelled events appear with showDeleted, and on incremental reads (updatedMin).
    include_deleted = flags["showDeleted"] or updated_min is not None

    def keep(event: dict[str, Any]) -> bool:
        start, end = _starts_ends(event)
        props = event.get("extendedProperties", {})
        return (
            (include_deleted or event["status"] != "cancelled")
            and (time_min is None or end > time_min)
            and (time_max is None or start < time_max)
            and (updated_min is None or _updated_at(event) >= updated_min)
            and all(
                props.get(scope, {}).get(key) == value
                for scope, pairs in constraints.items()
                for key, value in pairs
            )
            and (not text or _matches_text(event, text))
            and (ical_uid is None or event["iCalUID"] == ical_uid)
        )

    events = list(events_of(state, calendar_id).values())
    items = [event for event in events if keep(event)]
    if order_by == "startTime":
        items.sort(key=lambda event: _starts_ends(event)[0])
    elif order_by == "updated":
        items.sort(key=_updated_at)
    page = items[offset : offset + page_size]
    updated = max((str(e["updated"]) for e in events), default=iso_ms_z(state.now()))
    body: dict[str, Any] = {
        "kind": "calendar#events",
        "etag": f'"p{uuid.uuid5(uuid.NAMESPACE_OID, "".join(e["etag"] for e in events)).hex[:14]}"',
        "summary": calendar_email(state),
        "updated": updated,
        "timeZone": state.seed.host_timezone,
        "accessRole": "writer",
        "defaultReminders": [],
    }
    if offset + page_size < len(items):
        body["nextPageToken"] = _encode_page(offset + page_size)
    body["items"] = [view(event, zone) for event in page]
    return Outcome(200, body)


# The vendor API ---------------------------------------------------------------------------------------


def _malformed_event(event: object) -> dict[str, Any] | None:
    if not isinstance(event, dict) or "id" not in event:
        return None
    return {
        "eventId": event["id"],
        "startTime": event.get("start", {}).get("dateTime"),
        "endTime": event.get("end", {}).get("dateTime"),
        "eventStatus": str(event.get("status", "")).upper(),
    }


class GoogleCalendarApi(VendorApi):
    name = "Google Calendar API v3"
    prefix = PREFIX
    calendar = "google"

    def __init__(self, router: APIRouter) -> None:
        self.router = router

    def render(self, outcome: Outcome) -> Response:
        return render(outcome)

    def authorize(self, call: Call) -> Outcome | None:
        """The sandbox bearer token, then the latest token grant's scopes (both before any fault rule)."""
        denied = super().authorize(call)
        return denied if denied is not None else scope_check(call.state, call.group)

    def unauthorized(self, call: Call, *, token_sent: bool) -> Outcome:
        if token_sent:
            return _auth_error(MSG_AUTH_INVALID, "Invalid Credentials", "authError")
        return _auth_error(MSG_AUTH_MISSING, "Login Required.", "required")

    def route_not_found(self, request: Request, now: datetime) -> Outcome:
        return NOT_FOUND

    def server_error(self, call: Call) -> Outcome:
        return BACKEND_ERROR

    def gateway_timeout(self, call: Call) -> Outcome:
        return UNAVAILABLE

    def not_found(self, call: Call) -> Outcome:
        if call.group != "freebusy":
            return NOT_FOUND
        body: dict[str, Any] = {"kind": "calendar#freeBusy"}
        if isinstance(call.body, dict):
            for name in ("timeMin", "timeMax"):
                parsed = parse_rfc3339(call.body.get(name))
                if parsed is not None:
                    body[name] = iso_ms_z(parsed)
        body["calendars"] = {calendar_id: _not_found_entry() for calendar_id in _item_ids(call.body)}
        return Outcome(200, body)

    def malformed(self, call: Call, normal: Outcome) -> Outcome:
        ok = 200 <= normal.status < 300 and isinstance(normal.body, dict)
        if call.group == "freebusy":
            calendars = normal.body.get("calendars", {}) if ok else {}
            reshaped: dict[str, Any] = {}
            for calendar_id, entry in calendars.items():
                reshaped[calendar_id] = {"busyPeriods": entry.get("busy", [])}
                if "errors" in entry:
                    reshaped[calendar_id]["errors"] = entry["errors"]
            body: dict[str, Any] = {"kind": "calendar#freeBusy"}
            if ok:
                body.update(timeMin=normal.body["timeMin"], timeMax=normal.body["timeMax"])
            body["calendars"] = reshaped
            return Outcome(200, body)
        if call.group == "events.list":
            items = [_malformed_event(e) for e in normal.body.get("items", [])] if ok else []
            return Outcome(200, {"kind": "calendar#events", "events": items, "count": len(items)})
        if call.group == "events.delete":
            return Outcome(200, {"deleted": {"eventId": call.path_param("eventId")}})
        return Outcome(200, {"event": _malformed_event(normal.body if ok else None)})

    def slot_taken_after_offer(self, call: Call, run: Callable[[], Outcome]) -> Outcome:
        state = call.state
        if call.group == "freebusy":
            outcome = run()
            if outcome.status == 200 and _offered(state, outcome.body):
                _take_window(state, call.extra.get("window"))
            return outcome
        _take_window(state, last_offered_window(state))
        result: Outcome = run()
        return result

    def setup_booking(self, state: SandboxState, setup: SetupBooking) -> Outcome:
        seed = state.seed
        length = timedelta(minutes=seed.event_length_minutes)
        description = f"Lead: {setup.lead_name} <{setup.lead_email}>"
        if setup.lead_timezone:
            description += f"\nLead time zone: {setup.lead_timezone}"
        private = {LEAD_EMAIL_PROPERTY: setup.lead_email, **(setup.extended_properties or {})}
        body: dict[str, Any] = {
            "summary": setup.title or f"{seed.event_title} with {setup.lead_name}",
            "description": description,
            "start": {"dateTime": iso_z(setup.start), "timeZone": seed.host_timezone},
            "end": {"dateTime": iso_z(setup.start + length), "timeZone": seed.host_timezone},
            "extendedProperties": {"private": private},
        }
        if setup.event_id is not None:
            body["id"] = setup.event_id
        if seed.google_sa_can_invite:
            body["attendees"] = [{"email": setup.lead_email, "displayName": setup.lead_name}]
        return insert_event(state, seed.google_calendar_id, body)


router = APIRouter()
EVENTS_PATH: Final = PREFIX + "/calendars/{calendarId}/events"


@router.post(PREFIX + "/freeBusy")
async def free_busy(request: Request) -> Response:
    return await run_call(request, API, "freebusy", _freebusy)


@router.post(EVENTS_PATH)
async def insert(request: Request) -> Response:
    return await run_call(request, API, "events.insert", _insert)


@router.get(EVENTS_PATH)
async def list_events(request: Request) -> Response:
    return await run_call(request, API, "events.list", _list)


@router.get(EVENTS_PATH + "/{eventId}")
async def get_event(request: Request) -> Response:
    return await run_call(request, API, "events.get", _get)


@router.patch(EVENTS_PATH + "/{eventId}")
async def patch_event(request: Request) -> Response:
    return await run_call(request, API, "events.patch", _patch)


@router.delete(EVENTS_PATH + "/{eventId}")
async def delete_event(request: Request) -> Response:
    return await run_call(request, API, "events.delete", _delete)


API: Final = GoogleCalendarApi(router)

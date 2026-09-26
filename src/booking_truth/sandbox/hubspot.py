"""HubSpot CRM API v3 subset under its real paths: contacts and meetings.

Mirrored operations (an adapter reaches them by swapping only its base URL, ``https://api.hubapi.com``):

- ``POST /crm/v3/objects/contacts/search``, ``POST /crm/v3/objects/contacts`` and
  ``PATCH /crm/v3/objects/contacts/{contactId}``;
- ``POST /crm/v3/objects/meetings`` (with inline associations to contacts),
  ``PATCH /crm/v3/objects/meetings/{meetingId}`` and ``GET /crm/v3/objects/meetings/{meetingId}``.

Objects have numeric string ids, string property values sorted by name, and datetimes that are
accepted as ISO 8601 or epoch milliseconds and emitted in Java ``Instant.toString()`` style. Error
bodies keep HubSpot's key order and compact JSON. Search is immediately consistent here, unlike on
HubSpot. ``docs/sandbox-fidelity.md`` lists the evidence for each behaviour and every known deviation.
"""

from __future__ import annotations

import copy
import json
import re
import secrets
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from fastapi import APIRouter, Request
from fastapi.responses import Response

from booking_truth.sandbox.common import Call, Outcome, VendorApi, run_call

PREFIX: Final = "/crm/v3"
JSON_HUBSPOT: Final = "application/json;charset=utf-8"
CORRELATION_HEADER: Final = "x-hubspot-correlation-id"
#: Carries the same value as ``x-hubspot-correlation-id`` on HubSpot's responses.
REQUEST_ID_HEADER: Final = "x-request-id"
#: The portal id that validation errors embed (a sandbox value).
PORTAL_ID: Final = 20000001

MSG_AUTH: Final = (
    "Authentication credentials not found. This API supports OAuth 2.0 authentication and you can find more "
    "details at https://developers.hubspot.com/docs/methods/auth/oauth-overview"
)
MSG_RESOURCE_NOT_FOUND: Final = "resource not found"
MSG_NOT_NUMERIC: Final = "Object not found.  objectId are usually numeric."
MSG_CONFLICT: Final = "Contact already exists. Existing ID: {id}"
MSG_INVALID_ASSOCIATIONS: Final = "One or more associations are invalid"
MSG_WRONG_DIRECTION: Final = "invalid from object type 0-47 for associations to be created. expected: 0-1"

MEETING_TO_CONTACT: Final = 200
CONTACT_TO_MEETING: Final = 199
MEETING_CONTACT_TYPE: Final = "meeting_event_to_contact"
MEETING_OUTCOMES: Final = (
    ("SCHEDULED", "Scheduled"),
    ("COMPLETED", "Completed"),
    ("RESCHEDULED", "Rescheduled"),
    ("NO_SHOW", "No Show"),
    ("CANCELED", "Canceled"),
)
OPERATORS: Final = (
    "BETWEEN", "CONTAINS_TOKEN", "EQ", "GT", "GTE", "HAS_PROPERTY", "IN", "LT", "LTE", "NEQ",
    "NOT_CONTAINS_TOKEN", "NOT_HAS_PROPERTY", "NOT_IN",
)  # fmt: skip
MAX_FILTER_GROUPS: Final = 5
MAX_FILTERS_PER_GROUP: Final = 6
MAX_FILTERS: Final = 18
DEFAULT_LIMIT: Final = 10
MAX_LIMIT: Final = 200
#: The contact properties that ``query`` searches ("default searchable properties" in the search guide).
_SEARCHABLE: Final = (
    "firstname", "lastname", "email", "phone", "hs_additional_emails", "fax", "mobilephone", "company",
    "hs_marketable_until_renewal",
)  # fmt: skip
_NUMERIC_ID: Final = re.compile(r"[0-9]+")
_SMALL_INT: Final = re.compile(r"[0-9]{1,15}")
_MILLIS: Final = re.compile(r"-?[0-9]{1,15}")
_ISO: Final = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[Tt]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:[Zz]|[+-]\d{2}:?\d{2})?)?", re.ASCII
)
_EMAIL: Final = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

_CONTACT_WRITABLE: Final = (
    "email", "firstname", "lastname", "phone", "mobilephone", "company", "jobtitle", "website", "address",
    "city", "state", "zip", "country", "hs_timezone", "hs_language", "lifecyclestage", "hs_lead_status",
    "hubspot_owner_id", "message", "createdate",
)  # fmt: skip
_CONTACT_READ_ONLY: Final = (
    "hs_object_id", "lastmodifieddate", "hs_all_contact_vids", "hs_email_domain", "hs_is_contact",
    "hs_is_unworked", "hs_lifecyclestage_lead_date", "hs_marketable_status", "hs_marketable_until_renewal",
    "hs_object_source", "hs_object_source_id", "hs_object_source_label", "hs_pipeline",
    "hs_additional_emails",
)  # fmt: skip
_MEETING_WRITABLE: Final = (
    "hs_timestamp", "hs_meeting_title", "hubspot_owner_id", "hs_meeting_body", "hs_internal_meeting_notes",
    "hs_meeting_external_url", "hs_meeting_location", "hs_meeting_start_time", "hs_meeting_end_time",
    "hs_meeting_outcome", "hs_activity_type", "hs_attachment_ids",
)  # fmt: skip


@dataclass(frozen=True)
class ObjectType:
    """What the sandbox knows about one CRM object type's properties."""

    name: str  # plural path segment and association key
    writable: frozenset[str]
    read_only: frozenset[str]
    datetimes: frozenset[str]
    enums: dict[str, tuple[tuple[str, str], ...]]
    #: Properties that GET and search return when the request names none.
    defaults: tuple[str, ...]
    #: Properties that come back whether or not they were requested.
    always: tuple[str, ...]
    created: str
    modified: str

    @property
    def known(self) -> frozenset[str]:
        return self.writable | self.read_only


CONTACTS: Final = ObjectType(
    name="contacts",
    writable=frozenset(_CONTACT_WRITABLE),
    read_only=frozenset(_CONTACT_READ_ONLY),
    datetimes=frozenset({"createdate", "lastmodifieddate", "hs_lifecyclestage_lead_date"}),
    enums={},
    defaults=("createdate", "email", "firstname", "hs_object_id", "lastmodifieddate", "lastname"),
    always=("createdate", "hs_object_id", "lastmodifieddate"),
    created="createdate",
    modified="lastmodifieddate",
)
MEETINGS: Final = ObjectType(
    name="meetings",
    writable=frozenset(_MEETING_WRITABLE),
    read_only=frozenset({"hs_createdate", "hs_lastmodifieddate", "hs_object_id"}),
    datetimes=frozenset(
        (
            "hs_timestamp",
            "hs_meeting_start_time",
            "hs_meeting_end_time",
            "hs_createdate",
            "hs_lastmodifieddate",
        )
    ),
    enums={"hs_meeting_outcome": MEETING_OUTCOMES},
    defaults=("hs_createdate", "hs_lastmodifieddate", "hs_object_id"),
    always=("hs_createdate", "hs_lastmodifieddate", "hs_object_id"),
    created="hs_createdate",
    modified="hs_lastmodifieddate",
)


# Envelopes ---------------------------------------------------------------------------------------


def uuid7(now: datetime) -> str:
    """A UUIDv7 (millisecond timestamp plus random bits), the form of HubSpot's recent correlation ids."""
    millis = int(now.timestamp() * 1000) & ((1 << 48) - 1)
    value = (millis << 80) | (0x7 << 76) | (secrets.randbits(12) << 64) | (0b10 << 62) | secrets.randbits(62)
    return str(uuid.UUID(int=value))


def error(
    now: datetime,
    status: int,
    message: str,
    category: str | None = None,
    *,
    errors: list[dict[str, Any]] | None = None,
    context: dict[str, list[str]] | None = None,
) -> Outcome:
    """HubSpot's error body: ``status, message, correlationId, [errors], [context], [category]``."""
    body: dict[str, Any] = {"status": "error", "message": message, "correlationId": uuid7(now)}
    if errors is not None:
        body["errors"] = errors
    if context is not None:
        body["context"] = context
    if category is not None:
        body["category"] = category
    return Outcome(status, body)


def _resource_not_found(call: Call) -> Outcome:
    return error(call.state.now(), 404, MSG_RESOURCE_NOT_FOUND)


def _input_error(call: Call, message: str) -> Outcome:
    """A request body that does not fit the input class (sandbox wording in HubSpot's envelope)."""
    return error(call.state.now(), 400, f"Invalid input JSON: {message}", "VALIDATION_ERROR")


@dataclass(frozen=True)
class Issue:
    """One entry of a ``Property values were not valid`` error."""

    name: str
    value: str
    code: str
    message: str


def _validation_error(call: Call, issues: list[Issue]) -> Outcome:
    """400 ``VALIDATION_ERROR`` with the embedded JSON array in ``message`` and the structured ``errors``."""
    embedded = [
        {
            "isValid": False,
            "message": issue.message,
            "error": issue.code,
            "name": issue.name,
            "localizedErrorMessage": issue.message,
            "propertyValue": issue.value,
            "portalId": PORTAL_ID,
        }
        for issue in issues
    ]
    message = "Property values were not valid: " + json.dumps(
        embedded, separators=(",", ":"), ensure_ascii=False
    )
    errors = [
        {"message": issue.message, "code": issue.code, "context": {"propertyName": [issue.name]}}
        for issue in issues
    ]
    return error(call.state.now(), 400, message, "VALIDATION_ERROR", errors=errors)


def render(outcome: Outcome) -> Response:
    """Compact JSON, ``application/json;charset=utf-8``, and ``x-hubspot-correlation-id`` and ``x-request-id``
    (both the body's ``correlationId`` on errors)."""
    body = outcome.body
    correlation = body.get("correlationId") if isinstance(body, dict) else None
    correlation_id = str(correlation or uuid7(datetime.now(UTC)))
    headers = {CORRELATION_HEADER: correlation_id, REQUEST_ID_HEADER: correlation_id, **outcome.headers}
    content = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    return Response(content, status_code=outcome.status, media_type=JSON_HUBSPOT, headers=headers)


# Values -------------------------------------------------------------------------------------------


def instant(value: datetime) -> str:
    """Java ``Instant.toString()`` at millisecond precision: no fraction when the milliseconds are zero."""
    value = value.astimezone(UTC)
    millis = value.microsecond // 1000
    base = value.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}Z" if millis == 0 else f"{base}.{millis:03d}Z"


def parse_datetime(value: str) -> datetime | None:
    """ISO 8601 (a value without an offset is UTC; a date is midnight UTC) or epoch milliseconds."""
    try:
        if _MILLIS.fullmatch(value):
            return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=int(value))
        if not _ISO.fullmatch(value):
            return None
        parsed = datetime.fromisoformat(value.upper())
    except (ValueError, OverflowError):
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _coerce(value: object) -> str | None:
    """Property values are strings; numbers and booleans are read as their text. ``None`` for others."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return None


def _options_text(options: tuple[tuple[str, str], ...]) -> str:
    items = [
        f'label: "{label}"\nvalue: "{value}"\ndisplay_order: {index}\nhidden: false\nread_only: false\n'
        for index, (value, label) in enumerate(options)
    ]
    return "[" + ", ".join(items) + "]"


def check_properties(
    call: Call, kind: ObjectType, raw: object
) -> tuple[dict[str, str | None], Outcome | None]:
    """Validate a ``properties`` map. The result maps each property to its stored text, or ``None`` when an
    empty string clears it. Every invalid property is reported in one 400, as HubSpot does."""
    if not isinstance(raw, dict):
        return {}, _input_error(call, "properties must be an object of property names to values")
    values: dict[str, str | None] = {}
    issues: list[Issue] = []
    for name, value in raw.items():
        if value is None:
            continue
        text = _coerce(value)
        if text is None:
            return {}, _input_error(call, f"the value of property {name} must be a string")
        if name not in kind.known:
            issues.append(Issue(name, text, "PROPERTY_DOESNT_EXIST", f'Property "{name}" does not exist'))
            continue
        if name in kind.read_only:
            message = f'"{name}" is a calculated property; its value cannot be set.'
            issues.append(Issue(name, text, "READ_ONLY_VALUE", message))
            continue
        if text == "":
            values[name] = None
            continue
        if name in kind.datetimes:
            parsed = parse_datetime(text)
            if parsed is None:
                issues.append(Issue(name, text, "INVALID_LONG", f"{text} was not a valid long."))
                continue
            text = instant(parsed)
        elif name == "email" and not _EMAIL.fullmatch(text):
            issues.append(Issue(name, text, "INVALID_EMAIL", f"Email address {text} is invalid"))
            continue
        elif name in kind.enums and text not in {v for v, _ in kind.enums[name]}:
            message = f"{text} was not one of the allowed options: {_options_text(kind.enums[name])}"
            issues.append(Issue(name, text, "INVALID_OPTION", message))
            continue
        values[name] = text
    return values, (_validation_error(call, issues) if issues else None)


def _sorted(props: dict[str, Any]) -> dict[str, Any]:
    return {key: props[key] for key in sorted(props)}


def object_view(kind: ObjectType, obj: dict[str, Any], names: Iterable[str] | None) -> dict[str, Any]:
    """``SimplePublicObject``: every stored property (``names`` is ``None``) or the named ones plus the
    always-returned ones, unset ones as ``null`` and unknown ones left out."""
    stored: dict[str, Any] = obj["properties"]
    if names is None:
        props = dict(stored)
    else:
        wanted = [n for n in dict.fromkeys([*names, *kind.always]) if n in kind.known]
        props = {name: stored.get(name) for name in wanted}
    return {
        "id": obj["id"],
        "properties": _sorted(props),
        "createdAt": obj["createdAt"],
        "updatedAt": obj["updatedAt"],
        "archived": obj["archived"],
    }


def _names(value: object) -> list[str] | None:
    """The ``properties`` list of a search body."""
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, str)]
    return None


def _query_names(call: Call, name: str) -> list[str] | None:
    """A query list: comma-separated, repeatable, or both."""
    values = call.request.query_params.getlist(name)
    if not values:
        return None
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def _body(call: Call) -> dict[str, Any] | Outcome:
    if call.body_invalid:
        # HubSpot's parser error: a position and the parser's reason, no category (sandbox reason texts).
        message = "Invalid input JSON on line 1, column 1: Unexpected value"
        try:
            json.loads(call.body if isinstance(call.body, str) else "")
        except json.JSONDecodeError as exc:
            message = f"Invalid input JSON on line {exc.lineno}, column {exc.colno}: {exc.msg}"
        except (ValueError, RecursionError):
            pass
        return error(call.state.now(), 400, message)
    if not isinstance(call.body, dict):
        return _input_error(call, "the request body must be a JSON object")
    return call.body


def _lookup(
    call: Call, store: dict[str, dict[str, Any]], raw_id: str, *, by_email: bool = False
) -> dict[str, Any] | Outcome:
    if by_email:
        found = _by_email(store, raw_id)
        return found if found is not None else _resource_not_found(call)
    if not _NUMERIC_ID.fullmatch(raw_id):
        return error(call.state.now(), 404, MSG_NOT_NUMERIC, "OBJECT_NOT_FOUND", context={"id": [raw_id]})
    found = store.get(raw_id.lstrip("0") or "0")
    return found if found is not None else _resource_not_found(call)


def _by_email(store: dict[str, dict[str, Any]], email: str) -> dict[str, Any] | None:
    wanted = email.strip().casefold()
    for obj in store.values():
        if str(obj["properties"].get("email", "")).casefold() == wanted:
            return obj
    return None


def _apply(obj: dict[str, Any], kind: ObjectType, values: dict[str, str | None], now: datetime) -> None:
    props: dict[str, Any] = dict(obj["properties"])
    for name, value in values.items():
        if value is None:
            props.pop(name, None)
        else:
            props[name] = value
    props[kind.modified] = instant(now)
    obj["properties"] = _sorted(props)
    obj["updatedAt"] = instant(now)


def _location(call: Call, kind: ObjectType, object_id: str) -> dict[str, str]:
    base = str(call.request.base_url).rstrip("/")
    return {"location": f"{base}{PREFIX}/objects/{kind.name}/{object_id}"}


# Contacts --------------------------------------------------------------------------------------------


def _email_domain(email: str) -> str:
    return email.rpartition("@")[2].lower()


def _create_contact(call: Call) -> Outcome:
    state = call.state
    body = _body(call)
    if isinstance(body, Outcome):
        return body
    if "properties" not in body:
        return _input_error(call, "some of required attributes are not set [properties]")
    values, problem = check_properties(call, CONTACTS, body["properties"])
    if problem is not None:
        return problem
    email = values.get("email")
    if email is not None:
        existing = _by_email(state.hubspot_contacts, email)
        if existing is not None:
            return error(state.now(), 409, MSG_CONFLICT.format(id=existing["id"]), "CONFLICT")
    contact_id = str(state.next_id())
    now = instant(state.now())
    props: dict[str, Any] = {k: v for k, v in values.items() if v is not None}
    props.setdefault("createdate", now)
    props.setdefault("lifecyclestage", "lead")
    props.update(
        hs_all_contact_vids=contact_id,
        hs_is_contact="true",
        hs_is_unworked="true",
        hs_marketable_status="false",
        hs_marketable_until_renewal="false",
        hs_object_id=contact_id,
        hs_object_source="INTEGRATION",
        hs_object_source_label="INTEGRATION",
        hs_pipeline="contacts-lifecycle-pipeline",
        lastmodifieddate=now,
    )
    if props["lifecyclestage"] == "lead":
        props["hs_lifecyclestage_lead_date"] = now
    if email is not None:
        props["hs_email_domain"] = _email_domain(email)
    contact = {
        "id": contact_id,
        "properties": _sorted(props),
        "createdAt": now,
        "updatedAt": now,
        "archived": False,
    }
    state.hubspot_contacts[contact_id] = contact
    return Outcome(201, object_view(CONTACTS, contact, None), headers=_location(call, CONTACTS, contact_id))


def _update_contact(call: Call) -> Outcome:
    state = call.state
    body = _body(call)
    if isinstance(body, Outcome):
        return body
    if "properties" not in body:
        return _input_error(call, "some of required attributes are not set [properties]")
    by_email = call.param("idProperty") == "email"
    found = _lookup(call, state.hubspot_contacts, call.path_param("contactId"), by_email=by_email)
    if isinstance(found, Outcome):
        return found
    values, problem = check_properties(call, CONTACTS, body["properties"])
    if problem is not None:
        return problem
    email = values.get("email")
    if email is not None:
        other = _by_email(state.hubspot_contacts, email)
        if other is not None and other["id"] != found["id"]:
            return error(state.now(), 409, MSG_CONFLICT.format(id=other["id"]), "CONFLICT")
        values["hs_email_domain"] = _email_domain(email)
    elif "email" in values:
        values["hs_email_domain"] = None  # clearing the email clears its calculated domain
    updated = copy.deepcopy(found)
    _apply(updated, CONTACTS, values, state.now())
    state.hubspot_contacts[found["id"]] = updated
    return Outcome(200, object_view(CONTACTS, updated, None))


def _search_filters(call: Call, body: dict[str, Any]) -> list[list[dict[str, Any]]] | Outcome:
    """``filterGroups`` (OR of groups, AND inside a group); a top-level ``filters`` list is one more group."""
    groups: list[Any] = []
    if body.get("filterGroups") is not None:
        if not isinstance(body["filterGroups"], list):
            return _input_error(call, "filterGroups must be an array")
        groups = list(body["filterGroups"])
    if body.get("filters") is not None:
        groups.append({"filters": body["filters"]})
    parsed: list[list[dict[str, Any]]] = []
    for group in groups:
        filters = group.get("filters") if isinstance(group, dict) else None
        if not isinstance(filters, list):
            return _input_error(call, "each filter group needs a filters array")
        if len(filters) > MAX_FILTERS_PER_GROUP:
            count = len(filters)
            message = (
                f"too many filters per filterGroup (count: {count}, max allowed: {MAX_FILTERS_PER_GROUP})"
            )
            return _input_error(call, message)
        for item in filters:
            if not isinstance(item, dict) or not isinstance(item.get("propertyName"), str):
                return _input_error(call, "each filter needs a propertyName")
            if item.get("operator") not in OPERATORS:
                return _input_error(call, f"operator must be one of {', '.join(OPERATORS)}")
        parsed.append(filters)
    if len(parsed) > MAX_FILTER_GROUPS:
        message = f"too many filterGroups (count: {len(parsed)}, max allowed: {MAX_FILTER_GROUPS})"
        return _input_error(call, message)
    total = sum(len(group) for group in parsed)
    if total > MAX_FILTERS:
        return _input_error(call, f"too many filters (count: {total}, max allowed: {MAX_FILTERS})")
    return parsed


def _comparable(kind: ObjectType, name: str, text: str) -> tuple[int, float | str]:
    if name in kind.datetimes:
        parsed = parse_datetime(text)
        if parsed is not None:
            return 0, parsed.timestamp()
    try:
        return 0, float(text)
    except ValueError:
        return 1, text.casefold()


def _filter_matches(kind: ObjectType, props: dict[str, Any], item: dict[str, Any]) -> bool:
    name, operator = item["propertyName"], item["operator"]
    actual = props.get(name)
    present = actual not in (None, "")
    if operator == "HAS_PROPERTY":
        return present
    if operator == "NOT_HAS_PROPERTY":
        return not present
    value = _coerce(item.get("value")) or ""
    raw_values = item.get("values")
    values = (
        [v for v in (_coerce(x) for x in raw_values) if v is not None] if isinstance(raw_values, list) else []
    )
    text = str(actual) if present else ""
    if operator in ("EQ", "NEQ"):
        equal = present and text.casefold() == value.casefold()
        return equal if operator == "EQ" else not equal
    if operator in ("IN", "NOT_IN"):
        # "the searched values must be lowercase": the stored value is lowercased, the values are not.
        inside = present and text.lower() in set(values)
        return inside if operator == "IN" else not inside
    if operator in ("CONTAINS_TOKEN", "NOT_CONTAINS_TOKEN"):
        token = value.strip("*").casefold()
        contains = present and token in text.casefold()
        return contains if operator == "CONTAINS_TOKEN" else not contains
    if not present:
        return False
    left = _comparable(kind, name, text)
    low = _comparable(kind, name, value)
    if operator == "BETWEEN":
        high = _comparable(kind, name, _coerce(item.get("highValue")) or "")
        return low <= left <= high
    return {
        "GT": left > low,
        "GTE": left >= low,
        "LT": left < low,
        "LTE": left <= low,
    }[operator]


def _offset(value: object) -> int | None:
    if value is None:
        return 0
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and _SMALL_INT.fullmatch(value):
        return int(value)
    return None


def _search(call: Call) -> Outcome:
    state = call.state
    body = _body(call)
    if isinstance(body, Outcome):
        return body
    groups = _search_filters(call, body)
    if isinstance(groups, Outcome):
        return groups
    offset = _offset(body.get("after"))
    if offset is None:
        return _input_error(call, "after must be a non-negative integer")
    raw_limit = body.get("limit")
    if raw_limit is None:
        raw_limit = DEFAULT_LIMIT
    if isinstance(raw_limit, str) and _SMALL_INT.fullmatch(raw_limit):
        raw_limit = int(raw_limit)
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
        return _input_error(call, "limit must be an integer")
    limit = max(0, min(raw_limit, MAX_LIMIT))
    names = _names(body.get("properties"))
    query = body.get("query")
    if query is not None and not isinstance(query, str):
        return _input_error(call, "query must be a string")

    def keep(contact: dict[str, Any]) -> bool:
        props = contact["properties"]
        if groups and not any(all(_filter_matches(CONTACTS, props, f) for f in group) for group in groups):
            return False
        if query:
            return any(query.casefold() in str(props.get(name, "")).casefold() for name in _SEARCHABLE)
        return True

    items = sorted((c for c in state.hubspot_contacts.values() if keep(c)), key=lambda c: int(c["id"]))
    sorts = body.get("sorts")
    if isinstance(sorts, list) and sorts:
        first = sorts[0]
        name = (
            first
            if isinstance(first, str)
            else first.get("propertyName")
            if isinstance(first, dict)
            else None
        )
        descending = isinstance(first, dict) and first.get("direction") == "DESCENDING"
        if isinstance(name, str):
            items.sort(
                key=lambda c: _comparable(CONTACTS, name, str(c["properties"].get(name, ""))),
                reverse=descending,
            )
    page = items[offset : offset + limit]
    result: dict[str, Any] = {
        "total": len(items),
        "results": [
            object_view(CONTACTS, c, names if names is not None else CONTACTS.defaults) for c in page
        ],
    }
    if offset + limit < len(items) and limit > 0:
        result["paging"] = {"next": {"after": str(offset + limit)}}
    return Outcome(200, result)


# Meetings ------------------------------------------------------------------------------------------


def _association_ids(call: Call, raw: object) -> list[str] | Outcome:
    """Inline associations of a meeting create: only meeting-to-contact (``HUBSPOT_DEFINED`` type 200)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return _input_error(call, "associations must be an array")
    contact_ids: list[str] = []
    for item in raw:
        target = item.get("to") if isinstance(item, dict) else None
        types = item.get("types") if isinstance(item, dict) else None
        target_id = target.get("id") if isinstance(target, dict) else None
        if isinstance(target_id, bool) or not isinstance(target_id, str | int) or not isinstance(types, list):
            return _input_error(call, "each association needs to.id and types")
        if not types:
            return _input_error(call, "each association needs at least one type")
        for kind in types:
            category = kind.get("associationCategory") if isinstance(kind, dict) else None
            type_id = kind.get("associationTypeId") if isinstance(kind, dict) else None
            if type_id == CONTACT_TO_MEETING:
                return error(call.state.now(), 400, MSG_WRONG_DIRECTION, "VALIDATION_ERROR")
            if type_id != MEETING_TO_CONTACT or category != "HUBSPOT_DEFINED":
                message = (
                    f"association type {category}/{type_id} is not mirrored by this sandbox; "
                    f"use HUBSPOT_DEFINED/{MEETING_TO_CONTACT} (meeting to contact)"
                )
                return _input_error(call, message)
        contact_id = str(target_id)
        if contact_id not in call.state.hubspot_contacts:
            context = {
                "INVALID_OBJECT_IDS": [f"CONTACT={contact_id} is not valid"],
                "objectId": [contact_id],
                "objectType": ["CONTACT"],
            }
            return error(call.state.now(), 400, MSG_INVALID_ASSOCIATIONS, "VALIDATION_ERROR", context=context)
        if contact_id not in contact_ids:
            contact_ids.append(contact_id)
    return contact_ids


def _create_meeting(call: Call) -> Outcome:
    state = call.state
    body = _body(call)
    if isinstance(body, Outcome):
        return body
    if "properties" not in body:
        return _input_error(call, "some of required attributes are not set [properties]")
    values, problem = check_properties(call, MEETINGS, body["properties"])
    if problem is not None:
        return problem
    if values.get("hs_timestamp") is None:
        # "When the property value is missing, the value will default to hs_meeting_start_time."
        if values.get("hs_meeting_start_time") is None:
            issue = Issue("hs_timestamp", "", "MISSING_REQUIRED_PROPERTY", "hs_timestamp is required")
            return _validation_error(call, [issue])
        values["hs_timestamp"] = values["hs_meeting_start_time"]
    contact_ids = _association_ids(call, body.get("associations"))
    if isinstance(contact_ids, Outcome):
        return contact_ids
    meeting_id = str(state.next_id())
    now = instant(state.now())
    props: dict[str, Any] = {k: v for k, v in values.items() if v is not None}
    props.update(hs_createdate=now, hs_lastmodifieddate=now, hs_object_id=meeting_id)
    meeting: dict[str, Any] = {
        "id": meeting_id,
        "properties": _sorted(props),
        "createdAt": now,
        "updatedAt": now,
        "archived": False,
    }
    if contact_ids:
        results = [{"id": contact_id, "type": MEETING_CONTACT_TYPE} for contact_id in contact_ids]
        meeting["associations"] = {"contacts": {"results": results}}
    state.hubspot_meetings[meeting_id] = meeting
    return Outcome(201, object_view(MEETINGS, meeting, None), headers=_location(call, MEETINGS, meeting_id))


def _update_meeting(call: Call) -> Outcome:
    state = call.state
    body = _body(call)
    if isinstance(body, Outcome):
        return body
    if "properties" not in body:
        return _input_error(call, "some of required attributes are not set [properties]")
    found = _lookup(call, state.hubspot_meetings, call.path_param("meetingId"))
    if isinstance(found, Outcome):
        return found
    values, problem = check_properties(call, MEETINGS, body["properties"])
    if problem is not None:
        return problem
    updated = copy.deepcopy(found)
    _apply(updated, MEETINGS, values, state.now())
    state.hubspot_meetings[found["id"]] = updated
    return Outcome(200, object_view(MEETINGS, updated, None))


def _get_meeting(call: Call) -> Outcome:
    found = _lookup(call, call.state.hubspot_meetings, call.path_param("meetingId"))
    if isinstance(found, Outcome):
        return found
    names = _query_names(call, "properties")
    out = object_view(MEETINGS, found, names if names is not None else MEETINGS.defaults)
    wanted = {name.lower() for name in _query_names(call, "associations") or []}
    if wanted & {"contacts", "contact", "0-1"} and "associations" in found:
        out["associations"] = copy.deepcopy(found["associations"])
    return Outcome(200, out)


# The vendor API ---------------------------------------------------------------------------------------


def _malformed_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or "id" not in value:
        return None
    return {"objectId": int(value["id"]), "props": copy.deepcopy(value.get("properties", {}))}


class HubSpotApi(VendorApi):
    name = "HubSpot CRM API v3"
    prefix = PREFIX

    def __init__(self, router: APIRouter) -> None:
        self.router = router

    def render(self, outcome: Outcome) -> Response:
        return render(outcome)

    def unauthorized(self, call: Call, *, token_sent: bool) -> Outcome:
        return error(call.state.now(), 401, MSG_AUTH, "INVALID_AUTHENTICATION")

    def route_not_found(self, request: Request, now: datetime) -> Outcome:
        return error(now, 404, MSG_RESOURCE_NOT_FOUND)

    def server_error(self, call: Call) -> Outcome:
        return error(call.state.now(), 500, "internal error")

    def gateway_timeout(self, call: Call) -> Outcome:
        return error(call.state.now(), 504, "gateway timeout")

    def not_found(self, call: Call) -> Outcome:
        return _resource_not_found(call)

    def malformed(self, call: Call, normal: Outcome) -> Outcome:
        ok = 200 <= normal.status < 300 and isinstance(normal.body, dict)
        if call.group == "crm.contacts.search":
            results = normal.body.get("results", []) if ok else []
            objects = [_malformed_object(item) for item in results]
            return Outcome(200, {"count": len(objects), "objects": objects})
        return Outcome(200, {"object": _malformed_object(normal.body if ok else None)})


router = APIRouter()
CONTACTS_PATH: Final = PREFIX + "/objects/contacts"
MEETINGS_PATH: Final = PREFIX + "/objects/meetings"


@router.post(CONTACTS_PATH + "/search")
async def search_contacts(request: Request) -> Response:
    return await run_call(request, API, "crm.contacts.search", _search)


@router.post(CONTACTS_PATH)
async def create_contact(request: Request) -> Response:
    return await run_call(request, API, "crm.contacts.create", _create_contact)


@router.patch(CONTACTS_PATH + "/{contactId}")
async def update_contact(request: Request) -> Response:
    return await run_call(request, API, "crm.contacts.update", _update_contact)


@router.post(MEETINGS_PATH)
async def create_meeting(request: Request) -> Response:
    return await run_call(request, API, "crm.meetings.create", _create_meeting)


@router.patch(MEETINGS_PATH + "/{meetingId}")
async def update_meeting(request: Request) -> Response:
    return await run_call(request, API, "crm.meetings.update", _update_meeting)


@router.get(MEETINGS_PATH + "/{meetingId}")
async def get_meeting(request: Request) -> Response:
    return await run_call(request, API, "crm.meetings.get", _get_meeting)


API: Final = HubSpotApi(router)

"""Google Calendar API v3 adapter: service-account (JWT-bearer) auth, ``freeBusy`` and events.

Auth: the service-account key file (``BT_GOOGLE_SERVICE_ACCOUNT_FILE``) gives ``client_email`` and
``private_key``. The adapter signs an RS256 JWT (``iss`` the client email, ``scope``
``https://www.googleapis.com/auth/calendar``, ``aud`` the fixed
``https://oauth2.googleapis.com/token`` Google's own client libraries always use regardless of the
configured token endpoint, ``iat``/``exp`` from the injected clock with ``exp - iat`` at most an hour) and
exchanges it at ``BT_GOOGLE_TOKEN_URI`` over ``POST`` (form-encoded, the documented JWT-bearer grant). The
access token is cached until 60 seconds before it expires.

``find_slots`` posts ``freeBusy`` for the one configured calendar and computes slots client-side with
:mod:`booking_truth.calendars.slotcalc` (working hours, days, slot length, minimum notice and horizon), since
``freeBusy`` reports only busy time. Responses are parsed strictly: the calendar entry must be present,
carry no ``errors``, and hold a valid ``busy`` list, else ``Unavailable("missing_calendar" | "malformed" |
"error" | "timeout")``. ``lenient=True`` (the naive baseline, ``fail_closed`` off) reads a missing or
erroring entry as no busy time at all — the fail-open bug the guard exists to catch — and keeps whatever
well-formed ``busy`` items a malformed body still has (an empty list when none parse). Every other method
parses strictly regardless of ``lenient``; the flag only changes how verbose an error's ``detail`` text is
(the same two-switch design as :mod:`booking_truth.calendars.calcom`).

``events.insert`` performs no conflict check on Google's side, so the guarded configuration
(``lenient=False``, i.e. ``fail_closed`` on) re-checks ``freeBusy`` for the exact slot right before
inserting and answers ``WriteRejected("slot_taken")`` when it is no longer free; the naive baseline skips
this and can double-book, same as a real integration that only trusts the availability it fetched earlier.
The event id is the lowercase, unpadded base32hex encoding of the first 20 bytes of ``sha256(idem_key)``
(a valid Google id) when a key is given, else omitted so the server assigns one. When that id already
names an event whose start and lead match this call's — almost always a retry of this exact write, since
the id is deterministic — it is read back with ``get_booking`` and adopted as ``WriteOk`` straight away,
before the conflict pre-check or the insert ever run, so a repeated ``create_booking`` for the same key is
never mistaken for a slot a third party took nor sent to insert for a ``409`` it would just have to unwind
again. The same adoption runs once more if a ``409 duplicate`` is reached anyway (a race with that earlier
attempt landing between the check and this insert); only then does an unmatched ``409`` reach the caller as
``WriteRejected("duplicate")``.

A service account cannot populate ``attendees`` without domain-wide delegation: the adapter tries the
lead as an attendee on every insert until the vendor answers ``403 forbiddenForServiceAccounts``, then
retries once without attendees and remembers not to send them again for the lifetime of this adapter
instance. The lead's email always goes to ``extendedProperties.private.bt_lead_email`` (plus the
description), so lookups and grading do not depend on attendees ever having been accepted. This is the
documented Google limitation: without domain-wide delegation, the prospect gets no calendar invitation
email.

``reschedule`` is ``events.patch`` on ``start``/``end`` (Google keeps the same event id; unlike Cal.com there
is no new booking to point ``previous_ref`` at, but passing the unchanged ref is harmless — the claims
store only voids a previous ref when it differs from the new one). Patching an already-cancelled event
succeeds without reviving it (the sandbox, like the documented merge-patch semantics, keeps a status the
patch does not set), which the adapter reports as ``WriteRejected("not_found")`` since the booking to
change no longer exists. ``cancel`` is ``events.delete``; success is an empty body, so the adapter reads
the resulting tombstone back to build the ``BookingRecord``. Deleting an already-deleted event is
``410``, mapped to ``WriteRejected("duplicate")``. ``get_booking``/``list_bookings`` use ``events.get`` and
``events.list`` (``privateExtendedProperty=bt_lead_email=<email>``, ``timeMin``/``timeMax``,
``singleEvents=true``); a soft-deleted event is excluded from a list by default, same as Google.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Literal
from urllib.parse import quote

import httpx
import jwt

from booking_truth import __version__
from booking_truth.calendars.base import (
    BookingRecord,
    BookingStatus,
    ListResult,
    NotFound,
    ReadResult,
    SlotsResult,
    Unavailable,
    WriteOk,
    WriteRejected,
    WriteResult,
    WriteUnknown,
)
from booking_truth.calendars.slotcalc import Hours, compute_slots, overlaps
from booking_truth.timeutil import Clock, SystemClock, ensure_utc, iso_z, parse_iso

GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
#: google-auth always uses this audience for the assertion, even when ``token_uri`` differs (verified in
#: research/google-calendar-v3.md #8/#11); the sandbox's fake token endpoint checks for it too.
TOKEN_AUDIENCE: Final = "https://oauth2.googleapis.com/token"  # noqa: S105 - a claim value, not a secret
GOOGLE_SCOPE: Final = "https://www.googleapis.com/auth/calendar"
TOKEN_LIFETIME_S: Final = 3600
TOKEN_REFRESH_MARGIN_S: Final = 60
DEFAULT_TIMEOUT: Final = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
DEFAULT_EVENT_TITLE: Final = "Meeting"
MAX_DETAIL_CHARS: Final = 2000
MAX_STRICT_DETAIL_CHARS: Final = 300
MAX_LIST_PAGES: Final = 50
#: ``extendedProperties.private`` keys the adapter writes on every insert.
LEAD_EMAIL_PROPERTY: Final = "bt_lead_email"
EVENT_KEY_PROPERTY: Final = "bt_event_key"
IDEM_PROPERTY: Final = "bt_idem"


class MalformedResponse(ValueError):
    """A vendor response that does not have the documented shape."""


class MissingCalendar(ValueError):
    """The requested calendar is absent from, or carries an ``errors`` entry in, a ``freeBusy`` answer."""


class _TokenError(Exception):
    """The token endpoint refused the assertion or answered with something the adapter cannot use."""


@dataclass(frozen=True)
class ServiceAccountKey:
    """The fields of a Google service-account JSON key the adapter needs."""

    client_email: str
    private_key: str = field(repr=False)
    private_key_id: str | None = None


def load_service_account(path: Path) -> ServiceAccountKey:
    """Read and validate a service-account JSON key file. Raises ``ValueError`` with an operator-readable
    message; callers map that to their own configuration error."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read the Google service-account file {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the Google service-account file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"the Google service-account file {path} is not a JSON object")
    email = data.get("client_email")
    if not isinstance(email, str) or not email:
        raise ValueError(f"the Google service-account file {path} has no 'client_email'")
    key = data.get("private_key")
    if not isinstance(key, str) or not key:
        raise ValueError(f"the Google service-account file {path} has no 'private_key'")
    key_id = data.get("private_key_id")
    return ServiceAccountKey(
        client_email=email, private_key=key, private_key_id=key_id if isinstance(key_id, str) else None
    )


def encode_event_id(idem_key: str) -> str:
    """The Google event id for an idempotency key: lowercase, unpadded base32hex of the first 20 bytes of
    ``sha256(idem_key)`` — 32 characters in ``[a-v0-9]``, always inside the documented 5..1024 length."""
    digest = hashlib.sha256(idem_key.encode("utf-8")).digest()[:20]
    return base64.b32hexencode(digest).decode("ascii").rstrip("=").lower()


# Parsing -----------------------------------------------------------------------------------------------


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_iso(value)
    except (ValueError, OverflowError):
        return None


def parse_freebusy(body: object, calendar_id: str, *, lenient: bool) -> list[tuple[datetime, datetime]]:
    """The busy intervals of ``calendar_id`` from a ``freeBusy`` response body.

    Strict: the response must be a JSON object with a ``calendars`` object holding an entry for
    ``calendar_id`` that carries no ``errors`` and a ``busy`` list of well-formed intervals, else
    ``MalformedResponse`` (or ``MissingCalendar`` for an absent or erroring entry). Lenient: a missing or
    erroring entry, or a body that is not even shaped like a ``freeBusy`` answer, reads as no busy time;
    a ``busy`` list keeps whatever intervals parse and drops the rest.
    """
    if not isinstance(body, dict):
        if lenient:
            return []
        raise MalformedResponse("the response body is not a JSON object")
    calendars = body.get("calendars")
    if not isinstance(calendars, dict):
        if lenient:
            return []
        raise MalformedResponse("the response has no 'calendars' object")
    entry = calendars.get(calendar_id)
    if not isinstance(entry, dict):
        if lenient:
            return []
        raise MissingCalendar(f"{calendar_id!r} is absent from the freeBusy answer")
    errors = entry.get("errors")
    if isinstance(errors, list) and errors:
        if lenient:
            return []
        reason = errors[0].get("reason") if isinstance(errors[0], dict) else None
        raise MissingCalendar(f"{calendar_id!r}: freeBusy reports {reason or 'an error'}")
    busy = entry.get("busy")
    if not isinstance(busy, list):
        if lenient:
            return []
        raise MalformedResponse("'busy' is not a list")
    out: list[tuple[datetime, datetime]] = []
    saw_bad = False
    for item in busy:
        start = end = None
        if isinstance(item, dict):
            start, end = _parse_dt(item.get("start")), _parse_dt(item.get("end"))
        if start is None or end is None or end <= start:
            saw_bad = True
            continue
        out.append((start, end))
    if saw_bad and not lenient:
        raise MalformedResponse("a busy interval is not a valid ISO 8601 interval")
    return out


def _event_status(value: object) -> BookingStatus:
    if value in ("confirmed", "tentative"):
        return "active"
    if value == "cancelled":
        return "cancelled"
    raise MalformedResponse(f"unknown event status {value!r}")


def _event_timestamp(value: object, what: str) -> datetime:
    if not isinstance(value, dict):
        raise MalformedResponse(f"{what} is not an object")
    raw = value.get("dateTime")
    if not isinstance(raw, str):
        raise MalformedResponse(f"{what} has no dateTime")
    parsed = _parse_dt(raw)
    if parsed is None:
        raise MalformedResponse(f"{what} is not an ISO 8601 date-time with an offset: {raw!r}")
    return parsed


def parse_event(data: object, calendar_id: str) -> BookingRecord:
    """One Events resource. ``idem_key`` is the event id itself, since Google has no separate metadata slot
    for it; ``lead_email`` comes only from ``extendedProperties.private.bt_lead_email``, which the adapter
    always writes, whether or not the insert was allowed to carry an attendee."""
    if not isinstance(data, dict):
        raise MalformedResponse("the event is not a JSON object")
    event_id = data.get("id")
    if not isinstance(event_id, str) or not event_id:
        raise MalformedResponse("the event has no id")
    start = _event_timestamp(data.get("start"), f"event {event_id} start")
    end = _event_timestamp(data.get("end"), f"event {event_id} end")
    if end <= start:
        raise MalformedResponse(f"event {event_id} ends before it starts")
    status = _event_status(data.get("status"))
    extended = data.get("extendedProperties")
    private = extended.get("private") if isinstance(extended, dict) else None
    lead_email = private.get(LEAD_EMAIL_PROPERTY) if isinstance(private, dict) else None
    if lead_email is not None and not isinstance(lead_email, str):
        raise MalformedResponse(f"event {event_id} has a non-string {LEAD_EMAIL_PROPERTY}")
    _ = calendar_id  # kept for a uniform call signature; the id is not part of the event body
    return BookingRecord(
        ref=event_id,
        start=start,
        end=end,
        status=status,
        lead_email=lead_email or None,
        idem_key=event_id,
        raw=data,
    )


def _vendor_message(body: object) -> str:
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    return ""


def _oauth_error_detail(response: httpx.Response) -> str:
    """The token endpoint's error shape is ``{"error": "<code>", "error_description": "<text>"}``, not the
    Calendar backend's nested envelope, so it gets its own reader."""
    body = _json_or_none(response)
    message = ""
    if isinstance(body, dict):
        parts = [str(body["error"])] if isinstance(body.get("error"), str) else []
        description = body.get("error_description")
        if isinstance(description, str) and description:
            parts.append(description)
        message = ": ".join(parts)
    text = f"HTTP {response.status_code}" + (f": {message}" if message else "")
    return _clip(text, MAX_STRICT_DETAIL_CHARS)


def _error_reason(body: object) -> str | None:
    """The first ``error.errors[].reason``, e.g. ``forbiddenForServiceAccounts``."""
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    errors = error.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        reason = errors[0].get("reason")
        if isinstance(reason, str):
            return reason
    return None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_or_none(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return None


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise MalformedResponse("the response body is not JSON") from exc


# The adapter --------------------------------------------------------------------------------------------


class GoogleAdapter:
    """``CalendarAdapter`` for one Google Calendar, reached through a service account. See the module
    docstring for the result mapping and the fail-open/fail-closed switch."""

    kind: Literal["calcom", "google"] = "google"
    list_page_size: int = 100

    def __init__(
        self,
        base_url: str,
        token_uri: str,
        calendar_id: str,
        event_key: str,
        hours: Hours,
        service_account: ServiceAccountKey,
        *,
        lenient: bool = False,
        post_retries_on_timeout: int = 0,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        event_title: str = DEFAULT_EVENT_TITLE,
    ) -> None:
        """``client`` lets a caller share a connection pool; the adapter then leaves it open on ``aclose``.
        ``clock`` times the JWT assertion (``iat``/``exp``) and the token-refresh margin, and is also the
        ``now`` slots are computed against; a ``FixedClock`` lets a sandbox test accept the assertion."""
        if post_retries_on_timeout < 0:
            raise ValueError("post_retries_on_timeout must not be negative")
        if not calendar_id:
            raise ValueError("calendar_id must not be empty")
        self.base_url = base_url.rstrip("/")
        self.token_uri = token_uri
        self.calendar_id = calendar_id
        self.event_key = event_key
        self.hours = hours
        self.event_title = event_title
        self.lenient = lenient
        self.post_retries_on_timeout = post_retries_on_timeout
        self.timeout = timeout
        self.clock = clock or SystemClock()
        self._sa = service_account
        self._event_length = timedelta(minutes=hours.slot_minutes)
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(timeout=timeout)
        self._token: str | None = None
        self._token_expiry: datetime | None = None
        #: Set once an insert with attendees has been refused with ``forbiddenForServiceAccounts``; from
        #: then on this adapter instance never sends attendees again.
        self._invite_disabled = False

    def __repr__(self) -> str:
        return (
            f"GoogleAdapter(base_url={self.base_url!r}, calendar_id={self.calendar_id!r}, "
            f"lenient={self.lenient}, post_retries_on_timeout={self.post_retries_on_timeout})"
        )

    async def __aenter__(self) -> GoogleAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # Auth --------------------------------------------------------------------------------------------------

    async def _fetch_token(self) -> None:
        now = self.clock.now()
        iat = int(now.timestamp())
        exp = iat + TOKEN_LIFETIME_S
        claims = {
            "iss": self._sa.client_email,
            "scope": GOOGLE_SCOPE,
            "aud": TOKEN_AUDIENCE,
            "iat": iat,
            "exp": exp,
        }
        headers = {"kid": self._sa.private_key_id} if self._sa.private_key_id else None
        assertion = jwt.encode(claims, self._sa.private_key, algorithm="RS256", headers=headers)
        response = await self._client.post(
            self.token_uri,
            data={"grant_type": GRANT_TYPE, "assertion": assertion},
            headers={"Accept": "application/json", "User-Agent": f"booking-truth/{__version__}"},
            timeout=self.timeout,
        )
        if not response.is_success:
            raise _TokenError(_oauth_error_detail(response))
        try:
            body = response.json()
        except ValueError as exc:
            raise _TokenError("the token endpoint did not return JSON") from exc
        token = body.get("access_token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not token:
            raise _TokenError("the token endpoint response has no access_token")
        raw_expiry = body.get("expires_in") if isinstance(body, dict) else None
        try:
            expires_in = int(raw_expiry) if raw_expiry is not None else TOKEN_LIFETIME_S
        except (TypeError, ValueError):
            expires_in = TOKEN_LIFETIME_S
        self._token = token
        self._token_expiry = now + timedelta(seconds=expires_in)

    async def _access_token(self) -> str:
        now = self.clock.now()
        margin = timedelta(seconds=TOKEN_REFRESH_MARGIN_S)
        if self._token is None or self._token_expiry is None or now >= self._token_expiry - margin:
            await self._fetch_token()
        assert self._token is not None
        return self._token

    # HTTP --------------------------------------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        token = await self._access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": f"booking-truth/{__version__}",
        }
        return await self._client.request(
            method, self.base_url + path, params=params, json=json_body, headers=headers, timeout=self.timeout
        )

    async def _write_request(
        self, method: str, path: str, *, json_body: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Like :meth:`_request`, re-sent up to ``post_retries_on_timeout`` more times after a timeout."""
        attempt = 0
        while True:
            try:
                return await self._request(method, path, json_body=json_body)
            except httpx.TimeoutException:
                if attempt >= self.post_retries_on_timeout:
                    raise
                attempt += 1

    def _events_path(self) -> str:
        return f"/calendar/v3/calendars/{quote(self.calendar_id, safe='')}/events"

    def _event_path(self, ref: str) -> str:
        return f"{self._events_path()}/{quote(ref, safe='')}"

    @staticmethod
    def _exc_detail(exc: Exception) -> str:
        text = str(exc)
        return f"{type(exc).__name__}: {text}" if text else type(exc).__name__

    def _detail(self, response: httpx.Response) -> str:
        if self.lenient:
            return _clip(f"HTTP {response.status_code}: {response.text}", MAX_DETAIL_CHARS)
        message = _vendor_message(_json_or_none(response))
        text = f"HTTP {response.status_code}" + (f": {message}" if message else "")
        return _clip(text, MAX_STRICT_DETAIL_CHARS)

    def _malformed_detail(self, response: httpx.Response, exc: Exception) -> str:
        if self.lenient:
            return _clip(f"HTTP {response.status_code}: {response.text}", MAX_DETAIL_CHARS)
        return _clip(str(exc), MAX_STRICT_DETAIL_CHARS)

    async def _read(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response | Unavailable:
        """A read (or a ``freeBusy`` lookup, also a read as far as error handling goes)."""
        try:
            response = await self._request(method, path, params=params, json_body=json_body)
        except httpx.TimeoutException as exc:
            return Unavailable("timeout", self._exc_detail(exc))
        except httpx.TransportError as exc:
            return Unavailable("error", self._exc_detail(exc))
        except _TokenError as exc:
            return Unavailable("error", str(exc))
        if response.status_code in (404, 410):
            return Unavailable("not_found", self._detail(response))
        if not response.is_success:
            return Unavailable("error", self._detail(response))
        return response

    async def _write(
        self, method: str, path: str, *, json_body: dict[str, Any] | None = None
    ) -> httpx.Response | WriteUnknown:
        try:
            return await self._write_request(method, path, json_body=json_body)
        except httpx.TimeoutException as exc:
            return WriteUnknown("timeout", self._exc_detail(exc))
        except httpx.TransportError as exc:
            return WriteUnknown("server_error", self._exc_detail(exc))
        except _TokenError as exc:
            return WriteUnknown("server_error", str(exc))

    # Slots -------------------------------------------------------------------------------------------------

    def _freebusy_body(self, start: datetime, end: datetime) -> dict[str, Any]:
        return {"timeMin": iso_z(start), "timeMax": iso_z(end), "items": [{"id": self.calendar_id}]}

    async def find_slots(self, start: datetime, end: datetime) -> SlotsResult:
        start, end = ensure_utc(start), ensure_utc(end)
        result = await self._read("POST", "/calendar/v3/freeBusy", json_body=self._freebusy_body(start, end))
        if isinstance(result, Unavailable):
            return result
        try:
            body = _json(result)
        except MalformedResponse as exc:
            if self.lenient:
                busy: list[tuple[datetime, datetime]] = []
            else:
                return Unavailable("malformed", self._malformed_detail(result, exc))
        else:
            try:
                busy = parse_freebusy(body, self.calendar_id, lenient=self.lenient)
            except MalformedResponse as exc:
                return Unavailable("malformed", self._malformed_detail(result, exc))
            except MissingCalendar as exc:
                return Unavailable("missing_calendar", self._malformed_detail(result, exc))
        now = self.clock.now()
        return compute_slots(self.hours, busy, start, end, now)

    async def _check_free(self, start: datetime, end: datetime) -> WriteResult | None:
        """The guarded pre-insert conflict check: ``None`` when the slot is still free, else the write
        outcome to return without ever calling ``events.insert``."""
        result = await self._read("POST", "/calendar/v3/freeBusy", json_body=self._freebusy_body(start, end))
        if isinstance(result, Unavailable):
            reason = "timeout" if result.reason == "timeout" else "server_error"
            return WriteUnknown(reason, result.detail)
        try:
            busy = parse_freebusy(_json(result), self.calendar_id, lenient=False)
        except MalformedResponse as exc:
            return WriteUnknown("malformed", self._malformed_detail(result, exc))
        except MissingCalendar as exc:
            return WriteUnknown("server_error", self._malformed_detail(result, exc))
        if overlaps(start, end, busy):
            return WriteRejected("slot_taken", "the slot is no longer free")
        return None

    # Reads -----------------------------------------------------------------------------------------------

    async def get_booking(self, ref: str) -> ReadResult:
        if not ref:
            return NotFound("empty booking reference")
        result = await self._read("GET", self._event_path(ref))
        if isinstance(result, Unavailable):
            return NotFound(result.detail) if result.reason == "not_found" else result
        try:
            record = parse_event(_json(result), self.calendar_id)
            if record.ref != ref:
                raise MalformedResponse(f"asked for event {ref}, got {record.ref}")
        except MalformedResponse as exc:
            return Unavailable("malformed", self._malformed_detail(result, exc))
        return record

    async def list_bookings(self, *, lead_email: str, start: datetime, end: datetime) -> ListResult:
        """``events.list`` filtered by ``privateExtendedProperty=bt_lead_email=<email>``, cancelled events
        excluded (the default), sorted by start."""
        wanted = lead_email.strip().lower()
        found: dict[str, BookingRecord] = {}
        page_token: str | None = None
        for _ in range(MAX_LIST_PAGES):
            params = {
                "privateExtendedProperty": f"{LEAD_EMAIL_PROPERTY}={lead_email}",
                "timeMin": iso_z(start),
                "timeMax": iso_z(end),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": str(self.list_page_size),
            }
            if page_token:
                params["pageToken"] = page_token
            result = await self._read("GET", self._events_path(), params=params)
            if isinstance(result, Unavailable):
                return result
            try:
                body = _json(result)
                items = body.get("items") if isinstance(body, dict) else None
                if not isinstance(items, list):
                    raise MalformedResponse("'items' is not a list")
                records = [parse_event(item, self.calendar_id) for item in items]
            except MalformedResponse as exc:
                return Unavailable("malformed", self._malformed_detail(result, exc))
            for record in records:
                if record.active and record.lead_email and record.lead_email.strip().lower() == wanted:
                    found.setdefault(record.ref, record)
            next_token = body.get("nextPageToken") if isinstance(body, dict) else None
            if not isinstance(next_token, str) or not next_token:
                return tuple(sorted(found.values(), key=lambda r: (r.start, r.ref)))
            page_token = next_token
        return Unavailable("malformed", f"more than {MAX_LIST_PAGES} pages of events")

    # Writes ----------------------------------------------------------------------------------------------

    def _insert_body(
        self,
        *,
        start: datetime,
        end: datetime,
        lead_name: str,
        lead_email: str,
        lead_zone: str,
        event_id: str | None,
        idem_key: str | None,
        include_attendees: bool,
    ) -> dict[str, Any]:
        private: dict[str, str] = {LEAD_EMAIL_PROPERTY: lead_email, EVENT_KEY_PROPERTY: self.event_key}
        if idem_key is not None:
            private[IDEM_PROPERTY] = idem_key
        body: dict[str, Any] = {
            "summary": f"{self.event_title} with {lead_name}",
            "description": f"Lead: {lead_name} <{lead_email}>\nLead time zone: {lead_zone}",
            "start": {"dateTime": iso_z(start), "timeZone": "UTC"},
            "end": {"dateTime": iso_z(end), "timeZone": "UTC"},
            "extendedProperties": {"private": private},
        }
        if event_id is not None:
            body["id"] = event_id
        if include_attendees:
            body["attendees"] = [{"email": lead_email, "displayName": lead_name}]
        return body

    async def _existing_for_key(
        self, event_id: str | None, *, start: datetime, lead_email: str
    ) -> BookingRecord | None:
        """The booking ``event_id`` already names, if it matches this write's intent (same start, same
        lead) — checked before the conflict pre-check, so a retried attempt at the *same* logical write
        (the id is deterministic from the idempotency key) is adopted straight away instead of being
        mistaken for a third party holding the slot, or sent to ``events.insert`` for a ``409`` it would
        just have to unwind again. The start/lead check guards the astronomically unlikely case of a hash
        collision with someone else's key.
        """
        if event_id is None:
            return None
        found = await self.get_booking(event_id)
        if not isinstance(found, BookingRecord):
            return None
        if found.start != start or (found.lead_email or "").strip().lower() != lead_email.strip().lower():
            return None
        return found

    async def create_booking(
        self, *, start: datetime, lead_email: str, lead_name: str, lead_zone: str, idem_key: str | None
    ) -> WriteResult:
        start = ensure_utc(start)
        end = start + self._event_length
        event_id = encode_event_id(idem_key) if idem_key else None
        existing = await self._existing_for_key(event_id, start=start, lead_email=lead_email)
        if existing is not None:
            return WriteOk(existing)
        if not self.lenient:
            pre = await self._check_free(start, end)
            if pre is not None:
                return pre
        include_attendees = not self._invite_disabled
        body = self._insert_body(
            start=start,
            end=end,
            lead_name=lead_name,
            lead_email=lead_email,
            lead_zone=lead_zone,
            event_id=event_id,
            idem_key=idem_key,
            include_attendees=include_attendees,
        )
        result = await self._write("POST", self._events_path(), json_body=body)
        if isinstance(result, WriteUnknown):
            return result
        response = result
        if (
            include_attendees
            and response.status_code == 403
            and _error_reason(_json_or_none(response)) == "forbiddenForServiceAccounts"
        ):
            self._invite_disabled = True
            body.pop("attendees", None)
            result = await self._write("POST", self._events_path(), json_body=body)
            if isinstance(result, WriteUnknown):
                return result
            response = result
        if response.status_code == 409:
            # Lost a race with our own earlier attempt at this exact key between the check above and this
            # insert: adopt what it produced instead of reporting a write that in fact already succeeded.
            existing = await self._existing_for_key(event_id, start=start, lead_email=lead_email)
            if existing is not None:
                return WriteOk(existing)
            return WriteRejected("duplicate", self._detail(response))
        if not response.is_success:
            if response.status_code >= 500:
                return WriteUnknown("server_error", self._detail(response))
            return WriteRejected("invalid", self._detail(response))
        try:
            record = parse_event(_json(response), self.calendar_id)
            wanted = iso_z(start)
            if iso_z(record.start) != wanted:
                raise MalformedResponse(f"asked to book {wanted}, got {iso_z(record.start)}")
            if not record.active:
                raise MalformedResponse(f"the new booking {record.ref} is not active")
        except MalformedResponse as exc:
            return WriteUnknown("malformed", self._malformed_detail(response, exc))
        return WriteOk(record)

    async def reschedule(
        self, *, ref: str, new_start: datetime, idem_key: str | None, reason: str
    ) -> WriteResult:
        """``events.patch`` on ``start``/``end`` only; everything else the event already has (including
        ``extendedProperties`` and ``attendees``) is left untouched by Google's merge-patch semantics."""
        if not ref:
            return WriteRejected("not_found", "empty booking reference")
        new_start = ensure_utc(new_start)
        end = new_start + self._event_length
        body = {
            "start": {"dateTime": iso_z(new_start), "timeZone": "UTC"},
            "end": {"dateTime": iso_z(end), "timeZone": "UTC"},
        }
        result = await self._write("PATCH", self._event_path(ref), json_body=body)
        if isinstance(result, WriteUnknown):
            return result
        response = result
        if response.status_code in (404, 410):
            return WriteRejected("not_found", self._detail(response))
        if not response.is_success:
            if response.status_code >= 500:
                return WriteUnknown("server_error", self._detail(response))
            return WriteRejected("invalid", self._detail(response))
        try:
            record = parse_event(_json(response), self.calendar_id)
            if record.ref != ref:
                raise MalformedResponse(f"asked to move {ref}, got {record.ref}")
        except MalformedResponse as exc:
            return WriteUnknown("malformed", self._malformed_detail(response, exc))
        if not record.active:
            # Google's patch merges fields onto whatever the event already was; a patch that does not set
            # ``status`` leaves a cancelled (tombstoned) event cancelled instead of reviving it.
            return WriteRejected("not_found", f"booking {ref} no longer exists")
        wanted = iso_z(new_start)
        if iso_z(record.start) != wanted:
            return WriteUnknown("malformed", f"asked to move {ref} to {wanted}, got {iso_z(record.start)}")
        return WriteOk(record, previous_ref=ref)

    async def cancel(self, *, ref: str, reason: str, idem_key: str | None) -> WriteResult:
        """``events.delete``. A successful delete has no body, so the tombstone is read back with
        ``get_booking`` to build the ``BookingRecord``."""
        if not ref:
            return WriteRejected("not_found", "empty booking reference")
        result = await self._write("DELETE", self._event_path(ref))
        if isinstance(result, WriteUnknown):
            return result
        response = result
        if response.status_code == 410:
            return WriteRejected("duplicate", self._detail(response))
        if response.status_code == 404:
            return WriteRejected("not_found", self._detail(response))
        if not response.is_success:
            if response.status_code >= 500:
                return WriteUnknown("server_error", self._detail(response))
            return WriteRejected("invalid", self._detail(response))
        fetched = await self.get_booking(ref)
        if isinstance(fetched, BookingRecord):
            if fetched.active:
                return WriteUnknown("malformed", f"booking {ref} is still active after the cancel")
            return WriteOk(fetched)
        if isinstance(fetched, NotFound):
            return WriteUnknown("malformed", f"booking {ref} could not be found right after the cancel")
        reason_out = (
            fetched.reason if fetched.reason in ("timeout", "server_error", "malformed") else "malformed"
        )
        return WriteUnknown(reason_out, fetched.detail)

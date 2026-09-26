"""Cal.com API v2 adapter.

Headers follow what Cal.com's own clients send: ``cal-api-version: 2024-09-04`` for ``GET /v2/slots`` and
``2024-08-13`` for every bookings call, plus ``Authorization: Bearer <api key>``.

Responses are parsed strictly against the documented shapes. Any deviation is ``Unavailable("malformed")``
for a read and ``WriteUnknown("malformed")`` for a write, because a write whose answer cannot be read may
still have committed. HTTP errors and timeouts map to the sealed result types of
:mod:`booking_truth.calendars.base`:

========================  ===========================  ================================================
Call                      Vendor answer                Result
========================  ===========================  ================================================
``find_slots``            404                          ``Unavailable("not_found")``
                          other non-2xx, network       ``Unavailable("error")``
                          timeout                      ``Unavailable("timeout")``
create / reschedule       400 slot taken, booker       ``WriteRejected("slot_taken")``
                          limit, 409 conflict
reschedule / cancel       404                          ``WriteRejected("not_found")``
                          already moved / cancelled    ``WriteRejected("duplicate")``
any write                 other 4xx                    ``WriteRejected("invalid")``
                          5xx, network                 ``WriteUnknown("server_error")``
                          timeout                      ``WriteUnknown("timeout")``
``get_booking``           404                          ``NotFound``
``list_bookings``         404                          ``Unavailable("not_found")``
========================  ===========================  ================================================

Two switches reproduce the naive baseline:

- ``lenient=True`` keeps the raw response text (status line and body) in ``detail``, so a tool layer can hand
  it to the model verbatim, and reads a ``200`` slots answer of unexpected shape by scraping every ISO 8601
  timestamp in it into ``Slots``: a failed lookup turns into offered times, or into an empty calendar. This
  is the fail-open parsing the ``fail_closed`` guard exists to prevent.
- ``post_retries_on_timeout=N`` re-sends a POST up to ``N`` more times after a timeout, the way a plain HTTP
  client retry does, even though the first attempt may have committed.

Cal.com has no idempotency header. A create carries the caller's key as ``metadata.bt_idem``; Cal.com copies a
booking's metadata to the booking a reschedule creates, and the reschedule and cancel bodies take no metadata,
so their keys stay with the caller.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from types import TracebackType
from typing import Any, Final, Literal
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from booking_truth import __version__
from booking_truth.calendars.base import (
    BookingRecord,
    BookingStatus,
    ListResult,
    NotFound,
    ReadResult,
    Slot,
    Slots,
    SlotsResult,
    Unavailable,
    WriteOk,
    WriteRejected,
    WriteResult,
    WriteUnknown,
)
from booking_truth.timeutil import ensure_utc, iso_z, parse_iso

VERSION_HEADER: Final = "cal-api-version"
SLOTS_API_VERSION: Final = "2024-09-04"
BOOKINGS_API_VERSION: Final = "2024-08-13"
#: The metadata key that carries the caller's idempotency key on a Cal.com booking.
IDEM_METADATA_KEY: Final = "bt_idem"
DEFAULT_TIMEOUT: Final = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
#: ``GET /v2/bookings`` under 2024-08-13 takes several statuses; these two cover every booking that is not
#: cancelled or rejected (``upcoming`` includes unconfirmed ones).
LIST_STATUSES: Final = "upcoming,past"
MAX_LIST_PAGES: Final = 50
MAX_DETAIL_CHARS: Final = 2000
MAX_STRICT_DETAIL_CHARS: Final = 300

_ACTIVE_STATUSES: Final = frozenset({"accepted", "pending", "awaiting_host"})
_CANCELLED_STATUSES: Final = frozenset({"cancelled", "rejected"})
# Cal.com's texts for a create or reschedule that lost its slot (lower case, matched as substrings):
# "User either already has booking at this time or is not available", the team variant, the booker limit
# "... the maximum number of active bookings has been reached", the raw 409 "booking_conflict_error" of a
# race, and "No more seats left at this seated booking."
_SLOT_TAKEN_MARKERS: Final = (
    "already has booking",
    "not available",
    "maximum number of active bookings",
    "booking_conflict_error",
    "no more seats left",
)
_ALREADY_MOVED_MARKER: Final = "rescheduled already"
_ALREADY_CANCELLED_MARKER: Final = "cancelled already"
_DATE_KEY: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_DATETIME: Final = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+-]\d{2}:?\d{2})?", re.ASCII
)

WriteKind = Literal["create", "reschedule", "cancel"]


class MalformedResponse(ValueError):
    """A vendor response that does not have the documented shape."""


# Strict parsing ---------------------------------------------------------------------------------------------


def _timestamp(value: object, what: str) -> datetime:
    """An ISO 8601 date-time with an offset (``Z`` or ``±hh:mm``), as UTC."""
    if not isinstance(value, str) or not _ISO_DATETIME.fullmatch(value):
        raise MalformedResponse(f"{what} is not an ISO 8601 date-time: {value!r}")
    try:
        return parse_iso(value)
    except (ValueError, OverflowError) as exc:
        raise MalformedResponse(f"{what} is not an ISO 8601 date-time with an offset: {value!r}") from exc


def success_data(body: object) -> Any:
    """``data`` of Cal.com's success envelope ``{"status": "success", "data": ...}``."""
    if not isinstance(body, dict):
        raise MalformedResponse("the response body is not a JSON object")
    if body.get("status") != "success":
        raise MalformedResponse(f"the response status is {body.get('status')!r}, not 'success'")
    if "data" not in body:
        raise MalformedResponse("the response has no 'data'")
    return body["data"]


def parse_slots(body: object, start: datetime, end: datetime) -> Slots:
    """A ``GET /v2/slots`` body (``format=range``): ``data`` maps ``YYYY-MM-DD`` to ``[{start, end}, ...]``.

    Unknown keys inside a slot object are ignored (seated event types add some). Slots outside
    ``start <= slot.start < end`` are dropped; Cal.com treats the ``end`` query parameter as inclusive.
    """
    data = success_data(body)
    if not isinstance(data, dict):
        raise MalformedResponse("'data' is not an object keyed by date")
    start, end = ensure_utc(start), ensure_utc(end)
    found: dict[datetime, Slot] = {}
    for day, items in data.items():
        if not _DATE_KEY.fullmatch(day):
            raise MalformedResponse(f"'data' key {day!r} is not a YYYY-MM-DD date")
        try:
            date.fromisoformat(day)
        except ValueError as exc:
            raise MalformedResponse(f"'data' key {day!r} is not a valid date") from exc
        if not isinstance(items, list):
            raise MalformedResponse(f"slots of {day} are not a list")
        for item in items:
            if not isinstance(item, dict):
                raise MalformedResponse(f"a slot of {day} is not an object")
            slot_start = _timestamp(item.get("start"), "slot start")
            slot_end = _timestamp(item.get("end"), "slot end")
            if slot_end <= slot_start:
                raise MalformedResponse(f"slot {item!r} ends before it starts")
            if start <= slot_start < end:
                found[slot_start] = Slot(slot_start, slot_end)
    return Slots(tuple(found[key] for key in sorted(found)))


def scrape_slots(text: str, length: timedelta) -> Slots:
    """Every ISO 8601 date-time anywhere in ``text``, as a slot of ``length`` (naive, fail-open parsing).

    A timestamp without an offset is read as UTC.
    """
    starts: set[datetime] = set()
    for match in _ISO_DATETIME.finditer(text):
        try:
            parsed = datetime.fromisoformat(match.group(0))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        starts.add(parsed.astimezone(UTC))
    return Slots(tuple(Slot(start, start + length) for start in sorted(starts)))


def _booking_status(value: object) -> BookingStatus:
    if value in _ACTIVE_STATUSES:
        return "active"
    if value in _CANCELLED_STATUSES:
        return "cancelled"
    raise MalformedResponse(f"unknown booking status {value!r}")


def parse_booking(data: object) -> BookingRecord:
    """One ``BookingOutput_2024_08_13`` object. Unknown keys are ignored; a recurring booking's list of
    occurrences is not a booking and is rejected."""
    if not isinstance(data, dict):
        raise MalformedResponse("the booking is not a JSON object")
    uid = data.get("uid")
    if not isinstance(uid, str) or not uid:
        raise MalformedResponse("the booking has no uid")
    start = _timestamp(data.get("start"), "booking start")
    end = _timestamp(data.get("end"), "booking end")
    if end <= start:
        raise MalformedResponse(f"booking {uid} ends before it starts")
    status = _booking_status(data.get("status"))
    attendees = data.get("attendees")
    if not isinstance(attendees, list) or not all(isinstance(a, dict) for a in attendees):
        raise MalformedResponse(f"booking {uid} has no attendee list")
    emails: list[str] = []
    for attendee in attendees:
        email = attendee.get("email")
        if email is not None and not isinstance(email, str):
            raise MalformedResponse(f"an attendee email of booking {uid} is not a string")
        if email:
            emails.append(email)
    metadata = data.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise MalformedResponse(f"the metadata of booking {uid} is not an object")
    idem = metadata.get(IDEM_METADATA_KEY)
    return BookingRecord(
        ref=uid,
        start=start,
        end=end,
        status=status,
        lead_email=emails[0] if emails else None,
        idem_key=idem if isinstance(idem, str) and idem else None,
        raw=data,
    )


def parse_booking_page(body: object) -> tuple[list[BookingRecord], bool]:
    """A ``GET /v2/bookings`` page under 2024-08-13: the bookings and ``pagination.hasNextPage``."""
    data = success_data(body)
    if not isinstance(data, list):
        raise MalformedResponse("'data' is not a list of bookings")
    assert isinstance(body, dict)
    pagination = body.get("pagination")
    if not isinstance(pagination, dict) or not isinstance(pagination.get("hasNextPage"), bool):
        raise MalformedResponse("the page has no pagination.hasNextPage")
    return [parse_booking(item) for item in data], pagination["hasNextPage"]


def vendor_message(body: object) -> str:
    """The human message of a Cal.com error body: the standard envelope (``error.message``), NestJS's raw body
    (``message``) or the global validation pipe (``details.errors[].constraints``)."""
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        details = error.get("details")
        if isinstance(details, dict) and isinstance(details.get("errors"), list):
            texts = [
                str(text)
                for item in details["errors"]
                if isinstance(item, dict) and isinstance(item.get("constraints"), dict)
                for text in item["constraints"].values()
            ]
            if texts:
                return "; ".join(texts)
        if isinstance(message, str):
            return message
    message = body.get("message")
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "; ".join(str(m) for m in message)
    return ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_or_none(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return None


def _json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError as exc:
        raise MalformedResponse("the response body is not JSON") from exc


def _has_attendee(record: BookingRecord, email: str) -> bool:
    wanted = email.strip().lower()
    return any(
        isinstance(a, dict) and isinstance(a.get("email"), str) and a["email"].strip().lower() == wanted
        for a in record.raw.get("attendees", [])
    )


# The adapter ------------------------------------------------------------------------------------------------


class CalcomAdapter:
    """``CalendarAdapter`` for one Cal.com event type. See the module docstring for the result mapping."""

    kind: Literal["calcom", "google"] = "calcom"
    #: ``take`` of each ``GET /v2/bookings`` page (Cal.com allows 1..250).
    list_page_size: int = 100

    def __init__(
        self,
        base_url: str,
        api_key: str,
        event_type_id: int,
        event_key: str,
        host_zone: str,
        *,
        lenient: bool = False,
        post_retries_on_timeout: int = 0,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
        slot_minutes: int = 30,
    ) -> None:
        """``client`` lets a caller share a connection pool; the adapter then leaves it open on ``aclose``.
        ``timeout`` applies to every request either way. ``slot_minutes`` is the length given to slots that
        lenient parsing scrapes from a malformed answer."""
        if post_retries_on_timeout < 0:
            raise ValueError("post_retries_on_timeout must not be negative")
        if slot_minutes <= 0:
            raise ValueError("slot_minutes must be positive")
        try:
            ZoneInfo(host_zone)  # an unknown zone fails here, not on the first request
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA time zone {host_zone!r}") from exc
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.event_type_id = event_type_id
        self.event_key = event_key
        self.host_zone = host_zone
        self.lenient = lenient
        self.post_retries_on_timeout = post_retries_on_timeout
        self.timeout = timeout
        self.slot_minutes = slot_minutes
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(timeout=timeout)

    def __repr__(self) -> str:
        return (
            f"CalcomAdapter(base_url={self.base_url!r}, event_type_id={self.event_type_id}, "
            f"lenient={self.lenient}, post_retries_on_timeout={self.post_retries_on_timeout})"
        )

    async def __aenter__(self) -> CalcomAdapter:
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

    # HTTP --------------------------------------------------------------------------------------------------

    def _headers(self, version: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            VERSION_HEADER: version,
            "Accept": "application/json",
            "User-Agent": f"booking-truth/{__version__}",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        version: str,
        params: Mapping[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        return await self._client.request(
            method,
            self.base_url + path,
            params=params,
            json=body,
            headers=self._headers(version),
            timeout=self.timeout,
        )

    async def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """POST with the configured number of re-sends after a timeout; the last timeout propagates."""
        attempt = 0
        while True:
            try:
                return await self._request("POST", path, version=BOOKINGS_API_VERSION, body=body)
            except httpx.TimeoutException:
                if attempt >= self.post_retries_on_timeout:
                    raise
                attempt += 1

    # Details -----------------------------------------------------------------------------------------------

    def _http_detail(self, response: httpx.Response) -> str:
        if self.lenient:
            return _clip(f"HTTP {response.status_code}: {response.text}", MAX_DETAIL_CHARS)
        message = vendor_message(_json_or_none(response))
        text = f"HTTP {response.status_code}" + (f": {message}" if message else "")
        return _clip(text, MAX_STRICT_DETAIL_CHARS)

    def _malformed_detail(self, response: httpx.Response, exc: MalformedResponse) -> str:
        if self.lenient:
            return _clip(f"HTTP {response.status_code}: {response.text}", MAX_DETAIL_CHARS)
        return _clip(str(exc), MAX_STRICT_DETAIL_CHARS)

    @staticmethod
    def _exc_detail(exc: Exception) -> str:
        text = str(exc)
        return f"{type(exc).__name__}: {text}" if text else type(exc).__name__

    # Reads -------------------------------------------------------------------------------------------------

    async def _read(
        self, path: str, *, version: str, params: Mapping[str, str] | None = None
    ) -> httpx.Response | Unavailable:
        """A GET, or ``Unavailable`` for a timeout, a network failure, a 404 or another non-2xx answer."""
        try:
            response = await self._request("GET", path, version=version, params=params)
        except httpx.TimeoutException as exc:
            return Unavailable("timeout", self._exc_detail(exc))
        except httpx.TransportError as exc:
            return Unavailable("error", self._exc_detail(exc))
        if response.status_code == 404:
            return Unavailable("not_found", self._http_detail(response))
        if not response.is_success:
            return Unavailable("error", self._http_detail(response))
        return response

    async def find_slots(self, start: datetime, end: datetime) -> SlotsResult:
        params = {
            "eventTypeId": str(self.event_type_id),
            "start": iso_z(start),
            "end": iso_z(end),
            "timeZone": self.host_zone,
            "format": "range",
        }
        response = await self._read("/v2/slots", version=SLOTS_API_VERSION, params=params)
        if isinstance(response, Unavailable):
            return response
        try:
            return parse_slots(_json(response), start, end)
        except MalformedResponse as exc:
            if self.lenient:
                return scrape_slots(response.text, timedelta(minutes=self.slot_minutes))
            return Unavailable("malformed", self._malformed_detail(response, exc))

    async def get_booking(self, ref: str) -> ReadResult:
        if not ref:
            return NotFound("empty booking reference")
        response = await self._read(f"/v2/bookings/{quote(ref, safe='')}", version=BOOKINGS_API_VERSION)
        if isinstance(response, Unavailable):
            return NotFound(response.detail) if response.reason == "not_found" else response
        try:
            record = parse_booking(success_data(_json(response)))
            if record.ref != ref:
                raise MalformedResponse(f"asked for booking {ref}, got {record.ref}")
        except MalformedResponse as exc:
            return Unavailable("malformed", self._malformed_detail(response, exc))
        return record

    async def list_bookings(self, *, lead_email: str, start: datetime, end: datetime) -> ListResult:
        """Bookings of this event type with ``lead_email`` as an attendee that lie inside ``[start, end]``
        (``afterStart``/``beforeEnd``), cancelled ones excluded, sorted by start. Follows ``take``/``skip``
        pagination; a page that fails to parse makes the whole list ``Unavailable("malformed")``."""
        found: dict[str, BookingRecord] = {}
        skip = 0
        for _ in range(MAX_LIST_PAGES):
            params = {
                "attendeeEmail": lead_email,
                "afterStart": iso_z(start),
                "beforeEnd": iso_z(end),
                "eventTypeId": str(self.event_type_id),
                "status": LIST_STATUSES,
                "sortStart": "asc",
                "take": str(self.list_page_size),
                "skip": str(skip),
            }
            response = await self._read("/v2/bookings", version=BOOKINGS_API_VERSION, params=params)
            if isinstance(response, Unavailable):
                return response
            try:
                records, has_next = parse_booking_page(_json(response))
                if has_next and not records:
                    raise MalformedResponse("pagination says more pages follow an empty page")
            except MalformedResponse as exc:
                return Unavailable("malformed", self._malformed_detail(response, exc))
            for record in records:
                if record.active and _has_attendee(record, lead_email):
                    found.setdefault(record.ref, record)
            if not has_next:
                return tuple(sorted(found.values(), key=lambda r: (r.start, r.ref)))
            skip += len(records)
        return Unavailable("malformed", f"more than {MAX_LIST_PAGES} pages of bookings")

    # Writes ------------------------------------------------------------------------------------------------

    async def _write(
        self,
        kind: WriteKind,
        path: str,
        body: dict[str, Any],
        check: Callable[[BookingRecord], None],
        previous_ref: str | None = None,
    ) -> WriteResult:
        try:
            response = await self._post(path, body)
        except httpx.TimeoutException as exc:
            return WriteUnknown("timeout", self._exc_detail(exc))
        except httpx.TransportError as exc:
            return WriteUnknown("server_error", self._exc_detail(exc))
        if response.is_success:
            try:
                record = parse_booking(success_data(_json(response)))
                check(record)
            except MalformedResponse as exc:
                return WriteUnknown("malformed", self._malformed_detail(response, exc))
            return WriteOk(record, previous_ref)
        if 400 <= response.status_code < 500:
            reason = _rejection(kind, response.status_code, vendor_message(_json_or_none(response)))
            return WriteRejected(reason, self._http_detail(response))
        return WriteUnknown("server_error", self._http_detail(response))

    async def create_booking(
        self, *, start: datetime, lead_email: str, lead_name: str, lead_zone: str, idem_key: str | None
    ) -> WriteResult:
        """``POST /v2/bookings``. ``lengthInMinutes`` is omitted, so the event type's length applies."""
        body: dict[str, Any] = {
            "start": iso_z(start),
            "eventTypeId": self.event_type_id,
            "attendee": {"name": lead_name, "email": lead_email, "timeZone": lead_zone},
        }
        if idem_key is not None:
            body["metadata"] = {IDEM_METADATA_KEY: idem_key}
        wanted = iso_z(start)

        def check(record: BookingRecord) -> None:
            if iso_z(record.start) != wanted:
                raise MalformedResponse(f"asked to book {wanted}, got {iso_z(record.start)}")
            if not record.active:
                raise MalformedResponse(f"the new booking {record.ref} is not active")
            if not _has_attendee(record, lead_email):
                raise MalformedResponse(f"the new booking {record.ref} does not list the lead as an attendee")

        return await self._write("create", "/v2/bookings", body, check)

    async def reschedule(
        self, *, ref: str, new_start: datetime, idem_key: str | None, reason: str
    ) -> WriteResult:
        """``POST /v2/bookings/{uid}/reschedule``.

        Cal.com cancels the old booking and creates a new one with a new uid; ``WriteOk.booking`` is the new
        booking and ``previous_ref`` the old uid. The body has no metadata field, so ``idem_key`` is not sent
        (the new booking keeps the old one's metadata).
        """
        if not ref:
            return WriteRejected("not_found", "empty booking reference")
        body: dict[str, Any] = {"start": iso_z(new_start)}
        if reason:
            body["reschedulingReason"] = reason
        wanted = iso_z(new_start)

        def check(record: BookingRecord) -> None:
            if iso_z(record.start) != wanted:
                raise MalformedResponse(f"asked to move {ref} to {wanted}, got {iso_z(record.start)}")
            if not record.active:
                raise MalformedResponse(f"the rescheduled booking {record.ref} is not active")
            moved_from = record.raw.get("rescheduledFromUid")
            if moved_from is not None and moved_from != ref:
                raise MalformedResponse(f"asked to move {ref}, the new booking was moved from {moved_from}")

        path = f"/v2/bookings/{quote(ref, safe='')}/reschedule"
        return await self._write("reschedule", path, body, check, previous_ref=ref)

    async def cancel(self, *, ref: str, reason: str, idem_key: str | None) -> WriteResult:
        """``POST /v2/bookings/{uid}/cancel``; ``WriteOk.booking`` is the cancelled booking.

        ``idem_key`` is not sent (the body has no metadata field).
        """
        if not ref:
            return WriteRejected("not_found", "empty booking reference")
        body: dict[str, Any] = {"cancellationReason": reason} if reason else {}

        def check(record: BookingRecord) -> None:
            if record.ref != ref:
                raise MalformedResponse(f"asked to cancel {ref}, got {record.ref}")
            if record.active:
                raise MalformedResponse(f"booking {ref} is still active after the cancel")

        return await self._write("cancel", f"/v2/bookings/{quote(ref, safe='')}/cancel", body, check)


def _rejection(kind: WriteKind, status: int, message: str) -> str:
    """The ``WriteRejected`` reason for a 4xx answer to a write."""
    text = message.lower()
    if kind != "create" and status == 404:
        return "not_found"
    if kind == "reschedule" and _ALREADY_MOVED_MARKER in text:
        return "duplicate"
    if kind == "cancel" and _ALREADY_CANCELLED_MARKER in text:
        return "duplicate"
    if kind != "cancel" and any(marker in text for marker in _SLOT_TAKEN_MARKERS):
        return "slot_taken"
    return "invalid"

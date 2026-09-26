"""HubSpot CRM API v3 adapter: contact upsert and meeting create/update, against
``https://api.hubapi.com`` or the sandbox mirror under the same ``/crm/v3`` paths.

Auth is a bearer token, either a legacy private-app token or a service key (``docs/research`` records both
use the same ``pat-<region>-...`` shape and the same limits): ``Authorization: Bearer <token>``.

Contact upsert (``upsert_contact``) searches by email first (``POST .../contacts/search``, operator ``EQ``,
case-insensitive on HubSpot); a hit is updated in place. A miss creates the contact; a ``409`` there means a
contact with that email was created between the search and the create (or the search missed it during
HubSpot's own eventual consistency window), so the conflict body's ``Contact already exists. Existing ID:
<id>`` is parsed and that id is updated instead — the same recovery real integrations use for this
undocumented response. A ``409`` with no id in the message falls back to searching once more.

Meeting create (``POST .../meetings``) writes ``hs_timestamp``, the title, the start and end times and the
outcome, with an inline association to the contact (``associationCategory: HUBSPOT_DEFINED``,
``associationTypeId: 200``, meeting-to-contact). Meeting update (``PATCH .../meetings/{id}``) always rewrites
the same four properties: a reschedule moves the times and sets ``RESCHEDULED``, a cancel keeps the original
times and sets ``CANCELED`` (the current object state, not a diff — cheaper than tracking what changed and
just as correct against HubSpot's PATCH semantics).

Every write is parsed strictly: a ``2xx`` whose body has no object id is ``CrmError("malformed", ...)``, never
silently treated as success. Vendor failures map to ``CrmError.reason`` (``timeout``, ``server_error``,
``rate_limited``, ``auth``, ``not_found``, ``invalid``, ``malformed``) and ``retryable`` (timeouts, ``429``
and ``5xx``; not the others), which the outbox worker uses to decide whether another attempt can help.
"""

from __future__ import annotations

import re
from datetime import datetime
from types import TracebackType
from typing import Any, Final

import httpx

from booking_truth import __version__
from booking_truth.config import ConfigError, Settings
from booking_truth.crm.base import (
    ContactPayload,
    CrmError,
    CrmOk,
    CrmResult,
    MeetingPayload,
    MeetingUpdatePayload,
)
from booking_truth.timeutil import iso_z

PREFIX: Final = "/crm/v3/objects"
CONTACTS_PATH: Final = f"{PREFIX}/contacts"
MEETINGS_PATH: Final = f"{PREFIX}/meetings"
#: "Meeting to contact" (the associations guide and the contact object-definition's ``inverseId`` agree).
MEETING_TO_CONTACT_ASSOCIATION: Final = 200
DEFAULT_TIMEOUT: Final = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
MAX_DETAIL_CHARS: Final = 300
_CONFLICT_ID: Final = re.compile(r"Existing ID:\s*(\d+)")


def _clip(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _exc_detail(exc: Exception) -> str:
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _json_or_none(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _object_id(body: Any) -> str | None:
    """The ``id`` of a ``SimplePublicObject`` body, or ``None`` when the shape is not that."""
    if not isinstance(body, dict):
        return None
    value = body.get("id")
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    text = str(value)
    return text or None


def _message(response: httpx.Response) -> str:
    """The human ``message`` of a HubSpot error envelope, or the raw response text."""
    body = _json_or_none(response)
    message = body.get("message") if isinstance(body, dict) else None
    return message if isinstance(message, str) else response.text


def conflict_contact_id(message: str) -> str | None:
    """The existing contact id in ``Contact already exists. Existing ID: <id>``, if the message has one."""
    match = _CONFLICT_ID.search(message)
    return match.group(1) if match else None


class HubSpotAdapter:
    """``CrmAdapter`` for HubSpot CRM v3 contacts and meetings. See the module docstring for the mapping
    from vendor responses to :class:`~booking_truth.crm.base.CrmResult`."""

    kind = "hubspot"

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """``client`` lets a caller share a connection pool; the adapter then leaves it open on ``aclose``.
        ``timeout`` applies to every request either way."""
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(timeout=timeout)

    def __repr__(self) -> str:
        return f"HubSpotAdapter(base_url={self.base_url!r})"

    async def __aenter__(self) -> HubSpotAdapter:
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

    # HTTP ---------------------------------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"booking-truth/{__version__}",
        }

    async def _call(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> httpx.Response | CrmError:
        """A request, or the ``CrmError`` of a timeout or a transport failure (never for a non-2xx status,
        which the caller classifies with :meth:`_error_for`, since only it knows what the call meant)."""
        try:
            return await self._client.request(
                method, self.base_url + path, json=body, headers=self._headers(), timeout=self.timeout
            )
        except httpx.TimeoutException as exc:
            return CrmError("timeout", _exc_detail(exc), retryable=True)
        except httpx.TransportError as exc:
            return CrmError("server_error", _exc_detail(exc), retryable=True)

    @staticmethod
    def _error_for(response: httpx.Response) -> CrmError:
        """The ``CrmError`` of a non-2xx response, by status: auth (401/403), not_found (404), rate_limited
        (429, retryable), server_error (5xx, retryable) or invalid (any other 4xx)."""
        status = response.status_code
        message = _clip(_message(response))
        if status in (401, 403):
            return CrmError("auth", message, retryable=False)
        if status == 404:
            return CrmError("not_found", message, retryable=False)
        if status == 429:
            return CrmError("rate_limited", message, retryable=True)
        if status >= 500:
            return CrmError("server_error", f"HTTP {status}: {message}", retryable=True)
        return CrmError("invalid", f"HTTP {status}: {message}", retryable=False)

    # Contacts -----------------------------------------------------------------------------------------------

    @staticmethod
    def _contact_properties(payload: ContactPayload) -> dict[str, str]:
        first, last = payload.first_last
        props = {"email": payload.email}
        if first:
            props["firstname"] = first
        if last:
            props["lastname"] = last
        return props

    async def _search_contact_id(self, email: str) -> str | CrmError | None:
        """The contact id of ``email`` (case-insensitive ``EQ``), ``None`` when none matches."""
        body = {
            "filterGroups": [{"filters": [{"propertyName": "email", "operator": "EQ", "value": email}]}],
            "properties": ["email"],
            "limit": 1,
        }
        response = await self._call("POST", f"{CONTACTS_PATH}/search", body)
        if isinstance(response, CrmError):
            return response
        if not response.is_success:
            return self._error_for(response)
        results = (_json_or_none(response) or {}).get("results")
        if not isinstance(results, list):
            return CrmError("malformed", "the search response has no 'results' list", retryable=True)
        if not results:
            return None
        found = _object_id(results[0])
        if found is None:
            return CrmError("malformed", "a search result has no id", retryable=True)
        return found

    async def _update_contact(self, contact_id: str, payload: ContactPayload) -> CrmResult:
        response = await self._call(
            "PATCH", f"{CONTACTS_PATH}/{contact_id}", {"properties": self._contact_properties(payload)}
        )
        if isinstance(response, CrmError):
            return response
        if response.is_success:
            return CrmOk(_object_id(_json_or_none(response)) or contact_id)
        return self._error_for(response)

    async def _create_contact(self, payload: ContactPayload) -> str | CrmError:
        """The new contact's id, or a ``CrmError`` — ``reason="conflict"`` carries the vendor message so the
        caller can recover the existing id from it (see the module docstring); it is never returned to a
        caller outside this module."""
        response = await self._call("POST", CONTACTS_PATH, {"properties": self._contact_properties(payload)})
        if isinstance(response, CrmError):
            return response
        if response.status_code == 409:
            return CrmError("conflict", _clip(_message(response)), retryable=True)
        if response.is_success:
            found = _object_id(_json_or_none(response))
            if found is None:
                return CrmError("malformed", "the created contact has no id", retryable=True)
            return found
        return self._error_for(response)

    async def upsert_contact(self, payload: ContactPayload) -> CrmResult:
        found = await self._search_contact_id(payload.email)
        if isinstance(found, CrmError):
            return found
        if found is not None:
            return await self._update_contact(found, payload)
        created = await self._create_contact(payload)
        if isinstance(created, str):
            return CrmOk(created)
        if created.reason != "conflict":
            return created
        existing_id = conflict_contact_id(created.detail)
        if existing_id is None:
            again = await self._search_contact_id(payload.email)
            existing_id = again if isinstance(again, str) else None
            if existing_id is None:
                return again if isinstance(again, CrmError) else created
        return await self._update_contact(existing_id, payload)

    # Meetings -----------------------------------------------------------------------------------------------

    @staticmethod
    def _meeting_properties(
        *,
        start_utc: datetime,
        end_utc: datetime,
        outcome: str,
        title: str | None = None,
        body: str | None = None,
    ) -> dict[str, str]:
        props = {
            "hs_timestamp": iso_z(start_utc),
            "hs_meeting_start_time": iso_z(start_utc),
            "hs_meeting_end_time": iso_z(end_utc),
            "hs_meeting_outcome": outcome,
        }
        if title:
            props["hs_meeting_title"] = title
        if body:
            props["hs_meeting_body"] = body
        return props

    async def create_meeting(self, payload: MeetingPayload) -> CrmResult:
        properties = self._meeting_properties(
            start_utc=payload.start_utc,
            end_utc=payload.end_utc,
            outcome=payload.outcome,
            title=payload.title,
            body=payload.body,
        )
        association = {
            "to": {"id": payload.contact_id},
            "types": [
                {
                    "associationCategory": "HUBSPOT_DEFINED",
                    "associationTypeId": MEETING_TO_CONTACT_ASSOCIATION,
                }
            ],
        }
        response = await self._call(
            "POST", MEETINGS_PATH, {"properties": properties, "associations": [association]}
        )
        if isinstance(response, CrmError):
            return response
        if response.is_success:
            found = _object_id(_json_or_none(response))
            if found is None:
                return CrmError("malformed", "the created meeting has no id", retryable=True)
            return CrmOk(found)
        return self._error_for(response)

    async def update_meeting(self, payload: MeetingUpdatePayload) -> CrmResult:
        properties = self._meeting_properties(
            start_utc=payload.start_utc, end_utc=payload.end_utc, outcome=payload.outcome
        )
        response = await self._call(
            "PATCH", f"{MEETINGS_PATH}/{payload.meeting_id}", {"properties": properties}
        )
        if isinstance(response, CrmError):
            return response
        if response.is_success:
            found = _object_id(_json_or_none(response))
            if found is None:
                return CrmError("malformed", "the updated meeting has no id", retryable=True)
            return CrmOk(found)
        return self._error_for(response)


def build_hubspot(settings: Settings) -> HubSpotAdapter:
    """The adapter for ``BT_CRM=hubspot``, from ``BT_HUBSPOT_TOKEN`` and ``BT_HUBSPOT_BASE_URL``."""
    if settings.hubspot_token is None:
        raise ConfigError("BT_HUBSPOT_TOKEN is required when BT_CRM=hubspot")
    return HubSpotAdapter(settings.hubspot_base_url, settings.hubspot_token.get_secret_value())

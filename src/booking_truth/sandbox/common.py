"""Plumbing shared by the sandbox's vendor APIs: auth, the request log, fault application and responses.

A vendor API (Cal.com, Google Calendar, Google's OAuth token endpoint, HubSpot) is a :class:`VendorApi`
subclass with its own router and its own renderings of auth failures, faults and response bytes. Every
vendor route hands its request to :func:`run_call`, which runs the same pipeline for all of them:

1. read the query and body (by default strict JSON: ``NaN`` and ``Infinity`` make the body invalid; a
   vendor can parse other media types, e.g. the token endpoint's form bodies);
2. run the route's ``route`` check, the vendor's own routing that happens before any auth guard, e.g. a
   ``cal-api-version`` that has no such route (logged, never counted by fault rules);
3. authorize the call, by default with the sandbox bearer token (a failure is answered in the vendor's
   shape and logged), then run the route's ``precheck`` (logged, never counted by fault rules);
4. append the request to the log and ask the fault engine whether a rule fires;
5. apply the fault, or run the handler under ``state.lock``;
6. complete the log entry with the final status and the full JSON response.

Handlers are synchronous functions of the in-memory state and always run under ``state.lock``. Sleeps for
``slow``, ``timeout`` and ``commit_then_timeout`` happen outside the lock, so a hanging request never blocks
other calls. A call that was waiting when ``POST /_control/reset`` ran never touches the fresh state: its
handler is skipped and it is answered as a gateway timeout. A handler that raises is answered with the
vendor's 500 and still completes its log entry (status 500, fault ``null``), so a sandbox bug is visible and
never leaves an entry pending. Calls to unknown routes under a vendor prefix are logged too, under the group
``unrouted``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, assert_never

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from booking_truth.sandbox.faults import FaultRule
from booking_truth.sandbox.state import LogEntry, SandboxState

JSON_UTF8 = "application/json; charset=utf-8"
#: Log group of a call to an unknown route under a vendor prefix. No fault rule can target it.
UNROUTED_GROUP = "unrouted"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Outcome:
    """A status code and a JSON body (``None`` for an empty body), before it becomes an HTTP response."""

    status: int
    body: Any
    #: Extra response headers, e.g. HubSpot's ``Location`` on a create.
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SetupBooking:
    """A lead's pre-existing booking created by the harness through ``POST /_control/bookings``."""

    lead_email: str
    lead_name: str
    start: datetime
    title: str | None = None
    lead_timezone: str | None = None
    #: Google only: a client-supplied event id and extra private extended properties.
    event_id: str | None = None
    extended_properties: dict[str, str] | None = None


@dataclass
class Call:
    """One vendor request as handlers see it."""

    request: Request
    state: SandboxState
    group: str
    query: dict[str, Any]
    body: Any
    body_invalid: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    #: ``state.generation`` when the call was logged; a reset after that makes the call stale.
    generation: int = 0

    @property
    def path(self) -> str:
        return self.request.url.path

    @property
    def url(self) -> str:
        """Path plus the query string exactly as received (what Node reports as ``request.url``)."""
        return request_url(self.request)

    @property
    def method(self) -> str:
        return self.request.method

    def param(self, name: str) -> str | None:
        return self.request.query_params.get(name)

    @property
    def params(self) -> list[str]:
        return list(dict.fromkeys(self.request.query_params.keys()))

    def path_param(self, name: str) -> str:
        return str(self.request.path_params[name])

    def header(self, name: str) -> str | None:
        return self.request.headers.get(name)


class VendorApi:
    """One mirrored vendor API. Subclasses render auth failures and faults in the vendor's shape."""

    name: str = ""
    prefix: str = ""
    router: APIRouter
    #: The ``calendar`` value of ``POST /_control/bookings`` that this vendor serves, if any.
    calendar: str | None = None

    def owns(self, path: str) -> bool:
        return path == self.prefix or path.startswith(self.prefix + "/")

    def parse_body(self, raw: bytes, content_type: str) -> tuple[Any, bool]:
        """The request body as handlers see it, and whether it is invalid. Default: strict JSON."""
        return parse_body_json(raw)

    def log_view(self, value: Any) -> Any:
        """What the request log keeps of a request or response body. Default: the body itself."""
        return value

    def authorize(self, call: Call) -> Outcome | None:
        """``None`` when the call may proceed. Default: the sandbox bearer token."""
        if bearer_ok(call.request):
            return None
        return self.unauthorized(call, token_sent="authorization" in call.request.headers)

    def render(self, outcome: Outcome) -> Response:
        """The HTTP response for an outcome. Default: compact JSON, ``application/json; charset=utf-8``."""
        return vendor_response(outcome)

    def unauthorized(self, call: Call, *, token_sent: bool) -> Outcome:
        raise NotImplementedError

    def route_not_found(self, request: Request, now: datetime) -> Outcome:
        raise NotImplementedError

    def server_error(self, call: Call) -> Outcome:
        raise NotImplementedError

    def gateway_timeout(self, call: Call) -> Outcome:
        raise NotImplementedError

    def not_found(self, call: Call) -> Outcome:
        raise NotImplementedError

    def malformed(self, call: Call, normal: Outcome) -> Outcome:
        """A 200 with an unexpected schema. ``normal`` is what the handler produced (writes commit).

        Called with ``state.lock`` held."""
        raise NotImplementedError

    def slot_taken_after_offer(self, call: Call, run: Callable[[], Outcome]) -> Outcome:
        """Take the offered slots by a third party around ``run``. Called with ``state.lock`` held."""
        return run()

    def setup_booking(self, state: SandboxState, setup: SetupBooking) -> Outcome:
        """Create a lead's pre-existing booking through the vendor's own create path (no log, no faults)."""
        raise NotImplementedError(f"{self.name} does not create setup bookings")


Handler = Callable[[Call], Outcome]
Precheck = Callable[[Call], Outcome | None]


def request_url(request: Request) -> str:
    """Path and query exactly as the client sent them (Express's ``request.url``), percent-encoding kept."""
    raw_path = request.scope.get("raw_path")
    path = raw_path.decode("utf-8", errors="replace") if raw_path else request.url.path
    query = request.url.query
    return path + (f"?{query}" if query else "")


def bearer_ok(request: Request) -> bool:
    expected: str = request.app.state.sandbox_token
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return scheme == "Bearer" and secrets.compare_digest(token.encode(), expected.encode())


def query_dict(request: Request) -> dict[str, Any]:
    """Query parameters for the log; a repeated key becomes a list."""
    result: dict[str, Any] = {}
    for key, value in request.query_params.multi_items():
        if key in result:
            previous = result[key]
            result[key] = [*previous, value] if isinstance(previous, list) else [previous, value]
        else:
            result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} does not fit a finite number")
    return value


def parse_json(raw: bytes) -> Any:
    """Strict JSON, as the vendors' parsers read it: ``NaN``, ``Infinity`` and numbers that overflow to
    infinity are rejected. Stored state must stay serialisable, or ``GET /_state`` itself would fail."""
    try:
        return json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
    except RecursionError as exc:
        raise ValueError("JSON nested too deeply") from exc


def parse_body_json(raw: bytes) -> tuple[Any, bool]:
    """A strict JSON body; an invalid one is kept as text for the log."""
    if not raw:
        return None, False
    try:
        return parse_json(raw), False
    except ValueError:
        return raw.decode("utf-8", errors="replace"), True


async def read_call(request: Request, group: str, api: VendorApi) -> Call:
    state: SandboxState = request.app.state.sandbox
    raw = await request.body()
    body, invalid = api.parse_body(raw, request.headers.get("content-type", ""))
    return Call(
        request=request, state=state, group=group, query=query_dict(request), body=body, body_invalid=invalid
    )


def vendor_response(outcome: Outcome) -> Response:
    return JSONResponse(
        outcome.body, status_code=outcome.status, media_type=JSON_UTF8, headers=dict(outcome.headers)
    )


def _log(call: Call, api: VendorApi) -> LogEntry:
    return call.state.log(call.method, call.path, call.group, call.query, api.log_view(call.body))


def _finish(entry: LogEntry, outcome: Outcome, api: VendorApi) -> None:
    entry.status = outcome.status
    entry.response = copy.deepcopy(api.log_view(outcome.body))
    entry.completed = True


async def _log_only(call: Call, api: VendorApi, outcome: Outcome) -> Response:
    """Log a call that never reached the fault engine (auth failure, version routing, unknown route)."""
    async with call.state.lock:
        entry = _log(call, api)
        _finish(entry, outcome, api)
    return api.render(outcome)


async def log_unrouted(request: Request, api: VendorApi) -> Response:
    """Answer and log a call to an unknown path or method under ``api``'s prefix."""
    call = await read_call(request, UNROUTED_GROUP, api)
    return await _log_only(call, api, api.route_not_found(request, call.state.now()))


def _failed(call: Call, api: VendorApi) -> Outcome:
    logger.exception("sandbox error while answering %s %s", call.method, call.url)
    return api.server_error(call)


def _check(call: Call, api: VendorApi, check: Precheck | None) -> Outcome | None:
    if check is None:
        return None
    try:
        return check(call)
    except Exception:
        return _failed(call, api)


async def run_call(
    request: Request,
    api: VendorApi,
    group: str,
    handler: Handler,
    *,
    route: Precheck | None = None,
    precheck: Precheck | None = None,
) -> Response:
    """The pipeline every vendor route runs; see the module docstring."""
    call = await read_call(request, group, api)
    state = call.state
    early = _check(call, api, route)
    if early is None:
        early = _check(call, api, api.authorize)
    if early is None:
        early = _check(call, api, precheck)
    if early is not None:
        return await _log_only(call, api, early)
    async with state.lock:
        entry = _log(call, api)
        call.generation = state.generation
        rule = state.faults.on_call(group)
        entry.fault = rule.mode if rule is not None else None
    try:
        outcome = await _apply(call, api, handler, rule)
    except Exception:
        outcome = _failed(call, api)
    _finish(entry, outcome, api)
    return api.render(outcome)


async def _locked(call: Call, api: VendorApi, work: Callable[[], Outcome]) -> Outcome:
    """Run ``work`` under ``state.lock`` unless the sandbox was reset since the call was logged."""
    async with call.state.lock:
        if call.state.generation != call.generation:
            return api.gateway_timeout(call)
        return work()


async def _apply(call: Call, api: VendorApi, handler: Handler, rule: FaultRule | None) -> Outcome:
    if rule is None:
        return await _locked(call, api, lambda: handler(call))
    if rule.latency_ms:
        await asyncio.sleep(rule.latency_ms / 1000)
    match rule.mode:
        case "slow":
            return await _locked(call, api, lambda: handler(call))
        case "error_500":
            return api.server_error(call)
        case "timeout":
            await asyncio.sleep(rule.hang_s)
            return api.gateway_timeout(call)
        case "commit_then_timeout":
            outcome = await _locked(call, api, lambda: handler(call))
            await asyncio.sleep(rule.hang_s)
            return outcome
        case "not_found":
            return api.not_found(call)
        case "malformed":
            return await _locked(call, api, lambda: api.malformed(call, handler(call)))
        case "slot_taken_after_offer":
            return await _locked(call, api, lambda: api.slot_taken_after_offer(call, lambda: handler(call)))
        case _:
            assert_never(rule.mode)

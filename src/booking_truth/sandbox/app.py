"""The sandbox FastAPI application: mirrored vendor APIs plus the control API and the read-only UI."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from booking_truth import __version__
from booking_truth.sandbox import calcom, control, google, hubspot, oauth
from booking_truth.sandbox.common import VendorApi, log_unrouted
from booking_truth.sandbox.state import SandboxState
from booking_truth.timeutil import Clock, SystemClock

#: Mirrored vendor APIs, mounted in this order. Each one brings its own router and error shapes.
VENDOR_APIS: tuple[VendorApi, ...] = (calcom.API, google.API, oauth.API, hubspot.API)


def create_sandbox_app(
    token: str = "sandbox",  # noqa: S107 - the documented default of BT_SANDBOX_TOKEN
    *,
    clock: Clock | None = None,
    state: SandboxState | None = None,
) -> FastAPI:
    """Build a sandbox app. Its state is ``app.state.sandbox``; ``clock`` replaces the state's clock."""
    if not token:
        raise ValueError("the sandbox token must not be empty")
    if state is None:
        state = SandboxState(clock=clock if clock is not None else SystemClock())
    elif clock is not None:
        state.clock = clock
    app = FastAPI(
        title="booking-truth sandbox",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.sandbox = state
    app.state.sandbox_token = token
    app.state.vendor_apis = VENDOR_APIS
    for api in VENDOR_APIS:
        app.include_router(api.router)
    app.include_router(control.router)
    app.include_router(control.ui_router)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(control.Unauthorized, _control_unauthorized)
    return app


async def _http_error(request: Request, exc: Exception) -> Response:
    """Unknown routes: vendor-shaped and logged under a vendor prefix (a wrong method is answered like an
    unknown path), so the request log still shows an agent that calls an endpoint the sandbox does not
    mirror."""
    status = int(getattr(exc, "status_code", 500))
    if status in (404, 405):
        for api in request.app.state.vendor_apis:
            if api.owns(request.url.path):
                return await log_unrouted(request, api)
        return JSONResponse(
            {"error": "not_found" if status == 404 else "method_not_allowed"}, status_code=status
        )
    return JSONResponse({"error": str(getattr(exc, "detail", "error"))}, status_code=status)


async def _control_unauthorized(request: Request, exc: Exception) -> Response:
    return control.unauthorized_response()

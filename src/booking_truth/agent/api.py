"""The agent's HTTP API: ``create_agent_app(settings, ...)``.

Endpoints:

- ``POST /v1/chat``: server-to-server, bearer ``BT_API_KEY``;
- ``POST /v1/widget/chat``: always channel ``widget``, CORS allowlist ``BT_ALLOWED_ORIGINS``, 30 requests per
  minute per client (the socket peer; ``X-Forwarded-For`` only with ``BT_TRUST_PROXY=true``), a signed
  session token instead of the bearer;
- ``GET /v1/version``: the agent version, the model, the guard configuration and whether it runs offline;
- ``GET /v1/sessions/{id}/trace``: the session's ``agent-trace/v1`` record (``BT_EXPOSE_TRACES`` and bearer);
- ``GET /healthz``: liveness plus the CRM outbox backlog;
- ``GET /demo`` and ``GET /widget.js``: the demo page and the embeddable widget.

Without an LLM key (an empty value counts as unset) the agent runs the scripted offline policy.
"""

from __future__ import annotations

import contextlib
import hmac
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import ValidationError

from booking_truth import __version__
from booking_truth.agent.core import AgentCore, AgentDeps
from booking_truth.agent.guards import GuardConfig
from booking_truth.agent.loop import AGENT_TEMPERATURE
from booking_truth.agent.models import ChatRequest, ErrorBody
from booking_truth.agent.outbox_worker import OutboxWorker
from booking_truth.agent.ratelimit import SlidingWindowLimiter, client_key
from booking_truth.agent.scripted import FakeLLM
from booking_truth.agent.tools import HandoffNotifier, tool_specs
from booking_truth.agent.version import compute_agent_version, installed_source_hash, prompt_files, schemas_of
from booking_truth.calendars.base import CalendarAdapter
from booking_truth.calendars.factory import build_calendar, calendar_options
from booking_truth.config import Settings
from booking_truth.crm import CrmAdapter, build_crm
from booking_truth.llm.client import OpenAICompatClient
from booking_truth.llm.types import LLM
from booking_truth.resources import data_path
from booking_truth.store import Store
from booking_truth.timeutil import Clock, SystemClock

MAX_BODY_BYTES = 256 * 1024
RATE_LIMITED_REPLY = "You're sending messages quickly. Please try again in a moment."
WIDGET_PLACEHOLDER = "// booking-truth widget: not bundled in this installation\n"


def _error(status: int, error: str, **fields: Any) -> JSONResponse:
    return JSONResponse(ErrorBody(error=error, **fields).to_json(), status_code=status)


def _validation_detail(exc: ValidationError | RequestValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        location = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        message = str(err.get("msg", "invalid")).removeprefix("Value error, ")
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts)


def demo_page(*, offline: bool) -> str:
    banner = (
        '<p class="banner" role="status">Offline demo mode: no LLM key is configured, so a scripted policy '
        "answers. It handles simple booking, reschedule and cancel requests in English.</p>"
        if offline
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>booking-truth agent demo</title>
<style>
body {{ margin: 0; font: 16px/1.5 system-ui, sans-serif; color: #1d2522; background: #fbfaf6; }}
main {{ max-width: 720px; margin: 0 auto; padding: 32px 16px; }}
.banner {{ background: #fff4d6; border: 1px solid #e6c56b; border-radius: 8px; padding: 10px 14px; }}
</style>
</head>
<body>
<main>
<h1>Book a 30-minute intro call</h1>
{banner}
<p>Use the chat button to book, move or cancel a call. Bookings go to the calendar this agent is configured
with.</p>
</main>
<script src="/widget.js" data-agent="/" async></script>
</body>
</html>
"""


def create_agent_app(
    settings: Settings,
    *,
    llm: LLM | None = None,
    calendar: CalendarAdapter | None = None,
    crm: CrmAdapter | None = None,
    clock: Clock | None = None,
) -> FastAPI:
    """The agent as a FastAPI app. Raises ``ConfigError`` for an invalid configuration."""
    settings.validate_for_agent()
    guards = GuardConfig(settings.enabled_guards)
    clock = clock or SystemClock()
    closers: list[Callable[[], Awaitable[None]]] = []
    if llm is None:
        if settings.offline:
            llm = FakeLLM()
        else:
            client = OpenAICompatClient.from_settings(settings, component="agent")
            closers.append(client.aclose)
            llm = client
    offline = isinstance(llm, FakeLLM)
    model_id = llm.model_id if isinstance(llm, FakeLLM) else settings.llm_model
    if calendar is None:
        options = calendar_options(guards.enabled)
        calendar = build_calendar(
            settings,
            lenient=options.lenient,
            post_retries_on_timeout=options.post_retries_on_timeout,
            clock=clock,
        )
        closers.append(calendar.aclose)
    crm_note: str | None = None
    if crm is None:
        crm, crm_note = build_crm(settings)
        closers.append(crm.aclose)
    store = Store(settings.resolved_db_path, clock=clock)
    version = compute_agent_version(
        prompts=prompt_files(),
        tool_schemas=schemas_of(tool_specs(guards.enabled)),
        model_id=model_id,
        guards_label=guards.label,
        package_version=__version__,
        source_digest=installed_source_hash(),
    )
    handoffs = HandoffNotifier(settings.handoff_webhook_url, store)
    deps = AgentDeps(
        settings=settings,
        llm=llm,
        calendar=calendar,
        crm=crm,
        store=store,
        clock=clock,
        guards=guards,
        version=version,
        model_id=model_id,
        handoffs=handoffs,
        offline=offline,
        crm_note=crm_note,
    )
    core = AgentCore(deps)
    limiter = SlidingWindowLimiter()
    outbox_worker = OutboxWorker(store, crm)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        outbox_worker.start()
        try:
            yield
        finally:
            await outbox_worker.stop()
            await handoffs.drain()
            for close in closers:
                with contextlib.suppress(Exception):
                    await close()
            store.close()

    app = FastAPI(
        title="booking-truth agent",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.core = core
    app.state.deps = deps
    app.state.outbox_worker = outbox_worker
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
        max_age=600,
    )

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, "invalid_request", detail=_validation_detail(exc))

    def authorized(request: Request) -> bool:
        if settings.api_key is None:
            return True
        scheme, _, value = request.headers.get("authorization", "").partition(" ")
        expected = settings.api_key.get_secret_value()
        return scheme.lower() == "bearer" and hmac.compare_digest(value.strip().encode(), expected.encode())

    async def parse(request: Request) -> ChatRequest | JSONResponse:
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return _error(
                413, "input_too_long", detail=f"the request body is limited to {MAX_BODY_BYTES} bytes"
            )
        try:
            return ChatRequest.model_validate_json(raw)
        except ValidationError as exc:
            return _error(422, "invalid_request", detail=_validation_detail(exc))

    @app.post("/v1/chat")
    async def chat(request: Request) -> JSONResponse:
        if not authorized(request):
            return _error(401, "unauthorized")
        parsed = await parse(request)
        if isinstance(parsed, JSONResponse):
            return parsed
        outcome = await core.handle_turn(parsed)
        return JSONResponse(outcome.body, status_code=outcome.status)

    @app.post("/v1/widget/chat")
    async def widget_chat(request: Request) -> JSONResponse:
        peer = request.client.host if request.client is not None else None
        key = client_key(peer, request.headers, trust_proxy=settings.trust_proxy)
        if not limiter.allow(key):
            response = _error(429, "rate_limited", reply=RATE_LIMITED_REPLY)
            response.headers["Retry-After"] = str(max(1, int(limiter.retry_after(key) + 0.999)))
            return response
        parsed = await parse(request)
        if isinstance(parsed, JSONResponse):
            return parsed
        outcome = await core.handle_turn(parsed, widget=True)
        return JSONResponse(outcome.body, status_code=outcome.status)

    @app.get("/v1/version")
    async def version_info() -> dict[str, Any]:
        return {
            "agent_version": deps.version,
            "version": __version__,
            "model": deps.model_id,
            "guards": guards.label,
            "source_hash": installed_source_hash(),
            "offline": deps.offline,
            "temperature": AGENT_TEMPERATURE,
            "calendar": deps.calendar.kind,
            "crm": deps.crm.kind,
        }

    @app.get("/v1/sessions/{session_id}/trace")
    async def session_trace(session_id: str, request: Request) -> JSONResponse:
        if not settings.expose_traces:
            return _error(404, "not_found")
        if not authorized(request):
            return _error(401, "unauthorized")
        trace = core.session_trace(session_id)
        if trace is None:
            return _error(404, "not_found")
        return JSONResponse(trace)

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        backlog = store.outbox.backlog()
        body: dict[str, Any] = {
            "status": "ok",
            "agent_version": deps.version,
            "offline": deps.offline,
            "outbox": {"pending": backlog.pending, "failed": backlog.failed},
            "outbox_backlog": backlog.pending,
            "handoffs_undelivered": len(store.handoffs.items(delivered=False, limit=1000)),
            "crm": deps.crm.kind,
        }
        if deps.crm_note:
            body["crm_note"] = deps.crm_note
        return body

    @app.get("/demo", response_class=HTMLResponse)
    async def demo() -> HTMLResponse:
        return HTMLResponse(demo_page(offline=deps.offline))

    @app.get("/widget.js")
    async def widget_js() -> Response:
        try:
            body = (data_path("widget") / "widget.js").read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            body = WIDGET_PLACEHOLDER
        return Response(body, media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    return app

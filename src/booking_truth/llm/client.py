"""Chat client for any OpenAI-compatible endpoint (OpenRouter by default), with pricing and a budget stop.

Every HTTP request, including each retry, goes through the same gates before it leaves the process:

1. the model, and the pinned provider when ``provider`` is set, must be in the price table
   (:class:`PricingError` otherwise);
2. this process must be able to write its ledger file, and the ledger total plus the estimates of
   requests still in flight in this process plus this request's estimate must stay within
   ``budget_usd`` (:class:`LedgerError` or :class:`BudgetExceeded` otherwise);
3. when ``provider`` is set, the request pins that one upstream provider with fallbacks disabled.

Every attempt that may have been billed is written to the ledger exactly once: with the returned
token counts when the response has usage, and with the pre-call estimate (``usage_estimated``) when
it has none (a 200 without usage, a read timeout, a dropped connection, a cancelled call). Requests
that never reached the server (connection refused, connect timeout) and HTTP error responses
without usage are not recorded.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import httpx
import openai

from booking_truth.llm.ledger import CostLedger, LedgerError
from booking_truth.llm.pricing import CHARS_PER_TOKEN_ESTIMATE, PriceTable
from booking_truth.llm.types import (
    ChatMessage,
    LLMError,
    LLMResponse,
    PricingError,
    ToolCall,
    ToolSpec,
    Usage,
)

if TYPE_CHECKING:
    from booking_truth.config import Settings

_KIND_BY_STATUS = {
    400: "bad_request",
    401: "auth",
    402: "payment_required",
    403: "permission_denied",
    404: "not_found",
    408: "timeout",
    413: "payload_too_large",
    422: "bad_request",
    429: "rate_limit",
}
_RETRY_STATUSES = frozenset({408, 409, 429})
_INITIAL_RETRY_DELAY_S = 0.5
_MAX_RETRY_DELAY_S = 8.0
_MAX_RETRY_AFTER_S = 60.0
# Transport failures that happen before a complete request reaches the server, so nothing is billed.
# Class names are shared by httpx and httpx2. Any other transport failure may follow a processed request.
_NOT_SENT = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "PoolTimeout",
        "WriteError",
        "WriteTimeout",
        "LocalProtocolError",
        "ProxyError",
        "UnsupportedProtocol",
    }
)
_SECRETISH = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+)")
_MAX_DETAIL = 400


class OpenAICompatClient:
    """Implements :class:`booking_truth.llm.types.LLM` with the official ``openai`` SDK.

    ``http_client`` is an ``httpx.AsyncClient`` (the SDK accepts one in place of its own client), so
    tests can intercept traffic with respx and callers can share connection settings. When it is
    omitted the client creates and owns one with ``timeout_s``.

    ``max_retries`` is applied by this client, not by the SDK (whose own retries are off), so every
    retry passes the budget gate and is recorded like a first attempt. Timeouts, dropped
    connections, 408, 409, 429 and 5xx are retried with exponential backoff, honouring
    ``Retry-After`` up to 60 seconds and ``x-should-retry``.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        *,
        provider: str | None = None,
        pricing: PriceTable,
        ledger: CostLedger,
        budget_usd: float | None,
        timeout_s: float = 60,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMError("no LLM API key is configured, so live calls are off", kind="offline")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.provider = provider.strip() if provider and provider.strip() else None
        self.pricing = pricing
        self.ledger = ledger
        self.budget_usd = budget_usd
        self.timeout_s = float(timeout_s)
        self.max_retries = max_retries
        self._secret = api_key.strip()
        parsed = urlparse(self.base_url)
        # Host only: never the user:password part of a URL, which would otherwise reach error messages.
        self._host = parsed.hostname or "the configured LLM endpoint"
        self._url_password = parsed.password or ""
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s))
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        headers: dict[str, str] = {}
        if self._host == "openrouter.ai" or self._host.endswith(".openrouter.ai"):
            # Documented opt-in that adds `openrouter_metadata` (selected endpoint, attempts) to responses.
            headers["X-OpenRouter-Metadata"] = "enabled"
        self._sdk = openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self._secret,
            timeout=self.timeout_s,
            max_retries=0,
            default_headers=headers,
            http_client=cast(Any, self._http),
        )
        # The SDK reads OPENAI_ORG_ID / OPENAI_PROJECT_ID from the environment; never forward
        # OpenAI account ids to whichever endpoint BT_LLM_BASE_URL points at.
        self._sdk.organization = None
        self._sdk.project = None

    def __repr__(self) -> str:
        return (
            f"OpenAICompatClient(host={self._host!r}, default_model={self.default_model!r}, "
            f"provider={self.provider!r}, budget_usd={self.budget_usd!r})"
        )

    # Construction from settings ----------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        component: str,
        model: str | None = None,
        ledger_dir: Path | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> OpenAICompatClient:
        """Wire ``BT_LLM_BASE_URL``, ``BT_LLM_API_KEY``, ``BT_LLM_PROVIDER``, ``BT_PRICING_PATH``,
        ``BT_LEDGER_DIR`` and ``BT_BUDGET_USD``. Raises :class:`LLMError` (kind ``offline``) without a key."""
        key = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key is not None else ""
        if settings.offline or not key:
            raise LLMError(
                "no LLM API key is configured (BT_LLM_API_KEY is empty), so live LLM calls are off",
                kind="offline",
            )
        return cls(
            settings.llm_base_url,
            key,
            model or settings.llm_model,
            provider=settings.llm_provider,
            pricing=PriceTable.load(settings.pricing_path),
            ledger=CostLedger(
                ledger_dir if ledger_dir is not None else settings.resolved_ledger_dir, component
            ),
            budget_usd=settings.budget_usd,
            http_client=http_client,
        )

    # Lifecycle ---------------------------------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> OpenAICompatClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # Request building --------------------------------------------------------------------------

    def provider_preferences(self) -> dict[str, Any] | None:
        """OpenRouter provider routing that pins exactly one provider, with no fallbacks."""
        if self.provider is None:
            return None
        return {"order": [self.provider], "allow_fallbacks": False, "require_parameters": True}

    # The call ----------------------------------------------------------------------------------

    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float,
        model: str | None = None,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        component: str = "agent",
        run_id: str | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        model_id = model or self.default_model
        wire_messages = [message.to_openai() for message in messages]
        wire_tools = [tool.to_openai() for tool in tools] if tools else None
        prompt_chars = len(json.dumps(wire_messages, ensure_ascii=False))
        if wire_tools:
            prompt_chars += len(json.dumps(wire_tools, ensure_ascii=False))
        if response_format:
            prompt_chars += len(json.dumps(response_format, ensure_ascii=False))

        # Raises PricingError before anything is sent when the model or pinned provider has no price.
        estimate = self.pricing.estimate(model_id, self.provider, prompt_chars, max_tokens)

        request: dict[str, Any] = {
            "model": model_id,
            "messages": wire_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if wire_tools:
            request["tools"] = wire_tools
        if response_format is not None:
            request["response_format"] = response_format
        preferences = self.provider_preferences()
        if preferences is not None:
            request["extra_body"] = {"provider": preferences}

        call = _CallContext(
            model_id=model_id,
            component=component,
            run_id=run_id,
            prompt_chars=prompt_chars,
            max_tokens=max_tokens,
        )
        retries = 0
        while True:
            try:
                return await self._attempt(request, call, estimate)
            except _AttemptFailed as failed:
                if failed.retry_after is None or retries >= self.max_retries:
                    raise failed.error from None
                delay = _backoff(retries, failed.retry_after)
            retries += 1
            await self._sleep(delay)

    async def _attempt(self, request: dict[str, Any], call: _CallContext, estimate: float) -> LLMResponse:
        """One HTTP request behind the gate. The hold is released only after the attempt is recorded."""
        reservation = self.ledger.reserve(estimate, self.budget_usd)
        try:
            return await self._send(request, call)
        finally:
            reservation.release()

    async def _send(self, request: dict[str, Any], call: _CallContext) -> LLMResponse:
        started = time.perf_counter()
        try:
            raw = await self._sdk.chat.completions.with_raw_response.create(**request)
        except asyncio.CancelledError:
            # The request may already be with the provider, which bills it even if nobody reads it.
            # A failed write marks the ledger unusable, and cancellation must still propagate.
            with contextlib.suppress(LedgerError):
                self._record_estimate(call)
            raise
        except openai.APIStatusError as err:
            self._record_error_usage(err, call)
            retry_after = _status_retry_after(err.status_code, err.response.headers)
            raise _AttemptFailed(self._status_error(err), retry_after) from None
        except openai.APITimeoutError as err:
            if _may_have_been_processed(err):
                self._record_estimate(call)
            error = LLMError(
                f"LLM request to {self._host} timed out after {self.timeout_s:g}s", kind="timeout"
            )
            raise _AttemptFailed(error, 0.0) from None
        except openai.APIConnectionError as err:
            if _may_have_been_processed(err):
                self._record_estimate(call)
            cause = err.__cause__
            reason = f"{type(cause).__name__}: {cause}" if cause is not None else "connection failed"
            error = LLMError(
                f"could not reach the LLM endpoint {self._host}: {self._scrub(reason)}", kind="connection"
            )
            raise _AttemptFailed(error, 0.0) from None
        except openai.APIError as err:
            error = LLMError(
                f"LLM request failed: {type(err).__name__}: {self._scrub(str(err))}", kind="error"
            )
            raise _AttemptFailed(error, None) from None
        latency = time.perf_counter() - started
        try:
            body = raw.http_response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            # A 200 response was produced, so the call may have been billed.
            self._record_estimate(call)
            error = LLMError(
                f"LLM endpoint {self._host} returned a body that is not a JSON object", kind="malformed"
            )
            raise _AttemptFailed(error, None) from None
        return self._finish(body, raw.headers, call, latency)

    # Response handling -------------------------------------------------------------------------

    def _finish(
        self, body: dict[str, Any], headers: Mapping[str, str], call: _CallContext, latency: float
    ) -> LLMResponse:
        model_returned = body.get("model") if isinstance(body.get("model"), str) else None
        provider = _provider_from(body, headers)
        response_id = _str(body.get("id")) or _str(headers.get("x-generation-id")) or ""
        choices = body.get("choices")
        choice = (
            choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        )
        message = choice.get("message") if choice is not None else None
        finish_reason = _str(choice.get("finish_reason")) if choice is not None else None

        failure: LLMError | None = None
        error = body.get("error")
        if choice is None or finish_reason == "error" or not isinstance(message, dict):
            source = choice.get("error") if choice is not None and choice.get("error") else error
            detail, code = _error_fields(source)
            if detail:
                failure = LLMError(
                    f"LLM provider error{f' ({provider})' if provider else ''}: {self._scrub(detail)}",
                    kind="provider",
                    status_code=code,
                )
            else:
                failure = LLMError(
                    f"LLM endpoint {self._host} returned a response without a usable message",
                    kind="malformed",
                )

        # Any 200 response may have been billed: without usage, the pre-call estimate is recorded.
        usage = self._record(
            body,
            call,
            model_returned=model_returned,
            provider=provider,
            response_id=response_id,
            ok=failure is None,
            estimate_if_missing=True,
        )
        if failure is not None:
            raise failure
        assert isinstance(message, dict)
        return LLMResponse(
            content=_content_text(message.get("content")),
            tool_calls=_tool_calls(message.get("tool_calls"), response_id),
            usage=usage,
            model_requested=call.model_id,
            model_returned=model_returned or call.model_id,
            provider=provider,
            response_id=response_id,
            latency_s=latency,
            finish_reason=finish_reason,
        )

    def _record(
        self,
        body: Mapping[str, Any],
        call: _CallContext,
        *,
        model_returned: str | None,
        provider: str | None,
        response_id: str,
        ok: bool,
        estimate_if_missing: bool,
    ) -> Usage:
        """Price the attempt and write its one ledger entry. Returns the usage (zeros if nothing recorded).

        ``usd`` is tokens times the table price, raised to the provider-reported ``usage.cost`` when
        that is higher, so the budget never counts less than the endpoint says it charged.
        """
        raw_usage = body.get("usage")
        estimated = False
        if isinstance(raw_usage, Mapping) and _int(raw_usage.get("prompt_tokens")) is not None:
            prompt_tokens = _int(raw_usage.get("prompt_tokens")) or 0
            completion_tokens = _int(raw_usage.get("completion_tokens")) or 0
            details = raw_usage.get("prompt_tokens_details")
            cached_tokens = (_int(details.get("cached_tokens")) or 0) if isinstance(details, Mapping) else 0
            reported = _float(raw_usage.get("cost"))
        elif estimate_if_missing:
            estimated = True
            prompt_tokens = math.ceil(call.prompt_chars / CHARS_PER_TOKEN_ESTIMATE)
            completion_tokens = call.max_tokens
            cached_tokens = 0
            reported = None
        else:
            return Usage()

        cost = self._cost(call.model_id, model_returned, prompt_tokens, completion_tokens, cached_tokens)
        if reported is not None and reported > cost:
            cost = reported
        self.ledger.record(
            {
                "component": call.component,
                "run_id": call.run_id,
                "model_requested": call.model_id,
                "model_returned": model_returned,
                "provider": provider,
                "provider_requested": self.provider,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "usd": cost,
                "provider_reported_cost": reported,
                "response_id": response_id or None,
                "ok": ok,
                "usage_estimated": estimated,
            }
        )
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            usd=cost,
            provider_reported_cost=reported,
        )

    def _cost(
        self,
        model_requested: str,
        model_returned: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
    ) -> float:
        """Tokens times the table price of the model that answered, since routers bill that one.

        Falls back to the requested model, which the pre-call gate already priced, when the answering
        model or its pinned-provider row is not in the table. Never raises after a call was made.
        """
        if model_returned:
            returned_key = self.pricing.resolve(model_returned)
            if returned_key is not None and returned_key != self.pricing.resolve(model_requested):
                try:
                    return self.pricing.cost(
                        returned_key, self.provider, prompt_tokens, completion_tokens, cached_tokens
                    )
                except PricingError:
                    pass
        return self.pricing.cost(
            model_requested, self.provider, prompt_tokens, completion_tokens, cached_tokens
        )

    def _record_estimate(self, call: _CallContext) -> None:
        """Record the pre-call estimate for an attempt that may have been billed but returned no usage."""
        self._record(
            {}, call, model_returned=None, provider=None, response_id="", ok=False, estimate_if_missing=True
        )

    def _record_error_usage(self, err: openai.APIStatusError, call: _CallContext) -> None:
        try:
            body = err.response.json()
        except ValueError:
            return
        if isinstance(body, dict) and isinstance(body.get("usage"), Mapping):
            model_returned = body.get("model") if isinstance(body.get("model"), str) else None
            self._record(
                body,
                call,
                model_returned=model_returned,
                provider=_provider_from(body, err.response.headers),
                response_id=_str(body.get("id")) or "",
                ok=False,
                estimate_if_missing=False,
            )

    def _status_error(self, err: openai.APIStatusError) -> LLMError:
        status = err.status_code
        kind = _KIND_BY_STATUS.get(status) or ("server" if status >= 500 else "http_error")
        detail, _ = _error_fields(err.body)
        if not detail:
            detail = err.response.reason_phrase or "no detail"
        hint = ""
        if status == 401:
            hint = " (check BT_LLM_API_KEY)"
        elif status == 402:
            hint = " (the account has insufficient credits)"
        elif status in (404, 503) and self.provider is not None:
            hint = f" (BT_LLM_PROVIDER={self.provider} is pinned with fallbacks disabled)"
        return LLMError(
            f"LLM endpoint {self._host} returned HTTP {status}: {self._scrub(detail)}{hint}",
            kind=kind,
            status_code=status,
        )

    def _scrub(self, text: str) -> str:
        cleaned = text
        for secret in (self._secret, self._url_password):
            if secret:
                cleaned = cleaned.replace(secret, "[redacted]")
        cleaned = _SECRETISH.sub("[redacted]", cleaned)
        return cleaned if len(cleaned) <= _MAX_DETAIL else cleaned[: _MAX_DETAIL - 3] + "..."


class _CallContext:
    __slots__ = ("component", "max_tokens", "model_id", "prompt_chars", "run_id")

    def __init__(
        self, *, model_id: str, component: str, run_id: str | None, prompt_chars: int, max_tokens: int
    ) -> None:
        self.model_id = model_id
        self.component = component
        self.run_id = run_id
        self.prompt_chars = prompt_chars
        self.max_tokens = max_tokens


class _AttemptFailed(Exception):
    """One attempt failed. ``retry_after`` is ``None`` when it must not be retried, else the
    server-requested delay in seconds (``0`` means use the backoff)."""

    def __init__(self, error: LLMError, retry_after: float | None) -> None:
        super().__init__(str(error))
        self.error = error
        self.retry_after = retry_after


# Retry helpers ---------------------------------------------------------------------------------


def _backoff(retries_taken: int, retry_after: float) -> float:
    if 0 < retry_after <= _MAX_RETRY_AFTER_S:
        return retry_after
    exponential: float = _INITIAL_RETRY_DELAY_S * 2.0**retries_taken
    return min(exponential, _MAX_RETRY_DELAY_S) * (1 - 0.25 * random.random())


def _status_retry_after(status: int, headers: Mapping[str, str]) -> float | None:
    """``None`` when an HTTP error must not be retried, else the ``Retry-After`` delay (0 if absent)."""
    delay = _retry_after_seconds(headers)
    if delay is not None and delay > _MAX_RETRY_AFTER_S:
        return None
    should_retry = headers.get("x-should-retry")
    if should_retry == "false":
        return None
    if should_retry == "true" or status in _RETRY_STATUSES or status >= 500:
        return delay or 0.0
    return None


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    for name, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) / divisor
        except ValueError:
            continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def _may_have_been_processed(err: openai.APIError) -> bool:
    cause = err.__cause__
    return cause is None or type(cause).__name__ not in _NOT_SENT


# Parsing helpers -------------------------------------------------------------------------------


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float) and math.isfinite(value):
        return max(int(value), 0)
    return None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _error_fields(source: Any) -> tuple[str, int | None]:
    """``(message, code)`` from an OpenAI/OpenRouter error object, or ``("", None)``."""
    if isinstance(source, Mapping):
        inner = source.get("error")
        if isinstance(inner, Mapping):
            source = inner
        message = source.get("message")
        code = source.get("code")
        return (message if isinstance(message, str) else "", code if isinstance(code, int) else None)
    if isinstance(source, str):
        return source, None
    return "", None


def _provider_from(body: Mapping[str, Any], headers: Mapping[str, str]) -> str | None:
    """The upstream provider: top-level ``provider``, else router metadata, else ``X-Provider-Name``."""
    top = _str(body.get("provider"))
    if top:
        return top
    meta = body.get("openrouter_metadata")
    if isinstance(meta, Mapping):
        endpoints = meta.get("endpoints")
        available = endpoints.get("available") if isinstance(endpoints, Mapping) else None
        if isinstance(available, list):
            for endpoint in available:
                if isinstance(endpoint, Mapping) and endpoint.get("selected") is True:
                    name = _str(endpoint.get("provider"))
                    if name:
                        return name
        attempts = meta.get("attempts")
        if isinstance(attempts, list) and attempts and isinstance(attempts[-1], Mapping):
            name = _str(attempts[-1].get("provider"))
            if name:
                return name
    return _str(headers.get("x-provider-name"))


def _content_text(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        return "".join(parts)
    return str(content)


def _tool_calls(raw: Any, response_id: str) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        if not isinstance(function, Mapping) or not _str(function.get("name")):
            continue
        arguments = function.get("arguments")
        if arguments is None:
            arguments = "{}"
        elif not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        call_id = _str(item.get("id")) or f"call_{response_id or 'local'}_{index}"
        calls.append(ToolCall(id=call_id, name=str(function["name"]), arguments=arguments))
    return calls

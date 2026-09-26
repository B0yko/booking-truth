"""The harness side of the sandbox control API: reset, seed, faults, setup bookings, state and settling.

Every trial runs on a freshly reset sandbox. :meth:`SandboxClient.prepare` resets it, applies the scenario's
seed, creates the lead's setup booking (reschedule and cancel scenarios) through ``POST /_control/bookings``,
which is not logged as agent traffic, and installs the fault rules. Rules are written with Cal.com group names
and gain a Google Calendar twin each, so one list serves both calendar adapters.

After the conversation, :meth:`SandboxClient.settle` polls ``GET /_state`` until the snapshot is unchanged
for one second (at most ``settle_s``), and for a bundled agent until its ``/healthz`` outbox backlog is 0.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import httpx

from booking_truth.harness.grading import SetupInfo
from booking_truth.harness.scenarios import ResolvedScenario, expand_faults_for_google
from booking_truth.sandbox.faults import FaultRule
from booking_truth.timeutil import iso_z

DEFAULT_TOKEN = "sandbox"  # noqa: S105 - the documented default of BT_SANDBOX_TOKEN
STABLE_S = 1.0
POLL_S = 0.1

CalendarKind = Literal["calcom", "google"]
BacklogProbe = Callable[[], Awaitable[int | None]]


class SandboxError(RuntimeError):
    """The sandbox could not be reached or refused a control call: a harness-side failure."""


@dataclass(frozen=True)
class Settled:
    """The end state after settling. ``settled`` is false when ``settle_s`` ran out first."""

    state: dict[str, Any]
    settled: bool
    waited_s: float
    polls: int
    backlog: int | None = None


def _state_key(state: dict[str, Any]) -> str:
    """The snapshot without its clock reading, for the "unchanged" comparison."""
    return json.dumps({k: v for k, v in state.items() if k != "now"}, sort_keys=True, default=str)


class SandboxClient:
    def __init__(
        self,
        base_url: str,
        token: str = DEFAULT_TOKEN,
        *,
        timeout_s: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _call(self, method: str, path: str, body: Any = None) -> Any:
        try:
            response = await self._client.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise SandboxError(f"sandbox {method} {path} failed: {type(exc).__name__}") from exc
        if response.status_code == 401:
            raise SandboxError(f"sandbox {method} {path}: 401 unauthorized (check BT_SANDBOX_TOKEN)")
        if response.status_code >= 300:
            raise SandboxError(f"sandbox {method} {path}: HTTP {response.status_code} {response.text[:300]}")
        try:
            return response.json()
        except ValueError as exc:
            raise SandboxError(f"sandbox {method} {path}: the response is not JSON") from exc

    async def reset(self) -> dict[str, Any]:
        result = await self._call("POST", "/_control/reset")
        return result if isinstance(result, dict) else {}

    async def seed(self, overrides: dict[str, Any]) -> dict[str, Any]:
        result = await self._call("POST", "/_control/seed", overrides)
        return result if isinstance(result, dict) else {}

    async def set_faults(self, rules: Sequence[FaultRule]) -> list[dict[str, Any]]:
        """Install ``rules`` plus their Google Calendar twins (replacing any earlier rules)."""
        expanded = expand_faults_for_google(rules)
        # ``times: null`` means persistent, so ``None`` values must be sent, not dropped.
        body = {"rules": [rule.model_dump(mode="json") for rule in expanded]}
        result = await self._call("POST", "/_control/faults", body)
        faults = result.get("faults") if isinstance(result, dict) else None
        return faults if isinstance(faults, list) else []

    async def create_setup_booking(
        self,
        *,
        calendar: CalendarKind,
        lead_email: str,
        lead_name: str,
        start: datetime,
        lead_timezone: str | None = None,
    ) -> SetupInfo:
        body: dict[str, Any] = {
            "calendar": calendar,
            "lead_email": lead_email,
            "lead_name": lead_name,
            "start": iso_z(start),
        }
        if lead_timezone is not None:
            body["lead_timezone"] = lead_timezone
        result = await self._call("POST", "/_control/bookings", body)
        if not isinstance(result, dict):
            raise SandboxError("setup booking: unexpected response")
        ref = result.get("uid") if calendar == "calcom" else result.get("id")
        if not isinstance(ref, str) or not ref:
            raise SandboxError("setup booking: the response carries no booking reference")
        return SetupInfo(ref=ref, start_utc=start)

    async def state(self) -> dict[str, Any]:
        result = await self._call("GET", "/_state")
        if not isinstance(result, dict):
            raise SandboxError("GET /_state: unexpected response")
        return result

    async def prepare(
        self, scenario: ResolvedScenario, *, calendar: CalendarKind, lead_email: str, lead_name: str
    ) -> SetupInfo | None:
        """Reset, seed, create the setup booking and install the faults for one trial."""
        await self.reset()
        overrides = scenario.scenario.seed_overrides()
        if overrides:
            await self.seed(overrides)
        setup: SetupInfo | None = None
        if scenario.setup_start_utc is not None:
            setup = await self.create_setup_booking(
                calendar=calendar,
                lead_email=lead_email,
                lead_name=lead_name,
                start=scenario.setup_start_utc,
                lead_timezone=scenario.scenario.persona.true_zone,
            )
        await self.set_faults(scenario.scenario.faults)
        return setup

    async def settle(
        self,
        *,
        settle_s: float,
        stable_s: float = STABLE_S,
        poll_s: float = POLL_S,
        backlog: BacklogProbe | None = None,
    ) -> Settled:
        """Poll ``/_state`` until it is unchanged for ``stable_s`` (and the outbox backlog is 0), at most
        ``settle_s`` seconds. The last snapshot is the end state either way."""
        start = time.monotonic()
        deadline = start + max(settle_s, 0.0)
        last_key: str | None = None
        changed_at = start
        polls = 0
        pending: int | None = None
        while True:
            snapshot = await self.state()
            polls += 1
            now = time.monotonic()
            key = _state_key(snapshot)
            if key != last_key:
                last_key, changed_at = key, now
            if backlog is not None:
                pending = await backlog()
            stable = now - changed_at >= stable_s and (backlog is None or not pending)
            if stable or now >= deadline:
                return Settled(snapshot, stable, now - start, polls, pending)
            await asyncio.sleep(min(poll_s, max(deadline - now, 0.0)))

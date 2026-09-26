"""Fixtures for the CRM tests."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from booking_truth.crm.hubspot import HubSpotAdapter
from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer
from booking_truth.timeutil import FixedClock

# Test modules import the shared helpers by module name.
sys.path.insert(0, str(Path(__file__).parent))

from hubspot_env import TOKEN, HubEnv

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``BT_*`` setting from the developer's environment reaches these tests."""
    for name in list(os.environ):
        if name.startswith("BT_") or name == "OPENROUTER_API_KEY":
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def sandbox_server() -> Iterator[tuple[FastAPI, BackgroundServer]]:
    app = create_sandbox_app(TOKEN, clock=FixedClock(NOW))
    with BackgroundServer(app) as server:
        yield app, server


@pytest.fixture
def env(sandbox_server: tuple[FastAPI, BackgroundServer]) -> Iterator[HubEnv]:
    """A fresh sandbox state behind the shared server, its clock fixed at ``NOW``."""
    app, server = sandbox_server
    state = SandboxState(clock=FixedClock(NOW))
    app.state.sandbox = state
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with httpx.Client(base_url=server.url, headers=headers, timeout=5.0) as control:
        yield HubEnv(app=app, state=state, url=server.url, control=control)


@pytest.fixture
async def hubspot(env: HubEnv) -> AsyncIterator[HubSpotAdapter]:
    """A strict adapter (a generous timeout) pointed at the same sandbox server as ``env``."""
    async with env.adapter() as adapter:
        yield adapter

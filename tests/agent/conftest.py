"""Fixtures for the agent tests: one sandbox server per module, a fresh sandbox state and agent per test."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI

from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.sandbox.state import SandboxState
from booking_truth.serve import BackgroundServer
from booking_truth.timeutil import FixedClock

sys.path.insert(0, str(Path(__file__).parent))

from agent_env import NOW, TOKEN, AgentEnv, make_env


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LLM key and no ``BT_*`` setting from the developer's environment reaches these tests."""
    for name in list(os.environ):
        if name.startswith("BT_") or name == "OPENROUTER_API_KEY":
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def sandbox_server() -> Iterator[tuple[FastAPI, BackgroundServer]]:
    app = create_sandbox_app(TOKEN, clock=FixedClock(NOW))
    with BackgroundServer(app) as server:
        yield app, server


@pytest.fixture
def sandbox(
    sandbox_server: tuple[FastAPI, BackgroundServer],
) -> tuple[FastAPI, BackgroundServer, SandboxState]:
    app, server = sandbox_server
    state = SandboxState(clock=FixedClock(NOW))
    app.state.sandbox = state
    return app, server, state


@pytest.fixture
async def guarded(
    sandbox: tuple[FastAPI, BackgroundServer, SandboxState], tmp_path: Path
) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards="all"):
        yield env


@pytest.fixture
async def naive(
    sandbox: tuple[FastAPI, BackgroundServer, SandboxState], tmp_path: Path
) -> AsyncIterator[AgentEnv]:
    async for env in make_env(sandbox, tmp_path, guards="off"):
        yield env

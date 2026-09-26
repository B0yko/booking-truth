"""Fixtures for the component-eval tests: no LLM key or ``BT_*`` setting from the developer's environment
reaches these tests, so a fixture dataset is always scored offline unless a test wires in a fake LLM
itself."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("BT_") or name == "OPENROUTER_API_KEY":
            monkeypatch.delenv(name, raising=False)

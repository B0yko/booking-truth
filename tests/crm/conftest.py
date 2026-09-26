"""Fixtures for the CRM tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``BT_*`` setting from the developer's environment reaches these tests."""
    for name in list(os.environ):
        if name.startswith("BT_") or name == "OPENROUTER_API_KEY":
            monkeypatch.delenv(name, raising=False)

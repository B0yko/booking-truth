"""Shared helpers for the HubSpot adapter tests: a real sandbox over HTTP and an adapter aimed at it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI

from booking_truth.crm.hubspot import HubSpotAdapter
from booking_truth.sandbox.state import SandboxState

TOKEN = "hubspot-test-token"


@dataclass
class HubEnv:
    """One test's view of the shared sandbox server: its fresh state and a control client."""

    app: FastAPI
    state: SandboxState
    url: str
    control: httpx.Client

    def adapter(self, **overrides: Any) -> HubSpotAdapter:
        base_url = overrides.pop("base_url", self.url)
        token = overrides.pop("token", TOKEN)
        return HubSpotAdapter(base_url, token, **overrides)

    def faults(self, *rules: dict[str, Any]) -> None:
        response = self.control.post("/_control/faults", json={"rules": list(rules)})
        assert response.status_code == 200, response.text

    def snapshot(self) -> dict[str, Any]:
        response = self.control.get("/_state")
        assert response.status_code == 200, response.text
        data: dict[str, Any] = response.json()
        return data

    def contacts(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = self.snapshot()["hubspot"]["contacts"]
        return items

    def meetings(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = self.snapshot()["hubspot"]["meetings"]
        return items

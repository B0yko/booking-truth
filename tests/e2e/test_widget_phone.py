"""Phone-width booking through the demo widget (``GET /demo``), driven with Playwright Chromium.

Starts an in-process sandbox and the bundled guarded agent, offline (no LLM key, so the deterministic
scripted policy answers), then drives the real ``/demo`` page and ``/widget.js`` at a 375x812 viewport,
exactly as a phone visitor would: fill in a name and an email, ask for a call, pick an offered time. It
checks both what the visitor sees (a booking card in the chat) and the sandbox's own state (an active
booking for that email), so the test fails if the widget only *looks* booked without a matching write.

Requires Chromium installed for Playwright (``uv run playwright install chromium``); skipped otherwise.
Run with ``uv run pytest -m e2e tests/e2e/test_widget_phone.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from booking_truth.harness.builtin import BuiltinAgent, start_builtin_agent, start_sandbox
from booking_truth.serve import BackgroundServer

pytestmark = pytest.mark.e2e

playwright_sync = pytest.importorskip("playwright.sync_api", reason="Playwright is a dev-only dependency")

LEAD_NAME = "Priya K"
LEAD_EMAIL = "priya.k@example.com"
VIEWPORT = {"width": 375, "height": 812}
BOOK_MESSAGE = "Hi, I'd like to book a call, please."


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LLM key and no stray ``BT_*`` setting from the developer's own shell reaches this test."""
    for name in list(os.environ):
        if name.startswith("BT_") or name == "OPENROUTER_API_KEY":
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def stack() -> Iterator[tuple[BackgroundServer, str, BuiltinAgent]]:
    sandbox_server, token = start_sandbox()
    agent = start_builtin_agent("guarded", sandbox_url=sandbox_server.url, sandbox_token=token)
    try:
        yield sandbox_server, token, agent
    finally:
        agent.stop()
        sandbox_server.stop()


def sandbox_state(sandbox_url: str, token: str) -> dict[str, Any]:
    response = httpx.get(f"{sandbox_url}/_state", headers={"Authorization": f"Bearer {token}"}, timeout=5.0)
    response.raise_for_status()
    state: dict[str, Any] = response.json()
    return state


def active_calcom_bookings(state: dict[str, Any], email: str) -> list[dict[str, Any]]:
    return [
        booking
        for booking in state["calcom"]["bookings"]
        if booking["status"] == "accepted" and any(a.get("email") == email for a in booking["attendees"])
    ]


def test_booking_through_the_demo_widget_at_phone_width(
    stack: tuple[BackgroundServer, str, BuiltinAgent], tmp_path: Path
) -> None:
    sandbox_server, token, agent = stack
    assert agent.mode == "guarded"
    base_url = agent.server.url

    with playwright_sync.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            context = browser.new_context(viewport=VIEWPORT, timezone_id="America/New_York")
            page = context.new_page()
            page.goto(f"{base_url}/demo")

            launcher = page.get_by_role("button", name="Book a call", exact=True)
            launcher.click()

            # The widget fetches GET /v1/version on load and shows this banner when it reports offline.
            playwright_sync.expect(
                page.get_by_text("Offline demo mode: replies come from a scripted assistant.")
            ).to_be_visible(timeout=15_000)

            page.get_by_label("Name").fill(LEAD_NAME)
            page.get_by_label("Email").fill(LEAD_EMAIL)
            page.get_by_role("button", name="Start chat").click()

            composer = page.get_by_placeholder("Type a message")
            composer.fill(BOOK_MESSAGE)
            composer.press("Enter")

            slot_buttons = page.locator("[aria-label='Suggested replies'] button")
            slot_buttons.first.wait_for(state="visible", timeout=15_000)
            slot_buttons.first.click()

            card = page.locator("[role='group'][aria-label^='Booking: Booked']")
            card.wait_for(state="visible", timeout=15_000)

            screenshot_path = tmp_path / "widget-phone-booking.png"
            page.screenshot(path=str(screenshot_path))
        finally:
            browser.close()

    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 0

    state = sandbox_state(sandbox_server.url, token)
    matches = active_calcom_bookings(state, LEAD_EMAIL)
    assert len(matches) == 1, state["calcom"]["bookings"]

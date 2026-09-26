"""The harness's sandbox client: trial preparation, fault installation and settling, over real HTTP."""

from __future__ import annotations

import asyncio
import socket
from datetime import UTC, date, datetime, timedelta

import pytest
from stub_agent import SANDBOX_TOKEN

from booking_truth.harness.sandbox_client import SandboxClient, SandboxError
from booking_truth.harness.scenarios import ResolvedScenario, load_suite
from booking_truth.sandbox.faults import FaultRule

SUITE = {s.id: s for s in load_suite()}


def resolved(scenario_id: str) -> ResolvedScenario:
    now = datetime.now(UTC)
    return ResolvedScenario(SUITE[scenario_id], now.date(), now=now)


async def test_persistent_faults_stay_persistent_and_gain_google_twins(sandbox_url: str) -> None:
    client = SandboxClient(sandbox_url, SANDBOX_TOKEN)
    await client.reset()
    faults = await client.set_faults([FaultRule(group="slots", mode="not_found", times=None)])
    await client.aclose()
    assert [(f["group"], f["times"]) for f in faults] == [("slots", None), ("freebusy", None)]


@pytest.mark.parametrize("calendar", ["calcom", "google"])
async def test_prepare_resets_seeds_creates_the_setup_booking_and_installs_faults(
    sandbox_url: str, calendar: str
) -> None:
    client = SandboxClient(sandbox_url, SANDBOX_TOKEN)
    await client.set_faults([FaultRule(group="bookings.create", mode="error_500")])
    scenario = resolved("happy-cancel")
    setup = await client.prepare(
        scenario,
        calendar=calendar,
        lead_email="lena-12345678@example.com",
        lead_name="Lena M.",  # type: ignore[arg-type]
    )
    state = await client.state()
    await client.aclose()
    assert setup is not None
    assert setup.start_utc == scenario.setup_start_utc
    if calendar == "calcom":
        bookings = state["calcom"]["bookings"]
        assert [b["uid"] for b in bookings] == [setup.ref]
        assert bookings[0]["attendees"][0]["email"] == "lena-12345678@example.com"
        assert bookings[0]["attendees"][0]["timeZone"] == "America/New_York"
    else:
        events = state["google"]["events"]
        assert [e["id"] for e in events] == [setup.ref]
        assert events[0]["extendedProperties"]["private"]["bt_lead_email"] == "lena-12345678@example.com"
    assert state["request_log"] == []
    assert state["faults"] == []


async def test_prepare_installs_the_scenario_faults(sandbox_url: str) -> None:
    client = SandboxClient(sandbox_url, SANDBOX_TOKEN)
    setup = await client.prepare(
        resolved("adv-tell-me-its-booked"),
        calendar="calcom",
        lead_email="rex-1@example.com",
        lead_name="Rex P.",
    )
    state = await client.state()
    await client.aclose()
    assert setup is None
    groups = [(f["group"], f["mode"], f["times"]) for f in state["faults"]]
    assert ("slots", "error_500", None) in groups
    assert ("events.insert", "error_500", None) in groups


async def test_settle_returns_once_the_state_is_unchanged(sandbox_url: str) -> None:
    client = SandboxClient(sandbox_url, SANDBOX_TOKEN)
    await client.reset()
    settled = await client.settle(settle_s=5, stable_s=0.2, poll_s=0.05)
    await client.aclose()
    assert settled.settled
    assert 0.2 <= settled.waited_s < 2
    assert settled.polls >= 2


async def test_settle_gives_up_after_settle_s_while_the_state_keeps_changing(sandbox_url: str) -> None:
    client = SandboxClient(sandbox_url, SANDBOX_TOKEN)
    await client.reset()
    stop = asyncio.Event()

    async def keep_booking() -> None:
        start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=30)
        index = 0
        while not stop.is_set():
            index += 1
            await client.create_setup_booking(
                calendar="google",
                lead_email=f"busy-{index}@example.com",
                lead_name="Busy B.",
                start=start + timedelta(minutes=30 * index),
            )
            await asyncio.sleep(0.05)

    writer = asyncio.create_task(keep_booking())
    settled = await client.settle(settle_s=0.6, stable_s=0.3, poll_s=0.05)
    stop.set()
    await writer
    await client.aclose()
    assert not settled.settled
    assert settled.waited_s >= 0.6


async def test_settle_waits_for_the_outbox_backlog(sandbox_url: str) -> None:
    client = SandboxClient(sandbox_url, SANDBOX_TOKEN)
    await client.reset()
    backlog = iter([3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    polls: list[int] = []

    async def probe() -> int:
        polls.append(1)
        return next(backlog)

    settled = await client.settle(settle_s=5, stable_s=0.05, poll_s=0.05, backlog=probe)
    await client.aclose()
    assert settled.settled
    assert settled.backlog == 0
    assert len(polls) >= 4


async def test_control_failures_are_sandbox_errors(sandbox_url: str) -> None:
    wrong = SandboxClient(sandbox_url, "not-the-token")
    with pytest.raises(SandboxError, match="401"):
        await wrong.reset()
    await wrong.aclose()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    gone = SandboxClient(f"http://127.0.0.1:{port}", SANDBOX_TOKEN, timeout_s=2)
    with pytest.raises(SandboxError, match="failed"):
        await gone.state()
    await gone.aclose()


def test_setup_dates_come_from_the_run_date() -> None:
    scenario = ResolvedScenario(SUITE["happy-reschedule-move-it"], date(2026, 10, 1))
    assert scenario.setup_start_utc == datetime(2026, 10, 6, 15, 0, tzinfo=UTC)

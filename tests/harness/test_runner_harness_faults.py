"""``Runner.converse``'s harness-side fault injection (``docs/metrics.md``, "Harness-side faults"):
``concurrent_channel`` fires on the first pick that knows any offered slot at all (asking for the second
offered slot, or the same slot when only one was offered), defers only past a pick that knows none yet,
is never injected at all when no pick in the whole conversation ever does (instead of crashing, run-1
defect 2), and the trace/``TrialResult`` record that honestly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from booking_truth.harness.adapters import AgentReply, Lead, Turn
from booking_truth.harness.hfaults import FaultReport
from booking_truth.harness.personas import AgentView, Offer, PersonaTurn
from booking_truth.harness.runner import (
    AgentUnderTest,
    Conversation,
    RunConfig,
    _fault_ready,
    _harness_fault_meta,
    _Run,
)
from booking_truth.harness.scenarios import HarnessFault, ResolvedScenario, load_suite

SUITE = {s.id: s for s in load_suite()}
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
OFFER_0 = Offer(NOW + timedelta(hours=1), "first")
OFFER_1 = Offer(NOW + timedelta(hours=2), "second")
LEAD = Lead(email="test-0000@example.com", name="Test T.")


@dataclass
class _ScriptedFakePersona:
    """Hands back exactly the turns given, in order; ``choices`` on a pick controls how many offers this
    fault-gating test says are known at that point in the conversation."""

    steps: list[PersonaTurn]
    turns: int = 0
    usage_usd: float = 0.0

    async def next_turn(self, view: AgentView) -> PersonaTurn | None:
        if not self.steps:
            return None
        self.turns += 1
        return self.steps.pop(0)


@dataclass
class _FakeClient:
    supports_actions: bool = True
    sent: list[Turn] = field(default_factory=list)
    _clock: float = 0.0

    async def send(self, turn: Turn) -> AgentReply:
        self.sent.append(turn)
        self._clock += 0.01
        return AgentReply(status=200, reply="ok", sent_at=self._clock, received_at=self._clock + 0.001)

    async def aclose(self) -> None:
        return None


def make_runner() -> _Run:
    scenario = SUITE["fault-concurrent-channel"]
    config = RunConfig(agents=[AgentUnderTest(label="test", endpoints=[])], scenarios=[scenario], k=1)
    return _Run(config)


def resolved_concurrent_channel() -> ResolvedScenario:
    return ResolvedScenario(SUITE["fault-concurrent-channel"], date(2026, 10, 1), now=NOW)


# _fault_ready -------------------------------------------------------------------------------------------


def test_duplicate_delivery_is_always_ready() -> None:
    fault = HarnessFault(type="duplicate_delivery")
    assert _fault_ready(fault, PersonaTurn("pick", "x", choices=()))


def test_concurrent_channel_is_ready_once_any_offer_is_known() -> None:
    """Run-2 anomaly: deferring until a *second* offer was known meant the fault was skipped outright in
    3 of 5 naive trials that only ever offered one slot at a time (``harness_fault_not_injected``). It
    only needs to know *some* offer, so it can ask for that same one when there is no second."""
    fault = HarnessFault(type="concurrent_channel", pick=1)
    assert not _fault_ready(fault, PersonaTurn("pick", "x", choices=()))
    assert _fault_ready(fault, PersonaTurn("pick", "x", choices=(OFFER_0,)))
    assert _fault_ready(fault, PersonaTurn("pick", "x", choices=(OFFER_0, OFFER_1)))


# _harness_fault_meta -------------------------------------------------------------------------------------


def test_harness_fault_meta_is_none_without_a_configured_fault() -> None:
    assert _harness_fault_meta(None, None) == (None, None)


def test_harness_fault_meta_reports_a_fired_fault() -> None:
    fired = FaultReport("concurrent_channel", {"session_b": "s-b"})
    meta, injected = _harness_fault_meta(HarnessFault(type="concurrent_channel"), fired)
    assert meta == {"type": "concurrent_channel", "injected": True, "session_b": "s-b"}
    assert injected is True


def test_harness_fault_meta_reports_a_configured_fault_that_never_fired() -> None:
    meta, injected = _harness_fault_meta(HarnessFault(type="concurrent_channel", pick=1), None)
    assert meta == {"type": "concurrent_channel", "injected": False}
    assert injected is False


# Runner.converse: the fault-gating behaviour end to end -----------------------------------------------------


async def test_concurrent_channel_fires_on_the_first_pick_that_knows_any_offer() -> None:
    """Run-2 anomaly: the first pick already knows one offer (``pick`` defaults to 1, asking for a second
    one), so the fault must fire right there, asking for ``offered[0]`` (the only offer known) instead of
    waiting for a pick that knows two - the deferral this fix narrows to "no offer known at all"."""
    runner = make_runner()
    resolved = resolved_concurrent_channel()
    conversation = Conversation(session_a="sess-a")
    client = _FakeClient()
    persona = _ScriptedFakePersona(
        [PersonaTurn("pick", "first works", offer=OFFER_0, choices=(OFFER_0,), end=True)]
    )
    agent = AgentUnderTest(label="test", endpoints=[])
    await runner.converse(agent, client, persona, LEAD, resolved, conversation, random.Random(0))
    assert conversation.fault is not None
    assert conversation.fault.type == "concurrent_channel"
    assert conversation.fault.details["requested_offer"] == 1  # the scenario's configured pick
    assert conversation.fault.details["offer_used"] == 0  # only one offer known: the same slot instead
    # The single pick fires the fault immediately: both channels contacted on it, no earlier unfaulted turn.
    assert len(client.sent) == 2
    assert conversation.sessions == ["sess-a", "sess-a-b"]


async def test_concurrent_channel_is_never_injected_without_a_pick_that_qualifies() -> None:
    """No pick in the whole conversation ever knows any offer at all (a freeform acceptance the harness
    could not match to a known offer): the fault is skipped, not crashed (run-1 defect 2), and the record
    says so rather than looking like no fault was configured."""
    runner = make_runner()
    resolved = resolved_concurrent_channel()
    conversation = Conversation(session_a="sess-a")
    client = _FakeClient()
    persona = _ScriptedFakePersona([PersonaTurn("pick", "works for me", offer=OFFER_0, choices=(), end=True)])
    agent = AgentUnderTest(label="test", endpoints=[])
    await runner.converse(agent, client, persona, LEAD, resolved, conversation, random.Random(0))
    assert conversation.fault is None
    assert len(client.sent) == 1
    assert conversation.sessions == ["sess-a"]  # session B was never contacted
    meta, injected = _harness_fault_meta(resolved.scenario.harness_fault, conversation.fault)
    assert meta == {"type": "concurrent_channel", "injected": False}
    assert injected is False

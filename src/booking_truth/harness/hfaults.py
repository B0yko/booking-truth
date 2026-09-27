"""Faults the harness injects on the delivery side (``docs/metrics.md``, "Harness-side faults").

Both fire on one of the persona's picks of an offered slot:

- ``duplicate_delivery`` sends the same request again, with the same ``message_id``, 50 to 200 ms after the
  first, while the first may still be in flight. Both replies are recorded; the persona continues from the one
  that arrived last. It fires on the persona's first pick, unconditionally.
- ``concurrent_channel`` sends, at the same moment, a scripted message for the same lead on a new session over
  the ``webhook`` channel, asking for another offered slot. Both replies are recorded; the persona continues
  in its own session. It needs a pick that already knows enough offered slots to ask for a different one, so
  the runner (``runner.py``'s ``_fault_ready``) defers it past an earlier pick that does not, and skips it
  outright if none in the whole conversation ever qualifies.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from booking_truth.harness.adapters import AgentClient, AgentReply, Turn
from booking_truth.harness.personas import Offer

DUPLICATE_DELAY_S: tuple[float, float] = (0.05, 0.2)
CONCURRENT_MESSAGE = "Please book {label} for me instead."

SessionName = Literal["A", "B"]


@dataclass(frozen=True)
class Delivery:
    """One request and its reply. ``session`` is ``A`` for the persona's session, ``B`` for the second
    channel."""

    session: SessionName
    turn: Turn
    reply: AgentReply
    duplicate: bool = False


@dataclass(frozen=True)
class FaultReport:
    """What a harness fault did, for the trace's ``meta``."""

    type: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"type": self.type, "injected": True, **self.details}


def by_arrival(deliveries: Sequence[Delivery]) -> list[Delivery]:
    return sorted(deliveries, key=lambda d: d.reply.received_at)


def duplicate_delay(rng: random.Random) -> float:
    low, high = DUPLICATE_DELAY_S
    return round(rng.uniform(low, high), 3)


async def deliver_duplicate(
    client: AgentClient, turn: Turn, *, delay_s: float
) -> tuple[list[Delivery], AgentReply, FaultReport]:
    """Send ``turn`` twice, ``delay_s`` apart, concurrently. Returns both deliveries, the reply that arrived
    last (the one the persona continues from) and a report."""
    first = asyncio.create_task(client.send(turn))
    await asyncio.sleep(delay_s)
    in_flight = not first.done()
    second = asyncio.create_task(client.send(turn))
    one, two = await asyncio.gather(first, second)
    deliveries = [Delivery("A", turn, one), Delivery("A", turn, two, duplicate=True)]
    last = by_arrival(deliveries)[-1].reply
    report = FaultReport(
        "duplicate_delivery",
        {
            "message_id": turn.message_id,
            "delay_s": delay_s,
            "first_in_flight_at_resend": in_flight,
            "continued_from": "duplicate" if last is two else "original",
        },
    )
    return deliveries, last, report


def concurrent_message(offers: Sequence[Offer], pick: int) -> tuple[str, int]:
    """The second channel's text asking for ``offers[pick]`` (the last offer when fewer were made)."""
    if not offers:
        raise ValueError("concurrent_channel needs at least one offered slot")
    index = min(pick, len(offers) - 1)
    return CONCURRENT_MESSAGE.format(label=offers[index].label), index


async def deliver_concurrent(
    client: AgentClient, turn_a: Turn, turn_b: Turn, *, requested_index: int, used_index: int
) -> tuple[list[Delivery], AgentReply, FaultReport]:
    """Send the persona's pick (session A) and the second channel's message (session B) at the same moment."""
    reply_a, reply_b = await asyncio.gather(client.send(turn_a), client.send(turn_b))
    deliveries = [Delivery("A", turn_a, reply_a), Delivery("B", turn_b, reply_b)]
    order = [d.session for d in by_arrival(deliveries)]
    report = FaultReport(
        "concurrent_channel",
        {
            "session_b": turn_b.session_id,
            "channel_b": turn_b.channel,
            "message_b": turn_b.message,
            "requested_offer": requested_index,
            "offer_used": used_index,
            "arrival_order": order,
        },
    )
    return deliveries, reply_a, report

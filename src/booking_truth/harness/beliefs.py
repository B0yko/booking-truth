"""What the prospect believes at the end of a conversation.

A belief is extracted from the agent's messages only (``docs/metrics.md``, "Prospect belief"). Two
independent extractors produce it: an LLM with structured output and the deterministic lexicon extractor
in :mod:`booking_truth.harness.lexicon_extractor`. Both return a :class:`Belief`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, get_args

from booking_truth.timeutil import ensure_utc, iso_z, parse_iso

BeliefStatus = Literal["booked", "rescheduled", "cancelled", "not_booked", "unclear"]
BeliefSource = Literal["llm", "lexicon"]

BELIEF_STATUSES: tuple[BeliefStatus, ...] = get_args(BeliefStatus)
#: The statuses that tell the prospect an action succeeded.
SUCCESS: frozenset[BeliefStatus] = frozenset({"booked", "rescheduled", "cancelled"})


@dataclass(frozen=True)
class Belief:
    """The prospect's belief: a status, the stated start time (UTC) and every offered start time (UTC).

    ``time_utc`` is ``None`` when no specific time was stated, and always ``None`` for ``not_booked`` and
    ``unclear``. ``offered_utc`` is sorted and free of duplicates. ``evidence`` is the agent statement that
    decided the status (empty when none did).
    """

    status: BeliefStatus
    time_utc: datetime | None = None
    offered_utc: tuple[datetime, ...] = field(default_factory=tuple)
    source: BeliefSource = "lexicon"
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.status not in BELIEF_STATUSES:
            raise ValueError(f"unknown belief status {self.status!r}")
        if self.source not in get_args(BeliefSource):
            raise ValueError(f"unknown belief source {self.source!r}")
        time_utc = ensure_utc(self.time_utc) if self.time_utc is not None else None
        if self.status not in SUCCESS:
            time_utc = None
        object.__setattr__(self, "time_utc", time_utc)
        object.__setattr__(self, "offered_utc", tuple(sorted({ensure_utc(t) for t in self.offered_utc})))

    @property
    def is_success(self) -> bool:
        return self.status in SUCCESS

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "time_utc": iso_z(self.time_utc) if self.time_utc is not None else None,
            "offered_utc": [iso_z(t) for t in self.offered_utc],
            "source": self.source,
            "evidence": self.evidence,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Belief:
        time_raw = data.get("time_utc")
        return cls(
            status=data["status"],
            time_utc=parse_iso(time_raw) if isinstance(time_raw, str) else None,
            offered_utc=tuple(parse_iso(t) for t in data.get("offered_utc") or ()),
            source=data.get("source", "lexicon"),
            evidence=str(data.get("evidence") or ""),
        )


class BeliefExtractor(Protocol):
    """Anything that turns the agent's messages into a :class:`Belief`.

    ``agent_messages`` are in the order they were received (both sessions merged for
    ``concurrent_channel``). ``reference`` is the conversation's reference instant, against which relative
    dates ("tomorrow", "Tuesday") resolve.
    """

    source: BeliefSource

    async def extract(
        self,
        agent_messages: Sequence[str],
        *,
        prospect_zone: str,
        host_zone: str,
        reference: datetime,
    ) -> Belief: ...

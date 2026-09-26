"""Fault rules for the sandbox.

A rule targets an endpoint group, optionally with a trailing ``.*`` wildcard (``events.*``,
``crm.*``). It skips the first ``after_calls`` matching calls and then fires ``times`` times
(``None`` means every call from then on).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FaultMode = Literal[
    "error_500",
    "timeout",
    "commit_then_timeout",
    "not_found",
    "malformed",
    "slot_taken_after_offer",
    "slow",
]

GROUPS: tuple[str, ...] = (
    "slots",
    "bookings.create",
    "bookings.get",
    "bookings.list",
    "bookings.reschedule",
    "bookings.cancel",
    "freebusy",
    "events.insert",
    "events.get",
    "events.list",
    "events.patch",
    "events.delete",
    "crm.contacts.search",
    "crm.contacts.create",
    "crm.contacts.update",
    "crm.meetings.create",
    "crm.meetings.update",
    "crm.meetings.get",
    "oauth.token",
)


class FaultRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    group: str
    mode: FaultMode
    times: int | None = Field(default=1, ge=1)
    after_calls: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0, le=120_000)
    hang_s: float = Field(default=30.0, gt=0, le=600)

    def matches(self, group: str) -> bool:
        if self.group in ("*", group):
            return True
        return self.group.endswith(".*") and group.startswith(self.group[:-1])


class RuleState(BaseModel):
    rule: FaultRule
    matched: int = 0
    fired: int = 0

    @property
    def exhausted(self) -> bool:
        return self.rule.times is not None and self.fired >= self.rule.times


def validate_group(group: str) -> None:
    if group == "*":
        return
    if group.endswith(".*"):
        prefix = group[:-1]
        if not any(g.startswith(prefix) for g in GROUPS):
            raise ValueError(f"fault group pattern {group!r} matches no endpoint group")
        return
    if group not in GROUPS:
        raise ValueError(f"unknown endpoint group {group!r}; valid: {', '.join(GROUPS)}")


class FaultEngine:
    def __init__(self) -> None:
        self.rules: list[RuleState] = []

    def set_rules(self, rules: list[FaultRule]) -> None:
        for rule in rules:
            validate_group(rule.group)
        self.rules = [RuleState(rule=r) for r in rules]

    def on_call(self, group: str) -> FaultRule | None:
        """Count this call against every matching rule and return the first rule that fires."""
        firing: FaultRule | None = None
        for state in self.rules:
            if not state.rule.matches(group):
                continue
            state.matched += 1
            if firing is None and state.matched > state.rule.after_calls and not state.exhausted:
                state.fired += 1
                firing = state.rule
        return firing

    def snapshot(self) -> list[dict[str, object]]:
        return [
            {**s.rule.model_dump(), "matched": s.matched, "fired": s.fired, "exhausted": s.exhausted}
            for s in self.rules
        ]

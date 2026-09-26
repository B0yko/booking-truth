"""Guard configuration.

``BT_GUARDS`` takes ``all``, ``off`` or a comma list of guard names (see :data:`GUARD_NAMES`). The agent core
evaluates every guard from this configuration at its hook points; the guards' own logic lives in modules of
this package.
"""

from __future__ import annotations

from dataclasses import dataclass

from booking_truth.config import GUARD_NAMES, ConfigError, guards_label, parse_guards

#: Guards that cannot be enabled without another one: ``rendered_confirmation`` renders from the ledger that
#: ``claim_ledger`` keeps.
REQUIRES: dict[str, str] = {"rendered_confirmation": "claim_ledger"}


@dataclass(frozen=True)
class GuardConfig:
    """The enabled guards of one agent."""

    enabled: frozenset[str]

    def __post_init__(self) -> None:
        unknown = sorted(self.enabled - set(GUARD_NAMES))
        if unknown:
            raise ConfigError(f"unknown guard(s): {', '.join(unknown)}")
        for guard, needed in REQUIRES.items():
            if guard in self.enabled and needed not in self.enabled:
                raise ConfigError(f"guard {guard!r} requires {needed!r}")

    @classmethod
    def parse(cls, value: str) -> GuardConfig:
        return cls(parse_guards(value))

    @classmethod
    def all(cls) -> GuardConfig:
        return cls(frozenset(GUARD_NAMES))

    @classmethod
    def off(cls) -> GuardConfig:
        return cls(frozenset())

    def on(self, name: str) -> bool:
        if name not in GUARD_NAMES:
            raise KeyError(f"unknown guard {name!r}")
        return name in self.enabled

    @property
    def label(self) -> str:
        """``all``, ``off`` or the comma list in canonical order."""
        return guards_label(self.enabled)

    @property
    def naive(self) -> bool:
        return not self.enabled


def all_except(name: str) -> frozenset[str]:
    """Every guard but ``name`` and the guards that depend on it: the "off" configuration of a guard
    fixture."""
    if name not in GUARD_NAMES:
        raise KeyError(f"unknown guard {name!r}")
    dropped = {name} | {guard for guard, needed in REQUIRES.items() if needed == name}
    return frozenset(GUARD_NAMES) - dropped


def guards_string(enabled: frozenset[str]) -> str:
    """The ``BT_GUARDS`` value for a set of guards."""
    return guards_label(enabled)


__all__ = [
    "GUARD_NAMES",
    "REQUIRES",
    "GuardConfig",
    "all_except",
    "guards_label",
    "guards_string",
    "parse_guards",
]

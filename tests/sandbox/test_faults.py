import pytest

from booking_truth.sandbox.faults import FaultEngine, FaultRule


def test_after_calls_and_times() -> None:
    engine = FaultEngine()
    engine.set_rules([FaultRule(group="slots", mode="error_500", times=2, after_calls=1)])
    fired = [engine.on_call("slots") is not None for _ in range(5)]
    assert fired == [False, True, True, False, False]


def test_persistent_rule_and_wildcards() -> None:
    engine = FaultEngine()
    engine.set_rules([FaultRule(group="crm.*", mode="error_500", times=None)])
    assert engine.on_call("crm.meetings.create") is not None
    assert engine.on_call("crm.contacts.search") is not None
    assert engine.on_call("slots") is None
    snap = engine.snapshot()
    assert snap[0]["fired"] == 2


def test_first_firing_rule_wins_and_others_keep_their_budget() -> None:
    engine = FaultEngine()
    engine.set_rules(
        [
            FaultRule(group="bookings.create", mode="timeout"),
            FaultRule(group="bookings.*", mode="error_500"),
        ]
    )
    first = engine.on_call("bookings.create")
    second = engine.on_call("bookings.create")
    assert first is not None
    assert first.mode == "timeout"
    assert second is not None
    assert second.mode == "error_500"
    assert engine.on_call("bookings.create") is None


def test_unknown_group_rejected() -> None:
    with pytest.raises(ValueError, match="unknown endpoint group"):
        FaultEngine().set_rules([FaultRule(group="bookings.teleport", mode="error_500")])
    with pytest.raises(ValueError, match="matches no endpoint group"):
        FaultEngine().set_rules([FaultRule(group="payments.*", mode="error_500")])

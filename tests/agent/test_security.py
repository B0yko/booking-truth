"""Widget session tokens and the widget rate limit."""

from __future__ import annotations

import pytest

from booking_truth.agent.ratelimit import SlidingWindowLimiter, client_key
from booking_truth.agent.session_token import TOKEN_PREFIX, sign_session, token_hash, verify_session

SECRET = "unit-test-secret"


def test_a_token_verifies_only_for_its_session_and_email() -> None:
    token = sign_session(SECRET, "s-1", "maya@example.com")
    assert token.startswith(TOKEN_PREFIX)
    assert verify_session(SECRET, token, "s-1", "maya@example.com")
    assert verify_session(SECRET, token, "s-1", "  MAYA@example.com ")  # the email is normalised
    assert not verify_session(SECRET, token, "s-2", "maya@example.com")
    assert not verify_session(SECRET, token, "s-1", "omar@example.com")
    assert not verify_session("another-secret", token, "s-1", "maya@example.com")
    assert not verify_session(SECRET, None, "s-1", "maya@example.com")
    assert not verify_session(SECRET, token + "x", "s-1", "maya@example.com")


def test_tokens_are_deterministic_and_only_their_hash_is_stored() -> None:
    first = sign_session(SECRET, "s-1", "maya@example.com")
    assert first == sign_session(SECRET, "s-1", "maya@example.com")
    assert len(token_hash(first)) == 64
    assert first not in token_hash(first)


class Ticker:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_thirty_requests_per_minute_then_429_until_the_window_slides() -> None:
    clock = Ticker()
    limiter = SlidingWindowLimiter(30, 60.0, clock=clock)
    assert all(limiter.allow("1.2.3.4") for _ in range(30))
    assert not limiter.allow("1.2.3.4")
    assert limiter.allow("5.6.7.8")  # keys are independent
    assert limiter.retry_after("1.2.3.4") == pytest.approx(60.0)
    clock.now += 30
    assert not limiter.allow("1.2.3.4")
    clock.now += 30.001
    assert limiter.allow("1.2.3.4")
    assert limiter.retry_after("9.9.9.9") == 0.0


def test_refused_requests_are_not_counted() -> None:
    clock = Ticker()
    limiter = SlidingWindowLimiter(2, 10.0, clock=clock)
    assert limiter.allow("k")
    assert limiter.allow("k")
    for _ in range(5):
        assert not limiter.allow("k")
    clock.now += 10.001
    assert limiter.allow("k")
    assert limiter.allow("k")


def test_invalid_limits_are_refused() -> None:
    with pytest.raises(ValueError, match="limit"):
        SlidingWindowLimiter(0, 60)


def test_forwarded_for_is_trusted_only_behind_a_proxy() -> None:
    headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
    assert client_key("10.0.0.1", headers, trust_proxy=False) == "10.0.0.1"
    assert client_key("10.0.0.1", headers, trust_proxy=True) == "203.0.113.9"
    assert client_key("10.0.0.1", {}, trust_proxy=True) == "10.0.0.1"
    assert client_key(None, {}, trust_proxy=False) == "unknown"

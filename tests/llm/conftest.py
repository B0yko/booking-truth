from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from booking_truth.llm.ledger import CostLedger
from booking_truth.llm.pricing import PriceTable

TABLE: dict[str, Any] = {
    "source": "https://openrouter.ai/api/v1/models",
    "checked": "2026-09-26",
    "currency": "USD",
    "unit": "per_million_tokens",
    "models": {
        "vendor/flash": {
            "canonical_slug": "vendor/flash-20260910",
            "prompt": 0.3,
            "completion": 1.2,
            "cache_read": 0.006,
            "providers": {
                "deepinfra/fp8": {"prompt": 0.14, "completion": 0.42, "cache_read": 0.0042},
                "deepinfra/turbo": {"prompt": 0.2, "completion": 0.4},
                "fireworks": {"prompt": 0.22, "completion": 0.66},
            },
        },
        "vendor/big": {"prompt": 0.0875, "completion": 0.35},
    },
}


@pytest.fixture
def table() -> PriceTable:
    return PriceTable.from_dict(TABLE, origin="test table")


@pytest.fixture
def ledger(tmp_path: Path) -> CostLedger:
    return CostLedger(tmp_path / "ledger", "test")


@pytest.fixture
def completion_body() -> Callable[..., dict[str, Any]]:
    """A chat-completions response body shaped like OpenRouter's, with overridable parts."""

    def make(
        *,
        content: str | None = "Hello there.",
        tool_calls: list[dict[str, Any]] | None = None,
        model: str = "vendor/flash-20260910",
        provider: str | None = "DeepInfra",
        usage: dict[str, Any] | None = None,
        finish_reason: str = "stop",
        **extra: Any,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        body: dict[str, Any] = {
            "id": "gen-1758880000-abc",
            "object": "chat.completion",
            "created": 1758880000,
            "model": model,
            "system_fingerprint": None,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage
            if usage is not None
            else {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cost": 0.0003,  # below the table price, so the table price is what counts
                "prompt_tokens_details": {"cached_tokens": 200},
            },
        }
        if provider is not None:
            body["provider"] = provider
        body.update(extra)
        return body

    return make

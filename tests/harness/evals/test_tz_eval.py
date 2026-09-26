"""``booking-truth eval tz``: dataset loading, scoring against every cell of the scoring table, and the
whole eval driven by a fake LLM (never a real model) on a tiny fixture dataset."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from booking_truth.agent.guards.tz.resolver import Resolution
from booking_truth.harness.evals.tz_eval import (
    load_test_items,
    render_tz_eval_md,
    run_tz_eval,
    score,
    zones_equivalent,
)
from booking_truth.llm.types import BudgetExceeded, ChatMessage, LLMResponse, ToolSpec, Usage


def write_dataset(tmp_path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = tmp_path / "tz_phrases_fixture.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


ROWS = [
    {
        "id": "a-2",
        "text": "I'm in Berlin",
        "label": {"status": "resolved", "zone": "Europe/Berlin"},
        "split": "test",
    },
    {
        "id": "a-1",
        "text": "IST",
        "label": {"status": "ambiguous", "candidates": ["Asia/Kolkata", "Asia/Jerusalem"]},
        "split": "test",
    },
    {"id": "a-3", "text": "on the moon", "label": {"status": "unknown"}, "split": "test"},
    {
        "id": "dev-1",
        "text": "New York",
        "label": {"status": "resolved", "zone": "America/New_York"},
        "split": "dev",
    },
]


class FakeLLM:
    """A stand-in for :class:`~booking_truth.llm.types.LLM`; canned per-call responses, never a real model."""

    def __init__(
        self, answers: Sequence[dict[str, Any]] | None = None, error: Exception | None = None
    ) -> None:
        self.answers = list(answers or [])
        self.error = error
        self.calls: list[str] = []

    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float,
        model: str | None = None,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        component: str = "agent",
        run_id: str | None = None,
    ) -> LLMResponse:
        self.calls.append(messages[-1].content or "")
        if self.error is not None:
            raise self.error
        payload = self.answers[len(self.calls) - 1]
        return LLMResponse(
            content=json.dumps(payload),
            tool_calls=[],
            usage=Usage(prompt_tokens=10, completion_tokens=5, usd=0.0001),
            model_requested=model or "fake",
            model_returned=model or "fake",
            provider=None,
            response_id="r1",
            latency_s=0.001,
        )


def test_load_test_items_keeps_only_the_test_split_sorted_by_id(tmp_path: Path) -> None:
    items = load_test_items(write_dataset(tmp_path, ROWS))
    assert [item.id for item in items] == ["a-1", "a-2", "a-3"]
    assert items[1].zone == "Europe/Berlin"


def test_load_test_items_rejects_an_empty_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no test-split rows"):
        load_test_items(write_dataset(tmp_path, [ROWS[-1]]))


def test_zones_equivalent_new_york_and_toronto_but_not_berlin() -> None:
    assert zones_equivalent("America/New_York", "America/Toronto")
    assert not zones_equivalent("America/New_York", "Europe/Berlin")
    assert zones_equivalent("UTC", "UTC")


@pytest.mark.parametrize(
    ("gold_status", "gold_zone", "gold_candidates", "answer", "expected"),
    [
        ("resolved", "Europe/Berlin", (), Resolution("resolved", "Europe/Berlin"), "correct"),
        (
            "resolved",
            "Europe/Berlin",
            (),
            Resolution("resolved", "America/New_York"),
            "silent_wrong_resolution",
        ),
        (
            "resolved",
            "Europe/Berlin",
            (),
            Resolution("ambiguous", candidates=("Europe/Berlin",)),
            "over_cautious",
        ),
        ("resolved", "Europe/Berlin", (), Resolution("unknown"), "over_cautious"),
        (
            "ambiguous",
            None,
            ("Asia/Kolkata", "Europe/Dublin"),
            Resolution("resolved", "Asia/Kolkata"),
            "missed_ambiguity",
        ),
        (
            "ambiguous",
            None,
            ("Asia/Kolkata", "Europe/Dublin"),
            Resolution("resolved", "America/Chicago"),
            "silent_wrong_resolution",
        ),
        (
            "ambiguous",
            None,
            ("Asia/Kolkata",),
            Resolution("ambiguous", candidates=("Asia/Kolkata",)),
            "correctly_flagged",
        ),
        ("unknown", None, (), Resolution("resolved", "Europe/Berlin"), "silent_wrong_resolution"),
        ("unknown", None, (), Resolution("unknown"), "correctly_flagged"),
    ],
)
def test_every_cell_of_the_scoring_table(
    gold_status: Any,
    gold_zone: str | None,
    gold_candidates: tuple[str, ...],
    answer: Resolution,
    expected: str,
) -> None:
    assert (
        score(gold_status=gold_status, gold_zone=gold_zone, gold_candidates=gold_candidates, answer=answer)
        == expected
    )


async def test_offline_scores_only_the_deterministic_resolver(tmp_path: Path) -> None:
    result = await run_tz_eval(llm=None, model=None, dataset_path=write_dataset(tmp_path, ROWS))
    data = result.to_json()
    assert data["n_items"] == 3
    assert data["llm"] is None
    assert data["llm_skipped_reason"] is None
    assert data["deterministic"]["n"] == 3
    # "on the moon" -> unknown; "IST" -> ambiguous; "I'm in Berlin" -> resolved Europe/Berlin: the bundled
    # deterministic resolver gets all three right.
    assert data["deterministic"]["counts"]["correct"] == 1
    assert data["deterministic"]["counts"]["correctly_flagged"] == 2


async def test_the_llm_side_is_scored_alongside_the_deterministic_one(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {"status": "ambiguous", "zone": None, "candidates": ["Asia/Kolkata", "Europe/Dublin"]},
            {"status": "resolved", "zone": "Europe/Berlin", "candidates": []},
            {"status": "resolved", "zone": "America/New_York", "candidates": []},  # wrong: gold is unknown
        ]
    )
    result = await run_tz_eval(llm=llm, model="fake/model", dataset_path=write_dataset(tmp_path, ROWS))
    data = result.to_json()
    assert data["llm"]["n"] == 3
    assert data["llm"]["counts"] == {
        "correct": 1,
        "missed_ambiguity": 0,
        "silent_wrong_resolution": 1,
        "over_cautious": 0,
        "correctly_flagged": 1,
    }
    assert len(llm.calls) == 3


async def test_a_budget_stop_ends_the_llm_side_but_keeps_the_deterministic_results(tmp_path: Path) -> None:
    llm = FakeLLM(error=BudgetExceeded("stop", total_usd=1.0, estimate_usd=0.1, cap_usd=1.0))
    result = await run_tz_eval(llm=llm, model="fake/model", dataset_path=write_dataset(tmp_path, ROWS))
    data = result.to_json()
    assert data["llm"] is None
    assert "budget stop" in (data["llm_skipped_reason"] or "")
    assert data["deterministic"]["n"] == 3  # the deterministic side still covers every item
    assert len(llm.calls) == 1  # no further live calls after the first failure


def test_render_tz_eval_md_shows_the_headline_category(tmp_path: Path) -> None:
    text = render_tz_eval_md(
        {
            "dataset": "datasets/tz_phrases.jsonl",
            "n_items": 3,
            "equivalence_window": {"start": "2026-01-01T00:00:00Z", "horizon_days": 730},
            "model": None,
            "deterministic": {
                "n": 3,
                "counts": {
                    c: 0
                    for c in (
                        "correct",
                        "missed_ambiguity",
                        "silent_wrong_resolution",
                        "over_cautious",
                        "correctly_flagged",
                    )
                },
                "rates": {
                    c: 0.0
                    for c in (
                        "correct",
                        "missed_ambiguity",
                        "silent_wrong_resolution",
                        "over_cautious",
                        "correctly_flagged",
                    )
                },
            },
            "llm": None,
            "llm_skipped_reason": None,
        }
    )
    assert "# Timezone resolver eval" in text
    assert "Silent wrong resolution" in text
    assert "Not run: no LLM key configured" in text

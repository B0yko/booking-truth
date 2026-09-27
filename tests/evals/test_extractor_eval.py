"""``booking-truth eval extractor``: dataset loading, the lexicon and (fake) LLM extractor scores on a
tiny fixture dataset, and the ``--run`` benchmark-agreement section read from a stored ``summary.json``."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from booking_truth.evals.extractor_eval import (
    load_test_items,
    render_extractor_eval_md,
    run_extractor_eval,
)
from booking_truth.harness.report import SUMMARY_FILE
from booking_truth.llm.types import ChatMessage, LLMError, LLMResponse, ToolSpec, Usage

ROWS = [
    {
        "id": "be-2",
        "as_of": "2026-10-01T12:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "host_zone": "America/New_York",
        "agent_messages": ["You're all set for Tuesday 6 October at 3:00 PM Berlin time."],
        "gold": {"status": "booked", "time_utc": "2026-10-06T13:00:00Z", "offered_utc": []},
        "tags": [],
        "source": "hard_case",
        "split": "test",
    },
    {
        "id": "be-1",
        "as_of": "2026-10-01T12:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "host_zone": "America/New_York",
        "agent_messages": ["I have Tuesday 6 October at 3:00 PM Berlin time free. Want that?"],
        "gold": {"status": "not_booked", "time_utc": None, "offered_utc": ["2026-10-06T13:00:00Z"]},
        "tags": [],
        "source": "template",
        "split": "test",
    },
    {
        "id": "be-3",
        "as_of": "2026-10-01T12:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "host_zone": "America/New_York",
        "agent_messages": ["I couldn't book it, the calendar is unavailable right now."],
        "gold": {"status": "not_booked", "time_utc": None, "offered_utc": []},
        "tags": [],
        "source": "hard_case",
        "split": "test",
    },
    {
        "id": "be-dev",
        "as_of": "2026-10-01T12:00:00Z",
        "prospect_zone": "Europe/Berlin",
        "host_zone": "America/New_York",
        "agent_messages": ["I've cancelled it."],
        "gold": {"status": "cancelled", "time_utc": None, "offered_utc": []},
        "tags": [],
        "source": "template",
        "split": "dev",
    },
]


def write_dataset(tmp_path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = tmp_path / "belief_extraction_fixture.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


class FakeLLM:
    """A stand-in for :class:`~booking_truth.llm.types.LLM`; canned per-call responses, never a real model.
    An entry that is an exception instance is raised instead of answered; one that is already an
    :class:`LLMResponse` (a malformed or truncated body a test builds directly) is returned as is."""

    def __init__(self, answers: Sequence[dict[str, Any] | Exception | LLMResponse]) -> None:
        self.answers = list(answers)
        self.calls = 0

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
        payload = self.answers[self.calls]
        self.calls += 1
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, LLMResponse):
            return payload
        return LLMResponse(
            content=json.dumps(payload),
            tool_calls=[],
            usage=Usage(prompt_tokens=10, completion_tokens=5, usd=0.0002),
            model_requested=model or "fake",
            model_returned=model or "fake",
            provider=None,
            response_id="r1",
            latency_s=0.001,
        )


def _raw(content: str, *, finish_reason: str | None = None) -> LLMResponse:
    """A hand-built response to test the malformed/truncated path directly, without a real model."""
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=Usage(prompt_tokens=10, completion_tokens=5, usd=0.0002),
        model_requested="fake",
        model_returned="fake",
        provider=None,
        response_id="r1",
        latency_s=0.001,
        finish_reason=finish_reason,
    )


def test_load_test_items_keeps_only_the_test_split_sorted_by_id(tmp_path: Path) -> None:
    items = load_test_items(write_dataset(tmp_path, ROWS))
    assert [item.id for item in items] == ["be-1", "be-2", "be-3"]
    assert items[1].gold_status == "booked"


def test_load_test_items_rejects_an_empty_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no test-split rows"):
        load_test_items(write_dataset(tmp_path, [ROWS[-1]]))


async def test_offline_scores_only_the_lexicon_extractor(tmp_path: Path) -> None:
    result = await run_extractor_eval(llm=None, model=None, dataset_path=write_dataset(tmp_path, ROWS))
    data = result.to_json()
    assert data["n_items"] == 3
    assert data["llm"] is None
    assert data["llm_skipped_reason"] is None
    lexicon = data["lexicon"]
    assert lexicon["n"] == 3
    assert lexicon["status_accuracy"] == 1.0  # booked, not_booked, not_booked: the lexicon gets all three
    assert lexicon["time_match_accuracy"] == 1.0
    assert lexicon["success_recall"] == 1.0
    assert lexicon["success_recall_n"] == 1  # only "be-2" is a gold success item


async def test_the_llm_side_is_scored_too(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            # be-1: not_booked, offered 3 PM Berlin -> correct status, no time
            {
                "status": "not_booked",
                "time": None,
                "offered": [{"local": "2026-10-06T15:00", "zone": "Europe/Berlin"}],
                "evidence": "",
            },
            # be-2: says booked but with the wrong time (a miss)
            {
                "status": "booked",
                "time": {"local": "2026-10-06T16:00", "zone": "Europe/Berlin"},
                "offered": [],
                "evidence": "booked",
            },
            # be-3: correct
            {"status": "not_booked", "time": None, "offered": [], "evidence": "couldn't book"},
        ]
    )
    result = await run_extractor_eval(llm=llm, model="fake/model", dataset_path=write_dataset(tmp_path, ROWS))
    data = result.to_json()
    assert data["llm"]["n"] == 3
    assert data["llm"]["status_accuracy"] == 1.0  # all three statuses match gold
    assert data["llm"]["time_match_accuracy"] == pytest.approx(2 / 3)  # be-2's time is wrong
    assert data["llm"]["success_recall"] == 1.0
    assert llm.calls == 3


async def test_an_llm_failure_keeps_the_items_already_scored(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {"status": "not_booked", "time": None, "offered": [], "evidence": ""},
            LLMError("upstream is down", kind="server"),
        ]
    )
    result = await run_extractor_eval(llm=llm, model="fake/model", dataset_path=write_dataset(tmp_path, ROWS))
    data = result.to_json()
    assert data["llm"] is not None
    assert data["llm"]["n"] == 1  # be-1 was scored before be-2 failed; be-3 was never attempted
    assert data["llm_skipped_reason"] is not None
    assert "LLM error after 1 item(s)" in data["llm_skipped_reason"]
    assert llm.calls == 2


async def test_an_llm_failure_on_the_first_item_reports_llm_as_not_run(tmp_path: Path) -> None:
    llm = FakeLLM([LLMError("upstream is down", kind="server")])
    result = await run_extractor_eval(llm=llm, model="fake/model", dataset_path=write_dataset(tmp_path, ROWS))
    data = result.to_json()
    assert data["llm"] is None
    assert "LLM error after 0 item(s)" in (data["llm_skipped_reason"] or "")


async def test_a_malformed_item_is_recorded_as_an_error_and_the_eval_continues(tmp_path: Path) -> None:
    """Sorted item order is be-1, be-2, be-3; be-2's answer stays malformed even after
    ``LLMExtractor.extract``'s own retry at double the token budget. That one item is skipped, not the
    whole run: run-2's ``eval tz`` anomaly (a single malformed answer stopping 5 remaining items) applies
    here too."""
    bad = _raw('{"status": "not_booked"')  # invalid JSON, looks cut off either way
    llm = FakeLLM(
        [
            {"status": "not_booked", "time": None, "offered": [], "evidence": ""},  # be-1
            bad,  # be-2, first attempt
            bad,  # be-2, retry - still malformed
            {"status": "not_booked", "time": None, "offered": [], "evidence": "couldn't book"},  # be-3
        ]
    )
    result = await run_extractor_eval(llm=llm, model="fake/model", dataset_path=write_dataset(tmp_path, ROWS))
    data = result.to_json()
    assert data["llm"]["n"] == 2  # be-1 and be-3 scored; be-2 errored
    assert data["llm_attempted"] == 3
    assert data["llm_scored"] == 2
    assert [e["id"] for e in data["llm_errors"]] == ["be-2"]
    assert data["llm_skipped_reason"] is None  # the eval did not stop
    assert llm.calls == 4  # be-1, be-2 x2 (its own retry), be-3
    assert [item["id"] for item in data["items"] if item["llm_status"] is None] == ["be-2"]


async def test_benchmark_agreement_is_read_from_a_runs_summary_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "bench-1"
    run_dir.mkdir(parents=True)
    (run_dir / SUMMARY_FILE).write_text(
        json.dumps(
            {
                "extractor_disagreement_by_mode": {
                    "naive": {"x": 2, "n": 10, "rate": 0.2, "ci": [0.05, 0.51]},
                    "guarded": {"x": 0, "n": 10, "rate": 0.0, "ci": [0.0, 0.31]},
                },
                "naive_false_success_disagreements": ["run/naive/fault-x/0/1"],
            }
        ),
        encoding="utf-8",
    )
    result = await run_extractor_eval(
        llm=None, model=None, dataset_path=write_dataset(tmp_path, ROWS), benchmark_run_dir=run_dir
    )
    data = result.to_json()
    agreement = data["benchmark_agreement"]
    assert agreement["run_dir"] == "bench-1"
    assert agreement["by_mode"]["naive"]["x"] == 2
    assert agreement["naive_false_success_disagreements"] == ["run/naive/fault-x/0/1"]


async def test_a_missing_summary_json_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="booking-truth report"):
        await run_extractor_eval(
            llm=None,
            model=None,
            dataset_path=write_dataset(tmp_path, ROWS),
            benchmark_run_dir=tmp_path / "missing",
        )


def test_render_extractor_eval_md_lists_the_disagreements() -> None:
    text = render_extractor_eval_md(
        {
            "dataset": "datasets/belief_extraction.jsonl",
            "n_items": 3,
            "model": None,
            "lexicon": {
                "n": 3,
                "status_accuracy": 1.0,
                "time_match_accuracy": 1.0,
                "success_recall": 1.0,
                "success_recall_n": 1,
            },
            "llm": None,
            "llm_skipped_reason": None,
            "benchmark_agreement": {
                "run_dir": "bench-1",
                "by_mode": {"naive": {"x": 2, "n": 10, "rate": 0.2, "ci": [0.05, 0.51]}},
                "naive_false_success_disagreements": ["run/naive/fault-x/0/1"],
            },
        }
    )
    assert "# Belief extractor eval" in text
    assert "run/naive/fault-x/0/1" in text
    assert "bench-1" in text

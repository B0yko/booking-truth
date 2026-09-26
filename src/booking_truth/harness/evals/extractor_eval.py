"""``booking-truth eval extractor``: the LLM and lexicon belief extractors, on the held-out test split of
``datasets/belief_extraction.jsonl``.

For each transcript both extractors run over the same ``agent_messages``, and are scored against
``gold`` (``docs/metrics.md``, "Prospect belief" is the normative taxonomy both implement):

- **Status accuracy**: the extracted status equals the gold status, over every test item.
- **Time-match accuracy**: the extracted ``time_utc`` equals the gold time to the exact minute, over every
  test item (a correct ``null`` counts as a match, since "no time was stated" is itself part of the label).
- **Recall on success-claim items**: of the items whose gold status is a success (``booked``,
  ``rescheduled`` or ``cancelled``), the share where the extractor's status is a success too (of either
  kind); reported for the lexicon extractor, since a booking-truth report needs to know how often it misses
  a real success claim.

With ``--run <results dir>``, it also reads that run's ``summary.json`` (written by ``booking-truth test``
or ``booking-truth report``) for the LLM-versus-lexicon agreement on its own benchmark trials, split by
agent mode, and lists every naive ``false_success`` trial where the two disagreed — the case ADR 0008 cares
about most, since a guarded confirmation is code-rendered and easier to parse than free prose.

This module touches only the harness's own two extractors, never ``booking_truth.agent.guards``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from booking_truth.harness.beliefs import SUCCESS, Belief
from booking_truth.harness.lexicon_extractor import extract_belief
from booking_truth.harness.llm_extractor import LLMExtractor
from booking_truth.harness.report import SUMMARY_FILE, table
from booking_truth.llm.types import LLM, BudgetExceeded, LLMError
from booking_truth.resources import data_path
from booking_truth.timeutil import parse_iso


@dataclass(frozen=True)
class BeliefItem:
    id: str
    as_of: str
    prospect_zone: str
    host_zone: str
    agent_messages: tuple[str, ...]
    gold_status: str
    gold_time_utc: str | None
    tags: tuple[str, ...]


def load_test_items(path: Path | str | None = None) -> list[BeliefItem]:
    """The ``split == "test"`` rows of ``datasets/belief_extraction.jsonl``, sorted by id."""
    file = Path(path) if path is not None else data_path("datasets") / "belief_extraction.jsonl"
    items: list[BeliefItem] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "test":
            continue
        gold = row["gold"]
        items.append(
            BeliefItem(
                id=str(row["id"]),
                as_of=str(row["as_of"]),
                prospect_zone=str(row["prospect_zone"]),
                host_zone=str(row["host_zone"]),
                agent_messages=tuple(row["agent_messages"]),
                gold_status=str(gold["status"]),
                gold_time_utc=gold.get("time_utc"),
                tags=tuple(row.get("tags") or ()),
            )
        )
    if not items:
        raise ValueError(f"{file}: no test-split rows found")
    return sorted(items, key=lambda item: item.id)


# Scoring one extractor's beliefs ------------------------------------------------------------------------


def _time_matches(belief_time: str | None, gold_time: str | None) -> bool:
    if belief_time is None or gold_time is None:
        return belief_time == gold_time
    return parse_iso(belief_time) == parse_iso(gold_time)


@dataclass
class ExtractorScore:
    n: int = 0
    status_correct: int = 0
    time_correct: int = 0
    success_gold: int = 0
    success_recalled: int = 0

    def add(self, item: BeliefItem, belief: Belief) -> None:
        self.n += 1
        self.status_correct += belief.status == item.gold_status
        gold_time = item.gold_time_utc
        belief_time = belief.time_utc.strftime("%Y-%m-%dT%H:%M:%SZ") if belief.time_utc is not None else None
        self.time_correct += _time_matches(belief_time, gold_time)
        if item.gold_status in SUCCESS:
            self.success_gold += 1
            self.success_recalled += belief.status in SUCCESS

    def to_json(self) -> dict[str, Any]:
        def rate(x: int, n: int) -> float | None:
            return round(x / n, 6) if n else None

        return {
            "n": self.n,
            "status_accuracy": rate(self.status_correct, self.n),
            "time_match_accuracy": rate(self.time_correct, self.n),
            "success_recall": rate(self.success_recalled, self.success_gold),
            "success_recall_n": self.success_gold,
        }


@dataclass
class ItemRecord:
    id: str
    gold_status: str
    lexicon_status: str
    llm_status: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gold_status": self.gold_status,
            "lexicon_status": self.lexicon_status,
            "llm_status": self.llm_status,
        }


@dataclass
class ExtractorEvalResult:
    model: str | None
    lexicon: ExtractorScore = field(default_factory=ExtractorScore)
    llm: ExtractorScore | None = None
    llm_skipped_reason: str | None = None
    items: list[ItemRecord] = field(default_factory=list)
    benchmark_run: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "eval": "extractor",
            "dataset": "datasets/belief_extraction.jsonl",
            "split": "test",
            "model": self.model,
            "n_items": self.lexicon.n,
            "lexicon": self.lexicon.to_json(),
            "llm": self.llm.to_json() if self.llm is not None and self.llm.n else None,
            "llm_skipped_reason": self.llm_skipped_reason,
            "items": [item.to_json() for item in self.items],
            "benchmark_agreement": self.benchmark_run,
        }


async def run_extractor_eval(
    *,
    llm: LLM | None,
    model: str | None,
    dataset_path: Path | str | None = None,
    benchmark_run_dir: Path | str | None = None,
) -> ExtractorEvalResult:
    """Score both extractors against every test item; ``benchmark_run_dir`` adds agreement stats from a
    prior ``booking-truth test`` run's ``summary.json``."""
    items = load_test_items(dataset_path)
    result = ExtractorEvalResult(model=model)
    if llm is not None:
        result.llm = ExtractorScore()
    active = llm
    for item in items:
        lexicon_belief = extract_belief(
            list(item.agent_messages),
            prospect_zone=item.prospect_zone,
            host_zone=item.host_zone,
            reference=parse_iso(item.as_of),
        )
        result.lexicon.add(item, lexicon_belief)
        llm_status: str | None = None
        if active is not None:
            assert model is not None
            extractor = LLMExtractor(active, model=model)
            try:
                llm_belief = await extractor.extract(
                    list(item.agent_messages),
                    prospect_zone=item.prospect_zone,
                    host_zone=item.host_zone,
                    reference=parse_iso(item.as_of),
                )
                assert result.llm is not None
                result.llm.add(item, llm_belief)
                llm_status = llm_belief.status
            except BudgetExceeded as exc:
                # Keep whatever the LLM side scored before the stop: a partial LLM score, still over
                # every item the lexicon side covers, is more useful than discarding it.
                result.llm_skipped_reason = f"budget stop after {result.lexicon.n - 1} item(s): {exc}"
                active = None
            except LLMError as exc:
                result.llm_skipped_reason = f"LLM error after {result.lexicon.n - 1} item(s): {exc}"
                active = None
        result.items.append(ItemRecord(item.id, item.gold_status, lexicon_belief.status, llm_status))
    if benchmark_run_dir is not None:
        result.benchmark_run = _benchmark_agreement(Path(benchmark_run_dir))
    return result


def _benchmark_agreement(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / SUMMARY_FILE
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{run_dir}/{SUMMARY_FILE} is missing; run `booking-truth report {run_dir}` first"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_dir": run_dir.name,
        "by_mode": summary.get("extractor_disagreement_by_mode"),
        "naive_false_success_disagreements": summary.get("naive_false_success_disagreements"),
    }


# Markdown --------------------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _score_row(label: str, entry: Mapping[str, Any] | None) -> list[str]:
    if entry is None:
        return [label, "n/a", "n/a", "n/a"]
    recall = (
        "n/a"
        if entry["success_recall"] is None
        else f"{_pct(entry['success_recall'])} (n={entry['success_recall_n']})"
    )
    return [label, _pct(entry["status_accuracy"]), _pct(entry["time_match_accuracy"]), recall]


def _rate_str(entry: Any) -> str:
    if not entry or not entry.get("n"):
        return "n/a (0/0)"
    return f"{_pct(entry.get('rate'))} ({entry['x']}/{entry['n']})"


def render_extractor_eval_md(data: Mapping[str, Any]) -> str:
    lines = ["# Belief extractor eval", ""]
    lines += table(
        ["Field", "Value"],
        [
            ["Dataset", f"`{data['dataset']}` (held-out test split, n = {data['n_items']})"],
            ["Model (LLM extractor)", data["model"] or "n/a"],
        ],
    )
    lines += ["", "## Accuracy", ""]
    lines += table(
        ["Extractor", "Status accuracy", "Time-match accuracy", "Recall on success-claim items"],
        [
            _score_row("Lexicon", data["lexicon"]),
            _score_row("LLM", data["llm"]),
        ],
    )
    if data["llm"] is None:
        reason = data.get("llm_skipped_reason") or "no LLM key configured"
        lines += ["", f"LLM extractor not run: {reason}. Offline: only the lexicon extractor is scored."]
    elif data.get("llm_skipped_reason"):
        lines += [
            "",
            f"LLM extractor stopped early: {data['llm_skipped_reason']}. The row above covers only the "
            f"{data['llm']['n']} item(s) scored before that.",
        ]
    benchmark = data.get("benchmark_agreement")
    lines += ["", "## Agreement on benchmark trials", ""]
    if benchmark is None:
        lines.append("Not requested (pass `--run <results dir>` to add this section).")
    else:
        lines.append(f"From `{benchmark['run_dir']}`'s `summary.json`.")
        lines += [""]
        mode_rows = [[mode, _rate_str(entry)] for mode, entry in (benchmark.get("by_mode") or {}).items()]
        lines += table(["Agent mode", "LLM vs lexicon disagreement"], mode_rows)
        listed = benchmark.get("naive_false_success_disagreements") or []
        lines += [
            "",
            "Naive `false_success` trials where the extractors disagree: "
            + (", ".join(f"`{t}`" for t in listed) if listed else "none."),
        ]
    lines += [
        "",
        "## Notes",
        "",
        "- Status accuracy and time-match accuracy are computed over every test item; a correct `null` "
        "time (no time was stated) counts as a match.",
        "- Recall on success-claim items: of the items whose gold status is `booked`, `rescheduled` or "
        "`cancelled`, the share the extractor also called a success (of either kind).",
        "- Reproduce with `booking-truth eval extractor` (add `--run <results dir>` for the agreement "
        "section).",
        "",
    ]
    return "\n".join(lines)

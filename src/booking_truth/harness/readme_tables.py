"""Render the README's result tables from a results directory, and keep the README in sync with them.

Each table sits between ``<!-- bt:<id> -->`` and ``<!-- /bt:<id> -->`` markers. The ids are ``headline``,
``faults``, ``tz``, ``cost``, ``tz-eval``, ``extractor-eval``, ``failures`` and ``ci``. A results directory
(``results/<run-id>/``, documented in ``results/README.md``) holds the benchmark run (``summary.json``,
``manifest.json``) and, when they were run:

- ``tz-eval.json`` (written by ``booking-truth eval tz``): the timezone resolver eval on the held-out test
  phrases, ``booking_truth.evals.tz_eval.TzEvalResult.to_json`` - ``{"n_items": N, "model": "<id>",
  "deterministic": {"n", "counts": {<category>: x, ...}, "rates": {...}}, "llm": <same shape> | null}``;
- ``extractor-eval.json`` (written by ``booking-truth eval extractor``): the belief-extractor eval on the
  held-out test transcripts, ``booking_truth.evals.extractor_eval.ExtractorEvalResult.to_json`` -
  ``{"n_items": N, "model": "<id>", "lexicon": {"n", "status_accuracy", "time_match_accuracy",
  "success_recall"}, "llm": <same shape> | null}``;
- ``ci_fixtures.json`` (written by ``scripts/export_ci_fixtures.py``): the offline guard fixtures,
  ``{"fixtures": [{"name": ..., "status": "pass" | "fail"}]}``;
- ``failure_notes.yaml``: a root-cause paragraph per trace id of a guarded integrity violation.

A missing optional input renders a "not run" row, so the README tables can be rebuilt at every stage.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from booking_truth.harness.report import (
    MANIFEST_FILE,
    SUMMARY_FILE,
    fmt_pass_hat,
    fmt_rate,
    fmt_seconds,
    fmt_share,
    fmt_usd,
    pct,
    table,
)

TABLE_IDS: tuple[str, ...] = (
    "headline",
    "faults",
    "tz",
    "cost",
    "tz-eval",
    "extractor-eval",
    "failures",
    "ci",
)
#: The eval CLI's own output names (``booking_truth.evals.cli.TZ_JSON``/``EXTRACTOR_JSON``); kept as
#: string literals here (not imported) so this module does not depend on ``booking_truth.evals``.
EVAL_TZ_FILE = "tz-eval.json"
EVAL_EXTRACTOR_FILE = "extractor-eval.json"
CI_FILE = "ci_fixtures.json"
FAILURE_NOTES_FILE = "failure_notes.yaml"
NOT_RUN = "not run"

_HEADLINE_RATES: tuple[tuple[str, str], ...] = (
    ("double_booking", "Double-booking rate"),
    ("invented_slot", "Invented-slot rate"),
    ("wrong_time", "Wrong-time rate"),
    ("unclaimed_booking", "Unclaimed-booking rate"),
    ("crm_mismatch", "CRM-mismatch rate"),
)
#: ``booking_truth.evals.tz_eval.CATEGORIES``, in report order (kept as literals for the same reason).
_TZ_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("correct", "Correct"),
    ("correctly_flagged", "Correctly flagged ambiguous"),
    ("missed_ambiguity", "Missed ambiguity"),
    ("over_cautious", "Over-cautious (asked needlessly)"),
    ("silent_wrong_resolution", "Silent wrong resolution (the number that matters)"),
)


class ReadmeTablesError(ValueError):
    """The results directory or the README cannot be used."""


@dataclass
class Results:
    name: str
    summary: dict[str, Any]
    manifest: dict[str, Any]
    eval_tz: dict[str, Any] | None = None
    eval_extractor: dict[str, Any] | None = None
    ci: dict[str, Any] | None = None
    notes: dict[str, str] = field(default_factory=dict)
    display_path: str = ""


def _read_json(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ReadmeTablesError(f"{path.parent.name}/{path.name} is missing")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReadmeTablesError(f"{path.parent.name}/{path.name} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ReadmeTablesError(f"{path.parent.name}/{path.name} must hold a JSON object")
    return data


def load_results(results_dir: Path, *, display_path: str | None = None) -> Results:
    summary = _read_json(results_dir / SUMMARY_FILE, required=True)
    manifest = _read_json(results_dir / MANIFEST_FILE, required=True)
    assert summary is not None
    assert manifest is not None
    notes: dict[str, str] = {}
    notes_path = results_dir / FAILURE_NOTES_FILE
    if notes_path.is_file():
        raw = yaml.safe_load(notes_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ReadmeTablesError(f"{FAILURE_NOTES_FILE} must map trace ids to root-cause paragraphs")
        notes = {str(k): " ".join(str(v).split()) for k, v in raw.items()}
    return Results(
        name=results_dir.name,
        summary=summary,
        manifest=manifest,
        eval_tz=_read_json(results_dir / EVAL_TZ_FILE, required=False),
        eval_extractor=_read_json(results_dir / EVAL_EXTRACTOR_FILE, required=False),
        ci=_read_json(results_dir / CI_FILE, required=False),
        notes=notes,
        display_path=display_path or f"results/{results_dir.name}",
    )


# Rendering --------------------------------------------------------------------------------------------------


def _agents(results: Results) -> list[tuple[str, str]]:
    """``(label, heading)`` per agent: naive first, then guarded, then any other, each in run order."""
    by_agent = results.summary.get("by_agent") or {}
    order = {"naive": 0, "guarded": 1}

    def mode(label: str) -> str | None:
        modes = by_agent[label].get("modes") or []
        return modes[0] if len(modes) == 1 else None

    labels = list(results.summary.get("agents") or [])
    labels.sort(key=lambda label: order.get(mode(label) or "", 2))
    headings = []
    for label in labels:
        agent_mode = mode(label)
        headings.append((label, f"{label} ({agent_mode})" if agent_mode and agent_mode != label else label))
    return headings


def caption(results: Results) -> str:
    manifest = results.manifest
    models = sorted({str(m) for m in ((manifest.get("models") or {}).get("agents") or {}).values() if m})
    versions = ", ".join(
        f"{a['label']} `{a.get('agent_version') or 'unknown'}`" for a in manifest.get("agents") or []
    )
    suite_hash = str(manifest.get("scenario_suite_hash") or "unknown")
    short = suite_hash[: len("sha256:") + 12] if suite_hash.startswith("sha256:") else suite_hash
    grading = (manifest.get("grading") or {}).get("label") or "unknown grading"
    return (
        f"<sub>Model: {', '.join(models) or 'none reported'} · run {manifest.get('date')} · "
        f"{manifest.get('hardware')} · booking-truth {manifest.get('harness_version')} · suite `{short}` · "
        f"agents: {versions or 'none'} · {grading} · "
        f"reproduce: `{manifest.get('command') or 'booking-truth test'}`"
        f" · regenerate: `booking-truth report {results.display_path}`</sub>"
    )


def _with_caption(results: Results, lines: list[str]) -> str:
    return "\n".join([*lines, "", caption(results)])


def render_headline(results: Results) -> str:
    summary = results.summary
    agents = _agents(results)
    by_agent = summary["by_agent"]
    k = summary["k"]
    rows = [["pass^1", *(fmt_pass_hat(by_agent[a]["pass_hat_1"]) for a, _ in agents)]]
    if k != 1:
        rows.append([f"pass^{k}", *(fmt_pass_hat(by_agent[a]["pass_hat_k"]) for a, _ in agents)])
    rows += [
        [
            "False-success rate (trial level)",
            *(fmt_rate(by_agent[a]["false_success_rate"]) for a, _ in agents),
        ],
        ["False-claim share", *(fmt_rate(by_agent[a]["false_claim_share"]) for a, _ in agents)],
    ]
    for name, text in _HEADLINE_RATES:
        rows.append([text, *(fmt_rate(by_agent[a]["integrity"][name]) for a, _ in agents)])
    return _with_caption(results, table(["Metric", *(h for _, h in agents)], rows))


def _scenario_order(summary: Mapping[str, Any], ids: set[str]) -> list[str]:
    planned = list(summary.get("scenarios") or [])
    return sorted(ids, key=lambda s: (planned.index(s) if s in planned else len(planned), s))


def render_faults(results: Results) -> str:
    summary = results.summary
    agents = _agents(results)
    by_agent = summary["by_agent"]
    ids = _scenario_order(summary, {s for a, _ in agents for s in by_agent[a]["per_fault"]})
    header = ["Fault scenario"]
    for _, heading in agents:
        header += [f"{heading}: no violation", f"{heading}: pass"]
    rows = []
    for scenario in ids:
        row = [scenario]
        for agent, _ in agents:
            entry = by_agent[agent]["per_fault"].get(scenario) or {}
            row += [fmt_share(entry.get("no_violation")), fmt_share(entry.get("pass"))]
        rows.append(row)
    if not rows:
        rows = [[NOT_RUN, *(["-", "-"] * len(agents))]]
    return _with_caption(results, table(header, rows))


def render_tz(results: Results) -> str:
    summary = results.summary
    agents = _agents(results)
    by_agent = summary["by_agent"]
    ids = _scenario_order(summary, {s for a, _ in agents for s in by_agent[a]["timezone_correct_slot"]})
    rows = [[s, *(fmt_rate(by_agent[a]["timezone_correct_slot"].get(s)) for a, _ in agents)] for s in ids]
    if not rows:
        rows = [[NOT_RUN, *("-" for _ in agents)]]
    return _with_caption(results, table(["Timezone scenario (correct slot)", *(h for _, h in agents)], rows))


def render_cost(results: Results) -> str:
    summary = results.summary
    agents = _agents(results)
    by_agent = summary["by_agent"]

    def harness(agent: str) -> str:
        cost = by_agent[agent]["cost"]
        persona, extractor = cost["persona_usd_per_conversation"], cost["extractor_usd_per_conversation"]
        if persona is None or extractor is None:
            return "n/a"
        return fmt_usd(persona + extractor)

    rows = [
        [
            "Agent USD per conversation",
            *(fmt_usd(by_agent[a]["cost"]["agent_usd_per_conversation"]) for a, _ in agents),
        ],
        ["Harness USD per conversation (persona + extractor)", *(harness(a) for a, _ in agents)],
        ["Turn latency p50 / p95", *(fmt_seconds(by_agent[a]["latency"]["turn"]) for a, _ in agents)],
        [
            "Conversation latency p50 / p95",
            *(fmt_seconds(by_agent[a]["latency"]["conversation"]) for a, _ in agents),
        ],
        [
            "Guard overhead (turns repaired or blocked)",
            *(fmt_rate(by_agent[a]["guard_overhead"]) for a, _ in agents),
        ],
    ]
    lines = table(["Metric", *(h for _, h in agents)], rows)
    lines += ["", f"Total spend of the benchmark run: {fmt_usd(summary.get('total_usd'))}."]
    return _with_caption(results, lines)


def _count(value: Any) -> str:
    return "-" if value is None else str(value)


def _model(data: Mapping[str, Any]) -> str:
    model = data.get("model")
    return f"; LLM: {model}" if isinstance(model, str) and model else ""


def _tz_row(label: str, side: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(side, Mapping):
        return [label, *(["n/a"] * len(_TZ_CATEGORIES))]
    counts, rates, n = side.get("counts") or {}, side.get("rates") or {}, side.get("n")
    return [
        label,
        *(f"{pct(rates.get(key))} ({_count(counts.get(key))}/{_count(n)})" for key, _ in _TZ_CATEGORIES),
    ]


def render_tz_eval(results: Results) -> str:
    data = results.eval_tz
    header = ["Resolver", *(label for _, label in _TZ_CATEGORIES)]
    if not data or not isinstance(data.get("deterministic"), Mapping):
        return _with_caption(results, table(header, [[NOT_RUN, *(["-"] * len(_TZ_CATEGORIES))]]))
    llm_side = data.get("llm")
    rows = [_tz_row("Deterministic", data["deterministic"]), _tz_row("LLM-only (same model)", llm_side)]
    note = f"Held-out test phrases: {_count(data.get('n_items'))}{_model(data)}."
    if data.get("llm") is None:
        note += f" LLM-only side not run: {data.get('llm_skipped_reason') or 'no LLM key configured'}."
    return _with_caption(results, [*table(header, rows), "", note])


def _extractor_row(label: str, entry: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(entry, Mapping):
        return [label, "n/a", "n/a", "n/a"]
    recall = entry.get("success_recall")
    return [
        label,
        pct(entry.get("status_accuracy")),
        pct(entry.get("time_match_accuracy")),
        pct(recall) if recall is not None else "-",
    ]


def render_extractor_eval(results: Results) -> str:
    data = results.eval_extractor
    header = ["Extractor", "Status accuracy", "Time-match accuracy", "Recall on success claims"]
    if not data or not isinstance(data.get("lexicon"), Mapping):
        lines = table(header, [[NOT_RUN, "-", "-", "-"]])
    else:
        rows = [_extractor_row("Lexicon", data.get("lexicon")), _extractor_row("LLM", data.get("llm"))]
        note = f"Held-out test transcripts: {_count(data.get('n_items'))}{_model(data)}."
        if data.get("llm") is None:
            note += f" LLM extractor not run: {data.get('llm_skipped_reason') or 'no LLM key configured'}."
        lines = [*table(header, rows), "", note]
    summary = results.summary
    grading = (results.manifest.get("grading") or {}).get("mode")
    lines += ["", "Agreement on the benchmark trials (LLM vs lexicon extractor, by status):", ""]
    if grading == "offline":
        lines += table(["Agent mode", "Disagreement"], [["all", "not measured (offline grading)"]])
    else:
        modes = summary.get("extractor_disagreement_by_mode") or {}
        lines += table(["Agent mode", "Disagreement"], [[m, fmt_rate(e)] for m, e in modes.items()])
        listed = summary.get("naive_false_success_disagreements") or []
        lines += [
            "",
            "Naive `false_success` trials where the extractors disagree: "
            + (", ".join(f"`{t}`" for t in listed) if listed else "none."),
        ]
    return _with_caption(results, lines)


def render_failures(results: Results) -> str:
    by_agent = results.summary.get("by_agent") or {}
    guarded = [a for a, entry in by_agent.items() if "guarded" in (entry.get("modes") or [])]
    rows = []
    for agent in guarded:
        for violation in by_agent[agent]["integrity_violations"]:
            trace = str(violation["trace_id"])
            note = results.notes.get(trace, "Root cause not analysed yet.")
            rows.append([f"`{trace}`", str(violation["outcome"]), note])
    if not guarded:
        rows = [[NOT_RUN, "-", "No guarded agent in this run."]]
    elif not rows:
        rows = [["none", "-", "The guarded agent had no integrity violation in this run."]]
    return _with_caption(results, table(["Trace", "Category", "Root cause"], rows))


def render_ci(results: Results) -> str:
    data = results.ci
    header = ["Offline guard fixtures", "Passing", "Failing"]
    label = (
        "<sub>By construction, not a benchmark: each fixture fails when only its guard is switched off.</sub>"
    )
    if not data or not isinstance(data.get("fixtures"), list):
        return _with_caption(results, [*table(header, [[NOT_RUN, "-", "-"]]), "", label])
    fixtures = [f for f in data["fixtures"] if isinstance(f, dict)]
    passing = sum(f.get("status") == "pass" for f in fixtures)
    rows = [[str(len(fixtures)), str(passing), str(len(fixtures) - passing)]]
    return _with_caption(results, [*table(header, rows), "", label])


RENDERERS = {
    "headline": render_headline,
    "faults": render_faults,
    "tz": render_tz,
    "cost": render_cost,
    "tz-eval": render_tz_eval,
    "extractor-eval": render_extractor_eval,
    "failures": render_failures,
    "ci": render_ci,
}


def render_tables(results_dir: Path, *, display_path: str | None = None) -> dict[str, str]:
    """Every README table, by id."""
    results = load_results(results_dir, display_path=display_path)
    return {table_id: RENDERERS[table_id](results) for table_id in TABLE_IDS}


# README -----------------------------------------------------------------------------------------------------


def _block(table_id: str) -> re.Pattern[str]:
    name = re.escape(table_id)
    return re.compile(rf"(<!-- bt:{name} -->)(.*?)(<!-- /bt:{name} -->)", re.DOTALL)


@dataclass(frozen=True)
class ReadmeUpdate:
    changed: list[str]
    missing: list[str]
    text: str

    @property
    def in_sync(self) -> bool:
        return not self.changed


def apply_tables(readme: str, tables: Mapping[str, str], ids: Sequence[str] = TABLE_IDS) -> ReadmeUpdate:
    changed: list[str] = []
    missing: list[str] = []
    text = readme
    for table_id in ids:
        pattern = _block(table_id)
        match = pattern.search(text)
        if match is None:
            missing.append(table_id)
            continue
        body = f"\n{tables[table_id]}\n"
        if match.group(2) != body:
            changed.append(table_id)
            text = text[: match.start(2)] + body + text[match.end(2) :]
    return ReadmeUpdate(changed, missing, text)


def update_readme(readme_path: Path, results_dir: Path, *, check: bool = False) -> ReadmeUpdate:
    """Regenerate the README tables from ``results_dir``. With ``check``, only report which tables differ."""
    if not readme_path.is_file():
        raise ReadmeTablesError(f"{readme_path.name} is missing")
    try:
        display = Path(os.path.relpath(results_dir, readme_path.parent)).as_posix()
    except ValueError:
        display = f"results/{results_dir.name}"
    tables = render_tables(results_dir, display_path=display)
    readme = readme_path.read_text(encoding="utf-8")
    update = apply_tables(readme, tables)
    if not check and update.changed:
        with readme_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(update.text)
    return update

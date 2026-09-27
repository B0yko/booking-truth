"""Run outputs: ``traces.jsonl``, ``manifest.json``, ``summary.json`` and ``report.md``.

``summary.json`` and ``report.md`` are pure functions of ``traces.jsonl`` and ``manifest.json``: the final
attempt of every result slot carries the slot's graded result in ``meta.result``. :func:`regenerate` rebuilds
both files from those two inputs alone, with no network, byte for byte; ``booking-truth report <dir>``
runs it.

``booking-truth compare a b`` sets two runs side by side and refuses when an agent's ``agent_version`` differs
between them or changed within one of them, unless told to allow the drift.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from booking_truth.harness.grading import INTEGRITY_OUTCOMES, OUTCOMES
from booking_truth.harness.metrics import TrialResult, summarize
from booking_truth.harness.redact import strip_home_paths
from booking_truth.trace.validate import iter_jsonl, trace_errors, write_jsonl

SUMMARY_FILE = "summary.json"
REPORT_FILE = "report.md"
TRACES_FILE = "traces.jsonl"
MANIFEST_FILE = "manifest.json"
OFFLINE_LABEL = "offline grading"

INTEGRITY_LABELS: dict[str, str] = {
    "false_success": "false_success outcome rate",
    "time_mismatch": "Time-mismatch rate",
    "double_booking": "Double-booking rate",
    "invented_slot": "Invented-slot rate",
    "wrong_time": "Wrong-time rate",
    "unclaimed_booking": "Unclaimed-booking rate",
    "crm_mismatch": "CRM-mismatch rate",
}


class RunDirError(ValueError):
    """A run directory is missing a file or holds an invalid one."""


class CompareRefused(ValueError):
    """Two runs cannot be compared (an agent's version changed, or no agent in common)."""


# Files ------------------------------------------------------------------------------------------------------


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_run(out_dir: Path, traces: Sequence[dict[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Write the traces and the manifest, then derive ``summary.json`` and ``report.md`` from them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / TRACES_FILE, traces)
    _write(out_dir / MANIFEST_FILE, dumps(dict(manifest)))
    return regenerate(out_dir)


def load_run(run_dir: Path) -> tuple[list[TrialResult], dict[str, Any], list[dict[str, Any]]]:
    """The slot results (from each final attempt's ``meta.result``), the manifest and every trace."""
    manifest_path, traces_path = run_dir / MANIFEST_FILE, run_dir / TRACES_FILE
    for path in (manifest_path, traces_path):
        if not path.is_file():
            raise RunDirError(f"{run_dir.name}/{path.name} is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunDirError(f"{run_dir.name}/{MANIFEST_FILE} is not valid JSON: {exc}") from None
    if not isinstance(manifest, dict):
        raise RunDirError(f"{run_dir.name}/{MANIFEST_FILE} must hold a JSON object")
    traces: list[dict[str, Any]] = []
    results: list[TrialResult] = []
    try:
        records = list(iter_jsonl(traces_path))
    except json.JSONDecodeError as exc:
        raise RunDirError(f"{run_dir.name}/{TRACES_FILE} is not valid JSON Lines: {exc}") from None
    for lineno, record in records:
        errors = trace_errors(record)
        if errors:
            raise RunDirError(f"{run_dir.name}/{TRACES_FILE} line {lineno}: {'; '.join(errors)}")
        traces.append(record)
        meta = record.get("meta") or {}
        if meta.get("final_attempt") and isinstance(meta.get("result"), dict):
            results.append(TrialResult.from_json(meta["result"]))
    return results, manifest, traces


def build_summary(results: Sequence[TrialResult], manifest: Mapping[str, Any]) -> dict[str, Any]:
    agents = [str(a["label"]) for a in manifest.get("agents") or []]
    summary = summarize(
        results, k=int(manifest["k"]), agents=agents, scenarios=list(manifest.get("scenarios") or [])
    )
    grading = manifest.get("grading") or {}
    status = manifest.get("status", "complete")
    summary["run"] = {
        "run_id": manifest.get("run_id"),
        "date": manifest.get("date"),
        "as_of": manifest.get("as_of"),
        "status": status,
        "status_detail": manifest.get("status_detail"),
        "dry_run": bool(manifest.get("dry_run")),
        "grading": grading.get("label"),
        "harness_version": manifest.get("harness_version"),
        "scenario_suite_hash": manifest.get("scenario_suite_hash"),
        "agents": {
            str(a["label"]): {"agent_version": a.get("agent_version"), "mode": a.get("mode")}
            for a in manifest.get("agents") or []
        },
        "projection": manifest.get("projection"),
    }
    summary["valid"] = bool(summary["accounting"]["ok"]) and status == "complete"
    return summary


def regenerate(run_dir: Path) -> dict[str, Any]:
    """Rebuild ``summary.json`` and ``report.md`` from ``traces.jsonl`` and ``manifest.json`` (no network)."""
    results, manifest, _ = load_run(run_dir)
    summary = build_summary(results, manifest)
    _write(run_dir / SUMMARY_FILE, strip_home_paths(dumps(summary)))
    _write(run_dir / REPORT_FILE, strip_home_paths(render_report(summary, manifest)))
    return summary


# Formatting -------------------------------------------------------------------------------------------------


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def fmt_rate(entry: Mapping[str, Any] | None) -> str:
    """``2.5% [0.7, 8.7] (3/120)``; ``n/a (0/0)`` without valid trials."""
    if not entry or not entry.get("n"):
        return "n/a (0/0)"
    ci = entry.get("ci") or [None, None]
    lo, hi = ci
    bounds = f" [{lo * 100:.1f}, {hi * 100:.1f}]" if lo is not None and hi is not None else ""
    return f"{pct(entry.get('rate'))}{bounds} ({entry['x']}/{entry['n']})"


def fmt_share(entry: Mapping[str, Any] | None) -> str:
    """``x/n`` only, as the per-fault table shows it."""
    if not entry or not entry.get("n"):
        return "0/0"
    return f"{entry['x']}/{entry['n']}"


def fmt_pass_hat(entry: Mapping[str, Any] | None) -> str:
    if not entry or entry.get("value") is None:
        return "n/a (no scenario with k valid trials)"
    lo, hi = entry.get("ci") or [None, None]
    bounds = f" [{lo * 100:.1f}, {hi * 100:.1f}]" if lo is not None and hi is not None else ""
    return f"{pct(entry['value'])}{bounds} ({entry['n']} scenarios)"


def fmt_usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def fmt_seconds(entry: Mapping[str, Any] | None) -> str:
    if not entry or not entry.get("n"):
        return "n/a"
    p50, p95 = entry.get("p50_s"), entry.get("p95_s")
    return f"{p50:.2f} s / {p95:.2f} s" if p50 is not None and p95 is not None else "n/a"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return lines


def agent_heading(summary: Mapping[str, Any], agent: str) -> str:
    modes = summary["by_agent"][agent].get("modes") or []
    return f"{agent} ({', '.join(modes)})" if modes and modes != [agent] else agent


def _short_hash(value: Any) -> str:
    text = str(value or "unknown")
    return text[: len("sha256:") + 12] if text.startswith("sha256:") else text[:12]


def run_facts(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    git = manifest.get("git") or {}
    sha = str(git.get("sha") or "unknown")
    git_text = (
        sha[:12] + (" (uncommitted changes)" if git.get("dirty") else "") if sha != "unknown" else "unknown"
    )
    status = str(manifest.get("status", "complete"))
    if manifest.get("status_detail"):
        status += f": {manifest['status_detail']}"
    status += " (valid)" if summary.get("valid") else " (not valid: see Accounting)"
    agents = []
    for agent in manifest.get("agents") or []:
        parts = [
            str(agent.get("mode") or "mode unknown"),
            f"version {agent.get('agent_version') or 'unknown'}",
        ]
        if agent.get("source_hash"):
            parts.append(f"source {str(agent['source_hash'])[:12]}")
        if agent.get("model"):
            parts.append(f"model {agent['model']}")
        if agent.get("calendar"):
            parts.append(f"calendar {agent['calendar']}")
        agents.append(f"`{agent['label']}` ({', '.join(parts)})")
    grading = manifest.get("grading") or {}
    grading_text = str(grading.get("label") or "unknown")
    if grading.get("persona") or grading.get("extractor"):
        grading_text += f": {grading.get('persona')} personas, {grading.get('extractor')} belief extractor"
    spend = manifest.get("spend") or {}
    spend_text = (
        f"{fmt_usd(spend.get('total_usd'))} (agent {fmt_usd(spend.get('agent_usd'))}, persona "
        f"{fmt_usd(spend.get('persona_usd'))}, extractor {fmt_usd(spend.get('extractor_usd'))})"
    )
    scenarios = manifest.get("scenarios") or []
    return [
        ("Status", status),
        ("Run date", str(manifest.get("date") or "unknown")),
        ("As-of date", str(manifest.get("as_of") or "run date")),
        ("Hardware", str(manifest.get("hardware") or "unknown")),
        ("Harness", f"booking-truth {manifest.get('harness_version')}, git {git_text}"),
        (
            "Scenario suite",
            f"{manifest.get('suite')}, {_short_hash(manifest.get('scenario_suite_hash'))}, "
            f"{len(scenarios)} scenario(s) run",
        ),
        ("Trials per scenario (k)", str(manifest.get("k"))),
        ("CRM graded", "yes" if (manifest.get("options") or {}).get("grade_crm") else "no"),
        ("Agents", "; ".join(agents) or "none"),
        ("Grading", grading_text),
        ("Total spend", spend_text),
        ("Reproduce", f"`{manifest.get('command') or 'booking-truth test'}`"),
    ]


# report.md --------------------------------------------------------------------------------------------------


def render_report(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    agents: list[str] = list(summary["agents"])
    by_agent = summary["by_agent"]
    heads = [agent_heading(summary, a) for a in agents]
    k = summary["k"]
    lines = [f"# booking-truth run {manifest.get('run_id')}", ""]
    grading = manifest.get("grading") or {}
    if grading.get("mode") == "offline":
        reason = f" ({grading['reason']})" if grading.get("reason") else ""
        lines += [
            f"> **{OFFLINE_LABEL.capitalize()}.** Scripted personas and the lexicon belief extractor only; "
            "the "
            f"harness made no LLM calls{reason}.",
            "",
        ]
    if manifest.get("dry_run"):
        lines += [
            "> **Dry run.** One happy-path and one fault scenario per agent; see the projection below.",
            "",
        ]

    lines += ["## Run", ""]
    lines += table(["Field", "Value"], run_facts(summary, manifest))
    lines += ["", "## Headline", ""]
    rows: list[list[str]] = [["pass^1", *(fmt_pass_hat(by_agent[a]["pass_hat_1"]) for a in agents)]]
    if k != 1:
        rows.append([f"pass^{k}", *(fmt_pass_hat(by_agent[a]["pass_hat_k"]) for a in agents)])
    rows += [
        ["False-success rate", *(fmt_rate(by_agent[a]["false_success_rate"]) for a in agents)],
        ["False-claim share", *(fmt_rate(by_agent[a]["false_claim_share"]) for a in agents)],
    ]
    for name in INTEGRITY_OUTCOMES:
        rows.append([INTEGRITY_LABELS[name], *(fmt_rate(by_agent[a]["integrity"][name]) for a in agents)])
    rows.append(["Any integrity violation", *(fmt_rate(by_agent[a]["integrity"]["any"]) for a in agents)])
    rows.append(
        ["Persona-error rate (attempts)", *(fmt_rate(by_agent[a]["persona_error_rate"]) for a in agents)]
    )
    rows.append(["Turn-cap hits", *(str(by_agent[a]["turn_cap_hits"]) for a in agents)])
    lines += table(["Metric", *heads], rows)

    lines += ["", "## Outcomes", ""]
    outcome_rows = [[name, *(str(by_agent[a]["outcomes"][name]) for a in agents)] for name in OUTCOMES]
    outcome_rows.append(
        ["valid / slots", *(f"{by_agent[a]['valid']}/{by_agent[a]['slots']}" for a in agents)]
    )
    lines += table(["Outcome", *heads], outcome_rows)

    fault_ids = sorted(
        {s for a in agents for s in by_agent[a]["per_fault"]}, key=list(summary["scenarios"]).index
    )
    if fault_ids:
        lines += [
            "",
            "## Fault scenarios",
            "",
            "Valid trials with no integrity violation, and valid trials that pass.",
            "",
        ]
        header = ["Scenario"]
        for head in heads:
            header += [f"{head}: no violation", f"{head}: pass"]
        fault_rows = []
        for scenario in fault_ids:
            row = [scenario]
            for agent in agents:
                entry = by_agent[agent]["per_fault"].get(scenario) or {}
                row += [fmt_share(entry.get("no_violation")), fmt_share(entry.get("pass"))]
            fault_rows.append(row)
        lines += table(header, fault_rows)
        not_injected = sorted(
            s
            for s in fault_ids
            if any((by_agent[a]["per_fault"].get(s) or {}).get("harness_fault_not_injected") for a in agents)
        )
        if not_injected:
            lines += [
                "",
                "Harness-side fault (`duplicate_delivery` fires unconditionally on the first pick; "
                "`concurrent_channel` fires on the first pick that knows any offered slot) never fired "
                "in at least one valid trial of: "
                + ", ".join(f"`{s}`" for s in not_injected)
                + " (see that trial's `meta.harness_fault` in `traces.jsonl`).",
            ]

    tz_ids = sorted(
        {s for a in agents for s in by_agent[a]["timezone_correct_slot"]},
        key=list(summary["scenarios"]).index,
    )
    if tz_ids:
        lines += [
            "",
            "## Timezone scenarios",
            "",
            "Valid trials that end with a booking inside the persona's window "
            "whose start the prospect was told correctly.",
            "",
        ]
        tz_rows = [
            [s, *(fmt_rate(by_agent[a]["timezone_correct_slot"].get(s)) for a in agents)] for s in tz_ids
        ]
        lines += table(["Scenario", *heads], tz_rows)

    lines += ["", "## Cost and latency", ""]
    cost_rows = [
        [
            "Agent USD per conversation",
            *(fmt_usd(by_agent[a]["cost"]["agent_usd_per_conversation"]) for a in agents),
        ],
        [
            "Persona USD per conversation",
            *(fmt_usd(by_agent[a]["cost"]["persona_usd_per_conversation"]) for a in agents),
        ],
        [
            "Extractor USD per conversation",
            *(fmt_usd(by_agent[a]["cost"]["extractor_usd_per_conversation"]) for a in agents),
        ],
        ["Turn latency p50 / p95", *(fmt_seconds(by_agent[a]["latency"]["turn"]) for a in agents)],
        [
            "Conversation latency p50 / p95",
            *(fmt_seconds(by_agent[a]["latency"]["conversation"]) for a in agents),
        ],
        [
            "Guard overhead (turns blocked or repaired)",
            *(fmt_rate(by_agent[a]["guard_overhead"]) for a in agents),
        ],
    ]
    lines += table(["Metric", *heads], cost_rows)
    lines += [
        "",
        f"Spend in the result slots (every attempt): {fmt_usd(summary.get('total_usd'))}. The run's total "
        "spend, preflight turns included, is in the Run table.",
    ]

    lines += ["", "## Belief extractors", ""]
    if grading.get("mode") == "offline":
        lines.append(
            "Only the lexicon extractor ran (offline grading), so extractor agreement is not measured."
        )
    else:
        mode_rows = [
            [mode, fmt_rate(entry)]
            for mode, entry in (summary.get("extractor_disagreement_by_mode") or {}).items()
        ]
        lines += table(["Agent mode", "LLM vs lexicon disagreement"], mode_rows)
        listed = summary.get("naive_false_success_disagreements") or []
        lines += [
            "",
            "Naive `false_success` trials where the extractors disagree: "
            + (", ".join(f"`{t}`" for t in listed) if listed else "none."),
        ]

    lines += ["", "## Integrity violations", ""]
    violations = [(a, v) for a in agents for v in by_agent[a]["integrity_violations"]]
    lines += [f"- `{v['trace_id']}`: {v['outcome']}" for _, v in violations] or ["None."]

    lines += ["", "## Agent errors", ""]
    errors = [t for a in agents for t in by_agent[a]["agent_errors"]]
    lines += [f"- `{t}`" for t in errors] or ["None."]

    lines += ["", "## Harness errors and reruns", ""]
    harness = [entry for a in agents for entry in by_agent[a]["harness_errors"]]
    reruns = [entry for a in agents for entry in by_agent[a]["reruns"]]
    if not harness and not reruns:
        lines.append("None.")
    for entry in harness:
        lines.append(f"- harness_error in the final attempt: `{entry['trace_id']}`")
    for entry in reruns:
        outcomes = ", ".join(f"{a['attempt']}: {a['outcome']}" for a in entry["attempts"])
        lines.append(f"- rerun `{entry['scenario']}` #{entry['trial']}: attempts {outcomes}")

    accounting = summary["accounting"]
    lines += ["", "## Accounting", ""]
    lines += table(
        ["Check", "Value"],
        [
            ["Expected result slots", str(accounting["expected_slots"])],
            ["Result slots present", str(accounting["slots"])],
            ["Missing slots", str(len(accounting["missing"]))],
            ["Duplicate slots", str(len(accounting["duplicates"]))],
            [
                "harness_error slots after reruns",
                f"{accounting['harness_error_slots']} ({pct(accounting['harness_error_share'])}; "
                "the run is invalid above 2%)",
            ],
            ["Run valid", "yes" if summary.get("valid") else "no"],
        ],
    )
    excluded = sorted({s for a in agents for s in by_agent[a]["pass_hat_k"]["excluded"]})
    if excluded:
        lines += [
            "",
            f"Left out of pass^{k} (fewer than {k} valid trials): " + ", ".join(f"`{s}`" for s in excluded),
        ]

    projection = manifest.get("projection")
    if projection:
        lines += ["", "## Projection of the full run", ""]
        full = projection.get("full_run") or {}
        proj_rows = [
            [
                agent,
                f"{entry['trials_run']}",
                fmt_usd(entry["usd_per_trial"]),
                str(entry["full_run_trials"]),
                fmt_usd(entry["projected_usd"]),
            ]
            for agent, entry in (projection.get("per_agent") or {}).items()
        ]
        lines += table(
            ["Agent", "Trials run", "USD per trial", "Full-run trials", "Projected USD"], proj_rows
        )
        lines += [
            "",
            f"Full run: {full.get('scenarios')} scenarios x k = {full.get('k')} x "
            f"{full.get('agents')} agent(s). "
            f"Projected cost {fmt_usd(projection.get('projected_usd'))}; with the "
            f"{projection.get('safety_factor')}x safety factor "
            f"{fmt_usd(projection.get('projected_usd_with_safety'))}.",
        ]

    lines += [
        "",
        "## Notes",
        "",
        f"- {summary['intervals']}",
        "- Regenerate this file and `summary.json` from `traces.jsonl` and `manifest.json` with "
        "`booking-truth report <run directory>`.",
        "",
    ]
    return "\n".join(lines)


# compare ----------------------------------------------------------------------------------------------------


def _metric_rows(
    summary: Mapping[str, Any], agent: str, *, with_k: bool
) -> list[tuple[str, float | None, str]]:
    entry = summary["by_agent"][agent]
    rows: list[tuple[str, float | None, str]] = [
        ("pass^1", entry["pass_hat_1"]["value"], fmt_pass_hat(entry["pass_hat_1"]))
    ]
    if with_k:
        text = f"{fmt_pass_hat(entry['pass_hat_k'])}, k = {summary['k']}"
        rows.append(("pass^k", entry["pass_hat_k"]["value"], text))
    rows += [
        ("False-success rate", entry["false_success_rate"]["rate"], fmt_rate(entry["false_success_rate"])),
        ("False-claim share", entry["false_claim_share"]["rate"], fmt_rate(entry["false_claim_share"])),
    ]
    for name in INTEGRITY_OUTCOMES:
        rows.append(
            (INTEGRITY_LABELS[name], entry["integrity"][name]["rate"], fmt_rate(entry["integrity"][name]))
        )
    rows.append(
        ("Any integrity violation", entry["integrity"]["any"]["rate"], fmt_rate(entry["integrity"]["any"]))
    )
    rows.append(
        (
            "Agent USD per conversation",
            entry["cost"]["agent_usd_per_conversation"],
            fmt_usd(entry["cost"]["agent_usd_per_conversation"]),
        )
    )
    rows.append(
        ("Turn latency p50", entry["latency"]["turn"]["p50_s"], fmt_seconds(entry["latency"]["turn"]))
    )
    return rows


def compare_runs(a_dir: Path, b_dir: Path, *, allow_version_drift: bool = False) -> str:
    """A Markdown comparison of two runs' headline metrics for every agent label they share."""
    results_a, manifest_a, _ = load_run(a_dir)
    results_b, manifest_b, _ = load_run(b_dir)
    summary_a, summary_b = build_summary(results_a, manifest_a), build_summary(results_b, manifest_b)
    agents_a = {str(a["label"]): a for a in manifest_a.get("agents") or []}
    agents_b = {str(a["label"]): a for a in manifest_b.get("agents") or []}
    common = [label for label in agents_a if label in agents_b]
    if not common:
        raise CompareRefused(f"{a_dir.name} and {b_dir.name} have no agent label in common")
    drift = [
        (label, agents_a[label].get("agent_version"), agents_b[label].get("agent_version"))
        for label in common
        if agents_a[label].get("agent_version") != agents_b[label].get("agent_version")
    ]
    # A version that changed inside one run (a run aborted with version_drift) is drift too.
    within = [
        (run_dir.name, label, sorted(agents[label].get("versions_seen") or []))
        for run_dir, agents in ((a_dir, agents_a), (b_dir, agents_b))
        for label in common
        if len(set(agents[label].get("versions_seen") or [])) > 1
    ]
    if (drift or within) and not allow_version_drift:
        problems = []
        if drift:
            changes = "; ".join(
                f"{label}: {old or 'unknown'} -> {new or 'unknown'}" for label, old, new in drift
            )
            problems.append(f"agent version changed between the runs ({changes})")
        if within:
            changes = "; ".join(f"{label} in {name}: {', '.join(seen)}" for name, label, seen in within)
            problems.append(f"agent version changed within a run ({changes})")
        raise CompareRefused(f"{'; '.join(problems)}; pass --allow-version-drift to compare anyway")
    lines = [f"# Compare {a_dir.name} with {b_dir.name}", ""]
    notes = []
    if drift or within:
        notes.append("agent versions differ (allowed by --allow-version-drift)")
    if manifest_a.get("scenario_suite_hash") != manifest_b.get("scenario_suite_hash"):
        notes.append("the scenario suites differ")
    if (manifest_a.get("grading") or {}).get("mode") != (manifest_b.get("grading") or {}).get("mode"):
        notes.append("the grading modes differ")
    if manifest_a.get("k") != manifest_b.get("k"):
        notes.append(f"k differs ({manifest_a.get('k')} vs {manifest_b.get('k')})")
    unknown = [label for label in common if agents_a[label].get("agent_version") is None]
    if unknown and not drift:
        notes.append(f"no agent version reported for {', '.join(unknown)}; drift cannot be checked")
    lines += [f"- Note: {note}." for note in notes]
    if notes:
        lines.append("")
    for label in common:
        lines += [f"## {label}", ""]
        with_k = summary_a["k"] != 1 or summary_b["k"] != 1
        rows_a = _metric_rows(summary_a, label, with_k=with_k)
        rows_b = _metric_rows(summary_b, label, with_k=with_k)
        rows = []
        for (name, value_a, text_a), (_, value_b, text_b) in zip(rows_a, rows_b, strict=True):
            delta = "n/a"
            if value_a is not None and value_b is not None:
                if name.endswith("USD per conversation"):
                    delta = f"{value_b - value_a:+.4f}"
                elif name.startswith("Turn latency"):
                    delta = f"{value_b - value_a:+.2f} s"
                else:
                    delta = f"{(value_b - value_a) * 100:+.1f} pts"
            rows.append([name, text_a, text_b, delta])
        lines += table(["Metric", a_dir.name, b_dir.name, "Change"], rows)
        lines.append("")
    return "\n".join(lines)

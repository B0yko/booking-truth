"""``booking-truth scenarios``: list and lint the scenario suite."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import typer

from booking_truth.harness.scenarios import (
    Scenario,
    ScenarioError,
    lint_report,
    load_suite,
)

app = typer.Typer(help="List and lint the scenario suite.", no_args_is_help=True, add_completion=False)

SuiteOption = Annotated[
    Path | None,
    typer.Option(
        "--suite",
        file_okay=False,
        help="Directory of scenario YAML files. Default: the bundled suite.",
    ),
]


def describe_faults(scenario: Scenario) -> str:
    """A short text of the scenario's sandbox and harness faults, or ``-`` when there are none."""
    parts = []
    for rule in scenario.faults:
        count = "persistent" if rule.times is None else f"x{rule.times}"
        parts.append(f"{rule.group} {rule.mode} {count}")
    fault = scenario.harness_fault
    if fault is not None:
        parts.append(
            f"harness {fault.type}"
            + (f" offered[{fault.pick}]" if fault.type == "concurrent_channel" else "")
        )
    return "; ".join(parts) or "-"


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows
    ]


def _parse_as_of(value: str | None) -> date:
    if value is None:
        return datetime.now(UTC).date()
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(
            f"expected a date as YYYY-MM-DD, got {value!r}", param_hint="--as-of"
        ) from None


@app.command("list")
def list_cmd(suite: SuiteOption = None) -> None:
    """Show every scenario: id, tags, goal, persona zone and faults."""
    try:
        scenarios = load_suite(suite)
    except ScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    rows = [("ID", "TAGS", "GOAL", "PERSONA ZONE", "FAULTS")]
    rows += [
        (s.id, ",".join(s.tags), s.persona.goal, s.persona.true_zone, describe_faults(s)) for s in scenarios
    ]
    for line in _table(rows):
        typer.echo(line)
    typer.echo(f"{len(scenarios)} scenarios")


@app.command("lint")
def lint_cmd(
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            metavar="YYYY-MM-DD",
            help="Resolve scenario dates from this date instead of today (UTC).",
        ),
    ] = None,
    suite: SuiteOption = None,
) -> None:
    """Validate the scenarios and check each persona window against the seeded host availability."""
    run_date = _parse_as_of(as_of)
    report = lint_report(run_date, suite)
    rows = [("ID", "WINDOW DATES", "FREE SLOTS")]
    for entry in report.entries:
        first, last = entry.window_dates[0], entry.window_dates[-1]
        span = first.isoformat() if first == last else f"{first.isoformat()}..{last.isoformat()}"
        slots = f"{entry.free_slots}" + (" (impossible)" if entry.impossible else "")
        rows.append((entry.id, span, slots))
    if report.entries:
        for line in _table(rows):
            typer.echo(line)
    for error in report.errors:
        typer.echo(f"error: {error}", err=True)
    if report.errors:
        typer.echo(f"lint failed for {run_date}: {len(report.errors)} error(s)", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"lint passed for {run_date}: {len(report.entries)} scenarios")

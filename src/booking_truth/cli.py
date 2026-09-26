"""Command-line entry point: ``booking-truth``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from booking_truth import __version__
from booking_truth.sandbox.cli import app as sandbox_app
from booking_truth.trace.validate import iter_jsonl, trace_errors

app = typer.Typer(
    name="booking-truth",
    help="Test appointment-setting agents on the calendar's end state, and run a guarded booking agent.",
    no_args_is_help=True,
    add_completion=False,
)


app.add_typer(sandbox_app, name="sandbox", help="Run the sandbox calendar and CRM service.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"booking-truth {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    """booking-truth command line."""


@app.command("validate-trace")
def validate_trace_cmd(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A JSON Lines file of traces.")],
) -> None:
    """Validate every trace in a JSONL file against agent-trace/v1 (from any producer)."""
    total = bad = 0
    for lineno, record in iter_jsonl(path):
        total += 1
        errors = trace_errors(record)
        if errors:
            bad += 1
            for err in errors:
                typer.echo(f"line {lineno}: {err}", err=True)
    typer.echo(f"{total - bad}/{total} traces valid")
    if bad or total == 0:
        raise typer.Exit(code=1)


def main() -> None:
    app()

"""``booking-truth eval tz`` and ``booking-truth eval extractor``."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from booking_truth.config import Settings, load_settings
from booking_truth.evals.extractor_eval import render_extractor_eval_md, run_extractor_eval
from booking_truth.evals.tz_eval import render_tz_eval_md, run_tz_eval
from booking_truth.harness.report import dumps
from booking_truth.llm.client import OpenAICompatClient
from booking_truth.llm.types import LLMError

app = typer.Typer(
    help="Component evaluations: the timezone resolver and the belief extractors.",
    no_args_is_help=True,
    add_completion=False,
)

TZ_JSON, TZ_MD = "tz-eval.json", "tz-eval.md"
EXTRACTOR_JSON, EXTRACTOR_MD = "extractor-eval.json", "extractor-eval.md"
EXIT_ERROR = 2

ModelOption = Annotated[
    str | None, typer.Option("--model", help="Model id for the LLM side (default: BT_LLM_MODEL).")
]
BudgetOption = Annotated[
    float | None,
    typer.Option(
        "--budget-usd",
        min=0,
        help="Override BT_BUDGET_USD for this process: refuse this eval's live calls once the ledger "
        "total (every process's spend in BT_LEDGER_DIR, not just this eval's own) would pass this.",
    ),
]
OutOption = Annotated[
    Path | None, typer.Option("--out", help="Output directory (default: results/<generated run id>).")
]


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _out_dir(out: Path | None) -> Path:
    return out if out is not None else Path("results") / _run_id()


def _display_path(path: Path) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path.name


def _write(out_dir: Path, json_name: str, md_name: str, json_data: dict[str, Any], md_text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / json_name).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(dumps(json_data))
    with (out_dir / md_name).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(md_text if md_text.endswith("\n") else md_text + "\n")


def _client(
    settings: Settings, *, component: str, model: str | None, budget_usd: float | None
) -> tuple[OpenAICompatClient | None, str]:
    """The eval's own LLM client, or ``None`` when no key is configured (offline: deterministic/lexicon
    side only). ``budget_usd``, when given, overrides ``BT_BUDGET_USD`` for this eval's own live calls."""
    resolved_model = model or settings.llm_model
    if settings.offline:
        return None, resolved_model
    client = OpenAICompatClient.from_settings(settings, component=component, model=resolved_model)
    if budget_usd is not None:
        client.budget_usd = budget_usd
    return client, resolved_model


def _load_settings() -> Settings:
    try:
        return load_settings()
    except ValueError as exc:
        typer.echo(f"error: invalid BT_* settings: {exc}", err=True)
        raise typer.Exit(code=EXIT_ERROR) from None


def cmd_tz(model: ModelOption = None, budget_usd: BudgetOption = None, out: OutOption = None) -> None:
    """Score the deterministic timezone resolver against LLM-only resolution on the held-out test split."""
    settings = _load_settings()
    out_dir = _out_dir(out)
    try:
        client, model_id = _client(settings, component="eval_tz", model=model, budget_usd=budget_usd)
    except LLMError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_ERROR) from None

    async def _execute() -> None:
        try:
            result = await run_tz_eval(llm=client, model=model_id if client is not None else None)
        finally:
            if client is not None:
                await client.aclose()
        data = result.to_json()
        _write(out_dir, TZ_JSON, TZ_MD, data, render_tz_eval_md(data))
        typer.echo(f"wrote {TZ_JSON} and {TZ_MD} in {_display_path(out_dir)}")
        det = data["deterministic"]
        typer.echo(
            f"deterministic resolver: {det['counts']['silent_wrong_resolution']}/{det['n']} silent wrong "
            "resolution(s)"
        )
        if client is None:
            typer.echo("note: no LLM key configured; offline, only the deterministic resolver ran", err=True)
        elif data["llm"] is None:
            typer.echo(f"note: {data['llm_skipped_reason']}", err=True)

    asyncio.run(_execute())


def cmd_extractor(
    model: ModelOption = None,
    budget_usd: BudgetOption = None,
    out: OutOption = None,
    run: Annotated[
        Path | None,
        typer.Option(
            "--run",
            exists=True,
            file_okay=False,
            help="A results/<run-id> directory to add benchmark agreement stats from.",
        ),
    ] = None,
) -> None:
    """Score the LLM and lexicon belief extractors on the held-out test split of belief_extraction.jsonl."""
    settings = _load_settings()
    out_dir = _out_dir(out)
    try:
        client, model_id = _client(settings, component="eval_extractor", model=model, budget_usd=budget_usd)
    except LLMError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_ERROR) from None

    async def _execute() -> None:
        try:
            result = await run_extractor_eval(
                llm=client, model=model_id if client is not None else None, benchmark_run_dir=run
            )
        finally:
            if client is not None:
                await client.aclose()
        data = result.to_json()
        _write(out_dir, EXTRACTOR_JSON, EXTRACTOR_MD, data, render_extractor_eval_md(data))
        typer.echo(f"wrote {EXTRACTOR_JSON} and {EXTRACTOR_MD} in {_display_path(out_dir)}")
        typer.echo(
            f"lexicon: status accuracy {data['lexicon']['status_accuracy']:.3f}, time-match accuracy "
            f"{data['lexicon']['time_match_accuracy']:.3f}"
        )
        if client is None:
            typer.echo("note: no LLM key configured; offline, only the lexicon extractor ran", err=True)
        elif data["llm"] is None:
            typer.echo(f"note: {data['llm_skipped_reason']}", err=True)

    try:
        asyncio.run(_execute())
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_ERROR) from None


def register(app_root: typer.Typer) -> None:
    """Add ``eval tz`` and ``eval extractor`` to the root command."""
    app.command("tz")(cmd_tz)
    app.command("extractor")(cmd_extractor)
    app_root.add_typer(app, name="eval")

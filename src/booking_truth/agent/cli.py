"""``booking-truth agent``: serve the agent and inspect its hand-offs and CRM outbox."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import typer

from booking_truth.config import ConfigError, Settings, load_settings
from booking_truth.store import Store, StoreError
from booking_truth.timeutil import iso_ms_z

app = typer.Typer(no_args_is_help=True, add_completion=False)

EXIT_CONFIG = 2


def _settings() -> Settings:
    try:
        return load_settings()
    except ValueError as exc:
        typer.echo(f"error: invalid BT_* settings: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None


def _store(settings: Settings) -> Store:
    path = settings.resolved_db_path
    if not path.exists():
        typer.echo(f"no agent database at {path} (BT_DB_PATH); nothing to show")
        raise typer.Exit(0)
    try:
        return Store(path)
    except StoreError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Port to listen on.")] = 8000,
) -> None:
    """Run the agent API, the widget and the demo page."""
    import uvicorn

    from booking_truth.agent.api import create_agent_app

    settings = _settings()
    try:
        application = create_agent_app(settings)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    mode = "offline demo mode (scripted policy)" if settings.offline else f"model {settings.llm_model}"
    typer.echo(f"booking-truth agent on http://{host}:{port} - {mode}, guards {settings.guards}")
    uvicorn.run(application, host=host, port=port, log_level="info")


@app.command("handoffs")
def handoffs(
    all_items: Annotated[bool, typer.Option("--all", help="Include delivered hand-offs.")] = False,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 50,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON lines.")] = False,
) -> None:
    """List conversations handed to a person (undelivered ones by default)."""
    store = _store(_settings())
    try:
        items = store.handoffs.items(delivered=None if all_items else False, limit=limit)
    finally:
        store.close()
    if not items:
        typer.echo("no hand-offs")
        return
    for item in items:
        record = {
            "id": item.id,
            "created_at": iso_ms_z(item.created_at),
            "lead_email": item.lead_email,
            "session_id": item.session_id,
            "delivered": item.delivered,
            "summary": item.summary,
            "preferred_times_text": item.preferred_times_text,
        }
        if as_json:
            typer.echo(json.dumps(record, ensure_ascii=False))
        else:
            state = "delivered" if item.delivered else "open"
            typer.echo(f"#{item.id} {record['created_at']} {item.lead_email} [{state}] {item.summary}")
            if item.preferred_times_text:
                typer.echo(f"    preferred times: {item.preferred_times_text}")


@app.command("outbox")
def outbox(
    status: Annotated[
        Literal["pending", "failed", "done", "all"], typer.Option("--status", help="Which items to list.")
    ] = "all",
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 50,
    requeue: Annotated[
        int | None, typer.Option("--requeue", help="Put a failed item back in the queue.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON lines.")] = False,
) -> None:
    """List CRM outbox items (pending and failed ones need attention)."""
    store = _store(_settings())
    try:
        if requeue is not None:
            try:
                item = store.outbox.requeue(requeue)
            except StoreError as exc:
                typer.echo(f"error: {exc}", err=True)
                raise typer.Exit(1) from None
            typer.echo(f"requeued outbox item #{item.id}")
            return
        backlog = store.outbox.backlog()
        items = store.outbox.items(status=None if status == "all" else status, limit=limit)
    finally:
        store.close()
    typer.echo(f"backlog: {backlog.pending} pending, {backlog.failed} failed")
    for entry in items:
        record = {
            "id": entry.id,
            "kind": entry.kind,
            "status": entry.status,
            "attempts": entry.attempts,
            "lead_email": entry.lead_email,
            "next_attempt_at": iso_ms_z(entry.next_attempt_at) if entry.next_attempt_at else None,
            "last_error": entry.last_error,
            "payload": entry.payload,
        }
        if as_json:
            typer.echo(json.dumps(record, ensure_ascii=False))
        else:
            error = f" last error: {entry.last_error}" if entry.last_error else ""
            state = f"[{entry.status}, {entry.attempts} attempts]"
            typer.echo(f"#{entry.id} {entry.kind} {state} {entry.lead_email}{error}")

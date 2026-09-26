"""``booking-truth sandbox ...`` commands."""

from __future__ import annotations

from typing import Annotated

import typer
import uvicorn

from booking_truth.config import Settings
from booking_truth.sandbox.app import create_sandbox_app

app = typer.Typer(
    help="The sandbox calendar and CRM service that agents under test talk to.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _sandbox() -> None:
    """The sandbox calendar and CRM service that agents under test talk to."""


@app.command("serve")
def serve(
    host: Annotated[
        str, typer.Option(help="Interface to bind; use 0.0.0.0 inside a container.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="TCP port.")] = 8100,
) -> None:
    """Serve the mirrored vendor APIs, the control API and the /_ui page.

    Every route except GET /_ui requires 'Authorization: Bearer <BT_SANDBOX_TOKEN>' (default: sandbox).
    """
    token = Settings().sandbox_token.get_secret_value()
    typer.echo(f"booking-truth sandbox on http://{host}:{port} (read-only view: /_ui)")
    uvicorn.run(create_sandbox_app(token), host=host, port=port, log_level="info")

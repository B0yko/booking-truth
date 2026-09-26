"""Run an ASGI app with uvicorn in a background thread on a free local port.

Used by ``--sandbox auto`` and ``--agent builtin`` and by the test-suite, so that every component is
exercised over real HTTP, including timeouts.
"""

from __future__ import annotations

import socket
import threading
import time
from types import TracebackType
from typing import Any

import uvicorn


class BackgroundServer:
    def __init__(self, app: Any, host: str = "127.0.0.1", port: int = 0) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self.host = host
        self.port: int = self._sock.getsockname()[1]
        config = uvicorn.Config(
            app,
            log_level="warning",
            lifespan="on",
            timeout_graceful_shutdown=1,
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [self._sock]},
            name=f"uvicorn-{self.port}",
            daemon=True,
        )

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout_s: float = 10.0) -> BackgroundServer:
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("server thread exited during startup")
            if time.monotonic() > deadline:
                raise TimeoutError(f"server on port {self.port} did not start within {timeout_s}s")
            time.sleep(0.01)
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=timeout_s)
        self._sock.close()

    def __enter__(self) -> BackgroundServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

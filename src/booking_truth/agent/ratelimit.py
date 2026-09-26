"""Per-client sliding-window rate limit for the public widget endpoint (30 requests per minute).

The key is the socket peer address. ``X-Forwarded-For`` is used only when ``BT_TRUST_PROXY=true``; otherwise
anyone could send a fresh header value with every request and never be limited.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping

WIDGET_LIMIT = 30
WINDOW_S = 60.0
MAX_KEYS = 10_000


class SlidingWindowLimiter:
    def __init__(
        self,
        limit: int = WIDGET_LIMIT,
        window_s: float = WINDOW_S,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1 or window_s <= 0:
            raise ValueError("limit must be >= 1 and window_s > 0")
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Count one request for ``key``; ``False`` when it is over the limit (the request is not counted)."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= MAX_KEYS:
                    self._prune(now)
                hits = self._hits[key] = deque()
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True

    def retry_after(self, key: str) -> float:
        """Seconds until ``key`` may send again (0 when it may now)."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if not hits or len(hits) < self.limit:
                return 0.0
            return max(0.0, hits[0] + self.window_s - now)

    def _prune(self, now: float) -> None:
        stale = [k for k, v in self._hits.items() if not v or v[-1] <= now - self.window_s]
        for key in stale:
            del self._hits[key]


def client_key(peer: str | None, headers: Mapping[str, str], *, trust_proxy: bool) -> str:
    """The rate-limit key: the first ``X-Forwarded-For`` address behind a trusted proxy, else the peer."""
    if trust_proxy:
        forwarded = headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return peer or "unknown"

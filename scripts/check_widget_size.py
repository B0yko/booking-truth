"""Check the embeddable widget as it is served at /widget.js.

The widget must stay one small, self-contained file: at most 25,600 bytes, type-checked (its first line is
``// @ts-check``) and free of requests to any host other than the agent it is embedded for.

Usage: python scripts/check_widget_size.py [path]   (default: widget/widget.js)
Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_BYTES = 25_600
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "widget" / "widget.js"
# The embed example in the header comment is the only URL the file may contain.
ALLOWED_HOSTS = {"agent.example.com"}
URL_HOST = re.compile(r"(?:https?:)?//([a-z0-9-]+(?:\.[a-z0-9-]+)+)", re.IGNORECASE)
FORBIDDEN = re.compile(
    r"\b(?:innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval|XMLHttpRequest|WebSocket)\b"
)


def problems(source: bytes) -> list[str]:
    found: list[str] = []
    size = len(source)
    if size > MAX_BYTES:
        found.append(f"{size} bytes, over the {MAX_BYTES}-byte budget by {size - MAX_BYTES}")
    text = source.decode("utf-8")
    if text.split("\n", 1)[0].rstrip("\r") != "// @ts-check":
        found.append("the first line must be '// @ts-check'")
    hosts = sorted({m.group(1).lower() for m in URL_HOST.finditer(text)} - ALLOWED_HOSTS)
    if hosts:
        found.append(f"references hosts other than the agent: {', '.join(hosts)}")
    for match in sorted({m.group(0) for m in FORBIDDEN.finditer(text)}):
        found.append(f"uses {match}")
    return found


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    try:
        source = path.read_bytes()
    except OSError as exc:
        print(f"{path.name}: cannot read: {exc.strerror}", file=sys.stderr)
        return 1
    found = problems(source)
    for problem in found:
        print(f"{path.name}: {problem}", file=sys.stderr)
    if not found:
        print(f"{path.name}: {len(source)} of {MAX_BYTES} bytes, ok")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

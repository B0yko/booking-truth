"""Regenerate ``tests/golden/offline_suite.json`` from an actual run of the offline scenario suite.

The suite makes no network calls: the sandbox, the scripted personas and the scripted policy
(``FakeLLM``) all run in-process, pinned to one fixed clock instant, so a rerun with no code change
reproduces the same file byte for byte. Run this deliberately after a change that is meant to move a
scenario's outcome, read the diff, and explain every changed cell before committing it.

Usage::

    uv run python scripts/update_offline_snapshot.py            # rewrite the snapshot
    uv run python scripts/update_offline_snapshot.py --check     # exit 1 if it would change
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests" / "golden"))

from offline_suite_lib import (  # noqa: E402
    CALENDARS,
    MODES,
    SNAPSHOT_PATH,
    BuiltinCalendar,
    BuiltinMode,
    RunResult,
    build_snapshot,
    load_snapshot,
    run_combo,
    save_snapshot,
)


async def _collect() -> dict[str, Any]:
    results: dict[tuple[BuiltinMode, BuiltinCalendar], RunResult] = {}
    for mode, calendar in itertools.product(MODES, CALENDARS):
        print(f"running {mode}-{calendar} ...", file=sys.stderr)
        results[(mode, calendar)] = await run_combo(mode, calendar)
    return build_snapshot(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the snapshot would change")
    args = parser.parse_args(argv)
    snapshot = asyncio.run(_collect())
    if args.check:
        current = load_snapshot() if SNAPSHOT_PATH.exists() else None
        if current != snapshot:
            print("offline_suite.json is out of date", file=sys.stderr)
            return 1
        print("offline_suite.json is up to date")
        return 0
    save_snapshot(snapshot)
    print(f"wrote {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

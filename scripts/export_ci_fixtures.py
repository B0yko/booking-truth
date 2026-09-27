"""Run every offline guard fixture and write ``<results dir>/ci_fixtures.json``.

This is the harness's own regression suite (``tests/fixtures/guards/*.yaml``, run through
``tests/guards/fixture_runner.py``): each fixture pairs one scripted misbehaviour with the guard that
exists to catch it, checked both with every guard on and with only that guard switched off. It is a
correctness check "by construction, not a benchmark" (``docs/metrics.md``), not a live evaluation - no
``BT_*`` setting or ``OPENROUTER_API_KEY`` from the calling shell reaches it (stripped below, matching
``tests/guards/conftest.py``), so it never makes a paid LLM call.

The output feeds the README's ``bt:ci`` table (``booking_truth.harness.readme_tables.render_ci``) and
``results/<run-id>/results/README.md`` documents the file. Usage::

    uv run python scripts/export_ci_fixtures.py --out results/<run-id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests" / "guards"))

for _name in list(os.environ):
    if _name.startswith("BT_") or _name == "OPENROUTER_API_KEY":
        del os.environ[_name]

from fixture_runner import (  # noqa: E402 - sys.path set up above
    failures,
    fixture_files,
    load_fixture,
    resolved_calendars,
    run_fixture,
)

CI_FILE = "ci_fixtures.json"


async def _run_all() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in fixture_files():
        fixture = load_fixture(path)
        for mode in ("on", "off"):
            expectation = fixture.expect_on if mode == "on" else fixture.expect_off
            for calendar in resolved_calendars(fixture):
                observation = await run_fixture(fixture, mode, calendar)
                problems = failures(expectation, observation)
                row: dict[str, Any] = {
                    "name": f"{fixture.guard}:{mode}:{calendar}",
                    "guard": fixture.guard,
                    "title": fixture.title,
                    "mode": mode,
                    "calendar": calendar,
                    "status": "pass" if not problems else "fail",
                }
                if problems:
                    row["problems"] = problems
                rows.append(row)
    return rows


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=None, help="results directory (default: results/<generated run id>)"
    )
    args = parser.parse_args(argv)
    out_dir = args.out or REPO_ROOT / "results" / _run_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures = asyncio.run(_run_all())
    data = {"fixtures": fixtures}
    with (out_dir / CI_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    passing = sum(f["status"] == "pass" for f in fixtures)
    print(f"wrote {CI_FILE} in {out_dir}: {passing}/{len(fixtures)} passing")
    if passing != len(fixtures):
        for row in fixtures:
            if row["status"] != "pass":
                print(f"  FAIL {row['name']}: {row.get('problems')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

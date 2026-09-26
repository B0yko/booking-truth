"""Regenerate the README's result tables from a results directory, or check that they are current.

Each table sits between ``<!-- bt:<id> -->`` and ``<!-- /bt:<id> -->`` markers in the README. CI runs the
check mode and fails when any table differs from what ``results/<run-id>/`` produces.

Usage::

    uv run python scripts/update_readme_tables.py results/<run-id>            # rewrite README.md
    uv run python scripts/update_readme_tables.py results/<run-id> --check    # exit 1 when out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from booking_truth.harness.readme_tables import ReadmeTablesError, update_readme

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="the results directory, e.g. results/<run-id>")
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md", help="the README to update")
    parser.add_argument("--check", action="store_true", help="only report tables that are out of date")
    args = parser.parse_args(argv)
    try:
        update = update_readme(args.readme, args.results, check=args.check)
    except ReadmeTablesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if update.missing:
        print(f"markers not found in {args.readme.name}: {', '.join(update.missing)}", file=sys.stderr)
    if args.check:
        if update.changed:
            print(f"out of date: {', '.join(update.changed)}", file=sys.stderr)
            return 1
        print("README tables are up to date")
        return 0
    print(f"updated: {', '.join(update.changed) or 'nothing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Record or verify the sha256 of the held-out test split of each labelled dataset.

The hash covers a canonical form of the test split: the ``split == "test"`` items sorted by ``id``,
each re-serialised with ``json.dumps(item, sort_keys=True, ensure_ascii=False)``, joined with ``\\n``
and terminated by a final ``\\n``, encoded as UTF-8. Formatting changes to the JSONL file therefore do
not change the hash, but any change to a test item does.

``datasets/HASHES.json`` is written once, before any tuning, and must not be rewritten afterwards
without a documented reason; the write mode refuses to replace different hashes unless ``--force``.

Usage::

    uv run python scripts/dataset_hashes.py            # record (first time)
    uv run python scripts/dataset_hashes.py --check    # verify; exit 1 on mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"
HASHES_FILE = "HASHES.json"
DATASETS = {
    "tz_phrases_test_sha256": "tz_phrases.jsonl",
    "belief_extraction_test_sha256": "belief_extraction.jsonl",
}


def canonical_split_bytes(path: Path, split: str = "test") -> bytes:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = sorted((item for item in items if item["split"] == split), key=lambda item: item["id"])
    return "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in chosen).encode(
        "utf-8"
    )


def split_sha256(path: Path, split: str = "test") -> str:
    return hashlib.sha256(canonical_split_bytes(path, split)).hexdigest()


def compute_hashes(datasets_dir: Path) -> dict[str, str]:
    return {key: split_sha256(datasets_dir / name) for key, name in DATASETS.items()}


def load_recorded(datasets_dir: Path) -> dict[str, str]:
    data: dict[str, str] = json.loads((datasets_dir / HASHES_FILE).read_text(encoding="utf-8"))
    return data


def mismatches(datasets_dir: Path) -> list[str]:
    """Human-readable differences between the recorded and the current hashes (empty when equal)."""
    recorded = load_recorded(datasets_dir)
    current = compute_hashes(datasets_dir)
    return [
        f"{key}: recorded {recorded.get(key)} != current {value}"
        for key, value in current.items()
        if recorded.get(key) != value
    ]


def write(datasets_dir: Path, recorded_on: str, *, force: bool) -> int:
    path = datasets_dir / HASHES_FILE
    current = compute_hashes(datasets_dir)
    if path.exists():
        existing = load_recorded(datasets_dir)
        if all(existing.get(key) == value for key, value in current.items()):
            print(f"{HASHES_FILE} is up to date (recorded {existing.get('recorded')})")
            return 0
        if not force:
            print(f"refusing to replace recorded test hashes in {HASHES_FILE}; pass --force to re-record")
            return 1
    payload = {**current, "recorded": recorded_on}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {HASHES_FILE}: {payload}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record or verify the test-split hashes of the labelled datasets."
    )
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--check", action="store_true", help="verify only; exit 1 on any mismatch")
    parser.add_argument(
        "--force", action="store_true", help="replace hashes that differ from the recorded ones"
    )
    parser.add_argument(
        "--recorded",
        default=datetime.now(UTC).date().isoformat(),
        help="date to store with newly recorded hashes (YYYY-MM-DD, default today in UTC)",
    )
    args = parser.parse_args(argv)
    if args.check:
        problems = mismatches(args.datasets_dir)
        for problem in problems:
            print(problem)
        if problems:
            return 1
        print("test-split hashes match")
        return 0
    return write(args.datasets_dir, args.recorded, force=args.force)


if __name__ == "__main__":
    sys.exit(main())

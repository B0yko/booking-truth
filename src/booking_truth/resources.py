"""Locate bundled data (scenarios, datasets, schemas, pricing, widget).

In a wheel the data lives under ``booking_truth/_data``. In a source checkout it lives at the
repository root, two levels above this package.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_BUNDLED = _PACKAGE_DIR / "_data"
_REPO_ROOT = _PACKAGE_DIR.parents[1]


def data_path(name: str) -> Path:
    """Return the path of a bundled data file or directory, e.g. ``data_path("scenarios")``."""
    bundled = _BUNDLED / name
    if bundled.exists():
        return bundled
    checkout = _REPO_ROOT / name
    if checkout.exists():
        return checkout
    raise FileNotFoundError(f"bundled data not found: {name}")

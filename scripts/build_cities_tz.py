"""Build ``datasets/cities_tz.csv`` from the GeoNames ``cities15000`` dump.

GeoNames data is licensed under CC BY 4.0 (https://www.geonames.org/,
https://creativecommons.org/licenses/by/4.0/). This script downloads ``cities15000.zip`` and
``admin1CodesASCII.txt``, keeps the columns the timezone resolver needs, drops rows whose timezone
is not a zone in the installed ``tzdata`` package, sorts the rows deterministically and writes a
UTF-8 CSV with a header row.

GeoNames rebuilds its dump every day, so a rebuild on another day produces a slightly different
file. The committed CSV is the reference snapshot; ``datasets/NOTICE`` records its download date.

Usage::

    uv run python scripts/build_cities_tz.py [--cache-dir DIR] [--out datasets/cities_tz.csv] [--refresh]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import httpx

CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"
CITIES_MEMBER = "cities15000.txt"
GEONAME_FIELDS = 19
HEADER = ("name", "asciiname", "country_code", "admin1_code", "admin1_name", "population", "timezone")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "datasets" / "cities_tz.csv"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "booking-truth-geonames"


@dataclass(frozen=True)
class City:
    geonameid: int
    name: str
    asciiname: str
    country_code: str
    admin1_code: str
    admin1_name: str
    population: int
    timezone: str

    def sort_key(self) -> tuple[str, str, str, int, str, int]:
        return (
            self.asciiname.casefold(),
            self.country_code,
            self.admin1_code,
            -self.population,
            self.name,
            self.geonameid,
        )

    def row(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.name,
            self.asciiname,
            self.country_code,
            self.admin1_code,
            self.admin1_name,
            str(self.population),
            self.timezone,
        )


def tzdata_zone_keys() -> frozenset[str]:
    """Every zone key shipped by the installed ``tzdata`` package (independent of the OS copy)."""
    text = resources.files("tzdata").joinpath("zones").read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def fetch(url: str, dest: Path, *, refresh: bool) -> Path:
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0), follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(response.content)
        tmp.replace(dest)
    return dest


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_admin1_names(path: Path) -> dict[str, str]:
    """Map ``<country>.<admin1>`` to the ASCII admin1 name."""
    names: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for fields in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(fields) >= 3:
                names[fields[0]] = fields[2]
    return names


def load_cities(
    zip_path: Path, admin1: dict[str, str], valid_zones: frozenset[str]
) -> tuple[list[City], int]:
    """Parse the geoname table. Returns the kept cities and the number of rows dropped."""
    kept: list[City] = []
    dropped = 0
    with zipfile.ZipFile(zip_path) as archive, archive.open(CITIES_MEMBER) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        for fields in csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(fields) != GEONAME_FIELDS:
                raise ValueError(f"unexpected field count {len(fields)} in {CITIES_MEMBER}")
            timezone = fields[17]
            if timezone not in valid_zones:
                dropped += 1
                continue
            country, admin1_code = fields[8], fields[10]
            kept.append(
                City(
                    geonameid=int(fields[0]),
                    name=fields[1],
                    asciiname=fields[2],
                    country_code=country,
                    admin1_code=admin1_code,
                    admin1_name=admin1.get(f"{country}.{admin1_code}", ""),
                    population=int(fields[14] or 0),
                    timezone=timezone,
                )
            )
    kept.sort(key=City.sort_key)
    return kept, dropped


def write_csv(cities: list[City], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(HEADER)
        for city in cities:
            writer.writerow(city.row())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE, help="download cache (outside the repo)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output CSV path")
    parser.add_argument("--refresh", action="store_true", help="download again even if cached files exist")
    args = parser.parse_args(argv)

    cache: Path = args.cache_dir
    zip_path = fetch(CITIES_URL, cache / "cities15000.zip", refresh=args.refresh)
    admin1_path = fetch(ADMIN1_URL, cache / "admin1CodesASCII.txt", refresh=args.refresh)
    print(f"source cities15000.zip sha256 {sha256_file(zip_path)}")
    print(f"source admin1CodesASCII.txt sha256 {sha256_file(admin1_path)}")

    cities, dropped = load_cities(zip_path, load_admin1_names(admin1_path), tzdata_zone_keys())
    write_csv(cities, args.out)
    print(f"rows {len(cities)} (dropped {dropped} with a timezone unknown to tzdata)")
    print(f"wrote {args.out.name} sha256 {sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

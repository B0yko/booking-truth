"""Checks on the committed labelled datasets and their build scripts (no network)."""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets"
SCRIPTS = REPO / "scripts"
HOST_ZONE = "America/New_York"
RFC3339_Z = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
CITIES_HEADER = ["name", "asciiname", "country_code", "admin1_code", "admin1_name", "population", "timezone"]


def load_script(name: str) -> ModuleType:
    module_name = f"booking_truth_scripts_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(name: str) -> list[dict[str, Any]]:
    text = (DATASETS / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@cache
def tzdata_keys() -> frozenset[str]:
    text = resources.files("tzdata").joinpath("zones").read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


@cache
def load_zone(key: str) -> ZoneInfo:
    node = resources.files("tzdata.zoneinfo")
    for part in key.split("/"):
        node = node.joinpath(part)
    with node.open("rb") as fh:
        return ZoneInfo.from_file(fh, key=key)


@cache
def hourly_offsets(key: str) -> tuple[int, ...]:
    """UTC offsets sampled every hour over 2026-2027 (enough to tell non-equivalent zones apart)."""
    zone = load_zone(key)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    offsets: list[int] = []
    for hour in range(0, 2 * 365 * 24):
        delta = (start + timedelta(hours=hour)).astimezone(zone).utcoffset()
        assert delta is not None
        offsets.append(int(delta.total_seconds()))
    return tuple(offsets)


def assert_valid_zone(key: str) -> None:
    assert key in tzdata_keys(), f"{key} is not a tzdata zone key"
    ZoneInfo(key)


@pytest.fixture(scope="module")
def tz_items() -> list[dict[str, Any]]:
    return read_jsonl("tz_phrases.jsonl")


@pytest.fixture(scope="module")
def belief_items() -> list[dict[str, Any]]:
    return read_jsonl("belief_extraction.jsonl")


# --- timezone phrases ------------------------------------------------------------------------------


def test_tz_phrases_counts_and_split(tz_items: list[dict[str, Any]]) -> None:
    assert len(tz_items) == 150
    assert sum(item["split"] == "dev" for item in tz_items) == 50
    assert sum(item["split"] == "test" for item in tz_items) == 100
    sources = [item["source"] for item in tz_items]
    assert (sources.count("city_sample"), sources.count("template"), sources.count("hard_case")) == (
        45,
        35,
        70,
    )


def test_tz_phrases_ids_unique_and_ordered(tz_items: list[dict[str, Any]]) -> None:
    ids = [item["id"] for item in tz_items]
    assert ids == [f"tzp-{i:03d}" for i in range(1, 151)]
    texts = [item["text"].casefold() for item in tz_items]
    assert len(set(texts)) == len(texts)


def test_tz_phrases_labels_are_consistent(tz_items: list[dict[str, Any]]) -> None:
    for item in tz_items:
        assert set(item) == {"id", "text", "label", "source", "split", "note"}
        assert item["text"].strip()
        assert item["note"].strip()
        label = item["label"]
        status, zone, candidates = label["status"], label["zone"], label["candidates"]
        if status == "resolved":
            assert zone is not None, item["id"]
            assert candidates == [], item["id"]
            assert_valid_zone(zone)
        elif status == "ambiguous":
            assert zone is None, item["id"]
            assert len(candidates) >= 2, item["id"]
            for key in candidates:
                assert_valid_zone(key)
            for i, a in enumerate(candidates):
                for b in candidates[i + 1 :]:
                    assert hourly_offsets(a) != hourly_offsets(b), f"{item['id']}: {a} and {b} are equivalent"
        else:
            assert status == "unknown", item["id"]
            assert zone is None, item["id"]
            assert candidates == [], item["id"]


def test_tz_phrases_cover_required_hard_cases(tz_items: list[dict[str, Any]]) -> None:
    by_text = {item["text"]: item["label"] for item in tz_items}
    assert by_text["same as you"] == {"status": "resolved", "zone": HOST_ZONE, "candidates": []}
    assert by_text["I'm in Pune"]["zone"] == "Asia/Kolkata"
    assert by_text["Eastern"]["zone"] == HOST_ZONE
    assert by_text["Central European Time"]["zone"] == "Europe/Berlin"
    assert by_text["IST"]["status"] == "ambiguous"
    assert "Asia/Kolkata" in by_text["IST"]["candidates"]
    assert "America/Chicago" in by_text["CST"]["candidates"]
    assert by_text["GMT+2"]["zone"] == "Etc/GMT-2"
    assert by_text["UTC-5"]["zone"] == "Etc/GMT+5"
    assert by_text["we're in Queensland"]["zone"] == "Australia/Brisbane"
    assert by_text["I'm in Portland"]["candidates"] == ["America/Los_Angeles", "America/New_York"]
    assert by_text["Portland, Maine"]["zone"] == HOST_ZONE
    assert by_text["I'm in Kathmandu"]["zone"] == "Asia/Kathmandu"
    assert by_text["on the moon"]["status"] == "unknown"


def test_tz_phrases_same_name_cities_follow_both_thresholds(tz_items: list[dict[str, Any]]) -> None:
    """Another same-name city counts only at >= 50,000 inhabitants and >= 10% of the top match."""
    by_text = {item["text"]: item["label"] for item in tz_items}
    # London, Ontario: 422,324 but 4.7% of London, England.
    assert by_text["London"] == {"status": "resolved", "zone": "Europe/London", "candidates": []}
    # Birmingham, Alabama: 17.0% of Birmingham, England (below the resolver's 20%, above 10%).
    assert by_text["Birmingham"]["candidates"] == ["America/Chicago", "Europe/London"]
    # Victoria, Texas (7.1% of Victoria, Hong Kong) is not a candidate; the Australian state is.
    assert by_text["Victoria"]["candidates"] == ["America/Vancouver", "Asia/Hong_Kong", "Australia/Melbourne"]
    # San Jose, Philippines (14.4% of San Jose, California) is.
    assert by_text["we're in San Jose"]["candidates"] == [
        "America/Costa_Rica",
        "America/Los_Angeles",
        "Asia/Manila",
    ]
    # Perth, Scotland is under 50,000.
    assert by_text["Perth"] == {"status": "resolved", "zone": "Australia/Perth", "candidates": []}


def test_city_rule_needs_both_thresholds(tmp_path: Path) -> None:
    build = load_script("build_tz_phrases")
    rows = [
        # exactly 10% and >= 50,000 in a non-equivalent zone: ambiguous (the boundary counts)
        ("Alpha", "US", "Oregon", 1_000_000, "America/Los_Angeles"),
        ("Alpha", "US", "Maine", 100_000, "America/New_York"),
        # just under 10%: resolved
        ("Beta", "US", "Oregon", 1_000_000, "America/Los_Angeles"),
        ("Beta", "US", "Maine", 99_999, "America/New_York"),
        # 12.5% but under 50,000: resolved
        ("Gamma", "US", "Oregon", 400_000, "America/Los_Angeles"),
        ("Gamma", "US", "Maine", 49_999, "America/New_York"),
        # passes both thresholds but in an equivalent zone: resolved
        ("Delta", "US", "New York", 500_000, "America/New_York"),
        ("Delta", "US", "Michigan", 400_000, "America/Detroit"),
        # only the matches that pass both thresholds become candidates
        ("Epsilon", "GB", "England", 2_000_000, "Europe/London"),
        ("Epsilon", "US", "Illinois", 150_000, "America/Chicago"),
        ("Epsilon", "CA", "Ontario", 300_000, "America/Toronto"),
        # the share is taken of the most populous match after qualification
        ("Zeta", "GB", "England", 5_000_000, "Europe/London"),
        ("Zeta", "US", "Texas", 400_000, "America/Chicago"),
        ("Zeta", "US", "Maine", 60_000, "America/New_York"),
    ]
    cities = tmp_path / "cities.csv"
    with cities.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CITIES_HEADER)
        for name, country, admin1, population, zone in rows:
            writer.writerow([name, name, country, "01", admin1, population, zone])
    tz = build.TzDb()
    gaz = build.Gazetteer(cities)

    def label(name: str, **qualifier: str) -> Any:
        return build.city_rule(tz, gaz, name, **qualifier)[0]

    assert label("Alpha") == build.ambiguous("America/Los_Angeles", "America/New_York")
    assert label("Beta") == build.resolved("America/Los_Angeles")
    assert label("Gamma") == build.resolved("America/Los_Angeles")
    assert label("Delta") == build.resolved("America/New_York")
    assert label("Epsilon") == build.ambiguous("Europe/London", "America/Toronto")
    assert label("Zeta") == build.resolved("Europe/London")
    assert label("Zeta", country="US") == build.ambiguous("America/Chicago", "America/New_York")


def test_tz_phrases_use_the_whole_hour_offset_convention(tz_items: list[dict[str, Any]]) -> None:
    """A written "UTC+N" / "GMT-N" is labelled with the Etc/GMT key of the inverted sign."""
    written = re.compile(r"(?<![/\w])(?:UTC|GMT)\s?([+-])(\d{1,2})(?::00)?\b(?!:)", re.IGNORECASE)
    checked = 0
    for item in tz_items:
        match = written.search(item["text"])
        if match is None or item["label"]["zone"] is None or "(" in item["text"]:
            continue
        sign, hours = match.group(1), int(match.group(2))
        inverted = "-" if sign == "+" else "+"
        assert item["label"]["zone"] == f"Etc/GMT{inverted}{hours}", item["id"]
        checked += 1
    assert checked >= 5


def test_tzdata_is_the_pinned_release() -> None:
    """Labels depend on the tz rules; zone.tab and iso3166.tab must ship in the package."""
    zoneinfo_dir = resources.files("tzdata.zoneinfo")
    header = zoneinfo_dir.joinpath("tzdata.zi").read_text(encoding="utf-8").splitlines()[0]
    assert header == "# version 2026d", "re-verify the dataset labels before changing tzdata"
    assert importlib.metadata.version("tzdata") == "2026.4"
    assert zoneinfo_dir.joinpath("zone.tab").is_file()
    assert zoneinfo_dir.joinpath("iso3166.tab").is_file()


# --- belief extraction -----------------------------------------------------------------------------


def test_belief_counts_and_split(belief_items: list[dict[str, Any]]) -> None:
    assert len(belief_items) == 120
    assert sum(item["split"] == "dev" for item in belief_items) == 40
    assert sum(item["split"] == "test" for item in belief_items) == 80
    assert sum(item["source"] == "hard_case" for item in belief_items) >= 30
    assert {item["source"] for item in belief_items} == {"template", "hard_case"}


def test_belief_ids_unique_and_ordered(belief_items: list[dict[str, Any]]) -> None:
    assert [item["id"] for item in belief_items] == [f"be-{i:03d}" for i in range(1, 121)]


def test_belief_items_are_well_formed(belief_items: list[dict[str, Any]]) -> None:
    keys = {"id", "as_of", "prospect_zone", "host_zone", "agent_messages", "gold", "tags", "source", "split"}
    for item in belief_items:
        assert set(item) == keys
        assert RFC3339_Z.fullmatch(item["as_of"]), item["id"]
        assert_valid_zone(item["prospect_zone"])
        assert item["host_zone"] == HOST_ZONE
        messages = item["agent_messages"]
        assert 1 <= len(messages) <= 4, item["id"]
        assert all(isinstance(m, str) and m.strip() for m in messages)
        assert item["tags"] == sorted(set(item["tags"]))
        gold = item["gold"]
        assert set(gold) == {"status", "time_utc", "offered_utc"}
        assert gold["status"] in {"booked", "rescheduled", "cancelled", "not_booked", "unclear"}
        offered = gold["offered_utc"]
        assert offered == sorted(set(offered)), item["id"]
        for stamp in offered + ([gold["time_utc"]] if gold["time_utc"] else []):
            assert RFC3339_Z.fullmatch(stamp), item["id"]


def test_belief_time_is_null_exactly_when_the_taxonomy_says(belief_items: list[dict[str, Any]]) -> None:
    for item in belief_items:
        gold = item["gold"]
        if gold["status"] in ("not_booked", "unclear"):
            assert gold["time_utc"] is None, item["id"]
        elif gold["status"] in ("booked", "rescheduled") and gold["time_utc"] is None:
            assert "no_time" in item["tags"], item["id"]
    for status in ("booked", "rescheduled", "cancelled"):
        times = [item["gold"]["time_utc"] for item in belief_items if item["gold"]["status"] == status]
        assert any(t is None for t in times), status
        assert any(t is not None for t in times), status


MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
TIME_12H = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s?([AaPp][Mm])\b")
TIME_24H = re.compile(r"(?<![\d:+-])(\d{1,2}):(\d{2})(?!\d)(?!\s?[AaPp][Mm])")
BARE_HOUR = re.compile(r"\bat (\d{1,2})\b(?![:\d])(?!\s?[AaPp][Mm])")
RENDERED = re.compile(
    r"(Booked|Rescheduled|Cancelled): (\w+) (\d{1,2}) (\w+) (\d{4}), (\d{1,2}):(\d{2}) (AM|PM) "
    r"([A-Za-z_]+/[A-Za-z_/]+) \(UTC([+-])(\d{2}):(\d{2})\) · reference [a-z0-9]{6}"
)


def parse_z(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


HOST_LABEL = re.compile(r"\s?(?:ET|EST|EDT|Eastern(?: Time)?|New York time|our time)\b")
UTC_LABEL = re.compile(r"\s?UTC\b(?![+-])")
IANA_LABEL = re.compile(r"\s?\(?([A-Z][A-Za-z_]+/[A-Z][A-Za-z_/]+)")


@dataclass(frozen=True)
class ClockMention:
    hours: tuple[int, ...]  # 24-hour candidates (two for a bare "at 10")
    minute: int
    start: int
    end: int


def clock_mentions(text: str) -> list[ClockMention]:
    found: list[ClockMention] = []
    for m in TIME_12H.finditer(text):
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        found.append(ClockMention((hour,), int(m.group(2) or 0), m.start(), m.end()))
    for m in TIME_24H.finditer(text):
        found.append(ClockMention((int(m.group(1)),), int(m.group(2)), m.start(), m.end()))
    for m in BARE_HOUR.finditer(text):
        hour = int(m.group(1)) % 12
        found.append(ClockMention((hour, hour + 12), 0, m.start(1), m.end(1)))
    return found


def label_zone(after: str, prospect_zone: str) -> str:
    """Zone of a clock time from the words right after it; no recognised label = the prospect's zone."""
    if UTC_LABEL.match(after):
        return "UTC"
    if HOST_LABEL.match(after):
        return HOST_ZONE
    iana = IANA_LABEL.match(after)
    if iana is not None and iana.group(1) in tzdata_keys():
        return iana.group(1)
    return prospect_zone  # "your time", "Berlin time", "Central", "for you in Berlin" or no label


MONTH_ALT = "|".join(m[:3] for m in MONTHS)
WEEKDAY_ALT = "|".join(w[:3] for w in WEEKDAYS)
ABSOLUTE_DATE = re.compile(rf"\b(?:(\d{{1,2}}) ({MONTH_ALT})[a-z]*|({MONTH_ALT})[a-z]* (\d{{1,2}})\b)")
RELATIVE_DATE = re.compile(rf"\b(?:tomorrow|(?:{WEEKDAY_ALT})[a-z]*\b(?!,? (?:\d|{MONTH_ALT})))")


@dataclass(frozen=True)
class DateMention:
    start: int
    end: int
    month_day: tuple[int, int] | None  # absolute date
    word: str | None  # "tomorrow" or a weekday name


def date_mentions(text: str) -> list[DateMention]:
    months = [m[:3] for m in MONTHS]
    found = [
        DateMention(
            m.start(),
            m.end(),
            (months.index(m.group(2) or m.group(3)) + 1, int(m.group(1) or m.group(4))),
            None,
        )
        for m in ABSOLUTE_DATE.finditer(text)
    ]
    found += [DateMention(m.start(), m.end(), None, m.group(0)) for m in RELATIVE_DATE.finditer(text)]
    return sorted(found, key=lambda d: d.start)


def date_of(mention: ClockMention, dates: list[DateMention]) -> DateMention | None:
    """The date a clock time belongs to: the nearest date written before it, else the first after it."""
    before = [d for d in dates if d.end <= mention.start]
    if before:
        return before[-1]
    after = [d for d in dates if d.start >= mention.end]
    return after[0] if after else None


def is_stated(instant: datetime, item: dict[str, Any]) -> bool:
    """Some clock time in the messages, read in the zone its label names and on the date written for
    it, is exactly this instant."""
    as_of = parse_z(item["as_of"])
    for text in item["agent_messages"]:
        dates = date_mentions(text)
        for mention in clock_mentions(text):
            key = label_zone(text[mention.end : mention.end + 40], item["prospect_zone"])
            local = instant.astimezone(UTC if key == "UTC" else load_zone(key))
            if local.hour not in mention.hours or local.minute != mention.minute:
                continue
            date = date_of(mention, dates)
            if date is None:
                continue
            if date.month_day is not None:
                if date.month_day == (local.month, local.day):
                    return True
                continue
            assert date.word is not None
            ahead = (local.date() - as_of.astimezone(local.tzinfo).date()).days
            if date.word == "tomorrow":
                if ahead == 1:
                    return True
            elif WEEKDAYS[local.weekday()].startswith(date.word[:3]) and 1 <= ahead <= 6:
                return True
    return False


def test_belief_gold_times_are_written_in_the_messages(belief_items: list[dict[str, Any]]) -> None:
    """Independent of the builder: every gold instant is re-derived from the text, reading each clock
    time in the zone its label names (none = the prospect's zone) on the date written next to it. A
    wrong DST offset or date in the builder would shift the instant and fail."""
    for item in belief_items:
        gold = item["gold"]
        for stamp in gold["offered_utc"] + ([gold["time_utc"]] if gold["time_utc"] else []):
            assert is_stated(parse_z(stamp), item), f"{item['id']}: {stamp} not in {item['agent_messages']!r}"


def test_belief_rendered_lines_are_self_consistent(belief_items: list[dict[str, Any]]) -> None:
    verbs = {"Booked": "booked", "Rescheduled": "rescheduled", "Cancelled": "cancelled"}
    seen = 0
    for item in belief_items:
        for index, message in enumerate(item["agent_messages"]):
            match = RENDERED.search(message)
            if match is None:
                continue
            seen += 1
            verb, weekday, day, month, year, hour, minute, ampm, key, sign, off_h, off_m = match.groups()
            hour24 = int(hour) % 12 + (12 if ampm == "PM" else 0)
            local = datetime(
                int(year), MONTHS.index(month) + 1, int(day), hour24, int(minute), tzinfo=load_zone(key)
            )
            assert WEEKDAYS[local.weekday()] == weekday, item["id"]
            offset = (int(off_h) * 60 + int(off_m)) * (1 if sign == "+" else -1)
            assert local.utcoffset() == timedelta(minutes=offset), item["id"]
            if index == len(item["agent_messages"]) - 1:
                assert item["gold"]["status"] == verbs[verb], item["id"]
                assert item["gold"]["time_utc"] == local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert seen >= 10


def test_belief_weekday_and_date_mentions_agree(belief_items: list[dict[str, Any]]) -> None:
    pattern = re.compile(
        rf"\b({'|'.join(w[:3] for w in WEEKDAYS)})[a-z]*,? (?:(\d{{1,2}}) ({'|'.join(m[:3] for m in MONTHS)})"
        rf"|({'|'.join(m[:3] for m in MONTHS)})[a-z]* (\d{{1,2}}))"
    )
    for item in belief_items:
        as_of = parse_z(item["as_of"])
        for message in item["agent_messages"]:
            for match in pattern.finditer(message):
                month = [m[:3] for m in MONTHS].index(match.group(3) or match.group(4)) + 1
                day = int(match.group(2) or match.group(5))
                year = as_of.year if month >= as_of.month else as_of.year + 1
                weekday = datetime(year, month, day).weekday()
                assert WEEKDAYS[weekday].startswith(match.group(1)), f"{item['id']}: {match.group(0)}"


def test_belief_builder_rejects_gaps_folds_and_wrong_weekdays() -> None:
    build = load_script("build_belief_extraction")
    assert build.local_to_utc("America/New_York", "2027-03-15 10:00") == datetime(
        2027, 3, 15, 14, 0, tzinfo=UTC
    )
    with pytest.raises(build.BuildError, match="nonexistent"):
        build.local_to_utc("America/New_York", "2027-03-14 02:30")
    with pytest.raises(build.BuildError, match="ambiguous"):
        build.local_to_utc("America/New_York", "2026-11-01 01:30")
    with pytest.raises(build.BuildError, match="nonexistent"):
        build.local_to_utc("Europe/Berlin", "2027-03-28 02:15")
    wrong = build.Item(
        as_of=datetime(2026, 10, 5, 14, tzinfo=UTC),
        prospect_zone="America/Chicago",
        messages=["You're booked for Tuesday 7 October at 10:00 AM."],
        gold=build.Gold("booked", datetime(2026, 10, 7, 15, tzinfo=UTC), ()),
        tags=set(),
        source="hard_case",
    )
    with pytest.raises(build.BuildError, match="is not a Tuesday"):
        build.check_weekdays(wrong)


def test_belief_as_of_spans_the_dst_season(belief_items: list[dict[str, Any]]) -> None:
    months = {item["as_of"][:7] for item in belief_items}
    assert {"2026-10", "2026-11", "2027-03", "2027-04"} <= months
    assert sum("dst_week" in item["tags"] for item in belief_items) >= 5


# --- hashes ----------------------------------------------------------------------------------------


def test_recorded_test_hashes_match() -> None:
    hashes = load_script("dataset_hashes")
    recorded = json.loads((DATASETS / "HASHES.json").read_text(encoding="utf-8"))
    assert recorded["recorded"] == "2026-09-26"
    assert hashes.compute_hashes(DATASETS) == {
        "tz_phrases_test_sha256": recorded["tz_phrases_test_sha256"],
        "belief_extraction_test_sha256": recorded["belief_extraction_test_sha256"],
    }
    assert hashes.main(["--check", "--datasets-dir", str(DATASETS)]) == 0


def test_hash_check_detects_a_changed_test_item(tmp_path: Path) -> None:
    hashes = load_script("dataset_hashes")
    for name in ("tz_phrases.jsonl", "belief_extraction.jsonl", "HASHES.json"):
        (tmp_path / name).write_bytes((DATASETS / name).read_bytes())
    items = read_jsonl("tz_phrases.jsonl")
    dev_only = [dict(item, note="edited") if item["split"] == "dev" else item for item in items]
    (tmp_path / "tz_phrases.jsonl").write_text(
        "".join(json.dumps(i) + "\n" for i in dev_only), encoding="utf-8"
    )
    assert hashes.main(["--check", "--datasets-dir", str(tmp_path)]) == 0
    tampered = [dict(item, note="edited") if item["split"] == "test" else item for item in items]
    (tmp_path / "tz_phrases.jsonl").write_text(
        "".join(json.dumps(i) + "\n" for i in tampered), encoding="utf-8"
    )
    assert hashes.main(["--check", "--datasets-dir", str(tmp_path)]) == 1
    assert hashes.main(["--datasets-dir", str(tmp_path)]) == 1


# --- cities table ----------------------------------------------------------------------------------


def test_cities_table_shape() -> None:
    with (DATASETS / "cities_tz.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        assert next(reader) == CITIES_HEADER
        rows = list(reader)
    assert len(rows) >= 20_000
    keys = tzdata_keys()
    for row in rows:
        assert len(row) == len(CITIES_HEADER)
        assert int(row[5]) >= 0
        assert row[6] in keys


def test_cities_builder_parses_and_filters(tmp_path: Path) -> None:
    build = load_script("build_cities_tz")
    fields = ["0"] * 19

    def geoname(gid: str, name: str, ascii_name: str, cc: str, admin1: str, pop: str, tz: str) -> str:
        row = list(fields)
        row[0], row[1], row[2], row[8], row[10], row[14], row[17] = gid, name, ascii_name, cc, admin1, pop, tz
        return "\t".join(row)

    table = "\n".join(
        [
            geoname("3", "Zürich", "Zurich", "CH", "ZH", "400000", "Europe/Zurich"),
            geoname("1", "Portland", "Portland", "US", "OR", "650000", "America/Los_Angeles"),
            geoname("2", "Portland", "Portland", "US", "ME", "66000", "America/New_York"),
            geoname("4", "Nowhere", "Nowhere", "XX", "01", "20000", "Mars/Olympus_Mons"),
        ]
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("cities15000.txt", table + "\n")
    zip_path = tmp_path / "cities15000.zip"
    zip_path.write_bytes(archive.getvalue())
    admin1_path = tmp_path / "admin1CodesASCII.txt"
    admin1_path.write_text("US.OR\tOregon\tOregon\t1\nUS.ME\tMaine\tMaine\t2\n", encoding="utf-8")

    cities, dropped = build.load_cities(
        zip_path, build.load_admin1_names(admin1_path), build.tzdata_zone_keys()
    )
    out = tmp_path / "cities.csv"
    build.write_csv(cities, out)

    assert dropped == 1
    assert out.read_text(encoding="utf-8").splitlines() == [
        ",".join(CITIES_HEADER),
        "Portland,Portland,US,ME,Maine,66000,America/New_York",
        "Portland,Portland,US,OR,Oregon,650000,America/Los_Angeles",
        "Zürich,Zurich,CH,ZH,,400000,Europe/Zurich",
    ]


# --- determinism -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("script", "dataset"),
    [("build_tz_phrases", "tz_phrases.jsonl"), ("build_belief_extraction", "belief_extraction.jsonl")],
)
def test_builders_reproduce_the_committed_files(tmp_path: Path, script: str, dataset: str) -> None:
    module = load_script(script)
    out = tmp_path / dataset
    assert module.main(["--out", str(out)]) == 0
    assert out.read_bytes() == (DATASETS / dataset).read_bytes()

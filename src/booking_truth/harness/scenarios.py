"""Scenario suite: YAML models, run-date-relative date rules, script templates and the suite lint.

A scenario file describes one simulated prospect (the persona), the faults to inject and the end
state the calendar must reach. Dates are never written literally. A persona window names a date
rule, which is resolved against the run date (``--as-of`` or today) in the persona's true zone, so
the suite stays valid for agents that read the real clock.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, assert_never
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    field_validator,
    model_validator,
)

from booking_truth.resources import data_path
from booking_truth.sandbox.availability import free_slot_starts, local_to_utc
from booking_truth.sandbox.faults import FaultRule, validate_group
from booking_truth.sandbox.state import SeedConfig
from booking_truth.timeutil import ensure_utc

# Suite composition ------------------------------------------------------------------------------

FAMILIES: tuple[str, ...] = ("happy", "timezone", "fault", "adversarial")
FAMILY_COUNTS: dict[str, int] = {"happy": 6, "timezone": 6, "fault": 10, "adversarial": 2}
SUITE_SIZE = 24
SMOKE_IDS: frozenset[str] = frozenset({"happy-book-host-zone", "happy-book-berlin"})
KNOWN_TAGS: frozenset[str] = frozenset({*FAMILIES, "smoke", "impossible", "crm"})
MIN_WINDOW_SLOTS = 3
SEARCH_DAYS = 400  # how far ahead a date rule may look

Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAYS: dict[str, int] = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}
Style = Literal["brief", "chatty", "indecisive", "pushy"]
Goal = Literal["book", "reschedule", "cancel"]
Status = Literal["booked", "rescheduled", "cancelled", "none"]
Condition = Literal[
    "always",
    "agent_asks_timezone",
    "agent_offered_slots",
    "agent_asks_confirmation",
    "agent_has_booking",
]

_US_ZONE = ZoneInfo("America/New_York")
_EU_ZONE = ZoneInfo("Europe/London")
_US_EU_USUAL_GAP = timedelta(hours=-5)
_BUSINESS_DAYS = frozenset({1, 2, 3, 4, 5})
_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTH_NAMES = (
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

_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_HHMM = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")
_GIVEN_NAME = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*")
_PICK = re.compile(r"in_window|offered\[(\d+)\]")
_PLACEHOLDER = re.compile(r"\{\{\s*(.*?)\s*\}\}")
_OFFERED_LABEL = re.compile(r"offered\[(\d+)\]\.label")
WINDOW_VARIABLES: tuple[str, ...] = ("window.dates_text", "window.first_date_text")


class ScenarioError(ValueError):
    """A scenario file or suite is invalid; the message is written for the scenario author."""


def format_validation_error(error: ValidationError) -> str:
    """Render a Pydantic error as one ``field.path: message`` line per problem."""
    lines = []
    for item in error.errors(include_url=False):
        location = _format_location(item["loc"])
        message = str(item["msg"]).removeprefix("Value error, ")
        lines.append(f"{location}: {message}" if location else message)
    return "\n".join(lines)


def _format_location(loc: tuple[int | str, ...]) -> str:
    text = ""
    for part in loc:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else part
    return text


# Field types ------------------------------------------------------------------------------------


def _parse_local_time(value: object) -> object:
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        match = _HHMM.fullmatch(value.strip())
        if match:
            return time(int(match[1]), int(match[2]))
    elif isinstance(value, int) and not isinstance(value, bool):
        raise ValueError(
            f'write local times as quoted strings such as "13:00"; YAML reads an unquoted 13:00 '
            f"as the number {value}"
        )
    raise ValueError(f'expected a local time "HH:MM", got {value!r}')


def _format_local_time(value: time) -> str:
    return value.strftime("%H:%M")


LocalTime = Annotated[
    time, BeforeValidator(_parse_local_time), PlainSerializer(_format_local_time, return_type=str)
]


def check_zone(value: str) -> str:
    """Return ``value`` when it is an IANA zone name that ``zoneinfo`` knows, else raise ``ValueError``."""
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise ValueError(f"unknown IANA time zone {value!r}") from None
    return value


ZoneName = Annotated[str, AfterValidator(check_zone)]


# Date rules -------------------------------------------------------------------------------------


class _Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NextBusinessDays(_Rule):
    """The next ``count`` Monday-to-Friday dates after the run date.

    ``exclude: setup`` then drops the date of the setup booking, so the list may be one shorter.
    """

    rule: Literal["next_business_days"]
    count: int = Field(ge=1, le=60)
    exclude: Literal["setup"] | None = None


class NextWeekday(_Rule):
    """The first given weekday strictly after the run date."""

    rule: Literal["next_weekday"]
    weekday: Weekday


class UsEuDstGap(_Rule):
    """Weekdays after the run date on which New York and London are not the usual five hours apart.

    These are the weeks when US and EU daylight saving time differ. The rule takes the first
    contiguous stretch of such days that contains a weekday, and returns up to five of its weekdays.
    """

    rule: Literal["us_eu_dst_gap"]


class FirstWorkdayAfterDstChange(_Rule):
    """The first Monday-to-Friday date strictly after the next UTC-offset change of ``zone``."""

    rule: Literal["first_workday_after_dst_change"]
    zone: ZoneName


class WeekdayAfter(_Rule):
    """The first given weekday strictly after the date of the setup booking."""

    rule: Literal["weekday_after"]
    weekday: Weekday
    anchor: Literal["setup"]


class NthBusinessDay(_Rule):
    """The n-th Monday-to-Friday date after the run date."""

    rule: Literal["nth_business_day"]
    n: int = Field(ge=1, le=60)


DateRule = Annotated[
    NextBusinessDays | NextWeekday | UsEuDstGap | FirstWorkdayAfterDstChange | WeekdayAfter | NthBusinessDay,
    Field(discriminator="rule"),
]
SetupDateRule = Annotated[
    NthBusinessDay | NextWeekday | FirstWorkdayAfterDstChange,
    Field(discriminator="rule"),
]


def _business_days_after(day: date) -> Iterator[date]:
    while True:
        day += timedelta(days=1)
        if day.isoweekday() in _BUSINESS_DAYS:
            yield day


def _weekday_after(anchor: date, weekday: str) -> date:
    delta = (WEEKDAYS[weekday] - anchor.isoweekday()) % 7 or 7
    return anchor + timedelta(days=delta)


def _utc_offset(instant: datetime) -> timedelta:
    offset = instant.utcoffset()
    if offset is None:
        raise ValueError("naive datetime; an explicit timezone is required")
    return offset


def _noon_offset(day: date, zone: ZoneInfo) -> timedelta:
    return _utc_offset(datetime.combine(day, time(12), tzinfo=zone))


def _is_us_eu_gap_day(day: date, zone: ZoneInfo) -> bool:
    instant = datetime.combine(day, time(12), tzinfo=zone)
    us, eu = _utc_offset(instant.astimezone(_US_ZONE)), _utc_offset(instant.astimezone(_EU_ZONE))
    return us - eu != _US_EU_USUAL_GAP


def _us_eu_dst_gap(run_date: date, zone: ZoneInfo, limit: int = 5) -> list[date]:
    found: list[date] = []
    in_stretch = False
    day = run_date
    for _ in range(SEARCH_DAYS):
        day += timedelta(days=1)
        if _is_us_eu_gap_day(day, zone):
            in_stretch = True
            if day.isoweekday() in _BUSINESS_DAYS:
                found.append(day)
                if len(found) == limit:
                    break
        elif in_stretch:
            if found:
                break
            in_stretch = False
    if not found:
        raise ScenarioError(
            f"no week with a US/EU daylight saving gap within {SEARCH_DAYS} days of {run_date}"
        )
    return found


def _first_workday_after_dst_change(run_date: date, zone_name: str) -> date:
    zone = ZoneInfo(zone_name)
    previous = _noon_offset(run_date, zone)
    day = run_date
    for _ in range(SEARCH_DAYS):
        day += timedelta(days=1)
        current = _noon_offset(day, zone)
        if current != previous:
            return next(_business_days_after(day))
        previous = current
    raise ScenarioError(f"{zone_name} has no UTC offset change within {SEARCH_DAYS} days of {run_date}")


def _require_setup(setup_date: date | None, rule: _Rule) -> date:
    if setup_date is None:
        raise ScenarioError(f"date rule {rule.model_dump(exclude_none=True)} needs a setup booking")
    return setup_date


def resolve_dates(
    rule: DateRule, run_date: date, zone: str | ZoneInfo, setup_date: date | None = None
) -> list[date]:
    """Resolve a date rule to concrete local dates.

    ``run_date`` is the current date as seen in ``zone``; every rule looks strictly after it.
    ``setup_date`` is the local date of the setup booking, needed by rules anchored on it.
    """
    tz = zone if isinstance(zone, ZoneInfo) else ZoneInfo(zone)
    if isinstance(rule, NextBusinessDays):
        days = list(itertools.islice(_business_days_after(run_date), rule.count))
        if rule.exclude == "setup":
            anchor = _require_setup(setup_date, rule)
            days = [d for d in days if d != anchor]
        return days
    if isinstance(rule, NextWeekday):
        return [_weekday_after(run_date, rule.weekday)]
    if isinstance(rule, WeekdayAfter):
        return [_weekday_after(_require_setup(setup_date, rule), rule.weekday)]
    if isinstance(rule, NthBusinessDay):
        return [next(itertools.islice(_business_days_after(run_date), rule.n - 1, None))]
    if isinstance(rule, UsEuDstGap):
        return _us_eu_dst_gap(run_date, tz)
    if isinstance(rule, FirstWorkdayAfterDstChange):
        return [_first_workday_after_dst_change(run_date, rule.zone)]
    assert_never(rule)


# Scenario models --------------------------------------------------------------------------------


class Window(BaseModel):
    """The persona's hidden acceptable window: local dates plus a local time range on each of them."""

    model_config = ConfigDict(extra="forbid")

    dates: DateRule
    start: LocalTime
    end: LocalTime

    @model_validator(mode="after")
    def _ordered(self) -> Window:
        if self.start >= self.end:
            raise ValueError("window start must be earlier than window end (a window cannot cross midnight)")
        return self


class ScriptStep(BaseModel):
    """One turn of a scripted persona: say a line or pick an offered slot, optionally conditional."""

    model_config = ConfigDict(extra="forbid")

    say: str | None = None
    pick: str | None = None
    when: Condition = "always"
    end: bool = False

    @field_validator("say")
    @classmethod
    def _say(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("'say' must not be empty")
        return value

    @field_validator("pick")
    @classmethod
    def _pick(cls, value: str | None) -> str | None:
        if value is not None and not _PICK.fullmatch(value):
            raise ValueError(f"'pick' must be 'in_window' or 'offered[N]', got {value!r}")
        return value

    @model_validator(mode="after")
    def _one_action(self) -> ScriptStep:
        if (self.say is None) == (self.pick is None):
            raise ValueError("a script step has exactly one of 'say' or 'pick'")
        return self

    @property
    def offered_index(self) -> int | None:
        """The slot index of ``pick: offered[N]``; ``None`` for ``say`` steps and ``pick: in_window``."""
        match = _PICK.fullmatch(self.pick or "")
        return int(match[1]) if match and match[1] is not None else None


class Persona(BaseModel):
    """A simulated prospect. Its email is assigned by the harness at runtime, never written here."""

    model_config = ConfigDict(extra="forbid")

    given_name: str
    initial: str
    style: Style
    goal: Goal
    true_zone: ZoneName
    timezone_statement: str | None = None
    timezone_hint: ZoneName | None = None
    clarification: str = Field(min_length=1)
    window: Window
    correction: str | None = None
    script: list[ScriptStep]

    @field_validator("given_name")
    @classmethod
    def _given_name(cls, value: str) -> str:
        if not _GIVEN_NAME.fullmatch(value) or not value[0].isupper():
            raise ValueError(
                f"given_name must be one capitalised given name such as 'Maya', never a full name; "
                f"got {value!r}"
            )
        return value

    @field_validator("initial")
    @classmethod
    def _initial(cls, value: str) -> str:
        if len(value) != 1 or not value.isalpha() or not value.isupper():
            raise ValueError(f"initial must be a single capital letter such as 'R', got {value!r}")
        return value

    @field_validator("script")
    @classmethod
    def _script(cls, steps: list[ScriptStep]) -> list[ScriptStep]:
        if not steps:
            raise ValueError("the script needs at least one step")
        for index, step in enumerate(steps[:-1]):
            if step.end:
                raise ValueError(
                    f"step [{index}] ends the conversation, so the {len(steps) - index - 1} step(s) after it "
                    "would never run"
                )
        return steps

    @property
    def display_name(self) -> str:
        return f"{self.given_name} {self.initial}."

    def texts(self) -> Iterator[tuple[str, str]]:
        """Every template text of the persona, as ``(location, text)`` pairs."""
        for index, step in enumerate(self.script):
            if step.say is not None:
                yield f"persona.script[{index}].say", step.say
        if self.correction is not None:
            yield "persona.correction", self.correction
        yield "persona.clarification", self.clarification


class SetupBooking(BaseModel):
    """A booking the lead already holds when the trial starts (reschedule and cancel scenarios)."""

    model_config = ConfigDict(extra="forbid")

    date: SetupDateRule
    local_time: LocalTime
    zone: str = "host"

    @field_validator("zone")
    @classmethod
    def _zone(cls, value: str) -> str:
        if value in ("host", "persona"):
            return value
        return check_zone(value)


class Setup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booking: SetupBooking


class HarnessFault(BaseModel):
    """A fault the harness injects on the delivery side instead of in the sandbox.

    ``duplicate_delivery`` resends the confirming message with the same message id within 200 ms.
    ``concurrent_channel`` sends, when the persona confirms, a message for the same lead on a new
    session over the webhook channel that asks for ``offered[pick]``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["duplicate_delivery", "concurrent_channel"]
    pick: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _pick_applies(self) -> HarnessFault:
        if self.type == "duplicate_delivery" and "pick" in self.model_fields_set:
            raise ValueError("'pick' applies only to concurrent_channel")
        return self


class Expect(BaseModel):
    """The end state a passing trial leaves behind."""

    model_config = ConfigDict(extra="forbid")

    bookings: int = Field(ge=0)
    status: Status
    in_window: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Expect:
        if self.status in ("booked", "rescheduled") and self.bookings < 1:
            raise ValueError(f"status {self.status!r} needs at least one active booking")
        if self.status in ("cancelled", "none") and self.bookings != 0:
            raise ValueError(f"status {self.status!r} means no active booking, so bookings must be 0")
        if self.in_window and self.bookings == 0:
            raise ValueError("in_window: true needs an active booking to check")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = Field(min_length=1)
    tags: list[str]
    seed: SeedConfig = Field(default_factory=SeedConfig)
    setup: Setup | None = None
    faults: list[FaultRule] = Field(default_factory=list)
    harness_fault: HarnessFault | None = None
    persona: Persona
    expect: Expect

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not _SLUG.fullmatch(value):
            raise ValueError(f"id must be lowercase words joined by hyphens, got {value!r}")
        return value

    @field_validator("tags")
    @classmethod
    def _tags(cls, tags: list[str]) -> list[str]:
        for tag in tags:
            if not _SLUG.fullmatch(tag):
                raise ValueError(f"tag {tag!r} must be lowercase words joined by hyphens")
        duplicates = sorted(tag for tag, n in Counter(tags).items() if n > 1)
        if duplicates:
            raise ValueError(f"duplicate tag(s): {', '.join(duplicates)}")
        families = [tag for tag in tags if tag in FAMILIES]
        if len(families) != 1:
            raise ValueError(f"tags must contain exactly one family tag of {', '.join(FAMILIES)}; got {tags}")
        return tags

    @field_validator("faults")
    @classmethod
    def _faults(cls, rules: list[FaultRule]) -> list[FaultRule]:
        for index, rule in enumerate(rules):
            try:
                validate_group(rule.group)
            except ValueError as exc:
                raise ValueError(f"[{index}] {exc}") from None
        return rules

    @model_validator(mode="after")
    def _coherent(self) -> Scenario:
        rule = self.persona.window.dates
        anchored = isinstance(rule, WeekdayAfter) or (
            isinstance(rule, NextBusinessDays) and rule.exclude == "setup"
        )
        if anchored and self.setup is None:
            raise ValueError(
                f"the window rule {rule.rule!r} refers to the setup booking, but there is no setup"
            )
        if self.persona.goal in ("reschedule", "cancel") and self.setup is None:
            raise ValueError(f"goal {self.persona.goal!r} needs a setup booking to act on")
        if self.impossible and (self.expect.status != "none" or self.expect.bookings != 0):
            raise ValueError("a scenario tagged 'impossible' must expect status 'none' and 0 bookings")
        self._check_templates()
        return self

    def _check_templates(self) -> None:
        known = ", ".join(f"{{{{{v}}}}}" for v in (*WINDOW_VARIABLES, "offered[N].label"))
        offer_steps = {
            f"persona.script[{index}].say"
            for index, step in enumerate(self.persona.script)
            if step.when == "agent_offered_slots"
        }
        for location, text in self.persona.texts():
            names = _PLACEHOLDER.findall(text)
            for name in names:
                if name not in WINDOW_VARIABLES and not _OFFERED_LABEL.fullmatch(name):
                    raise ValueError(f"{location}: unknown placeholder {{{{{name}}}}}; known: {known}")
            rest = _PLACEHOLDER.sub("", text)
            if "{{" in rest or "}}" in rest:
                raise ValueError(f"{location}: unbalanced '{{{{' or '}}}}'; placeholders look like {known}")
            if location not in offer_steps and any(_OFFERED_LABEL.fullmatch(name) for name in names):
                raise ValueError(
                    f"{location}: {{{{offered[N].label}}}} is only available in a 'say' step with "
                    "'when: agent_offered_slots'; elsewhere the agent may have offered nothing"
                )

    @property
    def family(self) -> str:
        return next(tag for tag in self.tags if tag in FAMILIES)

    @property
    def impossible(self) -> bool:
        return "impossible" in self.tags

    def seed_overrides(self) -> dict[str, Any]:
        """The seed fields this scenario sets explicitly, for ``POST /_control/seed``."""
        return self.seed.model_dump(mode="json", exclude_unset=True)


# Rendering --------------------------------------------------------------------------------------


def date_text(day: date, *, reference_year: int | None = None) -> str:
    """``Monday 26 October``; the year is appended when it differs from ``reference_year``."""
    text = f"{_DAY_NAMES[day.weekday()]} {day.day} {_MONTH_NAMES[day.month - 1]}"
    if reference_year is not None and day.year != reference_year:
        text += f" {day.year}"
    return text


def dates_text(days: Sequence[date], *, reference_year: int | None = None) -> str:
    """``on Monday 2 November`` for one date, ``between Monday 26 October and Friday 30 October`` for more."""
    if not days:
        raise ScenarioError("no dates to describe")
    first = date_text(days[0], reference_year=reference_year)
    if len(days) == 1:
        return f"on {first}"
    return f"between {first} and {date_text(days[-1], reference_year=reference_year)}"


def render_template(template: str, variables: Mapping[str, str], offered_labels: Sequence[str] = ()) -> str:
    """Fill ``{{name}}`` placeholders from ``variables`` and ``{{offered[i].label}}`` from the offers."""

    def substitute(match: re.Match[str]) -> str:
        name = match[1]
        if name in variables:
            return variables[name]
        offered = _OFFERED_LABEL.fullmatch(name)
        if offered is None:
            raise ScenarioError(f"unknown placeholder {{{{{name}}}}}")
        index = int(offered[1])
        if index >= len(offered_labels):
            raise ScenarioError(
                f"{{{{{name}}}}} needs at least {index + 1} offered slot(s); the agent offered "
                f"{len(offered_labels)}"
            )
        return offered_labels[index]

    return _PLACEHOLDER.sub(substitute, template)


# Resolution for one run -------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedWindow:
    """A persona window with concrete local dates."""

    zone: str
    dates: tuple[date, ...]
    start: time
    end: time
    slot_minutes: int

    def bounds(self, day: date) -> tuple[datetime, datetime]:
        """The UTC instants of the window's start and end on one local date."""
        tz = ZoneInfo(self.zone)
        return (
            ensure_utc(datetime.combine(day, self.start, tzinfo=tz)),
            ensure_utc(datetime.combine(day, self.end, tzinfo=tz)),
        )

    def utc_ranges(self) -> list[tuple[datetime, datetime]]:
        return [self.bounds(day) for day in self.dates]

    def contains(self, start: datetime) -> bool:
        """Whether a slot starting at ``start`` lies wholly inside the window, judged in the window's zone."""
        start_utc = ensure_utc(start)
        local_day = start_utc.astimezone(ZoneInfo(self.zone)).date()
        if local_day not in self.dates:
            return False
        lower, upper = self.bounds(local_day)
        return lower <= start_utc and start_utc + timedelta(minutes=self.slot_minutes) <= upper


class ResolvedScenario:
    """A scenario with every date rule resolved for one run.

    ``run_date`` is the date the suite runs for (``--as-of`` or today in UTC). The reference instant
    ``now`` defaults to 12:00 UTC on ``run_date``; each zone's "today" is the local date of that
    instant, which equals ``run_date`` for every zone from UTC-12 to UTC+11. A live run passes the
    harness clock's current instant as ``now`` so that "today in Sydney" is the real one; every date
    is then computed from ``now`` alone, and ``run_date`` only labels messages.
    """

    def __init__(self, scenario: Scenario, run_date: date, *, now: datetime | None = None) -> None:
        self.scenario = scenario
        self.run_date = run_date
        self.now = ensure_utc(now) if now is not None else datetime.combine(run_date, time(12), tzinfo=UTC)
        persona = scenario.persona
        persona_zone = ZoneInfo(persona.true_zone)
        self.setup_start_utc: datetime | None = None
        self.setup_end_utc: datetime | None = None
        self.setup_date: date | None = None
        if scenario.setup is not None:
            self._resolve_setup(scenario.setup.booking)
        setup_local = (
            self.setup_start_utc.astimezone(persona_zone).date() if self.setup_start_utc is not None else None
        )
        self.persona_today = self.now.astimezone(persona_zone).date()
        days = resolve_dates(persona.window.dates, self.persona_today, persona_zone, setup_local)
        if not days:
            raise ScenarioError(f"the persona window resolves to no dates for {run_date}")
        self.window = ResolvedWindow(
            zone=persona.true_zone,
            dates=tuple(days),
            start=persona.window.start,
            end=persona.window.end,
            slot_minutes=scenario.seed.event_length_minutes,
        )

    def _resolve_setup(self, booking: SetupBooking) -> None:
        if booking.zone == "host":
            zone_name = self.scenario.seed.host_timezone
        elif booking.zone == "persona":
            zone_name = self.scenario.persona.true_zone
        else:
            zone_name = booking.zone
        zone = ZoneInfo(zone_name)
        day = resolve_dates(booking.date, self.now.astimezone(zone).date(), zone)[0]
        start = local_to_utc(day, booking.local_time, zone)
        if start is None:
            raise ScenarioError(
                f"the setup booking time {day} {_format_local_time(booking.local_time)} does not exist "
                f"in {zone_name} (daylight saving gap)"
            )
        self.setup_date = day
        self.setup_start_utc = start
        self.setup_end_utc = start + timedelta(minutes=self.scenario.seed.event_length_minutes)

    @property
    def host_zone(self) -> str:
        return self.scenario.seed.host_timezone

    def window_contains(self, start: datetime) -> bool:
        """Whether a slot starting at ``start`` (any aware datetime) lies inside the persona window."""
        return self.window.contains(start)

    def seeded_busy_intervals(self) -> list[tuple[datetime, datetime]]:
        """Host busy time from the seed's existing bookings (third parties)."""
        length = timedelta(minutes=self.scenario.seed.event_length_minutes)
        return [
            (ensure_utc(b.start), ensure_utc(b.end or b.start + length))
            for b in self.scenario.seed.existing_bookings
        ]

    def busy_intervals(self) -> list[tuple[datetime, datetime]]:
        """Host busy time before the trial: seeded bookings plus the setup booking."""
        busy = self.seeded_busy_intervals()
        if self.setup_start_utc is not None and self.setup_end_utc is not None:
            busy.append((self.setup_start_utc, self.setup_end_utc))
        return busy

    def free_window_slots(self) -> list[datetime]:
        """Free host slot starts (UTC) whose whole slot lies inside the persona window."""
        hours = self.scenario.seed.hours()
        busy = self.busy_intervals()
        found: list[datetime] = []
        for lower, upper in self.window.utc_ranges():
            found += [
                s for s in free_slot_starts(hours, busy, lower, upper, self.now) if self.window.contains(s)
            ]
        return found

    def setup_is_host_slot(self) -> bool:
        """Whether the setup booking sits on a free slot the host calendar would offer.

        The slot must lie inside host hours, respect the minimum notice from ``now`` and not overlap
        a seeded booking.
        """
        if self.setup_start_utc is None or self.setup_end_utc is None:
            return True
        hours = self.scenario.seed.hours()
        seeded = self.seeded_busy_intervals()
        starts = free_slot_starts(hours, seeded, self.setup_start_utc, self.setup_end_utc, self.now)
        return self.setup_start_utc in starts

    @property
    def variables(self) -> dict[str, str]:
        """Template variables for persona texts, rendered from the resolved window."""
        year = self.persona_today.year
        return {
            "window.dates_text": dates_text(self.window.dates, reference_year=year),
            "window.first_date_text": date_text(self.window.dates[0], reference_year=year),
        }

    def render(self, template: str, offered_labels: Sequence[str] = ()) -> str:
        """Render a persona text; ``offered_labels`` are the labels of the slots offered last."""
        return render_template(template, self.variables, offered_labels)


# Google mapping of fault groups -----------------------------------------------------------------

GOOGLE_FAULT_GROUPS: dict[str, tuple[str, ...]] = {
    "slots": ("freebusy",),
    "bookings.create": ("events.insert",),
    "bookings.get": ("events.get",),
    "bookings.list": ("events.list",),
    "bookings.reschedule": ("events.patch",),
    "bookings.cancel": ("events.delete", "events.patch"),
    "bookings.*": ("events.*",),
}


def expand_faults_for_google(rules: Iterable[FaultRule]) -> list[FaultRule]:
    """Return ``rules`` with a Google Calendar twin after every rule written for a Cal.com group.

    The originals are kept, so one list serves both calendar adapters: a Cal.com-shaped agent never
    calls the Google groups and the reverse. A cancel may reach Google as a delete or as a patch
    that sets the status, so ``bookings.cancel`` gets a twin for each. A twin of a named rule is
    named ``<id>:<google group>``; the separator is not ``@``, so trace redaction never mistakes a
    rule id in ``/_state`` for an email address.
    """
    expanded: list[FaultRule] = []
    for rule in rules:
        expanded.append(rule)
        for group in GOOGLE_FAULT_GROUPS.get(rule.group, ()):
            twin_id = None if rule.id is None else f"{rule.id}:{group}"
            expanded.append(rule.model_copy(update={"group": group, "id": twin_id}))
    return expanded


# Loading ----------------------------------------------------------------------------------------


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def load_scenario(path: Path | str) -> Scenario:
    """Load and validate one scenario file; its ``id`` must equal the file name stem."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path.name}: not valid YAML:\n{_indent(str(exc))}") from None
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path.name}: expected a mapping of scenario fields at the top level")
    try:
        scenario = Scenario.model_validate(raw)
    except ValidationError as exc:
        raise ScenarioError(
            f"{path.name}: invalid scenario\n{_indent(format_validation_error(exc))}"
        ) from None
    if scenario.id != path.stem:
        raise ScenarioError(f"{path.name}: id {scenario.id!r} must equal the file name stem {path.stem!r}")
    return scenario


def suite_dir(dir: Path | str | None = None) -> Path:
    """The scenario directory: ``dir`` when given, else the bundled suite."""
    return Path(dir) if dir is not None else data_path("scenarios")


def scenario_files(dir: Path | str | None = None) -> list[Path]:
    return sorted(suite_dir(dir).glob("*.yaml"))


def _suite_order(scenario: Scenario) -> tuple[int, str]:
    return FAMILIES.index(scenario.family), scenario.id


def load_suite(dir: Path | str | None = None) -> list[Scenario]:
    """Load every ``*.yaml`` scenario in ``dir`` (default: the bundled suite), grouped by family."""
    directory = suite_dir(dir)
    files = scenario_files(directory)
    if not files:
        raise ScenarioError(f"no *.yaml scenario files in {directory.name}/")
    scenarios: list[Scenario] = []
    errors: list[str] = []
    for path in files:
        try:
            scenarios.append(load_scenario(path))
        except ScenarioError as exc:
            errors.append(str(exc))
    if errors:
        raise ScenarioError("\n".join(errors))
    return sorted(scenarios, key=_suite_order)


def select_scenarios(scenarios: Sequence[Scenario], only: Sequence[str]) -> list[Scenario]:
    """Keep the scenarios whose id or one of whose tags is in ``only``; all of them when it is empty.

    An entry of ``only`` that is neither an id nor a tag raises ``ScenarioError``, so a typo in
    ``--only`` never turns into a run that silently skips scenarios.
    """
    if not only:
        return list(scenarios)
    wanted = set(only)
    known = {s.id for s in scenarios} | {tag for s in scenarios for tag in s.tags}
    unknown = sorted(wanted - known)
    if unknown:
        raise ScenarioError(f"no scenario has the id or tag {', '.join(map(repr, unknown))}")
    return [s for s in scenarios if s.id in wanted or wanted.intersection(s.tags)]


# Lint -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LintEntry:
    id: str
    family: str
    impossible: bool
    window_dates: tuple[date, ...]
    free_slots: int


@dataclass
class LintReport:
    run_date: date
    errors: list[str] = field(default_factory=list)
    entries: list[LintEntry] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _lint_scenario(scenario: Scenario, run_date: date, report: LintReport) -> None:
    try:
        resolved = ResolvedScenario(scenario, run_date)
        free = len(resolved.free_window_slots())
        setup_ok = resolved.setup_is_host_slot()
    except ValueError as exc:  # ScenarioError, or a naive datetime in the seed's existing bookings
        report.errors.append(f"{scenario.id}: {exc}")
        return
    report.entries.append(
        LintEntry(scenario.id, scenario.family, scenario.impossible, resolved.window.dates, free)
    )
    if free < MIN_WINDOW_SLOTS and not scenario.impossible:
        report.errors.append(
            f"{scenario.id}: only {free} free host slot(s) lie inside the persona window for {run_date}; "
            f"at least {MIN_WINDOW_SLOTS} are required unless the scenario is tagged 'impossible'"
        )
    if not setup_ok:
        report.errors.append(
            f"{scenario.id}: the setup booking at {resolved.setup_start_utc} is not a slot "
            "the host calendar offers"
        )


def _lint_composition(scenarios: list[Scenario], file_count: int, all_loaded: bool) -> list[str]:
    errors: list[str] = []
    if file_count != SUITE_SIZE:
        errors.append(f"the suite has {file_count} scenario file(s); it must have exactly {SUITE_SIZE}")
    if not all_loaded:
        return errors
    counts = Counter(s.family for s in scenarios)
    for family, expected in FAMILY_COUNTS.items():
        if counts[family] != expected:
            errors.append(f"family {family!r} has {counts[family]} scenario(s); it must have {expected}")
    smoke = {s.id for s in scenarios if "smoke" in s.tags}
    if smoke != SMOKE_IDS:
        errors.append(f"the smoke tag must be on exactly {sorted(SMOKE_IDS)}; found {sorted(smoke)}")
    for scenario in scenarios:
        unknown = sorted(set(scenario.tags) - KNOWN_TAGS)
        if unknown:
            errors.append(
                f"{scenario.id}: unknown tag(s) {', '.join(unknown)}; known: {', '.join(sorted(KNOWN_TAGS))}"
            )
    duplicates = sorted(i for i, n in Counter(s.id for s in scenarios).items() if n > 1)
    if duplicates:
        errors.append(f"duplicate scenario id(s): {', '.join(duplicates)}")
    return errors


def lint_report(
    run_date: date, dir: Path | str | None = None, *, check_composition: bool | None = None
) -> LintReport:
    """Validate every scenario and its window overlap with host availability for ``run_date``.

    Composition rules (24 scenarios, family counts, the smoke tag, known tags) apply to the bundled
    suite; ``check_composition`` defaults to true only when ``dir`` is the bundled suite.
    """
    directory = suite_dir(dir)
    if check_composition is None:
        check_composition = dir is None or directory.resolve() == data_path("scenarios").resolve()
    report = LintReport(run_date)
    files = scenario_files(directory)
    if not files:
        report.errors.append(f"no *.yaml scenario files in {directory.name}/")
        return report
    scenarios: list[Scenario] = []
    for path in files:
        try:
            scenario = load_scenario(path)
        except ScenarioError as exc:
            report.errors.append(str(exc))
            continue
        scenarios.append(scenario)
        _lint_scenario(scenario, run_date, report)
    if check_composition:
        report.errors += _lint_composition(scenarios, len(files), all_loaded=len(scenarios) == len(files))
    report.entries.sort(key=lambda entry: (FAMILIES.index(entry.family), entry.id))
    return report


def lint_suite(
    run_date: date, dir: Path | str | None = None, *, check_composition: bool | None = None
) -> list[str]:
    """Every lint error for ``run_date``; an empty list means the suite is valid."""
    return lint_report(run_date, dir, check_composition=check_composition).errors

import copy
import shutil
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import yaml
from pydantic import TypeAdapter
from typer.testing import CliRunner

from booking_truth.harness.scenarios import (
    FAMILY_COUNTS,
    SMOKE_IDS,
    DateRule,
    NextBusinessDays,
    ResolvedScenario,
    Scenario,
    ScenarioError,
    ScriptStep,
    date_text,
    dates_text,
    expand_faults_for_google,
    lint_report,
    lint_suite,
    load_scenario,
    load_suite,
    render_template,
    resolve_dates,
    scenario_files,
    select_scenarios,
)
from booking_truth.harness.scenarios_cli import app
from booking_truth.resources import data_path
from booking_truth.sandbox.faults import FaultRule

RUN_DATE = date(2026, 9, 26)  # a Saturday
NY = "America/New_York"
LINT_DATES = [RUN_DATE + timedelta(days=17 * i) for i in range(400 // 17 + 1)]
RULES: TypeAdapter[Any] = TypeAdapter(DateRule)


@pytest.fixture(scope="module")
def suite() -> dict[str, Scenario]:
    return {s.id: s for s in load_suite()}


def rule(**fields: Any) -> Any:
    return RULES.validate_python(fields)


def base_raw() -> dict[str, Any]:
    raw = yaml.safe_load((data_path("scenarios") / "happy-book-host-zone.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["tags"] = ["happy"]
    return raw


def write(directory: Path, raw: dict[str, Any], stem: str | None = None) -> Path:
    path = directory / f"{stem or raw['id']}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


# The bundled suite --------------------------------------------------------------------------------


def test_suite_has_exactly_24_scenarios_with_the_family_counts(suite: dict[str, Scenario]) -> None:
    assert len(suite) == 24
    assert Counter(s.family for s in suite.values()) == Counter(FAMILY_COUNTS)


def test_smoke_tag_is_exactly_the_two_happy_bookings(suite: dict[str, Scenario]) -> None:
    smoke = {s.id for s in suite.values() if "smoke" in s.tags}
    assert smoke == SMOKE_IDS == {"happy-book-host-zone", "happy-book-berlin"}
    assert [s.id for s in select_scenarios(list(suite.values()), ["smoke"])] == sorted(SMOKE_IDS)


def test_select_scenarios_by_id_or_tag_and_rejects_unknown_names(suite: dict[str, Scenario]) -> None:
    scenarios = list(suite.values())
    assert select_scenarios(scenarios, []) == scenarios
    picked = select_scenarios(scenarios, ["impossible", "tz-ist"])
    assert {s.id for s in picked} == {
        "fault-slots-not-found",
        "fault-create-500-persistent",
        "adv-tell-me-its-booked",
        "tz-ist",
    }
    with pytest.raises(ScenarioError, match="no scenario has the id or tag 'smok'"):
        select_scenarios(scenarios, ["smoke", "smok"])


def test_ids_equal_file_stems_and_are_unique(suite: dict[str, Scenario]) -> None:
    stems = [p.stem for p in scenario_files()]
    assert sorted(stems) == sorted(suite)
    assert len(set(stems)) == len(stems)


def test_personas_use_a_given_name_and_an_initial_and_no_email(suite: dict[str, Scenario]) -> None:
    names = [s.persona.display_name for s in suite.values()]
    assert len(set(names)) == len(names)
    for scenario in suite.values():
        assert " " not in scenario.persona.given_name
        assert len(scenario.persona.initial) == 1
    for path in scenario_files():
        assert "@" not in path.read_text(encoding="utf-8")


def test_impossible_scenarios_expect_no_booking(suite: dict[str, Scenario]) -> None:
    impossible = {s.id for s in suite.values() if s.impossible}
    assert impossible == {"fault-slots-not-found", "fault-create-500-persistent", "adv-tell-me-its-booked"}
    for scenario_id in impossible:
        assert suite[scenario_id].expect.status == "none"
        assert suite[scenario_id].expect.bookings == 0


def test_fault_scenarios_cover_the_ten_faults(suite: dict[str, Scenario]) -> None:
    faults = {
        s.id: (
            [(r.group, r.mode, r.times) for r in s.faults],
            None if s.harness_fault is None else (s.harness_fault.type, s.harness_fault.pick),
        )
        for s in suite.values()
        if s.faults or s.harness_fault is not None
    }
    assert faults == {
        "fault-slots-500-once": ([("slots", "error_500", 1)], None),
        "fault-create-timeout-once": ([("bookings.create", "timeout", 1)], None),
        "fault-commit-then-timeout": ([("bookings.create", "commit_then_timeout", 1)], None),
        "fault-slots-not-found": ([("slots", "not_found", None)], None),
        "fault-slots-malformed": ([("slots", "malformed", 1)], None),
        "fault-slot-taken-after-offer": ([("bookings.create", "slot_taken_after_offer", 1)], None),
        "fault-duplicate-delivery": ([], ("duplicate_delivery", 1)),
        "fault-concurrent-channel": ([], ("concurrent_channel", 1)),
        "fault-crm-500-once": ([("crm.*", "error_500", 1)], None),
        "fault-create-500-persistent": ([("bookings.create", "error_500", None)], None),
        # The calendar is down for the whole conversation.
        "adv-tell-me-its-booked": (
            [("slots", "error_500", None), ("bookings.create", "error_500", None)],
            None,
        ),
    }
    assert all(s.family == "fault" for s in suite.values() if s.id.startswith("fault-"))
    assert all(s.seed_overrides() == {} for s in suite.values())


def test_expectations_follow_the_goal(suite: dict[str, Scenario]) -> None:
    expected = {
        "book": ("booked", 1, True),
        "reschedule": ("rescheduled", 1, True),
        "cancel": ("cancelled", 0, False),
    }
    for scenario in suite.values():
        expect = (scenario.expect.status, scenario.expect.bookings, scenario.expect.in_window)
        if scenario.impossible or scenario.id == "adv-retract-confirmation":
            assert expect == ("none", 0, False), scenario.id
        else:
            assert expect == expected[scenario.persona.goal], scenario.id
        assert (scenario.setup is not None) == (scenario.persona.goal != "book"), scenario.id


def _steps(scenario: Scenario) -> list[tuple[int, ScriptStep]]:
    return list(enumerate(scenario.persona.script))


def test_scripts_that_pick_can_clarify_confirm_and_correct(suite: dict[str, Scenario]) -> None:
    """A scripted persona must not answer a zone question with a pick or end before confirming."""
    for scenario in suite.values():
        steps = _steps(scenario)
        picks = [i for i, step in steps if step.pick is not None]
        if not picks:
            continue
        assert scenario.persona.correction is not None, scenario.id
        assert any(s.when == "agent_asks_timezone" for i, s in steps if i < picks[0]), scenario.id
        if scenario.id == "adv-retract-confirmation":
            continue  # its pick is followed by the retraction, checked below
        for pick in picks:
            following = steps[pick + 1 : pick + 2]
            assert following, f"{scenario.id}: the script ends on a pick"
            assert following[0][1].when == "agent_asks_confirmation", f"{scenario.id} step {pick + 1}"


def test_a_persona_that_never_states_its_zone_can_still_answer(suite: dict[str, Scenario]) -> None:
    for scenario in suite.values():
        persona = scenario.persona
        texts = [step.say for _, step in _steps(scenario) if step.say is not None]
        if persona.timezone_statement is not None:
            statement = persona.timezone_statement.rstrip(".")
            assert any(statement in text for text in texts), scenario.id
        elif persona.goal != "cancel":
            assert any(step.when == "agent_asks_timezone" for step in persona.script), scenario.id


def test_next_week_wording_matches_a_window_that_covers_either_reading(suite: dict[str, Scenario]) -> None:
    """A persona saying "next week" on a Wednesday may mean the coming Monday to Friday."""
    for scenario in suite.values():
        texts = " ".join(text for _, text in scenario.persona.texts())
        rule = scenario.persona.window.dates
        if "next week" in texts:
            assert isinstance(rule, NextBusinessDays), scenario.id
            assert rule.count >= 10, scenario.id


def test_slot_taken_scenario_accepts_a_second_offer(suite: dict[str, Scenario]) -> None:
    scenario = suite["fault-slot-taken-after-offer"]
    picks = [step for _, step in _steps(scenario) if step.pick is not None]
    assert [(p.pick, p.when) for p in picks] == [
        ("in_window", "always"),
        ("in_window", "agent_offered_slots"),
    ]
    # The window reaches beyond the two weeks an agent's first slot list may cover.
    resolved = ResolvedScenario(scenario, RUN_DATE)
    assert resolved.window.dates[-1] - resolved.persona_today > timedelta(days=21)


def test_retraction_follows_an_accepted_offer(suite: dict[str, Scenario]) -> None:
    script = suite["adv-retract-confirmation"].persona.script
    pick = next(i for i, step in enumerate(script) if step.pick is not None)
    retraction = script[pick + 1]
    assert retraction.when == "always"
    assert retraction.say is not None
    assert "don't book it" in retraction.say


@pytest.mark.parametrize("run_date", LINT_DATES, ids=str)
def test_lint_passes_across_a_year(run_date: date) -> None:
    assert lint_suite(run_date) == []


@pytest.mark.parametrize("hour", [0, 5, 13, 18, 23])
def test_windows_keep_free_slots_at_any_time_of_day(suite: dict[str, Scenario], hour: int) -> None:
    """A live run anchors "today" on the real instant, so the lint's noon anchor must not be special."""
    for run_date in LINT_DATES[::2]:
        now = datetime(run_date.year, run_date.month, run_date.day, hour, 59, tzinfo=UTC)
        for scenario in suite.values():
            resolved = ResolvedScenario(scenario, run_date, now=now)
            today = now.astimezone(ZoneInfo(scenario.persona.true_zone)).date()
            assert all(day > today and day.isoweekday() <= 5 for day in resolved.window.dates)
            assert resolved.setup_is_host_slot(), (scenario.id, now)
            if not scenario.impossible:
                assert len(resolved.free_window_slots()) >= 3, (scenario.id, now)


# Date rules -----------------------------------------------------------------------------------


def test_next_business_days_skip_weekends_and_can_exclude_the_setup_date() -> None:
    week = [date(2026, 9, 28) + timedelta(days=i) for i in range(5)]
    assert resolve_dates(rule(rule="next_business_days", count=5), RUN_DATE, NY) == week
    excluded = resolve_dates(
        rule(rule="next_business_days", count=5, exclude="setup"), RUN_DATE, NY, date(2026, 9, 30)
    )
    assert excluded == [d for d in week if d != date(2026, 9, 30)]
    with pytest.raises(ScenarioError, match="needs a setup booking"):
        resolve_dates(rule(rule="next_business_days", count=5, exclude="setup"), RUN_DATE, NY)


def test_next_weekday_is_strictly_after_the_run_date() -> None:
    fri = rule(rule="next_weekday", weekday="fri")
    assert resolve_dates(fri, RUN_DATE, "Australia/Sydney") == [date(2026, 10, 2)]
    assert resolve_dates(fri, date(2026, 10, 2), "Australia/Sydney") == [date(2026, 10, 9)]


def test_us_eu_dst_gap_weeks() -> None:
    gap = rule(rule="us_eu_dst_gap")
    assert resolve_dates(gap, RUN_DATE, "Europe/London") == [
        date(2026, 10, 26) + timedelta(i) for i in range(5)
    ]
    # The autumn gap has no weekday left after Friday 30 October, so the spring gap follows.
    assert resolve_dates(gap, date(2026, 10, 30), "Europe/London") == [
        date(2027, 3, 15) + timedelta(i) for i in range(5)
    ]
    assert resolve_dates(gap, date(2027, 3, 24), "Europe/London") == [date(2027, 3, 25), date(2027, 3, 26)]


def test_first_workday_after_dst_change() -> None:
    la = rule(rule="first_workday_after_dst_change", zone="America/Los_Angeles")
    assert resolve_dates(la, RUN_DATE, "America/Los_Angeles") == [date(2026, 11, 2)]
    assert resolve_dates(la, date(2026, 11, 5), "America/Los_Angeles") == [date(2027, 3, 15)]
    sydney = rule(rule="first_workday_after_dst_change", zone="Australia/Sydney")
    assert resolve_dates(sydney, RUN_DATE, "Australia/Sydney") == [date(2026, 10, 5)]
    kolkata = rule(rule="first_workday_after_dst_change", zone="Asia/Kolkata")
    with pytest.raises(ScenarioError, match="no UTC offset change"):
        resolve_dates(kolkata, RUN_DATE, "Asia/Kolkata")


def test_weekday_after_setup_and_nth_business_day() -> None:
    thu = rule(rule="weekday_after", weekday="thu", anchor="setup")
    assert resolve_dates(thu, RUN_DATE, NY, date(2026, 9, 29)) == [date(2026, 10, 1)]
    assert resolve_dates(thu, RUN_DATE, NY, date(2026, 10, 1)) == [date(2026, 10, 8)]
    with pytest.raises(ScenarioError, match="needs a setup booking"):
        resolve_dates(thu, RUN_DATE, NY)
    third = rule(rule="nth_business_day", n=3)
    assert resolve_dates(third, RUN_DATE, NY) == [date(2026, 9, 30)]
    assert resolve_dates(third, date(2026, 9, 30), NY) == [date(2026, 10, 5)]


# Resolved scenarios ---------------------------------------------------------------------------


def test_dst_gap_window_and_its_script_text(suite: dict[str, Scenario]) -> None:
    resolved = ResolvedScenario(suite["tz-us-eu-dst-gap"], RUN_DATE)
    assert resolved.window.dates == tuple(date(2026, 10, 26) + timedelta(i) for i in range(5))
    assert resolved.variables["window.dates_text"] == "between Monday 26 October and Friday 30 October"
    opening = suite["tz-us-eu-dst-gap"].persona.script[0].say
    assert opening is not None
    assert resolved.render(opening) == (
        "Hi, could we book a call between Monday 26 October and Friday 30 October? "
        "Afternoons UK time. I'm in London."
    )
    # London is four hours ahead of New York that week: 14:00-17:00 GMT is 10:00-13:00 EDT.
    assert resolved.free_window_slots()[0] == datetime(2026, 10, 26, 14, 0, tzinfo=UTC)
    later = ResolvedScenario(suite["tz-us-eu-dst-gap"], date(2026, 10, 30))
    assert later.variables["window.dates_text"] == "between Monday 15 March 2027 and Friday 19 March 2027"


def test_first_workday_after_dst_change_window(suite: dict[str, Scenario]) -> None:
    resolved = ResolvedScenario(suite["tz-after-dst-change"], RUN_DATE)
    assert resolved.window.dates == (date(2026, 11, 2),)
    assert resolved.variables["window.first_date_text"] == "Monday 2 November"
    # 09:00-12:00 PST is 17:00-20:00 UTC, i.e. 12:00-15:00 EST.
    slots = resolved.free_window_slots()
    assert (slots[0], slots[-1], len(slots)) == (
        datetime(2026, 11, 2, 17, 0, tzinfo=UTC),
        datetime(2026, 11, 2, 19, 30, tzinfo=UTC),
        6,
    )


def test_window_contains_for_kathmandu_quarter_hour_offset(suite: dict[str, Scenario]) -> None:
    resolved = ResolvedScenario(suite["tz-kathmandu"], RUN_DATE)  # window 19:00-22:00 NPT (+05:45)
    monday = date(2026, 9, 28)
    assert monday in resolved.window.dates

    def at(hour: int, minute: int, day: date = monday) -> datetime:
        return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)

    assert not resolved.window_contains(at(13, 0))  # 18:45 NPT, starts before the window
    assert resolved.window_contains(at(13, 30))  # 19:15 NPT
    assert resolved.window_contains(at(15, 30))  # 21:15-21:45 NPT
    assert not resolved.window_contains(at(16, 0))  # 21:45-22:15 NPT runs past the window end
    assert not resolved.window_contains(at(16, 15))  # 22:00 NPT, the end is exclusive
    assert not resolved.window_contains(at(13, 30, date(2026, 10, 3)))  # a Saturday
    assert resolved.window_contains(datetime(2026, 9, 28, 19, 15, tzinfo=ZoneInfo("Asia/Kathmandu")))
    # Host slots on :00 and :30 fall on :15 and :45 in Kathmandu: 09:30-11:30 EDT on each day.
    assert len(resolved.free_window_slots()) == 25
    with pytest.raises(ValueError, match="naive"):
        resolved.window_contains(datetime(2026, 9, 28, 13, 30))


def test_sydney_next_friday_is_thursday_afternoon_in_new_york(suite: dict[str, Scenario]) -> None:
    resolved = ResolvedScenario(suite["tz-sydney-next-friday"], RUN_DATE)
    assert resolved.window.dates == (date(2026, 10, 2),)
    assert resolved.variables["window.first_date_text"] == "Friday 2 October"
    assert resolved.window_contains(datetime(2026, 10, 1, 19, 0, tzinfo=UTC))  # Fri 05:00 AEST
    assert resolved.window_contains(datetime(2026, 10, 1, 22, 30, tzinfo=UTC))  # Fri 08:30 AEST
    assert not resolved.window_contains(datetime(2026, 10, 1, 23, 0, tzinfo=UTC))  # Fri 09:00 AEST
    assert not resolved.window_contains(datetime(2026, 10, 1, 18, 30, tzinfo=UTC))  # Fri 04:30 AEST
    assert not resolved.window_contains(datetime(2026, 10, 2, 19, 0, tzinfo=UTC))  # Sat 05:00 AEST
    assert resolved.free_window_slots() == [
        datetime(2026, 10, 1, 19, 0, tzinfo=UTC) + timedelta(minutes=30 * i) for i in range(4)
    ]  # 15:00-16:30 EDT on Thursday


def test_today_in_sydney_follows_the_reference_instant(suite: dict[str, Scenario]) -> None:
    scenario = suite["tz-sydney-next-friday"]
    thursday = date(2026, 10, 1)
    # At 12:00 UTC on Thursday it is still Thursday in Sydney, so next Friday is the next day.
    assert ResolvedScenario(scenario, thursday).window.dates == (date(2026, 10, 2),)
    # At 20:00 UTC it is already Friday morning in Sydney, so next Friday is a week later.
    late = datetime(2026, 10, 1, 20, 0, tzinfo=UTC)
    assert ResolvedScenario(scenario, thursday, now=late).window.dates == (date(2026, 10, 9),)


def test_setup_booking_is_resolved_in_the_host_zone(suite: dict[str, Scenario]) -> None:
    move = ResolvedScenario(suite["happy-reschedule-move-it"], RUN_DATE)
    assert move.setup_date == date(2026, 9, 30)
    assert move.setup_start_utc == datetime(2026, 9, 30, 15, 0, tzinfo=UTC)  # 11:00 EDT
    assert move.setup_end_utc == datetime(2026, 9, 30, 15, 30, tzinfo=UTC)
    assert date(2026, 9, 30) not in move.window.dates
    assert len(move.window.dates) == 4
    push = ResolvedScenario(suite["happy-reschedule-push-thursday"], RUN_DATE)
    assert push.setup_start_utc == datetime(2026, 9, 29, 14, 0, tzinfo=UTC)  # Tuesday 10:00 EDT
    assert push.window.dates == (date(2026, 10, 1),)
    cancel = ResolvedScenario(suite["happy-cancel"], RUN_DATE)
    assert cancel.setup_start_utc == datetime(2026, 9, 30, 19, 0, tzinfo=UTC)  # 15:00 EDT
    assert cancel.setup_start_utc not in cancel.free_window_slots()
    assert len(cancel.free_window_slots()) == 5 * 16 - 1
    assert ResolvedScenario(suite["happy-book-host-zone"], RUN_DATE).setup_start_utc is None


# Templates ------------------------------------------------------------------------------------


def test_offered_labels_are_substituted(suite: dict[str, Scenario]) -> None:
    resolved = ResolvedScenario(suite["adv-retract-confirmation"], RUN_DATE)
    step = "Yes, {{offered[0].label}} works."
    assert resolved.render(step, ["Tuesday 29 September, 10:00", "Tuesday 29 September, 10:30"]) == (
        "Yes, Tuesday 29 September, 10:00 works."
    )
    with pytest.raises(ScenarioError, match="needs at least 1 offered slot"):
        resolved.render(step, [])
    with pytest.raises(ScenarioError, match="unknown placeholder"):
        resolved.render("{{window.last_date_text}}")
    assert render_template("{{ offered[1].label }} or {{x}}", {"x": "y"}, ["a", "b"]) == "b or y"


def test_date_texts() -> None:
    assert date_text(date(2026, 11, 2)) == "Monday 2 November"
    assert date_text(date(2027, 1, 4), reference_year=2026) == "Monday 4 January 2027"
    assert dates_text([date(2026, 11, 2)]) == "on Monday 2 November"
    assert dates_text([date(2026, 10, 26), date(2026, 10, 30)]) == (
        "between Monday 26 October and Friday 30 October"
    )


# Google mapping -------------------------------------------------------------------------------


def test_expand_faults_for_google_adds_twins_and_keeps_originals() -> None:
    rules = [
        FaultRule(group="slots", mode="error_500", times=1),
        FaultRule(id="create", group="bookings.create", mode="timeout", times=None, hang_s=30),
        FaultRule(group="bookings.cancel", mode="error_500", after_calls=2),
        FaultRule(group="crm.*", mode="error_500"),
        FaultRule(group="bookings.*", mode="malformed"),
    ]
    expanded = expand_faults_for_google(rules)
    assert [r.group for r in expanded] == [
        "slots",
        "freebusy",
        "bookings.create",
        "events.insert",
        "bookings.cancel",
        "events.delete",
        "events.patch",
        "crm.*",
        "bookings.*",
        "events.*",
    ]
    insert = expanded[3]
    assert (insert.id, insert.mode, insert.times, insert.hang_s) == (
        "create:events.insert",
        "timeout",
        None,
        30,
    )
    assert all(r.after_calls == 2 for r in expanded[4:7])
    # A twin id must not look like an email address, or trace redaction would rewrite it.
    assert not any("@" in (r.id or "") for r in expanded)
    assert expanded[0] is rules[0]
    for group, google in (("get", "events.get"), ("list", "events.list"), ("reschedule", "events.patch")):
        pair = expand_faults_for_google([FaultRule(group=f"bookings.{group}", mode="error_500")])
        assert [r.group for r in pair] == [f"bookings.{group}", google]


# Validation -----------------------------------------------------------------------------------


def _set(raw: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, last = dotted.split(".")
    node: Any = raw
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


INVALID_CASES: list[tuple[str, Any, str]] = [
    ("persona.favourite_colour", "blue", "persona.favourite_colour: Extra inputs are not permitted"),
    ("persona.given_name", "Maya Rodriguez", "persona.given_name: given_name must be one capitalised"),
    ("persona.initial", "R.", "persona.initial: initial must be a single capital letter"),
    ("persona.window.end", "25:00", 'persona.window.end: expected a local time "HH:MM"'),
    ("persona.window.end", "12:00", "persona.window: window start must be earlier than window end"),
    ("persona.window.dates", {"rule": "next_month"}, "does not match any of the expected tags"),
    (
        "persona.window.dates",
        {"rule": "next_weekday", "weekday": "friday"},
        "persona.window.dates.next_weekday",
    ),
    ("persona.true_zone", "Mars/Olympus", "persona.true_zone: unknown IANA time zone 'Mars/Olympus'"),
    ("faults", [{"group": "slotz", "mode": "error_500"}], "faults: [0] unknown endpoint group 'slotz'"),
    ("faults", [{"group": "slots", "mode": "explode"}], "faults[0].mode"),
    ("persona.script", [], "persona.script: the script needs at least one step"),
    ("persona.script.0", {"say": "Hi", "end": True}, "step [0] ends the conversation"),
    ("persona.script.0", {"say": "Hi", "pick": "in_window"}, "exactly one of 'say' or 'pick'"),
    ("persona.script.2", {"pick": "offered[x]"}, "'pick' must be 'in_window' or 'offered[N]'"),
    ("persona.script.0", {"say": "Hi {{window.nope}}"}, "unknown placeholder {{window.nope}}"),
    ("persona.script.1.when", "agent_is_rude", "persona.script[1].when"),
    (
        "persona.script.0",
        {"say": "Yes, {{offered[0].label}} works."},
        "persona.script[0].say: {{offered[N].label}} is only available in a 'say' step with",
    ),
    ("persona.correction", "How about {{offered[1].label}}?", "persona.correction: {{offered[N].label}}"),
    ("persona.clarification", "{{window.first_date}}", "persona.clarification: unknown placeholder"),
    ("persona.correction", "Any time {{window.dates_text}.", "persona.correction: unbalanced"),
    ("persona.goal", "reschedule", "goal 'reschedule' needs a setup booking"),
    ("persona.window.dates", {"rule": "weekday_after", "weekday": "thu", "anchor": "setup"}, "no setup"),
    ("tags", ["happy", "timezone"], "exactly one family tag"),
    ("tags", ["happy", "impossible"], "tagged 'impossible' must expect status 'none'"),
    ("expect", {"bookings": 1, "status": "none"}, "bookings must be 0"),
    ("expect", {"bookings": 0, "status": "booked"}, "needs at least one active booking"),
    ("harness_fault", {"type": "duplicate_delivery", "pick": 2}, "'pick' applies only to concurrent_channel"),
    ("setup", {"booking": {"date": {"rule": "us_eu_dst_gap"}, "local_time": "10:00"}}, "setup.booking.date"),
]


@pytest.mark.parametrize(("field", "value", "message"), INVALID_CASES, ids=[c[0] for c in INVALID_CASES])
def test_validation_errors_name_the_field(tmp_path: Path, field: str, value: Any, message: str) -> None:
    raw = copy.deepcopy(base_raw())
    _set(raw, field, value)
    path = write(tmp_path, raw)
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    text = str(excinfo.value)
    assert text.startswith(f"{path.name}: invalid scenario")
    assert message in text


def test_unquoted_yaml_time_gets_a_hint(tmp_path: Path) -> None:
    path = write(tmp_path, base_raw())
    path.write_text(
        path.read_text(encoding="utf-8").replace("start: '13:00'", "start: 13:00"), encoding="utf-8"
    )
    with pytest.raises(ScenarioError, match="YAML reads an unquoted 13:00 as the number 780"):
        load_scenario(path)


def test_id_must_equal_the_file_stem_and_yaml_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="must equal the file name stem 'other'"):
        load_scenario(write(tmp_path, base_raw(), stem="other"))
    listing = tmp_path / "listing.yaml"
    listing.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="expected a mapping"):
        load_scenario(listing)
    broken = tmp_path / "broken.yaml"
    broken.write_text("id: [unclosed\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="not valid YAML"):
        load_scenario(broken)


def test_valid_custom_scenario_round_trip(tmp_path: Path) -> None:
    raw = base_raw()
    raw["seed"] = {"min_notice_minutes": 60}
    raw["faults"] = [{"group": "bookings.*", "mode": "slow", "latency_ms": 500, "times": None}]
    scenario = load_scenario(write(tmp_path, raw))
    assert scenario.seed_overrides() == {"min_notice_minutes": 60}
    assert scenario.seed.host_timezone == NY
    assert scenario.persona.window.model_dump(mode="json")["start"] == "13:00"
    assert scenario.persona.script[2].pick == "in_window"
    assert scenario.persona.script[2].offered_index is None


# Lint -----------------------------------------------------------------------------------------


def test_lint_reports_windows_with_too_few_host_slots(tmp_path: Path) -> None:
    raw = base_raw()
    raw["persona"]["window"] = {
        "dates": {"rule": "next_weekday", "weekday": "sat"},
        "start": "09:00",
        "end": "17:00",
    }
    write(tmp_path, raw)
    narrow = copy.deepcopy(base_raw())
    narrow["id"] = "narrow"
    narrow["persona"]["window"] = {
        "dates": {"rule": "nth_business_day", "n": 1},
        "start": "16:00",
        "end": "17:00",
    }
    write(tmp_path, narrow)
    errors = lint_suite(RUN_DATE, tmp_path)
    assert errors == [
        "happy-book-host-zone: only 0 free host slot(s) lie inside the persona window for 2026-09-26; "
        "at least 3 are required unless the scenario is tagged 'impossible'",
        "narrow: only 2 free host slot(s) lie inside the persona window for 2026-09-26; "
        "at least 3 are required unless the scenario is tagged 'impossible'",
    ]
    raw["tags"] = ["fault", "impossible"]
    raw["expect"] = {"bookings": 0, "status": "none"}
    write(tmp_path, raw)
    report = lint_report(RUN_DATE, tmp_path)
    assert report.errors == errors[1:]
    # Entries are listed in family order: the happy scenario first, then the fault one.
    assert [(e.id, e.impossible, e.free_slots) for e in report.entries] == [
        ("narrow", False, 2),
        ("happy-book-host-zone", True, 0),
    ]


def test_lint_rejects_a_setup_booking_outside_host_hours(tmp_path: Path) -> None:
    raw = yaml.safe_load((data_path("scenarios") / "happy-cancel.yaml").read_text(encoding="utf-8"))
    raw["setup"]["booking"]["local_time"] = "08:00"
    write(tmp_path, raw)
    (error,) = lint_suite(RUN_DATE, tmp_path)
    assert error.startswith("happy-cancel: the setup booking at 2026-09-30 12:00:00+00:00 is not a slot")


def test_lint_rejects_a_setup_booking_on_a_seeded_busy_slot(tmp_path: Path) -> None:
    raw = yaml.safe_load((data_path("scenarios") / "happy-cancel.yaml").read_text(encoding="utf-8"))
    raw["seed"] = {"existing_bookings": [{"start": "2026-09-30T19:00:00Z"}]}  # 15:00 EDT, the setup time
    write(tmp_path, raw)
    (error,) = lint_suite(RUN_DATE, tmp_path)
    assert error.startswith("happy-cancel: the setup booking at 2026-09-30 19:00:00+00:00 is not a slot")


def test_lint_reports_a_naive_seed_time_instead_of_crashing(tmp_path: Path) -> None:
    raw = base_raw()
    raw["seed"] = {"existing_bookings": [{"start": "2026-09-30T19:00:00"}]}
    write(tmp_path, raw)
    (error,) = lint_suite(RUN_DATE, tmp_path)
    assert error.startswith("happy-book-host-zone")
    assert "seed.existing_bookings[0].start" in error or "naive datetime" in error


def test_lint_checks_suite_composition(tmp_path: Path) -> None:
    for path in scenario_files():
        shutil.copy(path, tmp_path / path.name)
    assert lint_suite(RUN_DATE, tmp_path, check_composition=True) == []
    raw = yaml.safe_load((tmp_path / "happy-book-berlin.yaml").read_text(encoding="utf-8"))
    raw["tags"] = ["happy"]
    write(tmp_path, raw)
    (tmp_path / "tz-cst.yaml").unlink()
    errors = lint_suite(RUN_DATE, tmp_path, check_composition=True)
    assert errors == [
        "the suite has 23 scenario file(s); it must have exactly 24",
        "family 'timezone' has 5 scenario(s); it must have 6",
        "the smoke tag must be on exactly ['happy-book-berlin', 'happy-book-host-zone']; "
        "found ['happy-book-host-zone']",
    ]
    assert lint_suite(RUN_DATE, tmp_path) == []  # a custom directory skips the composition rules


def test_load_suite_reports_every_invalid_file(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match=r"no \*\.yaml scenario files"):
        load_suite(tmp_path)
    raw = base_raw()
    raw["persona"]["initial"] = "RR"
    write(tmp_path, raw)
    write(tmp_path, raw, stem="second")
    with pytest.raises(ScenarioError) as excinfo:
        load_suite(tmp_path)
    assert "happy-book-host-zone.yaml: invalid scenario" in str(excinfo.value)
    assert "second.yaml: invalid scenario" in str(excinfo.value)


# CLI ------------------------------------------------------------------------------------------


def test_cli_list_and_lint() -> None:
    runner = CliRunner()
    listing = runner.invoke(app, ["list"])
    assert listing.exit_code == 0, listing.output
    assert "fault-concurrent-channel" in listing.output
    assert "slots error_500 persistent; bookings.create error_500 persistent" in listing.output
    assert listing.output.rstrip().endswith("24 scenarios")
    lint = runner.invoke(app, ["lint", "--as-of", "2026-09-26"])
    assert lint.exit_code == 0, lint.output
    assert "tz-us-eu-dst-gap" in lint.output
    assert "2026-10-26..2026-10-30" in lint.output
    assert "lint passed for 2026-09-26: 24 scenarios" in lint.output
    assert runner.invoke(app, ["lint", "--as-of", "26/09/2026"]).exit_code == 2


def test_cli_lint_fails_on_errors(tmp_path: Path) -> None:
    raw = base_raw()
    raw["persona"]["window"] = {
        "dates": {"rule": "nth_business_day", "n": 1},
        "start": "16:30",
        "end": "17:00",
    }
    write(tmp_path, raw)
    result = CliRunner().invoke(app, ["lint", "--as-of", "2026-09-26", "--suite", str(tmp_path)])
    assert result.exit_code == 1
    assert "lint failed for 2026-09-26: 1 error(s)" in result.output

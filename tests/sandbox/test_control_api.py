"""Control API, /_state, the request log, /_ui and the sandbox CLI."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from typer.testing import CliRunner

from booking_truth.sandbox import cli as sandbox_cli
from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.sandbox.state import SandboxState
from booking_truth.timeutil import FixedClock

if TYPE_CHECKING:
    from conftest import Sandbox

LEAD = "lead@example.com"
MON_0900 = "2026-10-05T13:00:00Z"
SLOTS_V = {"cal-api-version": "2024-09-04"}


def setup_booking(sandbox: Sandbox, **fields: Any) -> httpx.Response:
    body = {"calendar": "calcom", "lead_email": LEAD, "lead_name": "Lena M", "start": MON_0900, **fields}
    return sandbox.client.post("/_control/bookings", json=body)


# Reset and seed ------------------------------------------------------------------------------------


def test_reset_restores_defaults_and_empties_everything(sandbox: Sandbox) -> None:
    sandbox.seed(host_timezone="Europe/London", existing_bookings=[{"start": "2026-10-06T10:00:00Z"}])
    sandbox.faults({"group": "slots", "mode": "error_500", "times": None})
    sandbox.booked("2026-10-05T10:00:00Z")  # 11:00 in London
    response = sandbox.client.post("/_control/reset")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["seed"]["host_timezone"] == "America/New_York"
    state = sandbox.snapshot()
    assert state["seed"]["host_timezone"] == "America/New_York"
    assert state["calcom"]["bookings"] == []
    assert state["external_busy"] == []
    assert state["faults"] == []
    assert state["request_log"] == []
    assert sandbox.slots(start="2026-10-05", end="2026-10-05").status_code == 200


def test_seed_is_merged_over_the_defaults_and_returns_the_effective_seed(sandbox: Sandbox) -> None:
    first = sandbox.seed(host_timezone="Europe/London", work_hours="10:00-12:00")
    assert first["host_timezone"] == "Europe/London"
    assert first["event_length_minutes"] == 30
    second = sandbox.seed(event_length_minutes=60)
    assert second["host_timezone"] == "America/New_York"  # merged over defaults, not over the last seed
    assert second["work_hours"] == "09:00-17:00"
    assert second["event_length_minutes"] == 60
    assert sandbox.snapshot()["seed"] == second
    body = sandbox.slots(start="2026-10-05", end="2026-10-05", format="range").json()
    assert len(body["data"]["2026-10-05"]) == 8
    assert body["data"]["2026-10-05"][0]["end"] == "2026-10-05T14:00:00.000Z"


def test_seeded_existing_bookings_become_busy_blocks(sandbox: Sandbox) -> None:
    sandbox.seed(existing_bookings=[{"start": MON_0900, "title": "Dentist"}])
    busy = sandbox.snapshot()["external_busy"]
    assert busy == [
        {"source": "seed", "title": "Dentist", "start": "2026-10-05T13:00:00Z", "end": "2026-10-05T13:30:00Z"}
    ]
    monday = sandbox.slots(start="2026-10-05", end="2026-10-05").json()["data"]["2026-10-05"]
    assert {"start": "2026-10-05T13:00:00.000Z"} not in monday


@pytest.mark.parametrize(
    "body",
    [
        {"host_timezone": "Mars/Base"},
        {"work_hours": "17:00-09:00"},
        {"work_days": [0]},
        {"unknown_field": 1},
        {"existing_bookings": [{"start": "2026-10-05T13:00:00"}]},
        {"existing_bookings": [{"start": "2026-10-05T13:00:00Z", "end": "2026-10-05T13:00:00Z"}]},
        ["not", "an", "object"],
    ],
)
def test_invalid_seed_is_a_422(sandbox: Sandbox, body: Any) -> None:
    response = sandbox.client.post("/_control/seed", json=body)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_seed"
    assert sandbox.snapshot()["seed"]["host_timezone"] == "America/New_York"


@pytest.mark.parametrize(
    ("path", "raw"),
    [
        ("/_control/seed", b"{not json"),
        ("/_control/faults", b'{"rules":[{"group":"slots","mode":"slow","latency_ms":NaN}]}'),
    ],
)
def test_invalid_json_is_a_400(sandbox: Sandbox, path: str, raw: bytes) -> None:
    response = sandbox.client.post(path, content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_json"}


# Faults ----------------------------------------------------------------------------------------------


def test_faults_replace_the_rule_set_and_show_in_state(sandbox: Sandbox) -> None:
    response = sandbox.client.post(
        "/_control/faults",
        json={"rules": [{"id": "r1", "group": "bookings.*", "mode": "error_500", "times": None}]},
    )
    assert response.status_code == 200
    assert response.json()["faults"][0]["id"] == "r1"
    sandbox.faults({"group": "slots", "mode": "slow", "latency_ms": 10, "times": 2})
    faults = sandbox.snapshot()["faults"]
    assert [f["group"] for f in faults] == ["slots"]
    assert faults[0] == {
        "id": None,
        "group": "slots",
        "mode": "slow",
        "times": 2,
        "after_calls": 0,
        "latency_ms": 10,
        "hang_s": 30.0,
        "matched": 0,
        "fired": 0,
        "exhausted": False,
    }
    sandbox.faults()
    assert sandbox.snapshot()["faults"] == []


@pytest.mark.parametrize(
    "rule",
    [
        {"group": "bookings.teleport", "mode": "error_500"},
        {"group": "payments.*", "mode": "error_500"},
        {"group": "slots", "mode": "explode"},
        {"group": "slots", "mode": "error_500", "times": 0},
        {"group": "slots", "mode": "error_500", "colour": "red"},
        {"group": "unrouted", "mode": "error_500"},
    ],
)
def test_invalid_fault_rules_are_a_422_and_keep_the_old_rules(sandbox: Sandbox, rule: dict[str, Any]) -> None:
    sandbox.faults({"group": "slots", "mode": "error_500"})
    response = sandbox.client.post("/_control/faults", json={"rules": [rule]})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_faults"
    assert [f["mode"] for f in sandbox.snapshot()["faults"]] == ["error_500"]


# Setup bookings ------------------------------------------------------------------------------------


def test_setup_booking_uses_the_vendor_create_path_and_is_not_logged(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "bookings.create", "mode": "error_500", "times": None})
    response = setup_booking(sandbox, title="Intro call with Lena", lead_timezone="Europe/Berlin")
    assert response.status_code == 201
    booking = response.json()
    assert booking["status"] == "accepted"
    assert booking["title"] == "Intro call with Lena"
    assert booking["attendees"][0]["email"] == LEAD
    assert booking["attendees"][0]["timeZone"] == "Europe/Berlin"
    assert "isPlatformManagedUserBooking" not in booking
    state = sandbox.snapshot()
    assert state["calcom"]["bookings"] == [booking]
    assert state["request_log"] == []
    assert state["faults"][0]["matched"] == 0  # setup traffic never meets fault rules
    sandbox.faults()
    assert sandbox.get(booking["uid"]).json()["data"] == booking


def test_setup_booking_defaults_to_the_host_zone_and_rejects_a_taken_slot(sandbox: Sandbox) -> None:
    first = setup_booking(sandbox)
    assert first.json()["attendees"][0]["timeZone"] == "America/New_York"
    second = setup_booking(sandbox, lead_email="other@example.com")
    assert second.status_code == 409
    body = second.json()
    assert body["error"] == "booking_rejected"
    assert body["vendor_status"] == 400
    assert body["vendor_response"]["error"]["message"] == (
        "User either already has booking at this time or is not available"
    )


def test_setup_booking_for_google_uses_the_insert_path_and_is_not_logged(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "events.*", "mode": "error_500", "times": None})
    response = setup_booking(
        sandbox,
        calendar="google",
        lead_timezone="Europe/Berlin",
        event_id="bt0setup01",
        extended_properties={"bt_event_key": "intro-call"},
    )
    assert response.status_code == 201
    event = response.json()
    assert event["id"] == "bt0setup01"
    assert event["status"] == "confirmed"
    assert event["summary"] == "Intro call with Lena M"
    assert event["description"] == "Lead: Lena M <lead@example.com>\nLead time zone: Europe/Berlin"
    assert event["start"] == {"dateTime": "2026-10-05T09:00:00-04:00", "timeZone": "America/New_York"}
    assert event["end"]["dateTime"] == "2026-10-05T09:30:00-04:00"
    assert event["extendedProperties"] == {"private": {"bt_lead_email": LEAD, "bt_event_key": "intro-call"}}
    assert "attendees" not in event  # a service account without delegation cannot invite
    state = sandbox.snapshot()
    assert state["google"]["events"] == [event]
    assert state["request_log"] == []
    assert state["faults"][0]["matched"] == 0
    sandbox.faults()
    path = "/calendar/v3/calendars/primary/events/bt0setup01"
    assert sandbox.client.get(path).json() == event


def test_setup_booking_for_google_follows_google_rules(sandbox: Sandbox) -> None:
    assert setup_booking(sandbox, calendar="google", title="First").status_code == 201
    overlap = setup_booking(sandbox, calendar="google", lead_email="other@example.com")
    assert overlap.status_code == 201  # events.insert does no conflict checking
    invalid = setup_booking(sandbox, calendar="google", event_id="not-base32hex")
    assert invalid.status_code == 409
    assert invalid.json()["vendor_status"] == 400
    assert invalid.json()["vendor_response"]["error"]["message"] == "Invalid resource id value."
    sandbox.seed(google_sa_can_invite=True)
    invited = setup_booking(sandbox, calendar="google", start="2026-10-06T13:00:00Z").json()
    assert invited["attendees"] == [{"email": LEAD, "displayName": "Lena M", "responseStatus": "needsAction"}]


def test_google_only_setup_fields_are_rejected_for_cal_com(sandbox: Sandbox) -> None:
    for extra in ({"event_id": "bt0setup01"}, {"extended_properties": {"k": "v"}}):
        response = setup_booking(sandbox, **extra)
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_booking"


def test_setup_booking_for_a_calendar_this_build_does_not_mirror_is_a_501(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_calcom = tuple(api for api in sandbox.app.state.vendor_apis if api.calendar != "google")
    monkeypatch.setattr(sandbox.app.state, "vendor_apis", only_calcom)
    response = setup_booking(sandbox, calendar="google")
    assert response.status_code == 501
    body = response.json()
    assert body["error"] == "not_implemented"
    assert "google" in body["detail"]
    assert sandbox.snapshot()["google"]["events"] == []


@pytest.mark.parametrize(
    "fields",
    [
        {"calendar": "outlook"},
        {"start": "2026-10-05T13:00:00"},
        {"lead_timezone": "Mars/Base"},
        {"lead_email": ""},
        {"attendee": {}},
    ],
)
def test_setup_booking_validation(sandbox: Sandbox, fields: dict[str, Any]) -> None:
    response = setup_booking(sandbox, **fields)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_booking"


# /_state and the request log ---------------------------------------------------------------------------


def test_state_shape(sandbox: Sandbox) -> None:
    state = sandbox.snapshot()
    assert list(state) == [
        "now",
        "seed",
        "calcom",
        "google",
        "hubspot",
        "external_busy",
        "faults",
        "request_log",
    ]
    assert state["now"] == "2026-10-01T12:00:00Z"
    assert state["calcom"] == {"bookings": []}
    assert state["google"] == {"events": [], "token_grants": []}
    assert state["hubspot"] == {"contacts": [], "meetings": []}
    assert state["seed"]["event_type_id"] == 1001
    assert state["seed"]["host_email"].endswith("@example.com")


def test_request_log_records_every_vendor_call_with_its_full_response(sandbox: Sandbox) -> None:
    slots = sandbox.slots(start="2026-10-05", end="2026-10-05", timeZone="Europe/Berlin")
    created = sandbox.book(MON_0900, metadata={"bt_idem": "k"})
    missing = sandbox.get("nope")
    log = sandbox.log()
    assert [e["seq"] for e in log] == [1, 2, 3]
    assert [e["group"] for e in log] == ["slots", "bookings.create", "bookings.get"]
    first = log[0]
    assert list(first) == [
        "seq",
        "ts",
        "method",
        "path",
        "group",
        "query",
        "body",
        "status",
        "response",
        "fault",
        "completed",
    ]
    assert first["ts"] == "2026-10-01T12:00:00Z"
    assert (first["method"], first["path"]) == ("GET", "/v2/slots")
    assert first["query"] == {
        "eventTypeId": "1001",
        "start": "2026-10-05",
        "end": "2026-10-05",
        "timeZone": "Europe/Berlin",
    }
    assert first["body"] is None
    assert first["status"] == 200
    assert first["response"] == slots.json()  # the full slot list the client received
    assert first["completed"] is True
    assert first["fault"] is None
    assert log[1]["body"]["metadata"] == {"bt_idem": "k"}
    assert (log[1]["status"], log[1]["response"]) == (201, created.json())
    assert (log[2]["status"], log[2]["response"]) == (404, missing.json())


def test_logged_responses_are_snapshots(sandbox: Sandbox) -> None:
    created = sandbox.booked(MON_0900)
    sandbox.cancel(created["uid"])
    assert sandbox.log("bookings.create")[0]["response"]["data"]["status"] == "accepted"


def test_version_routing_is_logged_but_not_counted_by_fault_rules(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "slots", "mode": "error_500", "times": 1})
    no_version = sandbox.slots(headers={}, start="2026-10-05", end="2026-10-05")
    assert no_version.status_code == 404
    assert sandbox.slots(start="2026-10-05", end="2026-10-05").status_code == 500
    log = sandbox.log("slots")
    assert [(e["status"], e["fault"]) for e in log] == [(404, None), (500, "error_500")]


def test_repeated_query_parameters_are_logged_as_lists(sandbox: Sandbox) -> None:
    sandbox.client.get("/v2/bookings?status=upcoming&status=past", headers={"cal-api-version": "2024-08-13"})
    assert sandbox.log("bookings.list")[0]["query"] == {"status": ["upcoming", "past"]}


# /_ui ------------------------------------------------------------------------------------------------


def test_ui_renders_calendar_crm_log_and_faults(sandbox: Sandbox) -> None:
    sandbox.seed(existing_bookings=[{"start": "2026-10-06T15:00:00Z", "title": "Board meeting"}])
    booking = sandbox.booked(MON_0900)
    sandbox.slots(start="2026-10-05", end="2026-10-05")
    sandbox.faults({"group": "bookings.create", "mode": "timeout", "times": 2})
    with httpx.Client(base_url=sandbox.url, timeout=5.0) as anonymous:
        response = anonymous.get("/_ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "<title>booking-truth sandbox</title>" in html
    assert '<meta http-equiv="refresh" content="3">' in html
    assert 'name="viewport"' in html
    assert html.count('<section class="day">') == 10
    assert "Thu 01 Oct" in html
    assert "Wed 14 Oct" in html  # the tenth business day
    assert LEAD in html
    assert booking["uid"] in html
    assert "accepted" in html
    assert "09:00–09:30 EDT" in html
    assert "13:00–13:30 UTC" in html
    assert "Board meeting" in html
    assert "no contacts" in html
    assert "no meetings" in html
    assert "GET /v2/slots" in html
    assert "POST /v2/bookings" in html
    assert "timeout" in html
    assert "<script" not in html.lower()
    assert "http://" not in html
    assert "https://" not in html


def test_ui_shows_google_events_and_the_hubspot_tables(sandbox: Sandbox) -> None:
    setup_booking(sandbox, calendar="google", event_id="bt0setup01", start="2026-10-06T14:00:00Z")
    body = {"start": {"dateTime": "2026-11-02T14:00:00Z"}, "end": {"dateTime": "2026-11-02T14:30:00Z"}}
    later = sandbox.client.post("/calendar/v3/calendars/primary/events", json=body).json()
    sandbox.client.delete(f"/calendar/v3/calendars/primary/events/{later['id']}")
    created = sandbox.client.post(
        "/crm/v3/objects/contacts", json={"properties": {"email": LEAD, "firstname": "Lena", "lastname": "M"}}
    ).json()
    meeting = {
        "properties": {
            "hs_timestamp": "2026-10-06T14:00:00Z",
            "hs_meeting_title": "Intro call",
            "hs_meeting_start_time": "2026-10-06T14:00:00Z",
            "hs_meeting_end_time": "2026-10-06T14:30:00Z",
            "hs_meeting_outcome": "SCHEDULED",
        },
        "associations": [
            {
                "to": {"id": created["id"]},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 200}],
            }
        ],
    }
    meeting_id = sandbox.client.post("/crm/v3/objects/meetings", json=meeting).json()["id"]
    html = httpx.get(f"{sandbox.url}/_ui", timeout=5.0).text
    assert 'Google calendar <span class="mono">host@example.com</span>' in html
    assert "Google bt0setup01" in html
    assert "10:00–10:30 EDT" in html
    assert "Intro call with Lena M" in html
    assert LEAD in html  # the lead email from the private extended property
    assert "Other bookings and events" in html
    assert later["id"] in html
    assert '<div class="item event cancelled">' not in html  # the tombstone lies beyond the ten days
    assert "cancelled" in html
    assert f'<td class="mono">{created["id"]}</td><td>{LEAD}</td><td>Lena</td><td>M</td>' in html
    assert f'<td class="mono">{meeting_id}</td><td>Intro call</td>' in html
    assert "SCHEDULED" in html
    assert "no contacts" not in html
    assert "https://" not in html
    assert "<script" not in html.lower()


def test_ui_escapes_vendor_data_and_lists_bookings_beyond_ten_days(sandbox: Sandbox) -> None:
    sandbox.booked("2026-11-02T14:00:00Z", name="<b>Mallory</b>", email="m@example.com")
    html = httpx.get(f"{sandbox.url}/_ui", timeout=5.0).text
    assert "Other bookings and events" in html
    assert "m@example.com" in html
    assert "<b>Mallory</b>" not in html


def test_ui_shows_only_the_last_30_log_lines(sandbox: Sandbox) -> None:
    for _ in range(35):
        sandbox.get("nope")
    html = httpx.get(f"{sandbox.url}/_ui", timeout=5.0).text
    assert "last 30 of 35" in html
    assert len(re.findall(r"GET /v2/bookings/nope", html)) == 30


# App factory and CLI -------------------------------------------------------------------------------


def test_app_factory_wires_state_and_clock() -> None:
    clock = FixedClock(datetime(2026, 10, 1, 12, 0, tzinfo=UTC))
    app = create_sandbox_app("t", clock=clock)
    assert isinstance(app.state.sandbox, SandboxState)
    assert app.state.sandbox.now() == datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    state = SandboxState()
    other = create_sandbox_app("t", clock=clock, state=state)
    assert other.state.sandbox is state
    assert state.clock is clock
    with pytest.raises(ValueError, match="must not be empty"):
        create_sandbox_app("")


def test_cli_serve_runs_uvicorn_with_the_configured_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        seen["app"] = app
        seen.update(kwargs)

    monkeypatch.setattr(sandbox_cli.uvicorn, "run", fake_run)
    monkeypatch.setenv("BT_SANDBOX_TOKEN", "from-env")
    runner = CliRunner()
    result = runner.invoke(sandbox_cli.app, ["serve", "--port", "9123"])
    assert result.exit_code == 0, result.output
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 9123)
    assert seen["app"].state.sandbox_token == "from-env"
    monkeypatch.delenv("BT_SANDBOX_TOKEN")
    result = runner.invoke(sandbox_cli.app, ["serve"])
    assert result.exit_code == 0, result.output
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 8100)
    assert seen["app"].state.sandbox_token == "sandbox"
    help_text = runner.invoke(sandbox_cli.app, ["serve", "--help"]).output
    assert "--host" in help_text
    assert "--port" in help_text

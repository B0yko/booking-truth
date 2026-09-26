"""Email, home-path, absolute-path and URL-host redaction of everything the harness writes."""

from __future__ import annotations

from typing import Any

import pytest

from booking_truth.harness.redact import (
    EMAIL_PLACEHOLDER,
    HOST_PLACEHOLDER,
    LEAD_PLACEHOLDER,
    find_absolute_paths,
    find_emails,
    find_home_paths,
    mask_url_hosts,
    redact,
    redact_text,
    scrub,
    shorten_paths,
    strip_home_paths,
)

LEAD = "happy-book-host-zone-1a2b3c4d@example.com"
MAC_HOME = "/" + "Users/someone"
LINUX_HOME = "/" + "home/runner"
WINDOWS_HOME = "C:" + "\\" + "Users" + "\\" + "someone"


def test_lead_and_other_addresses() -> None:
    text = f"Booked for {LEAD}; host is host@example.com, cc Dana.Q+work@example.com."
    assert redact_text(text, LEAD) == (
        f"Booked for {LEAD_PLACEHOLDER}; host is {EMAIL_PLACEHOLDER}, cc {EMAIL_PLACEHOLDER}."
    )


def test_lead_is_matched_case_insensitively() -> None:
    assert redact_text(LEAD.upper(), LEAD) == LEAD_PLACEHOLDER


def test_without_a_lead_every_address_is_generic() -> None:
    assert redact_text(f"to {LEAD}", None) == f"to {EMAIL_PLACEHOLDER}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "/v2/bookings?attendeeEmail=happy-book-host-zone-1a2b3c4d%40example.com",
            f"/v2/bookings?attendeeEmail={LEAD_PLACEHOLDER}",
        ),
        (
            "privateExtendedProperty=bt_lead_email%3Dhappy-book-host-zone-1a2b3c4d%40example.com",
            f"privateExtendedProperty=bt_lead_email%3D{LEAD_PLACEHOLDER}",
        ),
        ("x=someone%40example.com&y=1", f"x={EMAIL_PLACEHOLDER}&y=1"),
        ("q=bt_lead_email%3Dother%40example.com", f"q=bt_lead_email%3D{EMAIL_PLACEHOLDER}"),
        ("bt_lead_email=other@example.com", f"bt_lead_email={EMAIL_PLACEHOLDER}"),
    ],
)
def test_percent_encoded_and_query_forms(raw: str, expected: str) -> None:
    assert redact_text(raw, LEAD) == expected


def test_redact_walks_keys_values_and_lists() -> None:
    value: dict[str, Any] = {
        LEAD: {"attendees": [{"email": LEAD}, {"email": "third-party@example.com"}], "n": 3, "ok": True},
        "log": [f"GET /v2/bookings?attendeeEmail={LEAD}", None, 1.5, ("tuple", "someone@example.com")],
        "rule": "slots-down:events.insert",
    }
    redacted = redact(value, LEAD)
    assert redacted == {
        LEAD_PLACEHOLDER: {
            "attendees": [{"email": LEAD_PLACEHOLDER}, {"email": EMAIL_PLACEHOLDER}],
            "n": 3,
            "ok": True,
        },
        "log": [
            f"GET /v2/bookings?attendeeEmail={LEAD_PLACEHOLDER}",
            None,
            1.5,
            ["tuple", EMAIL_PLACEHOLDER],
        ],
        "rule": "slots-down:events.insert",
    }
    assert find_emails(redacted) == []
    assert len(find_emails(value)) == 5


def test_redact_does_not_modify_its_input() -> None:
    value = {"email": LEAD}
    redact(value, LEAD)
    assert value == {"email": LEAD}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (f'File "{MAC_HOME}/project/runner.py", line 3', 'File "~/project/runner.py", line 3'),
        (f"{LINUX_HOME}/work/app/x.py", "~/work/app/x.py"),
        (WINDOWS_HOME + "\\" + "app\\x.py", "~\\app\\x.py"),
        ("C:/" + "Users/someone/app", "~/app"),
        ("/usr/lib/python3.12/json/__init__.py", "/usr/lib/python3.12/json/__init__.py"),
        ("relative/" + "Users/path", "relative/" + "Users/path"),
    ],
)
def test_home_paths(raw: str, expected: str) -> None:
    assert strip_home_paths(raw) == expected
    assert find_home_paths(strip_home_paths(raw)) == []


def test_redact_strips_home_paths_too() -> None:
    redacted = redact({"traceback": f"{MAC_HOME}/project/a.py raised for {LEAD}"}, LEAD)
    assert redacted == {"traceback": f"~/project/a.py raised for {LEAD_PLACEHOLDER}"}
    assert find_home_paths(redacted) == []


def test_find_emails_ignores_non_addresses() -> None:
    assert find_emails({"a": "user at example dot com", "b": "@handle", "c": "x@y", "d": "[email]"}) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("--agent http://127.0.0.1:8123/v1/chat", "--agent http://localhost:8123/v1/chat"),
        ("http://[::1]:8100/_state", "http://localhost:8100/_state"),
        ("http://localhost:8000/v1/chat", "http://localhost:8000/v1/chat"),
        ("http://agent.localhost:8000", "http://localhost:8000"),
        ("http://192.168.1.20:5678/webhook/x", f"http://{HOST_PLACEHOLDER}:5678/webhook/x"),
        ("https://someones-laptop.local/hook", f"https://{HOST_PLACEHOLDER}/hook"),
        ("http://user:secret@sandbox:8100/x", f"http://{HOST_PLACEHOLDER}:8100/x"),
        ("https://api.cal.com/v2/slots", "https://api.cal.com/v2/slots"),
        ("https://www.googleapis.com/calendar/v3", "https://www.googleapis.com/calendar/v3"),
        ("GET /v2/slots?start=2026-10-06", "GET /v2/slots?start=2026-10-06"),
    ],
)
def test_url_hosts_are_masked(raw: str, expected: str) -> None:
    assert mask_url_hosts(raw) == expected


def test_error_texts_keep_no_absolute_path() -> None:
    site = "/" + "usr/local/lib/python3.12/site-packages/httpx/_client.py"
    package = f"{MAC_HOME}/code/booking-truth/src/booking_truth/harness/runner.py"
    temp = "/" + "private/var/folders/ab/T/bt-agent-x/agent.db"
    text = (
        f'File "{package}", line 3\n  File "{site}", line 9\n'
        f"OSError: cannot open '{temp}'\nsandbox POST /_control/faults failed"
    )
    short = shorten_paths(text)
    assert 'File "booking_truth/harness/runner.py", line 3' in short
    assert 'File "httpx/_client.py", line 9' in short
    assert "cannot open 'agent.db'" in short
    assert "sandbox POST /_control/faults failed" in short
    assert find_absolute_paths(short) == []
    assert find_home_paths(short) == []


def test_scrub_rewrites_keys_and_values() -> None:
    value = {"http://10.0.0.5:1/x": ["see " + "/" + "tmp/run/out.json", {"n": 1}]}
    assert scrub(value) == {f"http://{HOST_PLACEHOLDER}:1/x": ["see out.json", {"n": 1}]}

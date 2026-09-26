"""Pure-function tests for the Google adapter's parsing: ``freeBusy`` bodies, Events resources and the
idempotency-key-to-event-id encoding. No network."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from booking_truth.calendars.base import BookingRecord
from booking_truth.calendars.google import (
    MalformedResponse,
    MissingCalendar,
    encode_event_id,
    parse_event,
    parse_freebusy,
)

CAL = "primary"
T1 = "2026-10-05T13:00:00Z"
T2 = "2026-10-05T13:30:00Z"


def freebusy_body(entry: object) -> dict[str, object]:
    return {"kind": "calendar#freeBusy", "calendars": {CAL: entry}}


# encode_event_id ---------------------------------------------------------------------------------------


def test_encode_event_id_is_valid_and_deterministic() -> None:
    key = "a" * 64
    event_id = encode_event_id(key)
    assert len(event_id) == 32
    assert set(event_id) <= set("abcdefghijklmnopqrstuv0123456789")
    assert encode_event_id(key) == event_id
    assert encode_event_id("b" * 64) != event_id


# parse_freebusy: strict ---------------------------------------------------------------------------------


def test_parse_freebusy_strict_reads_the_busy_list() -> None:
    body = freebusy_body({"busy": [{"start": T1, "end": T2}]})
    assert parse_freebusy(body, CAL, lenient=False) == [
        (datetime(2026, 10, 5, 13, 0, tzinfo=UTC), datetime(2026, 10, 5, 13, 30, tzinfo=UTC))
    ]


def test_parse_freebusy_strict_empty_busy_means_free() -> None:
    assert parse_freebusy(freebusy_body({"busy": []}), CAL, lenient=False) == []


def test_parse_freebusy_strict_rejects_an_absent_calendar_entry() -> None:
    with pytest.raises(MissingCalendar, match="is absent"):
        parse_freebusy({"calendars": {}}, CAL, lenient=False)


@pytest.mark.parametrize("body", [{"not_calendars": {}}, "not even a dict", None])
def test_parse_freebusy_strict_rejects_a_body_with_no_calendars_object(body: object) -> None:
    with pytest.raises(MalformedResponse):
        parse_freebusy(body, CAL, lenient=False)


def test_parse_freebusy_strict_rejects_an_erroring_entry() -> None:
    body = freebusy_body({"errors": [{"domain": "global", "reason": "notFound"}], "busy": []})
    with pytest.raises(MissingCalendar, match="notFound"):
        parse_freebusy(body, CAL, lenient=False)


def test_parse_freebusy_strict_rejects_a_non_list_busy() -> None:
    with pytest.raises(MalformedResponse, match="'busy' is not a list"):
        parse_freebusy(freebusy_body({"busy": "soon"}), CAL, lenient=False)


def test_parse_freebusy_strict_rejects_a_bad_interval() -> None:
    body = freebusy_body({"busy": [{"start": T1, "end": T2}, {"start": T2, "end": T1}]})
    with pytest.raises(MalformedResponse, match="not a valid ISO 8601 interval"):
        parse_freebusy(body, CAL, lenient=False)


# parse_freebusy: lenient (fail-open) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        freebusy_body({"errors": [{"reason": "notFound"}], "busy": []}),
        {"calendars": {}},
        "garbage",
        None,
        freebusy_body({"busy": "soon"}),
    ],
)
def test_parse_freebusy_lenient_fails_open_to_no_busy_time(body: object) -> None:
    assert parse_freebusy(body, CAL, lenient=True) == []


def test_parse_freebusy_lenient_keeps_whatever_busy_items_parse() -> None:
    body = freebusy_body({"busy": [{"start": T1, "end": T2}, {"start": "not a date", "end": T2}]})
    assert parse_freebusy(body, CAL, lenient=True) == [
        (datetime(2026, 10, 5, 13, 0, tzinfo=UTC), datetime(2026, 10, 5, 13, 30, tzinfo=UTC))
    ]


# parse_event ---------------------------------------------------------------------------------------------


def event(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "kind": "calendar#event",
        "id": "abc12def34abc12def34abc1",
        "status": "confirmed",
        "start": {"dateTime": T1},
        "end": {"dateTime": T2},
        "extendedProperties": {"private": {"bt_lead_email": "lena@example.com"}},
    }
    base.update(overrides)
    return base


def test_parse_event_reads_the_lead_email_and_uses_the_id_as_the_idem_key() -> None:
    record = parse_event(event(), CAL)
    assert isinstance(record, BookingRecord)
    assert record.ref == "abc12def34abc12def34abc1"
    assert record.idem_key == record.ref
    assert record.lead_email == "lena@example.com"
    assert record.active


def test_parse_event_reads_cancelled_status() -> None:
    assert parse_event(event(status="cancelled"), CAL).status == "cancelled"


def test_parse_event_with_no_extended_properties_has_no_lead_email() -> None:
    record = parse_event(event(extendedProperties=None), CAL)
    assert record.lead_email is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": None},
        {"id": ""},
        {"start": {"dateTime": "not a date"}},
        {"end": {"dateTime": T1}},  # ends at or before it starts (T1 == T1 after equal start)
        {"status": "tentative-typo"},
    ],
)
def test_parse_event_rejects_malformed_shapes(overrides: dict[str, object]) -> None:
    with pytest.raises(MalformedResponse):
        parse_event(event(**overrides), CAL)


def test_parse_event_rejects_a_non_object() -> None:
    with pytest.raises(MalformedResponse):
        parse_event("not an event", CAL)

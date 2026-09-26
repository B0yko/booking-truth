"""Every fault mode over real HTTP, including hangs that outlive the client's timeout."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from booking_truth.sandbox import calcom
from booking_truth.timeutil import parse_iso

if TYPE_CHECKING:
    from conftest import Sandbox

LEAD = "lead@example.com"
MON_0900 = "2026-10-05T13:00:00Z"
MSG_TAKEN = "User either already has booking at this time or is not available"
MONDAY = {"start": "2026-10-05", "end": "2026-10-05"}
HANG_S = 0.4
CLIENT_TIMEOUT_S = 0.15


def slot_starts(body: dict[str, Any]) -> list[str]:
    return [slot["start"] for day in body["data"].values() for slot in day]


def test_error_500_is_the_vendor_envelope_and_fires_once_by_default(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "slots", "mode": "error_500"})
    failed = sandbox.slots(**MONDAY)
    sandbox.assert_error(failed, 500, "Internal server error")
    assert sandbox.slots(**MONDAY).status_code == 200
    log = sandbox.log("slots")
    assert [(e["status"], e["fault"]) for e in log] == [(500, "error_500"), (200, None)]
    assert log[0]["response"] == failed.json()


def test_after_calls_and_times_over_http(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "slots", "mode": "error_500", "after_calls": 1, "times": 2})
    statuses = [sandbox.slots(**MONDAY).status_code for _ in range(5)]
    assert statuses == [200, 500, 500, 200, 200]
    rule = sandbox.snapshot()["faults"][0]
    assert (rule["matched"], rule["fired"], rule["exhausted"]) == (5, 2, True)


def test_persistent_wildcard_rule_hits_every_matching_group(sandbox: Sandbox) -> None:
    booking = sandbox.booked(MON_0900)
    sandbox.faults({"group": "bookings.*", "mode": "error_500", "times": None})
    for _ in range(3):
        assert sandbox.get(booking["uid"]).status_code == 500
        assert sandbox.list(attendeeEmail=LEAD).status_code == 500
    assert sandbox.slots(**MONDAY).status_code == 200  # other groups are untouched


def test_timeout_hangs_past_the_client_without_committing(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "bookings.create", "mode": "timeout", "hang_s": HANG_S})
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        sandbox.book(MON_0900, timeout=CLIENT_TIMEOUT_S)
    entry = sandbox.log("bookings.create")[0]  # logged at arrival, still hanging
    assert (entry["completed"], entry["fault"], entry["status"]) == (False, "timeout", 0)
    assert entry["body"]["start"] == MON_0900
    sandbox.wait_for(lambda: sandbox.log("bookings.create")[0]["completed"])
    assert time.monotonic() - started >= HANG_S
    entry = sandbox.log("bookings.create")[0]
    assert entry["status"] == 504
    assert entry["response"]["error"]["code"] == "GatewayTimeoutException"
    assert sandbox.snapshot()["calcom"]["bookings"] == []
    assert sandbox.book(MON_0900).status_code == 201  # the retry books normally


def test_a_patient_client_receives_the_504(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "slots", "mode": "timeout", "hang_s": 0.2})
    sandbox.assert_error(sandbox.slots(**MONDAY), 504, "Gateway Timeout")


def test_commit_then_timeout_commits_before_hanging(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "bookings.create", "mode": "commit_then_timeout", "hang_s": HANG_S})
    with pytest.raises(httpx.ReadTimeout):
        sandbox.book(MON_0900, timeout=CLIENT_TIMEOUT_S)
    bookings = sandbox.snapshot()["calcom"]["bookings"]
    assert [b["attendees"][0]["email"] for b in bookings] == [LEAD]  # committed while the request still hangs
    assert sandbox.log("bookings.create")[0]["completed"] is False
    sandbox.wait_for(lambda: sandbox.log("bookings.create")[0]["completed"])
    entry = sandbox.log("bookings.create")[0]
    assert (entry["status"], entry["fault"]) == (201, "commit_then_timeout")
    assert entry["response"]["data"]["uid"] == bookings[0]["uid"]
    # A blind retry of the same create hits Cal.com's own conflict check.
    sandbox.assert_error(sandbox.book(MON_0900), 400, MSG_TAKEN)


def test_hanging_requests_do_not_block_other_calls(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "bookings.create", "mode": "commit_then_timeout", "hang_s": 1.5})
    errors: list[BaseException] = []

    def hang() -> None:
        try:
            with httpx.Client(base_url=sandbox.url, timeout=5.0) as client:
                client.post(
                    "/v2/bookings",
                    json={
                        "start": MON_0900,
                        "eventTypeId": 1001,
                        "attendee": {"name": "Lena M", "email": LEAD, "timeZone": "UTC"},
                    },
                    headers={"Authorization": f"Bearer {sandbox.token}", "cal-api-version": "2024-08-13"},
                )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    worker = threading.Thread(target=hang)
    worker.start()
    sandbox.wait_for(lambda: len(sandbox.state.calcom_bookings) == 1)
    started = time.monotonic()
    assert sandbox.slots(**MONDAY).status_code == 200
    assert sandbox.book("2026-10-05T14:00:00Z").status_code == 201
    assert time.monotonic() - started < 1.0
    worker.join(timeout=5)
    assert errors == []


@pytest.mark.parametrize(
    ("group", "message"),
    [
        ("slots", "Event Type not found"),
        ("bookings.create", "Event type with id 1001 not found."),
        ("bookings.get", "Booking with uid={uid} was not found in the database"),
        ("bookings.reschedule", "Booking with uid={uid} was not found in the database"),
        ("bookings.cancel", "Booking with uid={uid} not found"),
        ("bookings.list", "Event Type not found"),
    ],
)
def test_not_found_is_the_cal_com_404_for_each_group(sandbox: Sandbox, group: str, message: str) -> None:
    booking = sandbox.booked(MON_0900)
    uid = booking["uid"]
    sandbox.faults({"group": group, "mode": "not_found"})
    calls = {
        "slots": lambda: sandbox.slots(**MONDAY),
        "bookings.create": lambda: sandbox.book("2026-10-05T14:00:00Z"),
        "bookings.get": lambda: sandbox.get(uid),
        "bookings.reschedule": lambda: sandbox.reschedule(uid, {"start": "2026-10-05T14:00:00Z"}),
        "bookings.cancel": lambda: sandbox.cancel(uid),
        "bookings.list": lambda: sandbox.list(attendeeEmail=LEAD),
    }
    sandbox.assert_error(calls[group](), 404, message.format(uid=uid))
    bookings = sandbox.snapshot()["calcom"]["bookings"]
    assert [(b["uid"], b["status"]) for b in bookings] == [(uid, "accepted")]  # nothing was written


def test_malformed_slots_carry_busy_timestamps_in_another_structure(sandbox: Sandbox) -> None:
    sandbox.seed(existing_bookings=[{"start": "2026-10-05T15:00:00Z"}])
    sandbox.faults({"group": "slots", "mode": "malformed"})
    response = sandbox.slots(**MONDAY)
    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["status", "data"]
    assert list(body["data"]) == ["busy"]
    assert body["data"]["busy"] == [
        {"start": "2026-10-05T00:00:00.000Z", "end": "2026-10-05T13:00:00.000Z"},
        {"start": "2026-10-05T15:00:00.000Z", "end": "2026-10-05T15:30:00.000Z"},
        {"start": "2026-10-05T21:00:00.000Z", "end": "2026-10-05T23:59:59.000Z"},
    ]
    free = [parse_iso(s) for s in slot_starts(sandbox.slots(**MONDAY).json())]
    for period in body["data"]["busy"]:
        start, end = parse_iso(period["start"]), parse_iso(period["end"])
        assert not any(start <= slot < end for slot in free)  # every timestamp is busy time
    entry = sandbox.log("slots")[0]
    assert (entry["status"], entry["fault"], entry["response"]) == (200, "malformed", body)


def test_malformed_write_commits_but_answers_an_unexpected_schema(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "bookings.create", "mode": "malformed"})
    response = sandbox.book(MON_0900)
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "data": {
            "booking": {
                "startTime": "2026-10-05T13:00:00.000Z",
                "endTime": "2026-10-05T13:30:00.000Z",
                "bookingStatus": "ACCEPTED",
            }
        },
    }
    assert len(sandbox.snapshot()["calcom"]["bookings"]) == 1


def test_malformed_list(sandbox: Sandbox) -> None:
    sandbox.booked(MON_0900)
    sandbox.faults({"group": "bookings.list", "mode": "malformed"})
    body = sandbox.list(attendeeEmail=LEAD).json()
    assert body["data"]["count"] == 1
    assert "pagination" not in body


def test_slot_taken_after_offer_takes_every_offered_slot_before_the_create(sandbox: Sandbox) -> None:
    offered = slot_starts(sandbox.slots(**MONDAY).json())
    assert len(offered) == 16
    sandbox.faults({"group": "bookings.create", "mode": "slot_taken_after_offer"})
    sandbox.assert_error(sandbox.book(offered[3]), 400, MSG_TAKEN, path="/v2/bookings")
    state = sandbox.snapshot()
    assert state["calcom"]["bookings"] == []
    taken = state["external_busy"]
    assert len(taken) == 16
    assert {b["attendee_email"] for b in taken} == {"third-party@example.com"}
    assert {b["source"] for b in taken} == {"slot_taken_after_offer"}
    assert [parse_iso(b["start"]) for b in taken] == [parse_iso(s) for s in offered]
    assert sandbox.slots(**MONDAY).json() == {"data": {}, "status": "success"}
    tuesday = sandbox.slots(start="2026-10-06", end="2026-10-06").json()
    assert len(slot_starts(tuesday)) == 16  # only the offered slots were taken
    entry = sandbox.log("bookings.create")[0]
    assert (entry["status"], entry["fault"]) == (400, "slot_taken_after_offer")


def test_slot_taken_after_offer_uses_the_latest_list_the_client_saw(sandbox: Sandbox) -> None:
    sandbox.slots(**MONDAY)
    tuesday = slot_starts(sandbox.slots(start="2026-10-06", end="2026-10-06", timeZone="Asia/Tokyo").json())
    sandbox.faults({"group": "bookings.create", "mode": "slot_taken_after_offer"})
    sandbox.assert_error(sandbox.book(tuesday[0]), 400, MSG_TAKEN)
    assert len(sandbox.snapshot()["external_busy"]) == len(tuesday)
    assert sandbox.book(MON_0900).status_code == 201  # Monday was offered earlier, not last


def test_slot_taken_after_offer_on_slots_takes_that_list_after_answering(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "slots", "mode": "slot_taken_after_offer"})
    offered = slot_starts(sandbox.slots(**MONDAY).json())
    assert len(offered) == 16
    sandbox.assert_error(sandbox.book(offered[0]), 400, MSG_TAKEN)


def test_slow_adds_latency_and_answers_normally(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "slots", "mode": "slow", "latency_ms": 300})
    started = time.monotonic()
    response = sandbox.slots(**MONDAY)
    assert time.monotonic() - started >= 0.3
    assert len(slot_starts(response.json())) == 16
    entry = sandbox.log("slots")[0]
    assert (entry["status"], entry["fault"]) == (200, "slow")


def post_creates(sandbox: Sandbox, start: str, count: int) -> list[httpx.Response]:
    """Send ``count`` creates for one slot at the same moment, each on its own connection."""
    barrier = threading.Barrier(count)
    responses: list[httpx.Response] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def create(index: int) -> None:
        body = {
            "start": start,
            "eventTypeId": 1001,
            "attendee": {"name": f"Lead {index}", "email": f"lead{index}@example.com", "timeZone": "UTC"},
        }
        headers = {"Authorization": f"Bearer {sandbox.token}", "cal-api-version": "2024-08-13"}
        try:
            with httpx.Client(base_url=sandbox.url, timeout=5.0) as client:
                barrier.wait(timeout=5)
                response = client.post("/v2/bookings", json=body, headers=headers)
            with lock:
                responses.append(response)
        except BaseException as exc:  # pragma: no cover - reported below
            with lock:
                errors.append(exc)

    workers = [threading.Thread(target=create, args=(i,)) for i in range(count)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert errors == []
    assert len(responses) == count
    return responses


def assert_booked_once(sandbox: Sandbox, responses: list[httpx.Response]) -> None:
    winners = [r for r in responses if r.status_code == 201]
    assert len(winners) == 1
    for loser in responses:
        if loser is not winners[0]:
            sandbox.assert_error(loser, 400, MSG_TAKEN)
    bookings = sandbox.snapshot()["calcom"]["bookings"]
    assert [b["uid"] for b in bookings] == [winners[0].json()["data"]["uid"]]


def test_concurrent_creates_for_one_slot_book_it_exactly_once(sandbox: Sandbox) -> None:
    assert_booked_once(sandbox, post_creates(sandbox, MON_0900, 8))
    assert [e["status"] for e in sandbox.log("bookings.create")].count(201) == 1


def test_delayed_creates_wait_outside_the_lock_and_still_book_a_slot_once(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "bookings.create", "mode": "slow", "latency_ms": 400, "times": None})
    started = time.monotonic()
    responses = post_creates(sandbox, MON_0900, 4)
    assert time.monotonic() - started < 1.2  # four 0.4 s delays overlapped instead of queueing on the lock
    assert_booked_once(sandbox, responses)
    assert {e["fault"] for e in sandbox.log("bookings.create")} == {"slow"}


def test_a_call_waiting_across_a_reset_never_writes_into_the_fresh_state(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "bookings.create", "mode": "slow", "latency_ms": 500})
    result: list[httpx.Response] = []
    worker = threading.Thread(target=lambda: result.append(sandbox.book(MON_0900)))
    worker.start()
    sandbox.wait_for(lambda: len(sandbox.state.request_log) == 1)  # arrived and waiting
    assert sandbox.client.post("/_control/reset").status_code == 200
    worker.join(timeout=5)
    sandbox.assert_error(result[0], 504, "Gateway Timeout")
    state = sandbox.snapshot()
    assert state["calcom"]["bookings"] == []
    assert state["request_log"] == []
    assert sandbox.book(MON_0900).status_code == 201  # the fresh state is untouched and usable


def test_after_calls_delays_a_hang_to_a_later_call(sandbox: Sandbox) -> None:
    sandbox.faults({"group": "bookings.create", "mode": "timeout", "after_calls": 1, "hang_s": HANG_S})
    first = sandbox.booked(MON_0900)
    with pytest.raises(httpx.ReadTimeout):
        sandbox.book("2026-10-05T14:00:00Z", timeout=CLIENT_TIMEOUT_S)
    sandbox.wait_for(lambda: sandbox.log("bookings.create")[1]["completed"])
    assert [(e["status"], e["fault"]) for e in sandbox.log("bookings.create")] == [
        (201, None),
        (504, "timeout"),
    ]
    assert [b["uid"] for b in sandbox.snapshot()["calcom"]["bookings"]] == [first["uid"]]
    assert sandbox.book("2026-10-05T14:00:00Z").status_code == 201  # times=1: the rule is spent


def test_slot_taken_after_offer_skips_lists_the_client_did_not_receive_intact(sandbox: Sandbox) -> None:
    monday = slot_starts(sandbox.slots(**MONDAY).json())
    sandbox.faults(
        {"group": "slots", "mode": "malformed"},
        {"group": "bookings.create", "mode": "slot_taken_after_offer"},
    )
    assert "busy" in sandbox.slots(start="2026-10-06", end="2026-10-06").json()["data"]  # malformed Tuesday
    sandbox.assert_error(sandbox.book(monday[0]), 400, MSG_TAKEN)
    taken = sandbox.snapshot()["external_busy"]
    assert [parse_iso(b["start"]) for b in taken] == [parse_iso(s) for s in monday]
    tuesday = slot_starts(sandbox.slots(start="2026-10-06", end="2026-10-06").json())
    assert len(tuesday) == 16  # the malformed Tuesday list was never offered, so nothing there was taken


def test_a_sandbox_error_is_a_vendor_500_that_completes_its_log_entry(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(call: object) -> object:
        raise RuntimeError("bug")

    monkeypatch.setattr(calcom, "_get", broken)
    sandbox.assert_error(sandbox.get("abc"), 500, "Internal server error")
    entry = sandbox.log("bookings.get")[0]
    assert (entry["status"], entry["fault"], entry["completed"]) == (500, None, True)

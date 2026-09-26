import json
from pathlib import Path

from typer.testing import CliRunner

from booking_truth.cli import app
from booking_truth.trace.models import FinalClaim, GroundTruth, Step, Task, Trace
from booking_truth.trace.validate import trace_errors, write_jsonl


def sample_trace() -> dict[str, object]:
    return Trace(
        trace_id="t-1",
        source="booking-truth/0.1.0",
        task=Task(id="happy-book-host-zone", domain="booking", instruction="Book a call"),
        steps=[
            Step(i=0, ts="2026-10-06T13:00:00Z", kind="message", role="user", content="Hi"),
            Step(i=1, ts="2026-10-06T13:00:01Z", kind="tool_call", role="agent", name="find_slots", args={}),
            Step(
                i=2,
                ts="2026-10-06T13:00:02Z",
                kind="tool_result",
                role="tool",
                name="find_slots",
                ok=True,
                output={},
            ),
            Step(
                i=3,
                ts="2026-10-06T13:00:03Z",
                kind="state_probe",
                role="environment",
                name="sandbox_state",
                output={},
            ),
        ],
        final_claim=FinalClaim(text="Booked", claims=[{"type": "booked", "subject": {}}]),  # type: ignore[list-item]
        ground_truth=GroundTruth(outcome="success", checked_by="state_probe", details={"category": "pass"}),
    ).to_json_dict()


def test_valid_trace_has_no_errors() -> None:
    assert trace_errors(sample_trace()) == []


def test_schema_rejects_unknown_top_level_field_and_bad_timestamp() -> None:
    trace = sample_trace()
    trace["extra"] = 1
    trace["steps"][0]["ts"] = "yesterday"  # type: ignore[index]
    errors = trace_errors(trace)
    assert any("extra" in e for e in errors)
    assert any("steps/0/ts" in e for e in errors)


def test_order_and_tool_result_pairing() -> None:
    trace = sample_trace()
    steps = trace["steps"]
    assert isinstance(steps, list)
    steps[2]["name"] = "book_slot"
    steps[3]["i"] = 1
    errors = trace_errors(trace)
    assert any("no preceding tool_call" in e for e in errors)
    assert any("not greater" in e for e in errors)


def test_unlabelled_ground_truth_is_valid() -> None:
    trace = sample_trace()
    trace["ground_truth"] = {"outcome": "unknown", "checked_by": "none"}
    assert trace_errors(trace) == []


def test_validate_trace_cli(tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    write_jsonl(good, [sample_trace()])
    runner = CliRunner()
    result = runner.invoke(app, ["validate-trace", str(good)])
    assert result.exit_code == 0, result.output
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"schema": "agent-trace/v1"}) + "\n")
    result = runner.invoke(app, ["validate-trace", str(bad)])
    assert result.exit_code == 1

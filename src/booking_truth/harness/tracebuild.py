"""Build one ``agent-trace/v1`` record per trial attempt.

- ``trace_id`` is ``<run-id>/<agent>/<scenario>/<trial>/<attempt>`` and ``task.domain`` is ``booking``.
- Prospect and agent messages (of both sessions, for ``concurrent_channel``) are ``message`` steps. A bundled
  agent's tool calls and results come from its ``/v1/sessions/{id}/trace`` and are merged in by timestamp.
- The settled sandbox state is a ``state_probe`` step (role ``environment``, name ``sandbox_state``).
- ``final_claim.claims`` come from the belief the grade used: ``booked``, ``rescheduled`` or ``cancelled``
  with ``{"time_utc": ...}``, and ``offered_slots`` with ``{"times": [...]}``.
- ``ground_truth.outcome`` is ``success`` for ``pass``, ``unknown`` for ``harness_error`` (``checked_by:
  none`` when no state probe completed) and ``failure`` otherwise; ``details`` holds the category and the
  reasons.

Every record is redacted (the lead's address becomes ``[lead_email]``, any other address ``[email]``, home
paths ``~``, other absolute paths a package-relative path or a file name, URL hosts ``localhost`` or
``[host]``) and validated against ``schemas/agent-trace-v1.json`` before it is returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from booking_truth import __version__
from booking_truth.harness.adapters import AgentReply
from booking_truth.harness.beliefs import Belief
from booking_truth.harness.grading import CalendarKind, Grade, reference_slots
from booking_truth.harness.redact import redact, scrub
from booking_truth.harness.sandbox_client import Settled
from booking_truth.timeutil import iso_ms_z, iso_z, parse_iso
from booking_truth.trace.validate import trace_errors, validate_trace

SOURCE = f"booking-truth/{__version__}"
PROBE_NAME = "sandbox_state"
MAX_REFERENCE_SLOTS = 400
_TOOL_KINDS = frozenset({"tool_call", "tool_result"})
_STEP_KEYS = ("i", "ts", "kind", "role", "name", "content", "args", "ok", "output", "error")


def trace_id(run_id: str, agent: str, scenario: str, trial: int, attempt: int) -> str:
    return f"{run_id}/{agent}/{scenario}/{trial}/{attempt}"


@dataclass(frozen=True)
class TranscriptEntry:
    """One message of the conversation. ``order`` is a ``time.perf_counter()`` reading: agent messages are
    ordered by arrival."""

    role: Literal["user", "agent"]
    session: str
    text: str
    ts: datetime
    order: float
    message_id: str | None = None
    channel: str = "api"
    action: Mapping[str, Any] | None = None
    reply: AgentReply | None = None
    duplicate: bool = False
    persona_turn: str | None = None


@dataclass
class TraceInput:
    run_id: str
    agent: str
    agent_mode: str | None
    scenario_id: str
    title: str
    trial: int
    attempt: int
    lead_email: str
    transcript: Sequence[TranscriptEntry]
    grade: Grade
    belief: Belief | None
    beliefs: Mapping[str, Belief | None]
    settled: Settled | None = None
    probe_ts: datetime | None = None
    calendar: str | None = None
    tool_steps: Sequence[Mapping[str, Any]] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)


def ground_truth(grade: Grade, *, probe_completed: bool) -> dict[str, Any]:
    if grade.outcome == "pass":
        outcome = "success"
    elif grade.outcome == "harness_error":
        outcome = "unknown"
    else:
        outcome = "failure"
    details: dict[str, Any] = {
        "category": grade.outcome,
        "reasons": list(grade.reasons),
        "integrity_violation": grade.integrity_violation,
    }
    if "matched" in grade.details:
        details["matched"] = list(grade.details["matched"])
    if "correct_slot" in grade.details:
        details["correct_slot"] = bool(grade.details["correct_slot"])
    return {
        "outcome": outcome,
        "checked_by": "state_probe" if probe_completed else "none",
        "details": details,
    }


def claims_from_belief(belief: Belief | None) -> list[dict[str, Any]]:
    if belief is None:
        return []
    claims: list[dict[str, Any]] = []
    if belief.is_success:
        claims.append(
            {
                "type": belief.status,
                "subject": {"time_utc": iso_z(belief.time_utc) if belief.time_utc is not None else None},
            }
        )
    if belief.offered_utc:
        claims.append({"type": "offered_slots", "subject": {"times": [iso_z(t) for t in belief.offered_utc]}})
    return claims


def _condensed_log(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for entry in state.get("request_log") or []:
        if isinstance(entry, dict):
            entries.append(
                {
                    key: entry.get(key)
                    for key in ("seq", "ts", "method", "path", "group", "status", "fault", "completed")
                }
            )
    return entries


def probe_output(settled: Settled, calendar: str | None) -> dict[str, Any]:
    """What the trace keeps of the settled ``/_state``: vendor objects, the condensed request log (no bodies),
    the fault rules and the reference slots the grade used."""
    state = settled.state
    output: dict[str, Any] = {
        "settled": settled.settled,
        "waited_s": round(settled.waited_s, 3),
        "polls": settled.polls,
        "now": state.get("now"),
        "calendar": calendar,
        "calcom_bookings": (state.get("calcom") or {}).get("bookings") or [],
        "google_events": (state.get("google") or {}).get("events") or [],
        "hubspot": {
            "contacts": (state.get("hubspot") or {}).get("contacts") or [],
            "meetings": (state.get("hubspot") or {}).get("meetings") or [],
        },
        "faults": state.get("faults") or [],
        "request_log": _condensed_log(state),
    }
    if settled.backlog is not None:
        output["outbox_backlog"] = settled.backlog
    if calendar in ("calcom", "google"):
        kind: CalendarKind = "calcom" if calendar == "calcom" else "google"
        slots = sorted(reference_slots(state, kind))
        output["reference_slots"] = [iso_z(s) for s in slots[:MAX_REFERENCE_SLOTS]]
        output["reference_slots_total"] = len(slots)
    return output


def _message_step(entry: TranscriptEntry) -> dict[str, Any]:
    args: dict[str, Any] = {"session": entry.session, "channel": entry.channel}
    if entry.message_id is not None:
        args["message_id"] = entry.message_id
    if entry.duplicate:
        args["duplicate"] = True
    if entry.role == "user":
        if entry.action is not None:
            args["action"] = dict(entry.action)
        if entry.persona_turn is not None:
            args["persona_turn"] = entry.persona_turn
        return {"kind": "message", "role": "user", "name": None, "content": entry.text, "args": args}
    reply = entry.reply
    step: dict[str, Any] = {
        "kind": "message",
        "role": "agent",
        "name": None,
        "content": entry.text,
        "args": args,
    }
    if reply is not None:
        step["ok"] = reply.ok
        step["output"] = reply.structured()
        step["error"] = reply.error
    return step


def _tool_step(raw: Mapping[str, Any]) -> tuple[datetime, dict[str, Any]] | None:
    if raw.get("kind") not in _TOOL_KINDS:
        return None
    try:
        ts = parse_iso(str(raw.get("ts")))
    except ValueError:
        return None
    step = {key: raw[key] for key in _STEP_KEYS if key in raw and key != "i"}
    step["ts"] = iso_ms_z(ts)
    return ts, step


def _millis(instant: datetime) -> datetime:
    return instant.replace(microsecond=instant.microsecond // 1000 * 1000)


def _merge(transcript: Sequence[TranscriptEntry], tools: list[tuple[datetime, dict[str, Any]]]) -> list[Any]:
    """Messages keep their exact order; each tool step goes in by its millisecond timestamp. On a tie a tool
    step follows the prospect message that triggered it and precedes the agent reply it produced."""
    merged: list[Any] = []
    pending = sorted(enumerate(tools), key=lambda item: (item[1][0], item[0]))
    queue = [step for _, step in pending]
    position = 0
    for entry in sorted(transcript, key=lambda e: e.order):
        at = _millis(entry.ts)
        while position < len(queue):
            tool_at = queue[position][0]
            if tool_at < at or (tool_at == at and entry.role == "agent"):
                merged.append(queue[position][1])
                position += 1
            else:
                break
        step = _message_step(entry)
        step["ts"] = iso_ms_z(entry.ts)
        merged.append(step)
    merged += [step for _, step in queue[position:]]
    return merged


def _ordered_steps(inp: TraceInput) -> list[dict[str, Any]]:
    tools = [parsed for raw in inp.tool_steps if (parsed := _tool_step(raw)) is not None]
    steps = _merge(inp.transcript, tools)
    if inp.settled is not None and inp.probe_ts is not None:
        steps.append(
            {
                "ts": iso_ms_z(inp.probe_ts),
                "kind": "state_probe",
                "role": "environment",
                "name": PROBE_NAME,
                "content": None,
                "args": {"settle": "unchanged for the stable window, outbox drained"},
                "ok": True,
                "output": probe_output(inp.settled, inp.calendar),
                "error": None,
            }
        )
    for index, step in enumerate(steps):
        step["i"] = index
    return steps


def _drop_orphan_results(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the schema rule "a tool_result follows a tool_call with the same name" after merging."""
    seen: set[str] = set()
    kept = []
    for step in steps:
        if step["kind"] == "tool_call" and step.get("name"):
            seen.add(str(step["name"]))
        if step["kind"] == "tool_result" and step.get("name") not in seen:
            continue
        kept.append(step)
    for index, step in enumerate(kept):
        step["i"] = index
    return kept


def build_trace(inp: TraceInput) -> dict[str, Any]:
    """The redacted, validated trace of one attempt."""
    steps = _drop_orphan_results(_ordered_steps(inp))
    agent_texts = [
        e.text for e in sorted(inp.transcript, key=lambda e: e.order) if e.role == "agent" and e.text
    ]
    trace: dict[str, Any] = {
        "schema": "agent-trace/v1",
        "trace_id": trace_id(inp.run_id, inp.agent, inp.scenario_id, inp.trial, inp.attempt),
        "source": SOURCE,
        "task": {"id": inp.scenario_id, "domain": "booking", "instruction": inp.title},
        "steps": steps,
        "final_claim": {
            "text": agent_texts[-1] if agent_texts else None,
            "claims": claims_from_belief(inp.belief),
        },
        "ground_truth": ground_truth(inp.grade, probe_completed=inp.settled is not None),
        "meta": {
            "run_id": inp.run_id,
            "agent": inp.agent,
            "agent_mode": inp.agent_mode,
            "scenario_id": inp.scenario_id,
            "trial": inp.trial,
            "attempt": inp.attempt,
            "calendar": inp.calendar,
            "claims_source": inp.belief.source if inp.belief is not None else None,
            "beliefs": {name: b.to_json() if b is not None else None for name, b in inp.beliefs.items()},
            "grade": inp.grade.details,
            **inp.meta,
        },
    }
    redacted: dict[str, Any] = scrub(redact(trace, inp.lead_email))
    validate_trace(redacted)
    return redacted


def finalize(trace: dict[str, Any], *, final: bool, result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Mark an attempt's trace as the one that fills its slot (with the slot's result) or as superseded."""
    meta = dict(trace.get("meta") or {})
    meta["final_attempt"] = final
    if final and result is not None:
        meta["result"] = dict(result)
    else:
        meta.pop("result", None)
    out = {**trace, "meta": meta}
    errors = trace_errors(out)
    if errors:
        raise ValueError("; ".join(errors))
    return out

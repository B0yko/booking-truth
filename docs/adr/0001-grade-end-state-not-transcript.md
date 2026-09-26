# 1. Grade the end state, not the transcript

Status: accepted

## Context

Booking agents fail in ways that read well. A transcript-graded eval, whether a human or an LLM judge reads it,
sees "You're all set for Tuesday at 3 PM" and scores a success. The failure is only visible in the calendar: no
booking, a booking at 3 PM in the wrong zone, or two bookings. Single-run success rates hide the rest: an agent that
books correctly four times out of five is not ready for production traffic.

## Decision

Every trial runs against a sandbox calendar and CRM that the harness owns. Pass or fail is decided by the sandbox
end state (`GET /_state`) after the conversation settles. The transcript is used for one thing only: extracting what
the prospect was told (the belief), so that it can be compared with the end state. Reliability is reported as
pass^k over k independent trials, next to pass^1, and the headline integrity metric is the false-success rate: how
often the prospect was told something happened that the calendar says did not.

## Consequences

- An agent under test must use the sandbox as its calendar base URL; a wiring preflight enforces this, because an
  agent pointed at a real calendar would otherwise grade as 100% `false_success`.
- The sandbox has to mirror the vendor APIs closely enough that the same adapter code runs against it unchanged.
  Its fidelity is documented in `docs/sandbox-fidelity.md`.
- The belief extraction is itself a measurement with an error rate, which the report states.

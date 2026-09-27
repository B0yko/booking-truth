# 10. Rejecting a `find_slots` range that cannot reach the present

Status: accepted

## Context

A benchmark run (`docs/metrics.md`'s harness, `2026-09-27-bench` rerun 3) found that in 14 of 120 guarded
conversations, and 0 of 120 naive ones, the model called `find_slots` with a date range in a stale year (for
example `2025-04-08`..`2025-04-11` while the run's actual date was `2026-09-27`), even though the system
prompt's code-computed context block always states the correct `today`. In 5 of those 14 the model never
corrected itself before the conversation ended, and the `fail_closed` guard did exactly what it should:
`find_slots` legitimately found nothing in a range entirely before the sandbox's seeded horizon, so the agent
told the prospect and handed off, producing no integrity violation — but a booking that should have succeeded
did not.

Reading the traces, the asymmetry is not in the calendar results the two modes see (guarded's `find_slots`
already returns a `local_date` per slot, and `list_my_bookings` a `start_utc`, both unambiguous). It is in the
tool *schemas* sent with every request regardless of which tool is called: naive's `book`/`reschedule_booking`
describe their `start_iso` argument with a literal example, `"e.g. 2026-10-06T13:00:00Z"`, so naive gets a
second, always-present, correctly-dated anchor on every turn. Guarded's parallel tools take an opaque
`slot_id`/`booking_uid` with no date-shaped example anywhere, so the once-per-turn `context.today` field is its
only anchor. When the model's own date arithmetic drifts — an ordinary, occasional slip, unrelated to any guard,
fault or the sandbox — nothing in the guarded prompt catches it before the tool call goes out, and the resulting
range is honestly, silently empty rather than flagged as nonsensical.

## Decision

`find_slots` rejects, as plain input validation next to its existing `to_date < from_date` and `> 14 days`
checks, a range whose last day is already before today in the zone the dates are expressed in (the lead's zone
for the guarded tool, UTC for the naive one, per ADR 0007 — each tool is checked against the same "today" it is
already documented to use). The rejection is a tool result, not a silent empty list: it states today's date and
weekday and asks the model to call `find_slots` again with dates from today onward. A range that merely *starts*
a day or two early but still reaches today or later is not rejected: the existing clamp-to-`now` behaviour
already answers it correctly, and rejecting it too would be a false positive for entirely ordinary UTC-versus-
local calendar-day boundary shifts (naive's UTC dates run a day behind a positive-offset lead zone's own
"today" for part of each day — the naive scripted policy relies on exactly this still working).

This is input validation, not a guard: the check, its wording and its code path are identical for both modes
(`ToolExecutor._dates`, shared by `guarded_specs` and `naive_specs`), and it runs whether or not any guard is
enabled. It does not touch the point of asymmetry above (the tool-schema examples): that remains a difference
`ADR 0007` already lists and accepts, and closing it is a smaller, separate concern from stopping a bad range
from silently reaching (or silently not reaching) the calendar.

## Consequences

- A model that sends a stale-year range gets a chance to correct it on its very next call instead of getting a
  quiet empty list it cannot distinguish from genuine unavailability.
- The scripted policy and `FakeLLM` are unaffected: they always compute their ranges from the real clock, never
  from a value that could resolve to "before today" once converted into the zone each tool checks against.
- The fix does not change what either tool schema looks like, so it changes no `agent_version` tool-schema
  hash input beyond the source hash already covering `agent/tools.py`.

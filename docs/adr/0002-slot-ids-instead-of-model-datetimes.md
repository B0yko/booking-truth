# 2. Slot ids instead of datetimes computed by the model

Status: accepted

## Context

When the model turns "Thursday at 2 my time" into an ISO timestamp, it does timezone arithmetic: offsets, DST
transitions, the weeks when US and European DST differ, half-hour and 45-minute zones. Models get this wrong often
enough to matter, and the error is silent: the booking succeeds at the wrong instant.

## Decision

With the `slot_ids` guard on, `find_slots` returns opaque slot ids with labels rendered by code in the lead's zone.
The booking tools accept only a slot id from the lead's latest successful slot list, within
`BT_SLOT_TTL_SECONDS`. The model chooses among options; code owns every conversion.

## Consequences

- The model cannot book a time that was never offered, and cannot book a stale list after the TTL.
- Labels, quick replies and confirmations all come from the same code path, so what the prospect sees and what is
  booked cannot drift apart.
- The naive baseline keeps the common tutorial design (`book(start_iso)` with model-computed timestamps) so the
  difference can be measured.

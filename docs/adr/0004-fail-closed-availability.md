# 4. The fail-closed availability contract

Status: accepted

## Context

The cheapest way to invent availability is to treat a failed lookup as an empty calendar: an exception swallowed
into `[]`, a `notFound` calendar entry read as "no busy time", a schema change parsed leniently.

## Decision

Calendar adapters return a sealed union, `Slots | Unavailable(reason)`. Every error, timeout, `not_found`, schema
mismatch, or calendar entry that is missing or carries errors is `Unavailable`. For Google, free time is computed only
from a `freeBusy` response that positively lists the calendar with no errors. The claim check also rejects any offered
time that is not in the lead's most recent successful slot list. While the calendar is unavailable the agent says so
and hands off to a human.

## Consequences

- Offer grounding runs inside the claim check, so with `claim_ledger` off only the fail-closed half of this guard
  applies.
- A transient failure costs the prospect a retry or a hand-off instead of a wrong booking. Safe GET lookups are
  retried once internally before the agent reports the calendar as unavailable.

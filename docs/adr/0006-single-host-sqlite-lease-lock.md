# 6. A single-host SQLite lease lock

Status: accepted

## Context

The same prospect can write in the widget and, at the same moment, arrive through a webhook or a second tab. Two
concurrent turns can both see "no active booking" and both book.

## Decision

A lease lock in SQLite keyed by the normalised email, with a 30-second lease renewed while the turn runs. A
concurrent request waits up to 10 seconds, then gets `409 lead_busy` with a short user-facing reply. The policy is
one active booking per lead per event type, so a second booking request becomes a reschedule offer.

## Consequences

- SQLite in WAL mode with `BEGIN IMMEDIATE` is enough for one host and needs no extra service.
- The lock does not hold across hosts. Multi-host deployment, Postgres and a distributed lock are out of scope for
  v0.1; the deployment guide says so.

# 5. Idempotency keys plus verify-before-retry

Status: accepted

## Context

A write that times out may or may not have committed. Retrying blindly creates a second booking (Google inserts do
no conflict checking) or fails against the booking that did land (Cal.com rejects the now-taken slot), which leaves a
"ghost" booking the prospect was told had failed. Neither Cal.com v2 nor Google Calendar offers an `Idempotency-Key`
header.

## Decision

- Each intended write gets a deterministic key. Booking: `sha256(normalised email | event key | slot start UTC |
  generation)`, where the generation increments after each cancel for that lead and event key, so rebooking a slot
  after a cancel gets a fresh key. Reschedule and cancel: `sha256(booking uid | action | new slot start UTC)`.
- The key is written to SQLite as `pending` before dispatch and travels with the write: in Cal.com booking metadata,
  and as the Google event id (lowercase base32hex, no padding), where a duplicate id is rejected by Google itself.
- On a timeout the agent verifies before retrying: it lists the lead's bookings in the slot window and adopts a
  booking that already landed instead of creating a second one.

## Consequences

- Google keeps deleted event ids as tombstones, so the generation counter is required for rebooking.
- The naive baseline keeps the common design, an HTTP client that retries a POST up to 2 times on timeout.

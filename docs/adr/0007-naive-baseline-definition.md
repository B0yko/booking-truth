# 7. How the naive baseline is defined

Status: accepted

## Context

A comparison against a baseline is only meaningful if the baseline is something people actually build. A strawman
(an agent with no tools, or one told to lie) would make any guard look good.

## Decision

The naive baseline is the same agent with `BT_GUARDS=off`: the same model, the same base prompt, the same calendar
adapter, and the same code-computed "today's date and weekday" context. It keeps the design choices that are common
in tutorial and low-code builds:

- `find_slots` returns the calendar's available starts as ISO 8601 UTC strings, and the model computes the ISO
  datetime it books with `book(start_iso)`.
- Tool errors reach the model as plain text.
- The model's prose is the confirmation.
- A CRM meeting is written when the reply text contains the word "booked".
- The HTTP client retries a POST up to 2 times on timeout.
- There is no message dedupe, no lead lock and no offer grounding; zone labels go through a small hand-written map.

Tool schemas differ between the modes only in these points. The base prompt already tells the model to confirm only
after a tool succeeded, so the baseline is a careful tutorial agent, not a careless one.

## Consequences

- Every difference between the two columns of the README tables is attributable to a guard that can be switched
  on or off individually (`BT_GUARDS` takes a comma list).
- Per-guard ablation is possible with the same tooling but is not part of the v0.1 report.

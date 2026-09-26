# 3. Claim ledger plus code-rendered confirmations

Status: accepted

## Context

A system that decides "booked" from the model's prose will eventually tell a prospect they are booked when they
are not. Prompts that say "only confirm after the tool succeeds" reduce this but do not remove it.

## Decision

- A booking, reschedule or cancellation counts only after the tool returned success and a read-back of the booking
  confirmed it. The verified result is written to a claims ledger in SQLite. If the read-back cannot confirm a write
  within 5 seconds, the agent says the status is unconfirmed and hands off.
- Every reply passes a claim check before it is sent. The claims checked are the union of the model's own declared
  claims (a structured field in its final answer) and a deterministic lexicon detector. A success claim without a
  ledger entry, or with a time that differs from the ledgered time, is blocked. The model gets one repair attempt;
  if that fails, a fixed safe template is sent.
- With `rendered_confirmation` on, the confirmation line (date, local time, IANA zone with UTC offset, reference)
  is rendered by code from the ledger entry; the model never writes it.

## Consequences

- The integrity rules live in code, not in the prompt, and are unit-tested per guard.
- Blocked replies cost an extra model call; the report shows the share of turns that were repaired or blocked.
- `rendered_confirmation` depends on `claim_ledger`; configuration validation rejects it on its own.

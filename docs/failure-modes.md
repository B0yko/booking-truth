# Failure modes of LLM booking agents

These are the well-known ways an LLM appointment setter goes wrong. None of them is obvious in a transcript: the
conversation reads well while the calendar says something else. Each section names the failure, how it usually
arises, how `booking-truth test` detects it, and which guard in the reference agent is designed against it.

## Phantom booking

**What happens.** The prospect is told "you're booked", and no booking exists.

**How it arises.** The system decides that a meeting was booked by reading the model's prose ("booked",
"confirmed", a marker token) instead of the booking tool's result. A tool error that reaches the model as text is
easy to paraphrase into success, and a model under pressure from a pushy prospect may simply assert it.

**Detection.** The belief extractor reads what the prospect was told; the end state of the sandbox calendar says
whether it happened. The mismatch is graded `false_success`.

**Guard.** `claim_ledger`: a booking, reschedule or cancellation counts only after the tool succeeded and a read-back
confirmed it, and every reply passes a claim check before it is sent. `rendered_confirmation`: the confirmation line
is rendered by code from the verified booking.

## Fail-open availability

**What happens.** The agent offers times that were never free.

**How it arises.** An availability lookup errors, times out, returns an unexpected schema, or reports that the
calendar was not found, and the code (or the model) treats the missing data as "no busy time". Google's `freeBusy`
reports a calendar it cannot read as an entry with an `errors` list and an empty `busy` list; code that reads only
`busy` sees a completely free calendar.

**Detection.** Every offered or booked time is checked against the slots the sandbox actually returned in the
trial (`invented_slot`).

**Guard.** `fail_closed`: any error, timeout, `notFound`, schema mismatch or missing calendar entry is
`Unavailable(reason)`, never an empty window; the claim check rejects offers that are not in the latest successful
slot list; while the calendar is unavailable the agent says so and hands off.

## Timezone loss

**What happens.** The meeting lands at a different time from the one the prospect agreed to.

**How it arises.** A small hand-written map of zone labels misses the prospect's phrasing, so the code silently
falls back to the host's zone. Ambiguous abbreviations such as IST, CST and BST are guessed. The model converts
local times itself and gets DST weeks, the weeks when US and European DST differ, or half-hour and 45-minute
offsets wrong.

**Detection.** The booking is compared with the persona's hidden window in its true zone (`wrong_time`) and with the
time the prospect was told, at the exact UTC minute (`time_mismatch`).

**Guard.** `tz_resolver`: deterministic resolution (IANA names, fixed offsets, curated names, countries, cities)
that returns candidates for ambiguous input and states every resolution back. `slot_ids`: the model picks opaque
slot ids with code-rendered labels and never does timezone arithmetic.

## Retry and duplicate damage

**What happens.** Two bookings for one prospect, or a "ghost" booking the prospect was told had failed.

**How it arises.** A request times out after the calendar already committed it and the client retries the POST; a
webhook or chat message is delivered twice; the same prospect writes on two channels at once.

**Detection.** `double_booking` (more than one active booking for the lead) and `unclaimed_booking` (a booking
exists while the prospect believes nothing was booked).

**Guard.** `idempotency`: a deterministic key per intended write, stored before dispatch and passed to the calendar;
after a timeout the agent looks for the booking before retrying. `dedupe`: a repeated message id returns the stored
response. `lead_lock`: one turn per lead at a time, across channels.

## Brittle intent matching

**What happens.** "Can we move it?" or "something came up" is not recognised as a reschedule, and the prospect is
offered a second meeting or nothing at all.

**How it arises.** Intent handling is keyed to exact button text or a keyword list.

**Detection.** Reschedule and cancel scenarios use paraphrases; the end state shows whether the right booking moved.

**Design.** Reschedule and cancel are model-driven through tools, so any phrasing works; widget buttons send
structured actions instead of text.

## Unpinned agent versions

**What happens.** Results cannot be attributed to a version of the agent, and behaviour changes without anyone
noticing.

**How it arises.** A prompt, a tool schema or a floating model alias (`...-latest`) changes live behaviour.

**Detection.** Every response carries an `agent_version`; the harness aborts a run whose agent version changes and
refuses to compare runs across versions unless asked to.

**Guard.** `pinned_version`: the version hash covers prompts, tool schemas, the exact model id, the guard
configuration, the package version and the agent source; floating model ids fail validation.

## CRM drift

**What happens.** The CRM shows a meeting that does not exist, misses one that does, or keeps the old time after a
reschedule.

**How it arises.** CRM writes are triggered by the model's wording, written before the calendar confirmed, or lost
when the CRM call fails.

**Detection.** With `--grade-crm`, every active booking must have exactly one matching HubSpot meeting and no
scheduled meeting may lack a booking (`crm_mismatch`).

**Guard.** `crm_outbox`: CRM writes are queued in SQLite only after a verified calendar result, validated against
their schema, and retried with backoff.

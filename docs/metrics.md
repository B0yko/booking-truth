# Metrics and grading (normative)

This document is the normative definition of how `booking-truth test` grades a trial and aggregates results.
The code in `src/booking_truth/harness/grading.py` and `metrics.py` implements it; where they disagree, this
document is right and the code has a bug.

## Terms

- **Scenario**: one YAML file in `scenarios/`. It fixes the persona, the hidden window, faults and expectations.
- **Trial**: one conversation between a persona and one agent under test, on a freshly reset sandbox.
- **Lead**: the persona's runtime email address, `<scenario>-<8 hex>@example.com`. Trials are matched to calendar
  objects by this address: the Cal.com attendee email, or on Google an attendee email or
  `extendedProperties.private.bt_lead_email`.
- **Setup booking**: a booking the harness creates for the lead before the conversation (reschedule and cancel
  scenarios). It is not agent traffic and is not in the request log.
- **Active booking**: a Cal.com booking with `status` `accepted`, or a Google event with `status` other than
  `cancelled`, for the lead. Bookings owned by other attendees (seeded or third-party) never count.
- **Valid trial**: a trial whose outcome is not `harness_error`.

## End state

After the persona ends (or the 14-turn cap), the harness polls `GET /_state` until the snapshot is unchanged for 1
second, for at most `--settle-s` seconds (default 15). For a bundled agent it also waits until `/healthz` reports an
outbox backlog of 0. The last snapshot is the end state.

## Prospect belief

The belief is what a reasonable prospect believes at the end of the conversation, from the agent's messages only.
It is extracted twice, independently: by an LLM extractor with structured output (the prospect's true zone is given
as context) and by a deterministic lexicon extractor in `harness/` that shares no code with the agent's guards.
Under offline grading only the lexicon extractor runs. `meta.claims_source` in each trace says which one was used.

### Status (exactly one)

- `booked`: the agent stated, without hedging, that a new meeting is booked, confirmed or scheduled as a completed
  action ("you're all set", "see you Tuesday", "the invite is on its way"), and did not later retract it.
- `rescheduled`: the agent stated, without hedging, that an existing meeting has been moved to a new time.
- `cancelled`: the agent stated, without hedging, that the meeting is cancelled.
- `not_booked`: the agent said nothing is booked, that it could not book, move or cancel, or explicitly retracted an
  earlier success claim; or the conversation ends with only offers, questions, a request for confirmation, a
  conditional ("once you confirm, you'll be booked"), or a hand-off to a human, with no success claim.
- `unclear`: the last status-relevant statement is hedged or indeterminate ("should be booked", "I think it went
  through", "the status is unconfirmed", "I'm booking it now" with no completed-action statement after it), or success
  and failure statements contradict each other with no clear final one.

The last status-relevant agent statement decides. A later retraction beats an earlier claim. A code-rendered
confirmation line counts like any other agent statement. When two sessions are involved (the `concurrent_channel`
fault), the agent messages of both sessions are merged in the order they were received, so the last agent message
wins.

### Time

`time_utc` is the start time the agent stated for the booked or rescheduled meeting (for `cancelled`, the time of
the cancelled meeting when one is stated), converted to UTC, or `null` when no specific time was stated. An explicit
zone label wins ("Berlin time", `Europe/Berlin`, "EDT", "UTC+2"; "our time" means the host zone). With no label, the
prospect's true zone is assumed. Relative dates resolve against the conversation date in the zone of the stated
time.

### Offered times

`offered_utc` is every specific start time the agent proposed as available, in any message, converted to UTC. The
time of an existing booking under discussion and times the prospect proposed are not offers.

## Harness-side faults

Two faults are injected by the harness, not the sandbox, on one of the persona's picks of an offered slot (a
`select_slot` action for the bundled protocol, or the slot's label as text):

- `duplicate_delivery` sends the same request again, with the same `message_id`, 50 to 200 ms after the first, while
  the first is still in flight. Both replies are recorded; the persona continues from the one that arrived last. It
  fires on the persona's first pick, unconditionally.
- `concurrent_channel` sends, at the same moment, a scripted message for the same lead on a new `session_id` with
  `channel=webhook`, asking for the second offered slot, or the same slot when only one was offered. A pass means
  exactly one active booking that matches the belief taken from both sessions. It needs at least one slot to have
  been offered by the pick it fires on, so it fires on the first pick that knows any, deferring past an earlier one
  that knows none; if no pick in the whole conversation ever does, it is skipped rather than injected. The trial's
  record says whether a configured harness fault actually fired (`harness_fault_injected`).

## Outcome of a trial

Each trial gets exactly one outcome: the first rule below that matches.

1. `harness_error`: a failure on the harness side only. A persona or extractor LLM error after retries, a
   `persona_error` (the persona accepted a time outside its hidden window), a sandbox crash, or a harness bug.
   Excluded from rates; counted and listed. The harness reruns such a trial up to twice; every attempt stays listed
   and the last attempt fills the slot.
2. `false_success`: the belief is `booked`, `rescheduled` or `cancelled`, but no matching state exists:
   - `booked`: no active booking for the lead other than the setup booking.
   - `rescheduled`: no active booking for the lead at a start time different from the setup booking's original
     start, or the setup booking is still active at its original time.
   - `cancelled`: the lead still has an active booking.
3. `time_mismatch`: the belief has a `time_utc`, and an active booking created or moved in this trial exists, but no
   such booking starts at that exact UTC minute.
4. `double_booking`: more than one active booking for the lead's email and the event key at trial end.
5. `invented_slot`: a specific time was offered (in `offered_utc`) or booked (the start of an active booking created
   or moved in this trial) that the sandbox never returned as available during this trial. The reference set is the
   union of all slot starts returned by the Cal.com slots endpoint in the request log; for Google it is the set of
   free slots computed from each logged `freeBusy` response (only calendar entries without `errors`) plus the seeded
   working hours, event length and minimum notice, inside that query's window. The setup booking's time is exempt.
   A booking created or moved in this trial that overlaps time held by someone else at the end of the trial (a seeded
   or third-party block, another attendee's booking) is also `invented_slot`: that time was not available when it was
   booked. Cal.com rejects such a booking; a Google insert does no conflict checking, so only a client that re-checks
   free/busy before inserting avoids it.
6. `wrong_time`: an active booking created or moved in this trial lies outside the persona's hidden window (checked
   in the persona's true zone).
7. `unclaimed_booking`: an active booking created or moved in this trial exists, but the belief is `not_booked`.
8. `crm_mismatch` (only with `--grade-crm`): for the lead's HubSpot contact, every active booking must have exactly one
   associated meeting with the same start and end and a non-cancelled outcome; every associated meeting with outcome
   `SCHEDULED` or `RESCHEDULED` must match an active booking (else it is a phantom); a cancelled booking's meeting,
   if any, must not be `SCHEDULED`. Any violation is a mismatch. With no active booking and no meeting, CRM is
   consistent.
9. `agent_error`: the agent returned a 5xx or a non-JSON body, timed out past `timeout_s`, or refused the connection
   (a `409 lead_busy` is a normal reply, not an error). Counts as a failure and is listed.
10. `goal_not_met`: the scenario's `expect:` block is not satisfied: the number of active bookings, the expected
    status (`booked`, `rescheduled`, `cancelled`, `none`) or `in_window`. A belief of `unclear` while an active
    booking exists is `goal_not_met`. For `expect.status: none`, the belief must be `not_booked`, or `cancelled`
    when a booking made during the trial was cancelled again before the end (the prospect retracted).
11. `pass`.

Outcomes 2 to 8 are **integrity violations**. In scenarios tagged `impossible`, a pass means no active booking, no
success belief and no offered time outside the reference set.

### When the belief is `unclear`

`unclear` never produces `false_success` or `time_mismatch`, because the prospect was not told a result. It produces
`goal_not_met` when an active booking exists or the goal was a booking; with no booking and `expect.status: none`, an
`unclear` belief still fails the goal, because the prospect was not told that nothing was booked.

## Aggregates

- **pass^k** (the tau-bench estimator): for each scenario with `n` valid trials of which `c` passed,
  `pass^k = C(c, k) / C(n, k)`; the suite value is the mean over scenarios with `n >= k`. Scenarios with fewer than `k`
  valid trials are left out and listed. With `n = k = 5`, pass^5 is the share of scenarios where all five trials
  passed. pass^1 is reported next to it.
- **False-success rate** (trial level): valid trials whose outcome is `false_success` or `time_mismatch`, divided by
  all valid trials.
- **False-claim share**: trials whose belief is a success belief (`booked`, `rescheduled`, `cancelled`) and whose
  outcome is `false_success` or `time_mismatch`, divided by all valid trials with a success belief.
- **Integrity category rates**: each of outcomes 2 to 8 divided by valid trials.
- **Per fault scenario**: the share of valid trials with no integrity violation, and the share that pass.
- **Timezone correct-slot rate**: in each timezone scenario, valid trials that end with an active booking inside the
  persona window whose start the prospect was told correctly (no `time_mismatch`), divided by valid trials.
- **Persona-error rate**: `persona_error` attempts divided by all attempts.
- **Extractor disagreement**: trials where the LLM and lexicon beliefs differ in status, reported separately for
  naive and guarded trials. Every naive `false_success` where they disagree is listed.
- **Latency**: p50 and p95 of agent turn latency (request sent to response received) and of conversation latency
  (first request to last response), nearest-rank method.
- **Cost per conversation**: agent USD (as reported in each response's `usage.usd`), persona USD and extractor USD.

### Intervals

Every aggregate rate carries a Wilson score 95% interval (z = 1.96). For trial-level rates, `n` is the number of
valid trials and `x` the count. For pass^k, `n` is the number of included scenarios and the point estimate is the
mean; the interval uses that mean as `p̂` with `n` scenarios. Trials within one scenario are correlated, so the
trial-level intervals are optimistic; the report says so.

## Accounting

The summary contains exactly scenarios × trials × agents result slots. Crashed trials are graded `harness_error` and
listed, never dropped. If more than 2% of slots are still `harness_error` after reruns, the run is marked invalid.
If an agent's `agent_version` changes during a run, the run aborts and is marked `version_drift`.

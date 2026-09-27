# booking-truth run 2026-09-27-bench

## Run

| Field | Value |
|---|---|
| Status | complete (valid) |
| Run date | 2026-09-27 |
| As-of date | 2026-09-27 |
| Hardware | MacBook Air M5, 24 GB |
| Harness | booking-truth 0.1.0, git c30e8c54c5ca (uncommitted changes) |
| Scenario suite | bundled, sha256:78bfae434212, 24 scenario(s) run |
| Trials per scenario (k) | 5 |
| CRM graded | yes |
| Agents | `guarded` (guarded, version a7e96fe8fee9, source 45a852f88d21, model deepseek/deepseek-v4-flash, calendar calcom); `naive` (naive, version f946e239d19a, source 45a852f88d21, model deepseek/deepseek-v4-flash, calendar calcom) |
| Grading | LLM grading: llm personas, llm belief extractor |
| Total spend | $0.2427 (agent $0.1643, persona $0.0522, extractor $0.0262) |
| Reproduce | `booking-truth test --pool bench-pool.yaml --k 5 --grade-crm --as-of 2026-09-27 --hardware 'MacBook Air M5, 24 GB' --budget-usd 2` |

## Headline

| Metric | guarded | naive |
|---|---|---|
| pass^1 | 100.0% [86.2, 100.0] (24 scenarios) | 34.2% [18.6, 54.1] (24 scenarios) |
| pass^5 | 100.0% [86.2, 100.0] (24 scenarios) | 16.7% [6.7, 35.9] (24 scenarios) |
| False-success rate | 0.0% [0.0, 3.1] (0/120) | 14.2% [9.0, 21.5] (17/120) |
| False-claim share | 0.0% [0.0, 3.6] (0/104) | 17.3% [11.1, 26.0] (17/98) |
| false_success outcome rate | 0.0% [0.0, 3.1] (0/120) | 0.8% [0.1, 4.6] (1/120) |
| Time-mismatch rate | 0.0% [0.0, 3.1] (0/120) | 13.3% [8.4, 20.6] (16/120) |
| Double-booking rate | 0.0% [0.0, 3.1] (0/120) | 8.3% [4.6, 14.7] (10/120) |
| Invented-slot rate | 0.0% [0.0, 3.1] (0/120) | 8.3% [4.6, 14.7] (10/120) |
| Wrong-time rate | 0.0% [0.0, 3.1] (0/120) | 0.0% [0.0, 3.1] (0/120) |
| Unclaimed-booking rate | 0.0% [0.0, 3.1] (0/120) | 0.0% [0.0, 3.1] (0/120) |
| CRM-mismatch rate | 0.0% [0.0, 3.1] (0/120) | 30.8% [23.3, 39.6] (37/120) |
| Any integrity violation | 0.0% [0.0, 3.1] (0/120) | 61.7% [52.7, 69.9] (74/120) |
| Persona-error rate (attempts) | 3.2% [1.3, 7.9] (4/125) | 0.0% [0.0, 3.1] (0/120) |
| Turn-cap hits | 0 | 0 |

## Outcomes

| Outcome | guarded | naive |
|---|---|---|
| harness_error | 0 | 0 |
| false_success | 0 | 1 |
| time_mismatch | 0 | 16 |
| double_booking | 0 | 10 |
| invented_slot | 0 | 10 |
| wrong_time | 0 | 0 |
| unclaimed_booking | 0 | 0 |
| crm_mismatch | 0 | 37 |
| agent_error | 0 | 0 |
| goal_not_met | 0 | 5 |
| pass | 120 | 41 |
| valid / slots | 120/120 | 120/120 |

## Fault scenarios

Valid trials with no integrity violation, and valid trials that pass.

| Scenario | guarded: no violation | guarded: pass | naive: no violation | naive: pass |
|---|---|---|---|---|
| fault-commit-then-timeout | 5/5 | 5/5 | 0/5 | 0/5 |
| fault-concurrent-channel | 5/5 | 5/5 | 1/5 | 1/5 |
| fault-create-500-persistent | 5/5 | 5/5 | 1/5 | 1/5 |
| fault-create-timeout-once | 5/5 | 5/5 | 4/5 | 4/5 |
| fault-crm-500-once | 5/5 | 5/5 | 1/5 | 1/5 |
| fault-duplicate-delivery | 5/5 | 5/5 | 0/5 | 0/5 |
| fault-slot-taken-after-offer | 5/5 | 5/5 | 1/5 | 1/5 |
| fault-slots-500-once | 5/5 | 5/5 | 5/5 | 1/5 |
| fault-slots-malformed | 5/5 | 5/5 | 0/5 | 0/5 |
| fault-slots-not-found | 5/5 | 5/5 | 5/5 | 5/5 |

Harness-side fault (`duplicate_delivery` fires unconditionally on the first pick; `concurrent_channel` fires on the first pick that knows any offered slot) never fired in at least one valid trial of: `fault-concurrent-channel` (see that trial's `meta.harness_fault` in `traces.jsonl`).

## Timezone scenarios

Valid trials that end with a booking inside the persona's window whose start the prospect was told correctly.

| Scenario | guarded | naive |
|---|---|---|
| tz-after-dst-change | 100.0% [56.6, 100.0] (5/5) | 20.0% [3.6, 62.4] (1/5) |
| tz-cst | 100.0% [56.6, 100.0] (5/5) | 100.0% [56.6, 100.0] (5/5) |
| tz-ist | 100.0% [56.6, 100.0] (5/5) | 100.0% [56.6, 100.0] (5/5) |
| tz-kathmandu | 100.0% [56.6, 100.0] (5/5) | 80.0% [37.6, 96.4] (4/5) |
| tz-sydney-next-friday | 100.0% [56.6, 100.0] (5/5) | 80.0% [37.6, 96.4] (4/5) |
| tz-us-eu-dst-gap | 100.0% [56.6, 100.0] (5/5) | 0.0% [0.0, 43.4] (0/5) |

## Cost and latency

| Metric | guarded | naive |
|---|---|---|
| Agent USD per conversation | $0.0006 | $0.0007 |
| Persona USD per conversation | $0.0002 | $0.0002 |
| Extractor USD per conversation | $0.0001 | $0.0001 |
| Turn latency p50 / p95 | 2.26 s / 13.29 s | 3.52 s / 13.99 s |
| Conversation latency p50 / p95 | 16.96 s / 33.09 s | 21.86 s / 43.05 s |
| Guard overhead (turns blocked or repaired) | 3.1% [1.8, 5.1] (14/453) | 0.0% [0.0, 0.8] (0/474) |

Spend in the result slots (every attempt): $0.2410. The run's total spend, preflight turns included, is in the Run table.

## Belief extractors

| Agent mode | LLM vs lexicon disagreement |
|---|---|
| naive | 10.0% [5.8, 16.7] (12/120) |
| guarded | 4.2% [1.8, 9.4] (5/120) |

Naive `false_success` trials where the extractors disagree: none.

## Integrity violations

- `2026-09-27-bench/naive/happy-book-browser-hint/0/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-book-browser-hint/4/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-book-host-zone/0/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-book-host-zone/2/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-book-host-zone/3/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-book-host-zone/4/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-move-it/0/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-move-it/1/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-move-it/2/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-move-it/3/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-move-it/4/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-push-thursday/0/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-push-thursday/1/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-push-thursday/2/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-push-thursday/3/1`: crm_mismatch
- `2026-09-27-bench/naive/happy-reschedule-push-thursday/4/1`: crm_mismatch
- `2026-09-27-bench/naive/tz-after-dst-change/0/1`: time_mismatch
- `2026-09-27-bench/naive/tz-after-dst-change/1/1`: crm_mismatch
- `2026-09-27-bench/naive/tz-after-dst-change/2/1`: time_mismatch
- `2026-09-27-bench/naive/tz-after-dst-change/4/1`: time_mismatch
- `2026-09-27-bench/naive/tz-cst/1/1`: crm_mismatch
- `2026-09-27-bench/naive/tz-cst/2/1`: crm_mismatch
- `2026-09-27-bench/naive/tz-ist/1/1`: crm_mismatch
- `2026-09-27-bench/naive/tz-ist/2/1`: crm_mismatch
- `2026-09-27-bench/naive/tz-ist/3/1`: invented_slot
- `2026-09-27-bench/naive/tz-ist/4/1`: invented_slot
- `2026-09-27-bench/naive/tz-kathmandu/0/1`: crm_mismatch
- `2026-09-27-bench/naive/tz-kathmandu/1/1`: time_mismatch
- `2026-09-27-bench/naive/tz-kathmandu/4/1`: crm_mismatch
- `2026-09-27-bench/naive/tz-sydney-next-friday/1/1`: invented_slot
- `2026-09-27-bench/naive/tz-sydney-next-friday/2/1`: invented_slot
- `2026-09-27-bench/naive/tz-sydney-next-friday/3/1`: time_mismatch
- `2026-09-27-bench/naive/tz-sydney-next-friday/4/1`: invented_slot
- `2026-09-27-bench/naive/tz-us-eu-dst-gap/0/1`: time_mismatch
- `2026-09-27-bench/naive/tz-us-eu-dst-gap/1/1`: time_mismatch
- `2026-09-27-bench/naive/tz-us-eu-dst-gap/2/1`: time_mismatch
- `2026-09-27-bench/naive/tz-us-eu-dst-gap/3/1`: time_mismatch
- `2026-09-27-bench/naive/tz-us-eu-dst-gap/4/1`: time_mismatch
- `2026-09-27-bench/naive/fault-commit-then-timeout/0/1`: double_booking
- `2026-09-27-bench/naive/fault-commit-then-timeout/1/1`: double_booking
- `2026-09-27-bench/naive/fault-commit-then-timeout/2/1`: double_booking
- `2026-09-27-bench/naive/fault-commit-then-timeout/3/1`: double_booking
- `2026-09-27-bench/naive/fault-commit-then-timeout/4/1`: double_booking
- `2026-09-27-bench/naive/fault-concurrent-channel/0/1`: double_booking
- `2026-09-27-bench/naive/fault-concurrent-channel/1/1`: time_mismatch
- `2026-09-27-bench/naive/fault-concurrent-channel/2/1`: time_mismatch
- `2026-09-27-bench/naive/fault-concurrent-channel/3/1`: time_mismatch
- `2026-09-27-bench/naive/fault-create-500-persistent/0/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-create-500-persistent/1/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-create-500-persistent/2/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-create-500-persistent/3/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-create-timeout-once/2/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-crm-500-once/0/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-crm-500-once/2/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-crm-500-once/3/1`: time_mismatch
- `2026-09-27-bench/naive/fault-crm-500-once/4/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-duplicate-delivery/0/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-duplicate-delivery/1/1`: double_booking
- `2026-09-27-bench/naive/fault-duplicate-delivery/2/1`: double_booking
- `2026-09-27-bench/naive/fault-duplicate-delivery/3/1`: double_booking
- `2026-09-27-bench/naive/fault-duplicate-delivery/4/1`: double_booking
- `2026-09-27-bench/naive/fault-slot-taken-after-offer/0/1`: time_mismatch
- `2026-09-27-bench/naive/fault-slot-taken-after-offer/1/1`: time_mismatch
- `2026-09-27-bench/naive/fault-slot-taken-after-offer/3/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-slot-taken-after-offer/4/1`: crm_mismatch
- `2026-09-27-bench/naive/fault-slots-malformed/0/1`: invented_slot
- `2026-09-27-bench/naive/fault-slots-malformed/1/1`: invented_slot
- `2026-09-27-bench/naive/fault-slots-malformed/2/1`: invented_slot
- `2026-09-27-bench/naive/fault-slots-malformed/3/1`: invented_slot
- `2026-09-27-bench/naive/fault-slots-malformed/4/1`: invented_slot
- `2026-09-27-bench/naive/adv-retract-confirmation/0/1`: crm_mismatch
- `2026-09-27-bench/naive/adv-retract-confirmation/1/1`: crm_mismatch
- `2026-09-27-bench/naive/adv-retract-confirmation/3/1`: false_success
- `2026-09-27-bench/naive/adv-retract-confirmation/4/1`: crm_mismatch

## Agent errors

None.

## Harness errors and reruns

- rerun `happy-reschedule-move-it` #0: attempts 1: harness_error, 2: pass
- rerun `tz-sydney-next-friday` #0: attempts 1: harness_error, 2: pass
- rerun `tz-sydney-next-friday` #2: attempts 1: harness_error, 2: pass
- rerun `tz-sydney-next-friday` #4: attempts 1: harness_error, 2: pass
- rerun `tz-us-eu-dst-gap` #4: attempts 1: harness_error, 2: pass

## Accounting

| Check | Value |
|---|---|
| Expected result slots | 240 |
| Result slots present | 240 |
| Missing slots | 0 |
| Duplicate slots | 0 |
| harness_error slots after reruns | 0 (0.0%; the run is invalid above 2%) |
| Run valid | yes |

## Notes

- Wilson score 95% intervals (z = 1.96). Trial-level intervals treat trials as independent, but trials within one scenario are correlated, so those intervals are optimistic. pass^k intervals use the mean over scenarios as the proportion and the number of included scenarios as n.
- Regenerate this file and `summary.json` from `traces.jsonl` and `manifest.json` with `booking-truth report <run directory>`.

# Timezone resolver eval

| Field | Value |
|---|---|
| Dataset | `datasets/tz_phrases.jsonl` (held-out test split, n = 100) |
| Equivalence window | 2026-01-01T00:00:00Z + 730 days |
| Model (LLM-only side) | deepseek/deepseek-v4-flash |

## Deterministic resolver

| Category | Count | Rate |
|---|---|---|
| Correct | 61 | 61.0% |
| Missed ambiguity | 5 | 5.0% |
| Silent wrong resolution | 0 | 0.0% |
| Over-cautious (asked needlessly) | 19 | 19.0% |
| Correctly flagged (ambiguous or unknown) | 15 | 15.0% |

## LLM-only resolution

| Category | Count | Rate |
|---|---|---|
| Correct | 74 | 74.0% |
| Missed ambiguity | 8 | 8.0% |
| Silent wrong resolution | 1 | 1.0% |
| Over-cautious (asked needlessly) | 5 | 5.0% |
| Correctly flagged (ambiguous or unknown) | 12 | 12.0% |

Attempted 100, scored 100, 0 error(s).

## Notes

- **Silent wrong resolution** is the number that matters: a resolver that answers with a single zone that is not what the prospect meant, without asking, silently mis-schedules the meeting.
- Candidate lists are not scored; only the resolved/ambiguous/unknown status and, when resolved, the zone.
- Reproduce with `booking-truth eval tz`.

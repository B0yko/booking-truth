# Belief extractor eval

| Field | Value |
|---|---|
| Dataset | `datasets/belief_extraction.jsonl` (held-out test split, n = 80) |
| Model (LLM extractor) | deepseek/deepseek-v4-flash |

## Accuracy

| Extractor | Status accuracy | Time-match accuracy | Recall on success-claim items |
|---|---|---|---|
| Lexicon | 93.8% | 91.2% | 100.0% (n=44) |
| LLM | 93.8% | 100.0% | 100.0% (n=44) |

LLM extractor: attempted 80, scored 80, 0 error(s).

## Agreement on benchmark trials

From `2026-09-27-bench`'s `summary.json`.

| Agent mode | LLM vs lexicon disagreement |
|---|---|
| guarded | 4.2% (5/120) |
| naive | 10.0% (12/120) |

Naive `false_success` trials where the extractors disagree: none.

## Notes

- Status accuracy and time-match accuracy are computed over every test item; a correct `null` time (no time was stated) counts as a match.
- Recall on success-claim items: of the items whose gold status is `booked`, `rescheduled` or `cancelled`, the share the extractor also called a success (of either kind).
- Reproduce with `booking-truth eval extractor` (add `--run <results dir>` for the agreement section).

# `results/`

Each subdirectory is one run of `booking-truth test`, named by its run id (for example
`results/2026-09-27-bench/`). Every number the README's `<!-- bt:... -->` tables show comes from one of
these directories, and nothing else: `scripts/update_readme_tables.py results/<run-id>` regenerates them,
and CI fails when the README differs from what a run directory produces.

## Files in a run directory

| File | Written by | What it holds |
|---|---|---|
| `summary.json` | `booking-truth test`, or `booking-truth report <dir>` | Every aggregate rate, pass^k, cost and latency number, per agent, with Wilson 95% intervals and raw counts. A pure function of `traces.jsonl` + `manifest.json` (`booking_truth.harness.report.build_summary`): recomputing it from those two files is byte-for-byte reproduction, no network. |
| `report.md` | same | The same numbers as `summary.json`, rendered as a human-readable Markdown report (not the README - a self-contained summary of this one run). |
| `manifest.json` | `booking-truth test` | What the run *was*: date, `--as-of`, hardware, harness version, git SHA, scenario-suite hash, each agent's version and source hash, model ids, the model/provider each call actually returned (flagged when either varied, per component: `agent:<label>`, `persona`, `extractor`), temperatures and total spend. |
| `traces.jsonl` | `booking-truth test` | One `agent-trace/v1` record per line (`schemas/agent-trace-v1.json`), every attempt of every trial. Email addresses are redacted before this file is committed. |
| `tz-eval.json`, `tz-eval.md` | `booking-truth eval tz` | The deterministic timezone resolver scored against LLM-only resolution, on the held-out test split of `datasets/tz_phrases.jsonl`. The number that matters is `silent_wrong_resolution`: a resolver that silently picks the wrong zone instead of asking. |
| `extractor-eval.json`, `extractor-eval.md` | `booking-truth eval extractor` | The LLM and lexicon belief extractors scored on the held-out test split of `datasets/belief_extraction.jsonl`, plus (with `--run <bench dir>`) their agreement rate on that benchmark run's own trials. |
| `ci_fixtures.json` | `scripts/export_ci_fixtures.py` | Every offline guard fixture (`tests/fixtures/guards/*.yaml`), both calendars, guard on and off, with a pass/fail status each. This is the harness's own regression suite exported as a results artifact - "by construction, not a benchmark" (`docs/metrics.md`): each fixture pairs one scripted misbehaviour with the guard that exists to catch it. |
| `failure_notes.yaml` | hand-written | One root-cause paragraph per trace id of a guarded integrity violation, keyed by the trace id exactly as it appears in `traces.jsonl` (for example `2026-09-27-bench/guarded/happy-book-host-zone/2/1: >`). Feeds the README's `bt:failures` table; a violation with no entry here is shown as "root cause not analysed yet." |

`eval_tz.json`/`eval_extractor.json` are not the right names: the `eval` CLI writes `tz-eval.json` and
`extractor-eval.json`, and that is what `booking_truth.harness.readme_tables` reads.

## Regenerating each file

```bash
# The benchmark run itself (summary.json, report.md, manifest.json, traces.jsonl)
booking-truth test --pool bench-pool.yaml --k 5 --grade-crm --as-of <date> --hardware "<hardware>" \
    --out results/<run-id>

# Recompute summary.json/report.md from the same traces.jsonl + manifest.json, no network
booking-truth report results/<run-id>

# The component evals
booking-truth eval tz --out results/<run-id>
booking-truth eval extractor --out results/<run-id> --run results/<run-id>

# The offline guard-fixture regression suite, as a results artifact
uv run python scripts/export_ci_fixtures.py --out results/<run-id>

# failure_notes.yaml: write one entry by hand per row of the README's "Remaining failures" table,
# keyed by that trial's trace id (see `summary.json`'s `by_agent.<agent>.integrity_violations`)

# The README's bt:* tables, from whichever of the above exist in the directory
uv run python scripts/update_readme_tables.py results/<run-id>            # rewrite README.md
uv run python scripts/update_readme_tables.py results/<run-id> --check    # exit 1 when out of date
```

A missing optional input (an eval not run yet, no `ci_fixtures.json`, no `failure_notes.yaml`) renders a
"not run" row rather than an error, so the tables can be rebuilt at every stage of a run.

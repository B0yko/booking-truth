# Datasets

Data files used by the timezone resolver and by the two component evaluations
(`booking-truth eval tz`, `booking-truth eval extractor`).

| File | Purpose | Source | Licence | Rebuild |
|---|---|---|---|---|
| `cities_tz.csv` | City gazetteer for timezone resolution (34,149 rows) | GeoNames `cities15000`, downloaded 2026-09-26 | CC BY 4.0 (GeoNames), see `NOTICE` | `uv run python scripts/build_cities_tz.py` (network) |
| `tz_phrases.jsonl` | 150 labelled prospect timezone phrases | written for this repository; city names from `cities_tz.csv` | Apache-2.0; GeoNames names CC BY 4.0 | `uv run python scripts/build_tz_phrases.py` |
| `belief_extraction.jsonl` | 120 labelled agent-side transcripts with the gold prospect belief | written for this repository | Apache-2.0 | `uv run python scripts/build_belief_extraction.py` |
| `HASHES.json` | sha256 of the held-out test split of both labelled sets | `scripts/dataset_hashes.py` | Apache-2.0 | see [Split and hash policy](#split-and-hash-policy) |
| `NOTICE` | Attribution for third-party data | | | |

All data is synthetic or public. No conversation data from real people is included; every transcript
and phrase was written for this repository. Timezone rules come from the IANA tz database (public
domain) through the `tzdata` package (2026.4, IANA 2026d); `zone.tab` and `iso3166.tab` are read from
that package, not copied here.

## `cities_tz.csv`

UTF-8 CSV with a header row:

| Column | Meaning |
|---|---|
| `name` | GeoNames name (UTF-8) |
| `asciiname` | GeoNames ASCII name |
| `country_code` | ISO 3166-1 alpha-2 |
| `admin1_code` | GeoNames first-level division code |
| `admin1_name` | ASCII name of that division from `admin1CodesASCII.txt` (empty when unknown) |
| `population` | GeoNames population |
| `timezone` | IANA zone key |

`scripts/build_cities_tz.py` downloads `cities15000.zip` and `admin1CodesASCII.txt` into a cache
directory outside the repository (`--cache-dir`, default a temporary directory), drops rows whose
timezone is not a `tzdata` zone key, sorts by ASCII name, country, admin1 code, descending population
and name, and prints the row count and the file's sha256. GeoNames rebuilds its dump daily and the
URLs carry no version, so the committed file is the reference snapshot: rebuilding it on another day
gives a slightly different table, and would also change the city samples in `tz_phrases.jsonl`. Do not
rebuild it without re-recording the dataset hashes and saying why.

## `tz_phrases.jsonl`

One JSON object per line:

```json
{"id": "tzp-108", "text": "I'm in Portland", "label": {"status": "ambiguous", "zone": null,
 "candidates": ["America/Los_Angeles", "America/New_York"]}, "source": "hard_case", "split": "dev",
 "note": "City rule: ..."}
```

- `label.status`: `resolved` (with `zone`, `candidates` empty), `ambiguous` (with `zone` null and at
  least two pairwise non-equivalent `candidates`, sorted) or `unknown` (no zone, no candidates).
- `source`: `city_sample` (45), `template` (35) or `hard_case` (70).
- `note`: the rule that produced the label.

**Sources.**

- `city_sample`: 45 GeoNames cities sampled with `random.Random(20260926)` from four population strata
  (>= 1M: 12, 300k-1M: 11, 100k-300k: 11, 15k-100k: 11), phrased with templates such as "I'm in X",
  "we're based in X", "X time works" or "calling from X, Y". 12 of the 45 carry a region or
  country qualifier; a city that is not the most populous city of its name is always qualified, so
  the phrase refers to the sampled city. Names that are also the name of a region or country in a
  different zone are not sampled.
- `template`: 12 fixed offsets (9 whole-hour, 3 fractional, written as `UTC+2`, `GMT -3`,
  `UTC+05:30`, ...), 12 zone names (the four ambiguous ones, "Mountain time", "Atlantic time",
  "Australian Eastern Time" and "Australian Central Time", plus a seeded sample of the others such
  as "Eastern time" and "Brasilia time") and 11 abbreviations (AST and PST plus a sample such as
  "HST", "MSK" and "PT").
- `hard_case`: the hand-written list `HARD_CASES` in `scripts/build_tz_phrases.py`: ambiguous
  abbreviations (IST, CST, BST), EST, "Eastern", "Central European Time", host-relative phrases,
  same-name cities (Portland, Springfield, Birmingham, Victoria, London, Hyderabad, San Jose, Perth),
  qualified cities, exonyms and colloquial names (Kiev, NYC, LA, the Bay Area), regions (Queensland,
  Arizona, Hawaii, Newfoundland, Chatham Islands, Indiana, Texas, Georgia, Lord Howe Island), countries
  (India, Nepal, China, Brazil, USA, Russia, Spain, Kazakhstan, the UK), fixed offsets and the
  `Etc/GMT` sign convention, typos, and phrases that name no zone. Where a rule below applies, the
  build recomputes the label and fails if it disagrees with the hand-written one.

### Labelling rules

The rules define what a phrase refers to. They do not depend on, and were written before, any
resolver in this repository.

1. **Equivalence.** Two zones are *equivalent* when their UTC offsets are identical at every instant
   from 2026-01-01T00:00Z to 2028-01-01T00:00Z, that is all of 2026 and 2027 (computed with
   `zoneinfo` from the `tzdata` package). A resolved zone is scored as correct when it is equivalent
   to the gold zone, so the gold zone is one canonical `zone.tab` zone of its class (for a GeoNames
   city, its GeoNames zone). Candidates are pairwise non-equivalent.
2. **Gold is the zone the phrase refers to.** Every resolved zone and every candidate is a valid
   `zoneinfo` key.
3. **Explicit IANA keys** are taken as written, including `Etc/GMT+3`, which is UTC-03:00.
4. **Fixed offsets** (`UTC-5`, `GMT+2`, `UTC+05:30`) are taken literally as fixed offsets, not as a
   regional zone. Whole-hour offsets map to `Etc/GMT` keys, whose sign is inverted (`GMT+2` ->
   `Etc/GMT-2`). No `Etc/GMT` key exists for other offsets, so gold is the `zone.tab` zone with that
   constant offset throughout the window (`UTC+05:30` -> `Asia/Kolkata`, `UTC+5:45` ->
   `Asia/Kathmandu`).
5. **Abbreviations and zone names** used for several non-equivalent regions are `ambiguous` with their
   common readings as candidates: IST (India, Israel, Ireland), CST (US Central, China), BST (British
   Summer Time, Bangladesh), AST (Atlantic Canada, Puerto Rico and the Caribbean, Arabia), "Atlantic
   time" (Atlantic Canada with summer time, Puerto Rico without), PST (Pacific, Philippines),
   "Mountain time" (Colorado and Utah with summer time, Arizona without), "Australian Eastern Time"
   (New South Wales with summer time, Queensland without), "Australian Central Time" (South Australia
   with summer time, the Northern Territory without). Candidate lists name the common readings; they
   are not exhaustive. A name with one dominant reading resolves: EST, ET, EDT, "Eastern", "Central
   time", CT, "Pacific time", PT and PDT refer to the US zones, and minority readings are not
   candidates (Saskatchewan keeps Central Standard Time all year and Alberta has kept UTC-06:00 all
   year since March 2026; British Columbia has kept UTC-07:00 all year since March 2026, which
   `tzdata` abbreviates MST, not Pacific time). CET/CEST and EET/EEST refer to the EU regions (their member zones are equivalent).
   A daylight-time abbreviation names its region, not a fixed offset.
6. **Host-relative phrases** ("same as you", "your time is fine") resolve to the host zone,
   `America/New_York`, the sandbox default host timezone.
7. **Cities.** Names match case- and accent-insensitively on the GeoNames name or ASCII name. A bare
   city name refers to the most populous match. Another city with the same name makes it `ambiguous`
   only when that city lies in a non-equivalent zone **and** passes both thresholds: a population of
   at least 50,000 and at least 10% of the most populous match's population. The candidates are the
   zones of the most populous match and of every other match that passes both thresholds, collapsed
   by equivalence. Otherwise the name resolves to the most populous match's zone. Rationale: a human
   booking agent would ask when a meaningful minority of people using that name mean another place.
   So "Portland" (Portland, Maine: 66,881, 10.2% of Portland, Oregon) and "Birmingham" (Birmingham,
   Alabama: 17.0% of Birmingham, England) are ambiguous, while "London" resolves to `Europe/London`
   (London, Ontario has 422,324 inhabitants but 4.7% of London, England's population) and "Perth"
   resolves to `Australia/Perth` (Perth, Scotland is under 50,000). A qualified city ("Portland,
   Maine", "London, Ontario") applies the same rule to the matches in that region or country, with
   the share taken of the most populous match there. A name that is also a region adds the region's
   zone as a candidate ("Victoria": Hong Kong, Victoria BC and the Australian state; Victoria, Texas is
   7.1% of Victoria, Hong Kong and is not a candidate). Misspelled names that a reader would recognise
   ("Chicgo") are labelled as the intended city.
8. **Regions and countries.** A region or country resolves when all its zones are equivalent, and is
   `ambiguous` with those zones otherwise. For a first-level region (a US state, an Australian state)
   the zones are those of its GeoNames cities in `cities_tz.csv` (population 15,000 or more): Arizona
   resolves to `America/Phoenix` because the Navajo Nation, which observes summer time, has no such
   city, while Texas (El Paso on Mountain time) and Indiana (north-west and south-west counties on
   Central time) are ambiguous. For countries the zones are the country's `zone.tab` entries: India
   (one zone) and Kazakhstan (a single UTC+05:00 time since 2024) resolve; China (Beijing and
   Xinjiang time), Spain (mainland and Canary Islands), Brazil, Russia and the USA are ambiguous. A
   place that is both a region and a country ("Georgia") has the zones of both.
9. **Unknown.** A phrase that names no place, zone or offset ("on the moon", "wherever works",
   "just use my local time") or only a continent is `unknown`.

These rules are deliberately independent of the resolver's own heuristics, so the evaluation is not
circular. The city rule (rule 7) is intentionally different from the resolver's specified city rule,
which is ambiguous when the runner-up match has at least 20% of the top match's population, so the
evaluation can expose where the two disagree. Of the 67 phrases labelled by the city rule (the 45
`city_sample` items and 22 hard cases), they disagree on two, where the other city passes the 10%
threshold but not the 20% one:

| Phrase | Split | Other same-name city | Share of the top match | This dataset | 20% rule |
|---|---|---|---|---|---|
| "I'm in Portland" | dev | Portland, Maine (66,881) | 10.2% | `ambiguous` | `America/Los_Angeles` |
| "Birmingham" | test | Birmingham, Alabama (196,357) | 17.0% | `ambiguous` | `Europe/London` |

A resolver that implements the 20% rule as specified scores a missed ambiguity on each (see
[Scoring](#scoring)). No phrase disagrees the other way (a same-name city with at least 20% of the top match's population
but fewer than 50,000 inhabitants, which the 20% rule would flag and this one would not). The result
is the same whether "runner-up" is read as the second most populous match or as the most populous
match in a non-equivalent zone. Both rules resolve "London" (London, Ontario is 4.7% of London,
England) and "Perth", and both flag "Springfield", "Victoria", "Hyderabad" and "San Jose". For "San
Jose" the gold candidates also include `Asia/Manila` (San Jose, Mimaropa: 14.4%), which the 20% rule
would leave out; candidate lists are not scored (see below).

### Scoring

A resolver's answer for one phrase is one of: a single zone, `ambiguous` (it asks the prospect,
optionally with candidates) or `unknown` (it asks for the timezone). Against the gold label:

| Gold | Single zone equivalent to gold (or to a candidate) | Single zone equivalent to nothing in gold | `ambiguous` or `unknown` |
|---|---|---|---|
| `resolved` | correct | **silent wrong resolution** | over-cautious (asked needlessly) |
| `ambiguous` | missed ambiguity (it guessed one real reading without asking) | **silent wrong resolution** | correctly flagged |
| `unknown` | (no gold zone) | **silent wrong resolution** (any single zone) | correctly flagged |

Candidate lists are not exhaustive, so "correctly flagged" does not require the resolver's candidates
to match the gold candidates.

## `belief_extraction.jsonl`

One JSON object per line:

```json
{"id": "be-001", "as_of": "2027-03-29T13:00:00Z", "prospect_zone": "Europe/Berlin",
 "host_zone": "America/New_York", "agent_messages": ["I have Wednesday 31 March at 10:00 AM ET
 (4:00 PM your time) or 1:00 PM ET (7:00 PM your time). Which do you prefer?"],
 "gold": {"status": "not_booked", "time_utc": null,
          "offered_utc": ["2027-03-31T14:00:00Z", "2027-03-31T17:00:00Z"]},
 "tags": ["dst_week", "dual_label", "offers"], "source": "hard_case", "split": "dev"}
```

- `as_of`: the reference instant of the conversation (RFC 3339, UTC), spread over 2026-10 to 2027-04
  with extra weight on the weeks around DST changes, so some items fall into weeks where the host and
  prospect offsets differ from usual.
- `agent_messages`: 1-4 agent messages, the end of a conversation. The prospect's messages are not
  included.
- `gold.status`: `booked`, `rescheduled`, `cancelled`, `not_booked` or `unclear`.
- `gold.time_utc`: the meeting start in UTC (`Z`), or `null`.
- `gold.offered_utc`: every offered start time in UTC, deduplicated and sorted.
- `tags`: what the item exercises (`label:none|prospect|host|utc`, `relative_date`, `dst_week`,
  `half_hour_zone`, `rendered_line`, `retracted`, `hedged`, `conditional`, `no_time`, ...).
- `source`: `template` (79) or `hard_case` (41).

**Sources.** `template` items come from phrasing templates filled with a seeded RNG
(`random.Random(20260926)`): plain confirmations, code-rendered confirmation lines
(`Booked: Tuesday 6 October 2026, 3:00 PM Europe/Berlin (UTC+02:00) · reference abc123`), offers only,
questions only, hand-offs, reschedule and cancel confirmations (plain and rendered), failures ("I
couldn't book it", "the calendar is unavailable right now") and pending statements. Offered times are
host working hours (09:00-17:00 New York, Monday to Friday) at least two hours after `as_of`. Each
time is rendered in the prospect's zone ("Berlin time", "your time", "(Europe/Berlin)"), the host's
zone ("our time", "ET", "Eastern", "EST"/"EDT" as appropriate), UTC, or with no label, with absolute
or relative dates. `hard_case` items are the hand-written list `HARD_CASES` in
`scripts/build_belief_extraction.py`; a comment on each gives the reason for its label. The build
rejects hand-written local times that do not exist or are ambiguous (DST gaps and folds), and any
"Weekday D Month" whose weekday does not match the date.

### Labelling rules

The gold label is what a reasonable prospect believes at the end of the conversation from the agent's
messages alone, as defined by the prospect-belief taxonomy in `docs/metrics.md` (normative):

1. **Status.** `booked`: the agent stated without hedging that a new meeting is booked, confirmed or
   scheduled ("you're all set", "see you Tuesday", "the invite is on its way"). `rescheduled`: an
   existing meeting was moved ("I've moved it"). `cancelled`: the meeting was cancelled. `not_booked`:
   the agent said nothing is booked, that it could not book, move or cancel, or retracted an earlier
   claim; or the conversation ends with only offers, questions, a request for confirmation, a
   conditional ("once you confirm, you'll be booked") or a hand-off. `unclear`: the last
   status-relevant statement is hedged or pending ("should be booked", "I think it went through", "the
   status is unconfirmed", "I'll book that now" with nothing after it), or success and failure
   statements contradict each other with no clear final one.
2. **Precedence.** The last status-relevant statement decides; a later retraction beats an earlier
   claim. A code-rendered confirmation line counts like any other statement.
3. **Time.** For `booked` and `rescheduled`, `time_utc` is the start time the agent stated for that
   meeting (possibly in an earlier message, as in "Tuesday at 3 PM is free" followed by "see you
   Tuesday"), or `null` when no time was stated anywhere ("You're all set, the invite is on its
   way"; tagged `no_time`). For `cancelled`, it is the cancelled meeting's time when stated, else
   `null`. For `not_booked` and `unclear` it is `null`.
4. **Zone of a stated time.** An explicit label wins: an IANA name, "Berlin time", "EDT", "ET",
   "UTC", "your time" (prospect zone), "our time" (host zone). A time with no label is in the
   prospect's zone.
5. **Dates.** Absolute dates are taken as stated. "Tomorrow" and weekday names resolve against
   `as_of` converted to the zone of the stated time; a weekday name means its next occurrence after
   that date (every weekday name used for a stated time is 2 to 6 days ahead, so this reading is
   never in doubt).
6. **Offered times.** Every specific start time the agent proposed as available in any message,
   including a slot that was later taken or booked. Not offered: the time of an existing booking being
   discussed, a time the prospect proposed, and a time the agent only reports trying to book.

## Split and hash policy

Each labelled set is split into dev (1/3) and held-out test (2/3): 50/100 phrases and 40/80
transcripts. The split is a seeded, stratified shuffle: items are grouped by `(source, label status)`,
each group is shuffled with `random.Random(20260926)`, the groups are concatenated in sorted order,
and every third item goes to dev. Every label class therefore keeps about the same share in both
splits.

**Report only on the held-out test split; the hashes were recorded before any tuning.** Use the dev
split to develop the resolver alias table, the lexicon extractor and the extractor prompt; never
inspect test items to tune them. `HASHES.json` stores the sha256 of each test split in canonical form:
the `split == "test"` items sorted by `id`, each serialised with
`json.dumps(item, sort_keys=True, ensure_ascii=False)` followed by `\n`, encoded as UTF-8. Verify with:

```sh
uv run python scripts/dataset_hashes.py --check    # exits 1 on any mismatch
```

The write mode (`uv run python scripts/dataset_hashes.py`) refuses to replace recorded hashes unless
`--force` is given. Re-recording them is a deliberate change that needs a stated reason.

The `tz_phrases.jsonl` hash was re-recorded on 2026-09-26, before any resolver existed and so before
any tuning, when the city rule gained the 10% share threshold (rule 7). Two labels changed: "London"
went from `ambiguous` (`America/Toronto`, `Europe/London`) to `Europe/London`, and "Victoria" lost
`America/Chicago` (Victoria, Texas) as a candidate. The split was recomputed with the same seeded
procedure; because it is stratified by label status, the new "London" label changed the seeded
shuffle and moved 46 items between dev and test (23 each way).

## Rebuilding

```sh
uv run python scripts/build_tz_phrases.py          # reads datasets/cities_tz.csv
uv run python scripts/build_belief_extraction.py
uv run python scripts/dataset_hashes.py --check
```

Both labelled builders are deterministic: with the committed `cities_tz.csv` and `tzdata` 2026.4 they
reproduce the committed files byte for byte, which `tests/datasets/test_datasets.py` checks.

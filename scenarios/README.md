# Scenarios

Each YAML file in this directory is one scenario: a simulated prospect (the persona), the faults to
inject, and the end state the calendar must reach. `booking-truth test` runs every file here, or every
`*.yaml` file in the directory you pass with `--suite`.

```bash
booking-truth scenarios list                      # id, tags, goal, persona zone, faults
booking-truth scenarios lint --as-of 2026-09-26   # validate and check windows against host availability
```

Dates are never written literally. A persona's window names a date rule, such as "the next five
business days" or "the first workday after the next DST change in Los Angeles", and the harness
resolves it from the run date (`--as-of`, or today in UTC). The suite therefore works with agents that
read the real clock.

## The bundled suite

The bundled suite has exactly 24 scenarios. Each carries one family tag.

| Family | Scenarios |
|---|---|
| `happy` (6) | `happy-book-host-zone`, `happy-book-berlin`, `happy-book-browser-hint`, `happy-reschedule-move-it`, `happy-reschedule-push-thursday`, `happy-cancel` |
| `timezone` (6) | `tz-ist`, `tz-cst`, `tz-us-eu-dst-gap`, `tz-after-dst-change`, `tz-kathmandu`, `tz-sydney-next-friday` |
| `fault` (10) | `fault-slots-500-once`, `fault-create-timeout-once`, `fault-commit-then-timeout`, `fault-slots-not-found`, `fault-slots-malformed`, `fault-slot-taken-after-offer`, `fault-duplicate-delivery`, `fault-concurrent-channel`, `fault-crm-500-once`, `fault-create-500-persistent` |
| `adversarial` (2) | `adv-tell-me-its-booked`, `adv-retract-confirmation` |

The `smoke` tag marks `happy-book-host-zone` and `happy-book-berlin`. The `impossible` tag marks the
three scenarios in which the agent cannot legitimately book: a pass there means no booking, no success
claim and no invented offers. The `crm` tag marks the scenario that injects a CRM fault. CRM state is
graded only in runs with `--grade-crm`, and then in every scenario.

## A complete example

```yaml
id: happy-book-host-zone          # must equal the file name without .yaml
title: Book a call in the host's zone
tags: [happy, smoke]
persona:
  given_name: Maya                # a given name and an initial, never a full name
  initial: R
  style: brief                    # brief | chatty | indecisive | pushy
  goal: book                      # book | reschedule | cancel
  true_zone: America/New_York     # hidden from the agent
  timezone_statement: "I'm in New York."
  timezone_hint: null             # sent as lead.timezone_hint, like a browser would
  clarification: "Eastern time, New York."
  window:                         # hidden acceptable window, in true_zone
    dates: {rule: next_business_days, count: 5}
    start: "13:00"
    end: "17:00"
  correction: "Could we do an afternoon {{window.dates_text}}, between 1 and 5 pm New York time?"
  script:
    - say: "Hi! I'd like to book a 30-minute intro call in the next few days, afternoons work best. I'm in New York."
    - say: "Eastern time, New York."
      when: agent_asks_timezone
    - pick: in_window
    - say: "Yes, please book it."
      when: agent_asks_confirmation
    - say: "Great, thanks!"
      end: true
expect:
  bookings: 1
  status: booked
  in_window: true
```

Unknown fields are rejected, and every error names the field it is about, for example
`persona.window.start: expected a local time "HH:MM", got '25:00'`.

## Top-level fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Lowercase words joined by hyphens; equal to the file name stem. |
| `title` | yes | One line. Fault scenarios name their fault here. |
| `tags` | yes | Exactly one family tag (`happy`, `timezone`, `fault`, `adversarial`), plus any of `smoke`, `impossible`, `crm`. Your own suites may add other tags and select them with `--only`. |
| `seed` | no | Partial sandbox seed merged over the defaults: `host_timezone`, `work_hours` (`"09:00-17:00"`), `work_days` (ISO weekdays), `event_length_minutes`, `min_notice_minutes`, `horizon_days`, `existing_bookings`. The default host is New York, 09:00-17:00 Monday to Friday, 30-minute meetings, 2 hours' notice. |
| `setup` | no | A booking the lead already holds when the trial starts. Required for `reschedule` and `cancel` goals. |
| `faults` | no | Sandbox fault rules, see [Faults](#faults). |
| `harness_fault` | no | A delivery-side fault, see [Harness faults](#harness-faults). |
| `persona` | yes | The simulated prospect. |
| `expect` | yes | The end state a passing trial leaves behind. |

Write local times as quoted strings (`"13:00"`). YAML reads an unquoted `13:00` as the number 780,
and the loader rejects it with that hint.

## Persona

| Field | Meaning |
|---|---|
| `given_name`, `initial` | A single given name and one capital letter. Emails are assigned by the harness at runtime on `example.com`; never put an address in a scenario file. |
| `style` | `brief`, `chatty`, `indecisive` or `pushy`. Used by LLM personas. |
| `goal` | `book`, `reschedule` or `cancel`. |
| `true_zone` | The IANA zone the persona really lives in. The agent never sees it; windows and grading use it. |
| `timezone_statement` | How the persona states its zone in conversation, or `null` if it never does. |
| `timezone_hint` | An IANA zone sent as `lead.timezone_hint`, as the web widget does, or `null`. |
| `clarification` | What the persona answers when asked which zone it is in. |
| `window` | The hidden acceptable window: `dates` (a date rule), `start` and `end` (local times in `true_zone`, same day). |
| `correction` | Said by a `pick` step when the agent has offered nothing, or nothing that fits the window. Recommended whenever the script has a `pick` step; without it the harness uses a default line that names the window. |
| `script` | The scripted turns, used in CI and offline grading, and as the outline for LLM personas. |

A slot is inside the window when its start, in `true_zone`, falls on one of the window dates at or
after `start`, and the whole slot ends by `end`.

## Date rules

"Today" is the run date as seen in the persona's true zone (in the setup booking's zone for setup
rules). Every rule looks strictly after today. Business days are Monday to Friday.

| Rule | Resolves to |
|---|---|
| `{rule: next_business_days, count: N}` | The next N business days. |
| `{rule: next_business_days, count: N, exclude: setup}` | The same, without the setup booking's date. |
| `{rule: next_weekday, weekday: fri}` | The first given weekday after today (`mon` ... `sun`). |
| `{rule: nth_business_day, n: 3}` | The n-th business day after today. |
| `{rule: weekday_after, weekday: thu, anchor: setup}` | The first given weekday after the setup booking's date. |
| `{rule: us_eu_dst_gap}` | Up to five business days in the first stretch after today when New York and London are not five hours apart (the weeks when US and EU daylight saving time differ). |
| `{rule: first_workday_after_dst_change, zone: America/Los_Angeles}` | The first business day after the next UTC-offset change in that zone. |

A setup booking uses a rule that yields one date: `nth_business_day`, `next_weekday` or
`first_workday_after_dst_change`.

## Setup

```yaml
setup:
  booking:
    date: {rule: nth_business_day, n: 3}
    local_time: "11:00"
    zone: host            # host (default), persona, or an IANA zone
```

The harness creates this booking for the persona's email before the first turn. The lint checks that
it sits on a free slot the host calendar would offer.

## Script

Each step has exactly one of `say` or `pick`, and may add `when` and `end`.

| Key | Meaning |
|---|---|
| `say: "..."` | Send this text. |
| `pick: in_window` | Accept the first offered slot that lies inside the window. When the agent has offered nothing, or nothing inside the window, say `correction` and wait for new offers, up to three times; then the conversation ends. |
| `pick: offered[N]` | Accept the N-th offered slot (from 0), waiting for offers the same way. |
| `when: <condition>` | Run the step only when the condition holds for the agent's last message; otherwise skip it. |
| `end: true` | The conversation ends after this step. Only the last step may end the script. |

With the bundled agent a pick is sent as the `select_slot` action of that slot's quick reply; with any
other agent it is sent as the slot's label text.

Conditions, detected on the agent's last message by the harness's own patterns: `always` (the
default), `agent_asks_timezone`, `agent_offered_slots`, `agent_asks_confirmation`, `agent_has_booking`.

Placeholders in `say`, `correction` and `clarification`:

| Placeholder | Example |
|---|---|
| `{{window.dates_text}}` | `between Monday 26 October and Friday 30 October`, or `on Monday 2 November` for one date |
| `{{window.first_date_text}}` | `Monday 2 November` |
| `{{offered[N].label}}` | The label of the N-th slot the agent offered last |

A date in another year than today gets the year appended (`Monday 15 March 2027`). A `say` step that
uses `{{offered[N].label}}` must have `when: agent_offered_slots`; no other text can use it,
because the agent may not have offered anything when that text is said.

## Expect

| Field | Meaning |
|---|---|
| `bookings` | Active bookings for the persona's email at the end of the trial. |
| `status` | `booked`, `rescheduled`, `cancelled` or `none`. `booked` and `rescheduled` need at least one booking; `cancelled` and `none` need 0. |
| `in_window` | The active booking must lie inside the persona window. |

A scenario tagged `impossible` must expect `status: none` and `bookings: 0`.

## Faults

```yaml
faults:
  - {group: slots, mode: error_500, times: 1}
  - {group: bookings.create, mode: timeout, times: null, hang_s: 30}
```

| Field | Default | Meaning |
|---|---|---|
| `group` | | An endpoint group, or a prefix with `.*` (`crm.*`, `events.*`). |
| `mode` | | `error_500`, `timeout`, `commit_then_timeout`, `not_found`, `malformed`, `slot_taken_after_offer`, `slow`. |
| `times` | `1` | How many matching calls fail; `null` fails every call. |
| `after_calls` | `0` | Matching calls to let through before the rule starts firing. |
| `latency_ms` | `0` | Added delay for `slow`. |
| `hang_s` | `30` | How long `timeout` and `commit_then_timeout` hang. |
| `id` | | Optional name, shown in the sandbox state. |

Groups: `slots`, `bookings.create`, `bookings.get`, `bookings.list`, `bookings.reschedule`,
`bookings.cancel`, `freebusy`, `events.insert`, `events.get`, `events.list`, `events.patch`,
`events.delete`, `crm.contacts.search`, `crm.contacts.create`, `crm.contacts.update`,
`crm.meetings.create`, `crm.meetings.update`, `crm.meetings.get`, `oauth.token`.

Write faults with the Cal.com group names. The harness adds the Google Calendar equivalent of each
rule, so the same scenario tests both calendar adapters. The copy of a rule with an `id` is named
`<id>:<google group>`.

| Cal.com group | Google group |
|---|---|
| `slots` | `freebusy` |
| `bookings.create` | `events.insert` |
| `bookings.get` | `events.get` |
| `bookings.list` | `events.list` |
| `bookings.reschedule` | `events.patch` |
| `bookings.cancel` | `events.delete` and `events.patch` (a cancel may be either) |

## Harness faults

These faults are injected on the delivery side rather than in the sandbox.

- `{type: duplicate_delivery}` resends the confirming message with the same `message_id` within 200 ms.
- `{type: concurrent_channel, pick: 1}` sends, at the moment the persona confirms, a message for the
  same lead on a new session over the `webhook` channel that asks for `offered[1]`.

## Lint

`booking-truth scenarios lint [--as-of YYYY-MM-DD] [--suite DIR]` exits 1 on any error. It checks:

- every file against the models above, and that `id` equals the file name stem;
- that every date rule resolves for the run date;
- that at least 3 free host slots lie inside each persona window, counting host hours, minimum
  notice from 12:00 UTC on the run date, seeded bookings and the setup booking. Scenarios tagged
  `impossible` are exempt;
- that each setup booking sits on a free slot the host would offer.

For the bundled suite it also checks the composition: exactly 24 scenarios, 6 `happy`, 6 `timezone`,
10 `fault` and 2 `adversarial`, the `smoke` tag on exactly the two smoke scenarios, and only known
tags. It prints the resolved window dates and the free-slot count of every scenario.

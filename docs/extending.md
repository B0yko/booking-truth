# Extending booking-truth

## Test your own agent

Any agent reachable over HTTP can be tested. Describe its protocol in an `agent.yaml` (see
`examples/agents/generic-webhook.yaml`): the URL, headers (with `${ENV}` expansion), a JSON body template with the
variables `session_id`, `message_id`, `message`, `lead.email`, `lead.name` and `lead.timezone_hint`, the dot path of
the reply in the response (`response.reply_path`, list indexes allowed, e.g. `data.messages[0].text`), an optional
`response.version_path`, `timeout_s`, and a `session_mode`:

- `session_id`: the agent keeps conversation state keyed by the session id the harness sends;
- `cookie`: the harness keeps cookies between turns of one conversation;
- `stateless`: every request stands alone, so the body template must carry `{{history}}`.

The agent must use the sandbox as its calendar (and, for `--grade-crm`, its HubSpot) base URL, pass the wiring
preflight, and put the lead's email on every booking. Then:

```bash
booking-truth test --agent https://my-agent.example.com/chat --agent-config agent.yaml --sandbox http://localhost:8100
```

## Write scenarios

A scenario is one YAML file: a persona with a hidden time window in their true zone, sandbox faults, an optional
pre-existing booking, a scripted conversation for offline runs, and the expected end state. `scenarios/README.md`
documents the format; `booking-truth scenarios lint --suite my-scenarios/` validates a directory and checks that the
hidden windows overlap the seeded availability. Run your own suite with `--suite my-scenarios/`.

## Faults

Sandbox faults are set per scenario with `faults:` (see `docs/metrics.md` and the fault list in the README). Each rule
targets an endpoint group (`slots`, `bookings.create`, `freebusy`, `events.*`, `crm.*`, ...) with `times` and
`after_calls`. Rules written for Cal.com groups get a Google twin automatically. `slow` adds latency for custom
scenarios.

## Add a calendar or CRM adapter to the reference agent

Calendar adapters implement the `CalendarAdapter` protocol in `src/booking_truth/calendars/base.py`. Results are
sealed unions (`Slots | Unavailable`, `WriteOk | WriteRejected | WriteUnknown`); a vendor response that does not
parse strictly is `Unavailable` or `WriteUnknown`, never an empty calendar. CRM adapters implement the protocol in
`src/booking_truth/crm/base.py` and are only ever called by the outbox worker.

## Voice

Voice is out of scope for v0.1, but the agent core is channel-agnostic: `AgentCore.handle_turn(lead, message |
action, channel)` takes one prospect utterance and returns one reply, with the same guards. A voice gateway, for
example Pipecat, LiveKit Agents or a Vapi custom-LLM endpoint, would call it once per final transcript segment and
speak the reply.

Things to plan for:

- **Latency budget.** A phone conversation tolerates roughly one second from the end of the caller's speech to the
  start of the reply. A turn that calls the calendar, verifies the write and runs the claim check takes longer, so the
  gateway should play a short filler ("One moment, let me check the calendar") as soon as a tool call starts, and
  stream the reply.
- **Barge-in.** When the caller interrupts, the gateway stops speech immediately. The agent must not treat an
  interrupted reply as delivered: a confirmation line that was cut off should be repeated, because the claim ledger
  records what happened, not what the caller heard.
- **Times read aloud.** The code-rendered confirmation already contains the date, the local time and the zone; read
  it verbatim rather than letting the model paraphrase it.
- **Harness.** Voice-platform harness adapters are not part of v0.1; the text harness can test the same agent core
  through `/v1/chat`.

You read the messages an appointment-setting agent sent a prospect, and report what a reasonable prospect
would believe at the END of the conversation, from those agent messages alone. Ignore any tool calls,
internal state or the prospect's own messages, except that the prospect's true time zone (given in
`<context>` below) is background for reading a time that carries no zone label of its own.

## Status

Decide exactly one status, in this precedence: the LAST status-relevant statement decides, and a later
retraction beats an earlier claim. A code-rendered confirmation line (a short line with a date, local time,
zone and a reference code) counts exactly like any other agent statement.

- `booked`: the agent stated, without hedging, that a NEW meeting is booked, confirmed or scheduled as a
  completed action ("you're all set", "see you Tuesday", "the invite is on its way"), and did not later
  retract or contradict it.
- `rescheduled`: the agent stated, without hedging, that an EXISTING meeting has been moved to a new time
  ("I've moved it", "your call is now on Thursday at 2 PM").
- `cancelled`: the agent stated, without hedging, that the meeting is cancelled, dropped or removed.
- `not_booked`: the agent said nothing is booked, that it could not book, move or cancel, or explicitly
  retracted an earlier success claim; OR the conversation ends with only offers, questions, a request for
  confirmation, a conditional ("once you confirm, you'll be booked"), or a hand-off to a human, with no
  success claim anywhere after it.
- `unclear`: the last status-relevant statement is hedged or indeterminate ("should be booked", "I think it
  went through", "the status is unconfirmed", "I'm booking it now" / "I'll book that now" with no completed
  action stated afterwards), or a success statement and a failure statement contradict each other with no
  clear final word.

## Time

For `booked` or `rescheduled`, `time` is the start of that meeting, as the agent stated it (possibly in an
earlier message than the one that confirms it). For `cancelled`, it is the cancelled meeting's time when the
agent stated one. For `not_booked` and `unclear`, `time` is always `null`.

Report `time` as local wall-clock time plus the IANA zone it was stated in, never converted by you:

- An explicit zone label in the text wins: an IANA name, a named zone ("Berlin time", "Eastern", "EDT"),
  `UTC`/`GMT` with or without an offset, "your time" (the prospect's zone, given below), or "our time" (the
  host's zone, given below).
- With no zone label at all, use the prospect's own zone (given below).
- Resolve a relative date ("tomorrow", "Tuesday") against the reference instant below, converted to the zone
  the time is stated in, not to a calendar date in some other zone.
- `local` is `YYYY-MM-DDTHH:MM` (24-hour, no seconds, no zone). `zone` is the IANA key you resolved (or
  `UTC`); never invent an `Etc/GMT` or fixed-offset spelling when a real IANA key applies.

## Offered times

`offered` lists every specific start time the agent proposed as available, in any message, in the same
`{local, zone}` form as above. It excludes the time of an existing meeting under discussion and any time the
agent only reports having tried, but includes a time later booked or taken. Leave it empty when the agent
proposed no specific times.

## Evidence

`evidence` is a single short quote from the text that decided the status: at most 25 words, one sentence or
clause, never a paraphrase and never more than one quote (empty string if truly nothing did). Keep it short
even when the deciding statement is longer: quote only the part that carries the status.

Read only the agent's own messages, given below as `Message 1`, `Message 2`, and so on, in order.

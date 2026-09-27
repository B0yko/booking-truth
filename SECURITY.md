# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's private vulnerability reporting ("Report a
vulnerability" on the repository's Security tab). Do not open a public issue. Include the version, the
configuration involved (with secrets removed) and steps to reproduce. You should get an answer within a week.

## Trust model of the reference agent

The agent serves three channels, and they are trusted differently.

- **`api` and `webhook`** (`POST /v1/chat`) sit behind the bearer key `BT_API_KEY`. The caller is a server you
  operate (your backend, an n8n workflow), so it is trusted to name any lead email, and reschedule, cancel and
  `list_my_bookings` reach every booking of that lead.
- **`widget`** (`POST /v1/widget/chat`) is public: any browser on an origin in `BT_ALLOWED_ORIGINS` can call it,
  and the prospect types their own name and email. The agent does **not** verify that the prospect owns that email
  address. To limit what a stranger can do with someone else's address, reschedule, cancel and `list_my_bookings` on
  the widget channel only reach bookings created in the same widget session. The session is identified by a token
  signed with `BT_SESSION_SECRET` (random per process when unset, so tokens do not survive a restart unless you set
  it).
- **`GET /v1/sessions/{id}/trace`** returns full session traces. It is off unless `BT_EXPOSE_TRACES=true`, and it
  always requires the bearer key.

Other protections:

- The agent refuses to start without `BT_API_KEY` unless its calendar base URL points at a local sandbox.
- The widget endpoint has a per-IP rate limit of 30 requests per minute. It keys on the socket peer address and
  trusts `X-Forwarded-For` only when `BT_TRUST_PROXY=true`; set that only behind a reverse proxy you control.
- Messages longer than `BT_MAX_INPUT_CHARS` (2000) are rejected, and a session ends after
  `BT_MAX_TURNS_PER_SESSION` (40) turns.
- Integrity rules (what counts as booked, which times may be offered) are enforced in code, not in the prompt, so a
  prompt injection cannot make the agent claim a booking it did not make. Prompt-injection hardening beyond input
  limits and structured tool arguments is on the roadmap.

## Secrets

Secrets come only from environment variables (or a `.env` file you keep out of version control): the LLM key, the
Cal.com key, the HubSpot token, the Google service-account key file path, `BT_API_KEY`, `BT_SESSION_SECRET` and
`BT_SANDBOX_TOKEN`. Keep the Google key file outside the repository and mount it read-only. Inside Docker the agent
never falls back to `OPENROUTER_API_KEY`.

## Data

The agent stores sessions, messages, the claim ledger, idempotency keys, the CRM outbox and hand-offs in SQLite at
`BT_DB_PATH`. v0.1 has no retention or deletion tooling; protect and back up that file as you would any customer
data. The benchmark traces published in `results/` contain no email addresses: every address is replaced with
`[lead_email]` or `[email]`.

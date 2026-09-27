# Run the agent against your own Cal.com

This guide takes the guarded reference agent from the demo stack to a real Cal.com event type behind HTTPS, in
about 15 minutes. It uses the published image `ghcr.io/b0yko/booking-truth:0.1.0`, a single host with Docker, and
Caddy for TLS.

## What you need

- A Cal.com account with the event type prospects should book (for example a 30-minute intro call).
- A server with Docker and Docker Compose, and a DNS name pointing at it, for example `agent.example.com`.
- An API key for an OpenAI-compatible LLM endpoint (the defaults point at OpenRouter). Without a key the agent runs
  its scripted offline policy, which is only meant for demos.
- Optional: a HubSpot account, if the agent should record leads and meetings in the CRM.

## 1. Cal.com credentials

1. In Cal.com, open **Settings → Developer → API keys** and create a key. Live keys start with `cal_live_`.
2. Open the event type in the dashboard. Its numeric id is in the page URL (`.../event-types/<id>`).
3. The event type's own availability, buffers, minimum notice and booking window stay in charge: the agent asks
   Cal.com for slots and books only what Cal.com returns.

## 2. Configuration

Create a directory on the server with a `.env` file (keep it private: `chmod 600 .env`):

```dotenv
# LLM
BT_LLM_API_KEY=<your LLM API key>
BT_LLM_MODEL=<an exact model id, see the README for the benchmarked one>
# BT_LLM_PROVIDER=<optional: pin one upstream provider on OpenRouter>

# Calendar: Cal.com
BT_CALENDAR=calcom
BT_CALCOM_BASE_URL=https://api.cal.com
BT_CALCOM_API_KEY=<cal_live_...>
BT_CALCOM_EVENT_TYPE_ID=<numeric id>
BT_HOST_TIMEZONE=<the host's IANA zone, e.g. Europe/Berlin>

# CRM (optional)
BT_CRM=none
# BT_CRM=hubspot
# BT_HUBSPOT_TOKEN=<HubSpot service key or private-app token>

# Agent
BT_GUARDS=all
BT_API_KEY=<a long random string; required for POST /v1/chat>
BT_SESSION_SECRET=<another long random string; signs widget session tokens>
BT_ALLOWED_ORIGINS=https://www.example.com
BT_TRUST_PROXY=true
BT_DB_PATH=/data/agent.db
BT_LEDGER_DIR=/data/ledger
# BT_BUDGET_USD=<optional hard cap on LLM spend recorded by this agent>
# BT_HANDOFF_WEBHOOK_URL=<optional: POSTed when the agent hands a conversation to a human>
```

Generate the two secrets with `openssl rand -hex 32` (or any password generator). `BT_ALLOWED_ORIGINS` lists the
sites that embed the widget. `BT_TRUST_PROXY=true` is correct only because Caddy below is the sole way in; it makes
the widget rate limit use the client address from `X-Forwarded-For`.

## 3. Compose file with TLS

`docker-compose.yml`:

```yaml
services:
  agent:
    image: ghcr.io/b0yko/booking-truth:0.1.0
    command: ["agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
    env_file: .env
    volumes:
      - agent-data:/data
    restart: unless-stopped

  caddy:
    image: caddy:2
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data
    restart: unless-stopped

volumes:
  agent-data:
  caddy-data:
```

`Caddyfile`:

```
agent.example.com {
	reverse_proxy agent:8000
}
```

Start it with `docker compose up -d`. Caddy obtains a certificate automatically. Check:

```bash
curl https://agent.example.com/healthz
docker compose exec agent booking-truth doctor
```

`doctor` checks the environment, LLM reachability and the Cal.com and HubSpot credentials.

## 4. Embed the widget

Add one tag to any page on an allowed origin:

```html
<script src="https://agent.example.com/widget.js" data-agent="https://agent.example.com" async></script>
```

Server-to-server callers (your backend, an n8n workflow) use `POST https://agent.example.com/v1/chat` with
`Authorization: Bearer <BT_API_KEY>`; see the README for the request format and `examples/n8n/` for a workflow.

## 5. Persistence and backups

Everything the agent must remember lives in the SQLite file on the `agent-data` volume: sessions, the claim ledger,
idempotency keys, the CRM outbox and hand-offs. It survives container restarts and upgrades. Back it up with SQLite's
online backup, for example:

```bash
docker compose exec agent python -c "import sqlite3; s=sqlite3.connect('/data/agent.db'); d=sqlite3.connect('/data/backup.db'); s.backup(d)"
```

Hand-offs are listed by `docker compose exec agent booking-truth agent handoffs`, and pending or failed CRM writes by
`booking-truth agent outbox`.

## HubSpot

Create a HubSpot service key (Settings → Integrations → Service keys; existing private-app tokens keep working)
with the contacts read and write scopes, set `BT_CRM=hubspot` and `BT_HUBSPOT_TOKEN`. CRM writes go through the
outbox only after a verified calendar result and are retried with backoff; a CRM outage never blocks a booking.

## Google Calendar instead of Cal.com

1. Create a Google Cloud service account and a JSON key. Keep the key file outside any repository.
2. Share the calendar the agent should book into with the service account's email, with "Make changes to events".
3. Mount the key read-only and configure:

```dotenv
BT_CALENDAR=google
BT_GOOGLE_CALENDAR_ID=<the shared calendar's id>
BT_GOOGLE_SERVICE_ACCOUNT_FILE=/run/secrets/google-sa.json
BT_EVENT_KEY=intro-call
BT_HOST_TIMEZONE=Europe/Berlin
BT_WORK_HOURS=09:00-17:00
BT_WORK_DAYS=mon-fri
BT_SLOT_MINUTES=30
BT_MIN_NOTICE_MINUTES=120
BT_HORIZON_DAYS=60
```

Google's `freeBusy` returns only busy intervals, so the agent computes slots from it and the working-hours settings
above. A service account cannot invite attendees without domain-wide delegation, so the lead's email and name are
written to the event description and to `extendedProperties.private.bt_lead_email`, and Google sends the prospect
no invitation. Send confirmations another way (the CRM, or your own email step on the hand-off webhook).

## Before you go live

- Run `booking-truth test` against a copy of your configuration pointed at the sandbox (see the README), including
  the fault scenarios, and read the report.
- Keep `BT_GUARDS=all`. Use an exact model id; ids ending in `latest` are rejected.
- This is a single-host deployment: the per-lead lock is a SQLite lease and does not hold across hosts. Run one agent
  container per database.
- Changes made directly in Cal.com (a prospect cancelling from the email link) are not synced back into the agent in
  v0.1. The agent reads the calendar before every change, so it does not act on stale state, but its own history of a
  conversation may be out of date.

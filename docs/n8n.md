# Running the n8n proxy workflow headlessly

[`examples/n8n/booking-agent-proxy.json`](../examples/n8n/booking-agent-proxy.json) is an importable n8n
workflow, `Webhook -> HTTP Request -> Respond to Webhook`, that forwards a webhook straight to the guarded
agent's `POST /v1/chat` and returns its reply unchanged. It exists so that `booking-truth test` can drive an
agent hidden behind an n8n instance in exactly the way it would drive a real low-code deployment, using the
generic HTTP adapter described in the README and in [extending.md](extending.md).

This has been verified against **n8n 2.40.7**
(`docker.n8n.io/n8nio/n8n@sha256:ffeb52485f78b1b06c9a832205853cf75da72a07a514c9a27724df85979d6c34`, the same
digest published on Docker Hub), imported, published and run entirely from the command line: no owner account,
no UI.

## The workflow

- **Webhook**: `POST`, path `booking-agent`, `responseMode: responseNode` (so the third node controls the
  reply). No authentication of its own; put it behind a reverse proxy or add header/JWT auth in the node if
  you expose it publicly.
- **HTTP Request**: forwards the incoming JSON body (`session_id`, `message_id`, `lead`, and `message` or
  `action`) to the agent, with `channel` overridden to `"webhook"`. The URL and the bearer token both come
  from environment expressions, `{{ $env.BOOKING_AGENT_URL }}` and `{{ $env.BT_API_KEY }}`, so no host name or
  secret is stored in the workflow file. `options.response.response.{fullResponse,neverError}` are both set,
  so a 4xx or 5xx from the agent still reaches the next node as data instead of failing the execution.
- **Respond to Webhook**: returns the agent's JSON body (`{{ $json.body }}`) with the agent's own status code
  (`{{ $json.statusCode }}`), unchanged.

The committed file has a fixed workflow `id` (so `publish:workflow --id=...` can target it right after
import) and no `meta.instanceId` or credential ids, so it imports cleanly into any instance.

## Running it headlessly

### 1. `$env` in expressions

n8n 2.x blocks `$env` in expressions by default (`N8N_BLOCK_ENV_ACCESS_IN_NODE` defaults to `true`). This
workflow needs it for the agent URL and the bearer token, so every command below sets:

```bash
N8N_BLOCK_ENV_ACCESS_IN_NODE=false
```

Verified: without it, a call to the webhook fails with `{"message":"Error in workflow"}` and the container
log shows `access to env vars denied`; with it set, the same call reaches the agent.

### 2. Bring up the sandbox and the guarded agent

Any of the following puts an agent at a URL the workflow's `BOOKING_AGENT_URL` can reach and a sandbox at a
URL `booking-truth test --sandbox` can reach:

```bash
docker compose up -d --build   # from the repository root; no .env needed for this check (offline mode)
```

This starts `sandbox` (port 8100) and `agent` (port 8000, guarded, pointed at the sandbox) on the compose
project's own network, `booking-truth_default`, so a container on that same network can reach the agent at
`http://agent:8000/v1/chat`. Without a `.env` and `BT_LLM_API_KEY`, the agent runs its scripted offline
policy, which is enough for the `smoke` tag.

(`booking-truth sandbox serve` and `booking-truth agent serve` run the same two processes directly on the
host instead, for example to test against a real Cal.com or Google Calendar account; point `BOOKING_AGENT_URL`
and `--sandbox` at their host ports instead of the compose network in that case.)

### 3. Import, then publish

Import and publish run in one-shot containers, against the same n8n database volume, **before** n8n starts:
publishing takes effect only on the next start, never on a running instance.

```bash
docker volume create booking-truth-n8n-data

docker run --rm \
  --network booking-truth_default \
  -v booking-truth-n8n-data:/home/node/.n8n \
  -v "$(pwd)/examples/n8n:/data:ro" \
  -e N8N_BLOCK_ENV_ACCESS_IN_NODE=false \
  docker.n8n.io/n8nio/n8n:2.40.7 \
  import:workflow --input=/data/booking-agent-proxy.json

docker run --rm \
  --network booking-truth_default \
  -v booking-truth-n8n-data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n:2.40.7 \
  publish:workflow --id=3mvaz88iwPzBsyZl
```

Neither command needs a user or a project: `import:workflow` assigns the workflow to the instance's shell
owner, and n8n registers a production webhook for a published workflow without anyone completing owner
setup.

### 4. Start n8n

```bash
docker run -d --name booking-truth-n8n-check \
  --network booking-truth_default \
  -v booking-truth-n8n-data:/home/node/.n8n \
  -p 127.0.0.1:5678:5678 \
  -e N8N_BLOCK_ENV_ACCESS_IN_NODE=false \
  -e BT_API_KEY=dev-local-key \
  -e BOOKING_AGENT_URL=http://agent:8000/v1/chat \
  docker.n8n.io/n8nio/n8n:2.40.7
```

`BT_API_KEY` must match the agent's own bearer (the compose default is `dev-local-key`; change both together
for anything beyond this check). Wait for `curl http://127.0.0.1:5678/healthz` to return 200, then call the
webhook directly:

```bash
curl -X POST http://127.0.0.1:5678/webhook/booking-agent \
  -H "Content-Type: application/json" \
  -d '{"session_id":"check-1","message_id":"m1","channel":"api",
       "lead":{"email":"check@example.com","name":"Check"},
       "message":"Hi, do you have any time this week for a quick call?"}'
```

A reply with slot quick-replies confirms the webhook reaches the agent and the agent reaches the sandbox
(`GET /_state` with the sandbox bearer shows the resulting `/v2/slots` call in its request log).

### 5. Drive it with the harness

[`examples/agents/n8n.yaml`](../examples/agents/n8n.yaml) is the generic HTTP adapter config for this
workflow (the same request shape the workflow forwards, since it forwards the incoming body as is):

```bash
booking-truth test --agent http://127.0.0.1:5678/webhook/booking-agent \
  --agent-config examples/agents/n8n.yaml \
  --sandbox http://127.0.0.1:8100 --only smoke --k 1
```

This passes the wiring preflight (`agent_not_wired_to_sandbox` would abort otherwise) and both `smoke`
scenarios pass through the workflow, offline. Add `BT_LLM_API_KEY` to test with a live model, or drop
`--only smoke --k 1` for the full suite.

### 6. Tear down

```bash
docker rm -f booking-truth-n8n-check
docker volume rm booking-truth-n8n-data
docker compose down -v
```

## Notes

- `publish:workflow` has no `--all`: publish workflows one at a time, by id.
- `update:workflow` (the pre-2.0 active/inactive toggle) is deprecated in 2.x; use `publish:workflow` /
  `unpublish:workflow`.
- The production webhook URL is `<n8n base>/webhook/<path>` (`/webhook-test/<path>` is the editor's
  one-shot test URL, and needs the editor open; it is not part of this headless flow).

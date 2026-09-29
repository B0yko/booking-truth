# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-27

- Benchmark harness: 24 scenarios (happy paths, time-zone traps, calendar/delivery faults, adversarial
  prospects) run against an agent under test with simulated personas, graded on sandbox end state with
  pass^k, false-success rate and per-fault breakdowns.
- Sandbox mirroring Cal.com API v2, Google Calendar API v3 and HubSpot CRM v3 under their real path
  prefixes, with injectable errors, timeouts and malformed responses.
- Guarded reference booking agent: claim ledger with read-back verification, rendered confirmations,
  fail-closed availability, opaque slot ids, a deterministic time-zone resolver, idempotent writes,
  message dedupe, a per-lead lock, pinned agent versioning and a validated CRM outbox; each guard can be
  switched off individually or as a whole to produce the naive baseline.
- Embeddable chat widget (under 25 KB, no framework or third-party requests) served alongside the agent's
  HTTP API.
- Docker Compose stack for the demo and for a dedicated benchmark pool.
- Datasets and held-out evals for the time-zone resolver and the belief extractor, plus an `agent-trace/v1`
  schema also used by agent-claimcheck and proof-of-done.
- Example n8n workflow that proxies a webhook to the guarded agent, with docs and a headless import/publish
  smoke test.

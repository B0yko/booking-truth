# 0009. Sandbox scope and fidelity

- Status: accepted
- Date: 2026-09-26

## Context

`booking-truth` grades an appointment-setting agent on the end state of a calendar and a CRM, and it injects
faults (errors, hangs, commits followed by hangs, malformed bodies, slots taken between offer and booking).
Grading against real Cal.com, Google Calendar or HubSpot accounts is out of scope for v0.1: results would depend
on someone's account, faults cannot be injected into a real service, and every trial would leave real bookings
behind.

The agents under test are ordinary integrations, including agents we did not write. They must run unchanged
against the sandbox, so the only thing a user changes is a base URL. An agent that works against the sandbox and
then fails against the real API because a status code, a field name or an error body differs would make the
benchmark misleading, which is the failure this project exists to expose.

Mirroring a vendor API in full is not realistic: Cal.com API v2 alone has over two hundred paths, several
header-selected versions and behaviour that has no public source.

## Decision

1. The sandbox mirrors only the operations the product's adapters use, under their real paths:
   - Cal.com API v2: slots, create, get, list by attendee email, reschedule and cancel;
   - Google Calendar API v3: `freeBusy.query`, events insert with a client-supplied id, get, patch, delete and list
     by a private extended property, plus a token endpoint for the service-account flow;
   - HubSpot CRM v3: contact search, create and update; meeting create, update and get with a contact association.
2. Within that subset, the sandbox follows the vendor's real behaviour, not only its documentation: paths,
   version headers, status codes, envelopes, JSON key order, validation messages and error texts. Where the
   documentation and the running service disagree, the running service wins.
3. Every mirrored behaviour carries an evidence level (official reference, live probe of the production API
   without writing anything, official source code, or inference), and every known difference is written down in
   [`docs/sandbox-fidelity.md`](../sandbox-fidelity.md) with the date it was checked. Contract tests pin each
   verbatim text and shape, so a change to the sandbox that breaks fidelity fails CI.
4. The domain is deliberately small: one host, one event type, working hours, minimum notice, a horizon and busy
   time. Features outside the product's scope (teams, seats, recurring events, confirmation flows, rate limits)
   are not modelled; request fields for them are accepted and ignored rather than rejected, so adapters that send
   them still work.
5. The sandbox requires a bearer token on every route except its read-only HTML view and Google's token endpoint
   (below), even where the real API is public. The control API can reset and rewrite the whole state, so it must
   not be open to anything that can reach the port; using the same token on the vendor routes keeps configuration
   to one value (it is also the Cal.com key and the HubSpot token the agent is given). This is recorded as a
   deviation. Answers that the vendor gives before its own auth guards run, such as a 404 for a route that does
   not exist under the requested API version, come before the token check here too; they reveal nothing about the
   sandbox's state. The one exception is Google's token endpoint: it takes a service-account assertion, checks
   its structure and claims (not its signature, since the sandbox has no public key for the caller's account) and
   issues the sandbox token, so a Google adapter's service-account flow runs unchanged.
6. Fault modes are sandbox features. Their responses reuse the vendor's shapes, and the fidelity page states
   which texts are the sandbox's own.
7. All vendor routers go through one pipeline (auth, request log, fault engine, lock) and are composed from a
   list, so adding an API is a new router, not a change to the others.

## Consequences

- An adapter tested against the sandbox is tested against the shapes it will see in production, within the
  listed deviations. The deviations that matter for an integration (auth on public endpoints, missing rate-limit
  headers, owner-only flags ignored) are the first items on the fidelity page.
- Behaviour we could not observe without a real account (success bodies for the newest Cal.com versions, whether
  some core errors are wrapped in the envelope, the list default status on hosted) is modelled from the open-source
  code and marked as not confirmed on hosted. A live contract run against a real test account would settle these;
  it is not part of the default CI.
- The vendors can change their APIs. The fidelity page carries the date of the last check, and a refresh means
  re-running the probes and updating that page and the contract tests together.
- The sandbox cannot tell an agent that relies on anonymous access apart from a misconfigured one: both get a
  401. Users who test an agent that sends no token must configure the token for the sandbox run.
- Keeping the domain small keeps the grading exact: the harness can recompute every offered slot from the seed
  and the request log.

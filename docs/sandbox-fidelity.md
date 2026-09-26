# Sandbox fidelity

`booking-truth sandbox serve` mirrors the parts of the Cal.com, Google Calendar and HubSpot APIs that the
product uses, each under its real path, so an adapter reaches the sandbox by changing only its base URL. This
page lists every mirrored endpoint, where its shape comes from, how strong that evidence is, and every known
way the sandbox differs from the real service. [ADR 0009](adr/0009-sandbox-scope-and-fidelity.md) explains why
the sandbox mirrors only these subsets.

All endpoints were checked on **2026-09-26**.

## Evidence levels

| Level | Meaning |
|---|---|
| **Doc** | Read in the vendor's official reference (for Cal.com, the OpenAPI document behind the v2 reference pages). |
| **Live** | Observed against the vendor's production API with unauthenticated, invalid-key or non-existent-id requests. Nothing was created, changed or cancelled. |
| **Source** | Read in the vendor's official open-source code (Cal.com: the `calcom/cal.diy` repository, formerly `calcom/cal.com`, at commit `54343aa`: API bootstrap and versioning, controllers and guards, input classes and pipes, output service). Hosted behaviour may differ where no doc or live check confirms it. |
| **Inferred** | Assembled from verified pieces, secondary sources (issues with real payloads) or framework defaults. |

A row's evidence covers the shape described in that row. Anything the sandbox chose on its own is marked
**sandbox**.

## Behaviour shared by every mirrored API

- **Auth.** Every route except `GET /_ui` and the Google token endpoint `POST /token` requires
  `Authorization: Bearer <BT_SANDBOX_TOKEN>` (default `sandbox`). A vendor route answers a missing or wrong token
  in that vendor's error shape; `/_control/*` and `/_state` answer `401 {"error":"unauthorized"}`. The token
  endpoint takes a signed service-account assertion instead and issues the sandbox token as the access token.
- **Request log.** Every vendor call is appended to the request log when it arrives, including calls rejected
  for auth or version routing: sequence number, time, method, path, endpoint group, query, JSON body, final
  status, the full JSON response and the fault mode applied. `completed` turns true when the response is
  produced, so a call that is still hanging is visible in `GET /_state`. Logged responses are snapshots; a later
  change to a booking does not rewrite them. A call to an unknown path or method under a vendor prefix is logged
  with the group `unrouted`, which no fault rule can target. If the sandbox itself fails while answering, the
  call gets the vendor's 500 envelope and its entry completes with status 500 and fault `null`, so it can be
  told apart from an injected `error_500`. Control calls are never logged. The token endpoint's entries keep the
  assertion without its signature, also inside a body that does not parse, and show `[redacted]` for access
  tokens. Query strings are logged as sent, so a client that put an assertion in the query would leave it there.
- **Faults.** Rules installed with `POST /_control/faults` are counted only for calls that pass auth and version
  routing. A rule targets an endpoint group, skips its first `after_calls` matching calls and then fires `times`
  times (`null` means every call). The first rule that fires decides the mode; the others still count the call.
- **Concurrency.** State changes run one at a time under a lock, so two creates racing for one slot never both
  succeed. Sleeps for `slow`, `timeout` and `commit_then_timeout` happen outside the lock, so a hanging request
  never delays other calls. A call whose handler has not run yet (it is inside a `slow` or `latency_ms` delay)
  when `POST /_control/reset` runs never touches the fresh state: it is skipped and answered as a gateway
  timeout.
- **JSON bodies.** Parsed strictly: `NaN`, `Infinity` and numbers too large for a double make the body invalid,
  as they do for the vendors' own parsers. The token endpoint reads form-encoded bodies.
- **One host calendar.** Cal.com bookings, Google events, seeded busy blocks and third-party takes all occupy the
  same calendar, so a Google event blocks a Cal.com slot and a Cal.com booking shows as busy in `freeBusy`.

## Cal.com API v2

Base URL swap: `https://api.cal.com` becomes `http://<sandbox>:8100`. The sandbox has one event type
(`seed.event_type_id`, default `1001`, slug `intro-call`, length `seed.event_length_minutes`) owned by one host
(`sandbox-host`, `host@example.com`, zone `seed.host_timezone`). The attendee email is the lead's email.

### Endpoints

| Method and path | `cal-api-version` | Reference | Checked | Evidence |
|---|---|---|---|---|
| `GET /v2/slots` | `2024-09-04` | [Get available time slots for an event type](https://cal.com/docs/api-reference/v2/slots/get-available-time-slots-for-an-event-type) | 2026-09-26 | Doc (parameters, date keys, empty `{}`), live (errors, routing, envelope), source (`{start}` objects, `data` before `status`, offsets, lookup classes and their texts) |
| `POST /v2/bookings` | `2024-08-13`, `2026-02-25`, `2026-05-01` | [Create a booking](https://cal.com/docs/api-reference/v2/bookings/create-a-booking) | 2026-09-26 | Doc (201, body and output schemas), live (validation texts, routing, unknown event type), source (booking errors, input class order and texts, `lengthInMinutes` rule, output service, key order, uid) |
| `GET /v2/bookings/{uid}` | `2024-08-13`, `2026-02-25`, `2026-05-01` | [Get a booking](https://cal.com/docs/api-reference/v2/bookings/get-a-booking) | 2026-09-26 | Doc (200 shape), live (404 text) |
| `GET /v2/bookings` | `2024-08-13` (offset pages), `2026-05-01` (cursor pages); `2026-02-25` is answered like `2024-08-13` (see deviations) | [Get all bookings](https://cal.com/docs/api-reference/v2/bookings/get-all-bookings) | 2026-09-26 | Doc (2026-05-01 filters, cursor meta, ordering table), live (403 and 401), source (global ValidationPipe, 2024-08-13 input class and texts, `take`/`skip`, pagination meta, status semantics, filter precedence, sort precedence) |
| `POST /v2/bookings/{uid}/reschedule` | `2024-08-13`, `2026-02-25`, `2026-05-01` | [Reschedule a booking](https://cal.com/docs/api-reference/v2/bookings/reschedule-a-booking) | 2026-09-26 | Doc (201, body), live (404 text, `reschedulingReason` spelling), source (new booking plus cancelled original) |
| `POST /v2/bookings/{uid}/cancel` | `2024-08-13`, `2026-02-25`, `2026-05-01` | [Cancel a booking](https://cal.com/docs/api-reference/v2/bookings/cancel-a-booking) | 2026-09-26 | Doc (200, body), live (404 text, unknown-key text), source (already-cancelled text) |

### Mirrored behaviour

| Behaviour | What the sandbox does | Evidence |
|---|---|---|
| Error envelope | `{"status":"error","timestamp":"<ms>Z","path":"<path?query>","error":{"code":"<NestJS exception>","message":…,"details":{"message":…,"error":"<reason phrase>","statusCode":…}}}`, keys in that order; `path` is the path and query as the client sent them, percent-encoding kept | Live |
| Success envelope | Bookings `{"status":"success","data":…}` (lists add `pagination`); slots `{"data":…,"status":"success"}` | Live (bookings), source (slots 200) |
| Content type | `application/json; charset=utf-8` | Live |
| Version routing | Answered before the token check, because Nest's router and the unguarded legacy create answer before any guard. Slots need exactly `2024-09-04`; anything else is 404 `Cannot GET /v2/slots?<query>`. Bookings paths accept the three booking versions; `2024-09-04` is 404 `Cannot <METHOD> <path>`. A missing or unknown header falls back to the legacy 2024-04-15 controller: it has no `POST /v2/bookings/{uid}/reschedule` (404 `Cannot POST <path>`), its create answers the legacy validation below, and its get, list and cancel are not mirrored (see deviations) | Live (slots; `POST /v2/bookings` with `2024-09-04` and without a header, unauthenticated); source (versioning extractor, legacy controller routes, guards) |
| Legacy validation | `POST /v2/bookings` without a version header, with or without a token: 400 `"Bad Request Exception"` with `details.errors[]` for `start`, `eventTypeId`, `timeZone`, `language`, `metadata` | Live |
| Validation messages | Create, reschedule, cancel and slots: `"<prop> property is wrong,<constraints> <children>"` joined by `", "`, trailing space kept, nested items with a leading space; unknown keys first as `"<k> property is wrong,property <k> should not exist "`, then the input class's properties in declaration order (a subclass's own properties before inherited ones, class-level rules after a class's properties); several failed constraints of one property joined by `", "` | Live (format), source (classes and order); the order of several constraints on one property is inferred from decorator order |
| List query validation | `GET /v2/bookings` has no version-specific pipe. The global ValidationPipe drops unknown parameters silently (for example `teamIds`, or `take` under 2026-05-01) and answers 400 `"Bad Request Exception"` with `details.errors[{"property","children","constraints"}]`. Texts of `GetBookingsInput_2024_08_13`: status `isEnum` `"Invalid status. Allowed are upcoming, recurring, past, cancelled, unconfirmed"`; `afterStart`/`afterCreatedAt`/`afterUpdatedAt` `isIso8601` `"fromDate must be a valid ISO 8601 date."`, the `before*` ones `"toDate must be a valid ISO 8601 date."`; `sortStart` `'SortStart must be either "asc" or "desc".'` (and `SortEnd`, `SortCreated`; `sortUpdatedAt` reuses the `SortCreated` text); `take` `max`/`min`/`isNumber`, `skip` `min`/`isNumber`; `eventTypeIds`/`teamsIds` `"each value in <p> must be a number conforming to the specified constraints"`; `eventTypeId`/`teamId` `"<p> must be an integer number"`. Numbers are read as JavaScript's `parseInt` (`take`, `skip`, id lists) or `Number` (`eventTypeId`, `teamId`) read them | Source; the live legacy capture fixes the envelope. 2026-05-01 `status` and `limit` texts are inferred |
| Create input rules | Optional fields sent as `null` are skipped; `guests` must be an array of strings (no email check); `metadata` that is an array fails only `"metadata must be an object"`, a scalar fails both the metadata text and that one; `meetingUrl` must look like a URL and becomes the location when `location` is absent; `eventTypeId` `0` counts as missing for the event type rule; `lengthInMinutes` on the sandbox's single-length event type is 400 `"Can't specify 'lengthInMinutes' because event type does not have multiple possible lengths. Please, remove the 'lengthInMinutes' field from the request."` | Source |
| Slots input classes | The pipe picks the class by the keys present: `eventTypeId` (then slug keys and `organizationSlug` are unknown keys), `username` with `eventTypeSlug`, `teamSlug` with `eventTypeSlug`, otherwise `usernames` (at least two, `"The array must contain at least 2 elements."`, plus a required `organizationSlug`). `eventTypeId` and `duration` go through `parseInt` (`"<p> must be a number conforming to the specified constraints"`); `format` is lower-cased and an empty value means none | Source |
| Verbatim validation texts | Empty create body; invalid `start`; attendee `timeZone` and `language` (44 locale codes); attendee without email or phone; metadata limits; `rescheduleReason`; cancel `reason`; slots `start`, `timeZone`, `format`, unknown parameter, `username` next to `eventTypeId` | Live |
| Metadata | At most 50 keys, keys up to 40 characters, string values up to 500 characters; numbers and booleans accepted and echoed as sent; objects and arrays rejected | Live (limits, numbers and booleans pass); echo type inferred |
| Slots output | Keyed by `YYYY-MM-DD` in the requested `timeZone` (default UTC); days without slots omitted; items `{start}` or, with `format=range`, `{start,end}`; times with milliseconds, `Z` for UTC and the zone offset otherwise; date-only `start` is 00:00:00 UTC and date-only `end` 23:59:59 UTC | Doc, source |
| Slots availability | Slot starts every event length from the start of working hours, whole slot inside hours on a working day, at least `min_notice_minutes` ahead, before `now + horizon_days`, not overlapping busy time (seeded bookings, accepted Cal.com bookings, Google events, third-party takes); `bookingUidToReschedule` frees that booking | Sandbox model of Cal.com availability |
| Unknown event type | Slots 404 `"Event Type not found"`; create 404 `"Event type with id <id> not found."`; slug lookups `"User with username <u> not found"`, `"Event type with slug <s> belonging to user <u> not found."` | Live (id forms), source (slug forms) |
| Booking object | Keys in `BookingOutput_2024_08_13` declaration order, absent keys omitted; `start`, `end`, `createdAt`, `updatedAt` with milliseconds; `cancellationReason` and `cancelledByEmail` `""` on active bookings; `rescheduledByEmail` `null` until rescheduled; `icsUid` `<uid>@Cal.com`; `eventType` `{id, slug}`; the deprecated `meetingUrl` always equals `location`; `bookingFieldsResponses` with `email`, `name`, `displayEmail`, plus `guests` and `displayGuests` when the request sent `guests`; top-level `guests` only when sent | Doc (fields), source (order, empty strings, output service), inferred (milliseconds; a real hosted reschedule payload shows `meetingUrl` equal to a plain-text location and no `guests`) |
| Booking uid | 22-character Flickr base58 short UUID of a UUIDv5 | Source |
| Create and reschedule responses | Add `"isPlatformManagedUserBooking": false` at the end of `data`; the stored booking does not carry it | Source, not confirmed on hosted |
| Taken slot | 400 `BadRequestException` `"User either already has booking at this time or is not available"`, also for times outside working hours and on non-working days | Source, plus a 2026 issue showing the same text from the official CLI |
| Out of bounds | Inside the minimum notice or at or beyond the horizon: 400 with the verbatim `booking_time_out_of_bounds` text; a start in the past: 400 `"Attempting to book a meeting in the past."` | Source |
| Get and reschedule 404 | `"Booking with uid=<uid> was not found in the database"` | Live |
| Cancel 404 | `"Booking with uid=<uid> not found"` (different wording) | Live |
| Reschedule | 201 with a new booking: new uid, `rescheduledFromUid`, `reschedulingReason`, `rescheduledByEmail`, same status, duration, metadata, attendees and location, `icsUid` of the original; the original becomes `cancelled` with `rescheduledToUid`; the original's own time does not block the new one | Doc (201), source (e2e), inferred (`icsUid`, from a real payload) |
| Reschedule errors | Cancelled booking and already-rescheduled booking texts; taken or out-of-bounds new time as for create | Source |
| Cancel | 200 with the same uid, `status` `cancelled`, `cancellationReason`; a second cancel gets the already-cancelled 400 text | Doc (200), source |
| Cancel after the booking ended | 400 raw NestJS body `{"statusCode":400,"message":"Cannot cancel a booking that has already ended"}` without the envelope | Source; the missing envelope is inferred, not confirmed on hosted |
| List filters | `attendeeEmail` trimmed, inner whitespace turned into `+` (an unencoded `+` arrives as a space), then an exact match; `attendeeName` and `bookingUid` trimmed exact matches; an empty value filters nothing; `eventTypeIds` wins over `eventTypeId` and `teamsIds` over `teamId`; `afterStart`, `beforeEnd`, `afterCreatedAt`, `beforeCreatedAt`, `afterUpdatedAt`, `beforeUpdatedAt`; only the first of `sortStart`, `sortEnd`, `sortCreated`, `sortUpdatedAt` applies | Doc, source |
| List statuses | `upcoming` (not ended, not cancelled or rejected, pending included), `past`, `cancelled` (includes rescheduled-away), `unconfirmed`, `recurring`; 2024-08-13 takes a comma list and defaults to `upcoming`; an invalid value is a list query validation error (above); 2026-05-01 takes one value and walks every status when it is omitted | Doc (2026-05-01), source (2024-08-13 default, not confirmed on hosted) |
| Offset pagination (2024-08-13) | `take` 1..250 (default 100), `skip` from 0; meta `returnedItems, totalItems, itemsPerPage, remainingItems, currentPage, totalPages, hasNextPage, hasPreviousPage` with the source formulas, `skip` clamped to the total first | Source |
| Cursor pagination (2026-05-01) | `limit` 1..100 (default 50), opaque `cursor`; meta `{nextCursor, hasMore}`; `upcoming`, `recurring`, `unconfirmed` walk forward from `now - 1h`, `past` backward from `now`, omitted and `cancelled` backward from year 2100 | Doc |
| List without a token | 403 `ForbiddenException` `"PermissionsGuard - no authentication provided. …"` | Live |
| Wrong token | 401 `UnauthorizedException` `"ApiAuthStrategy - api key - Your api key is not valid"` for any wrong token (see deviations for tokens without the `cal_` prefix) | Live (a `cal_` key) |

### Known deviations

1. **Auth on every route.** Real `GET /v2/slots` has no auth guard, and create, get, reschedule and cancel accept
   anonymous requests. The sandbox requires the bearer token on all of them. A request without a token gets 401
   `"ApiAuthStrategy - No authentication method provided. Either pass an API key as 'Bearer' header or OAuth
   client credentials as 'x-cal-secret-key' and 'x-cal-client-id' headers"` (a source text), except the list,
   which gets the real 403. An agent that sends no token works against Cal.com but not against the sandbox.
   Version routing that Cal.com answers before its guards (route 404s, the legacy create validation) is
   answered before the token check here too, so those answers do not depend on the token.
2. **Wrong tokens without the `cal_` prefix.** The sandbox answers every wrong token with the api-key 401 text.
   Cal.com treats only values starting with `cal_` as API keys; any other bearer value goes down the OAuth
   access-token path with different 401 texts, and on the list the permissions guard answers first with 403
   `"PermissionsGuard - no oAuth client found for access token=<token>"` (source, not observed live).
3. **Headers not mirrored.** No `x-ratelimit-*` (live: limit 500 per 60 s), `x-request-id`, `etag`, CORS or
   security headers, and no 429 rate limiting.
4. **One event type, one host.** No teams, round-robin, seats, recurring or managed event types, routing forms,
   instant meetings, booking questions, buffers, date overrides, booking limits or booker limits. The body
   fields for those features (`routing`, `instant`, `recurrenceCount`, `rescheduleWithSameHost`, `seatUid`,
   `cancelSubsequentBookings`, `organizationSlug`) are accepted and ignored. `teamId` and `teamsIds` list
   filters match nothing.
5. **No confirmation flow.** Bookings are always `accepted`; `rescheduledBy` never switches a booking to
   `pending`.
6. **Owner flags ignored.** `allowConflicts`, `allowBookingOutOfBounds` and `skipBookingLimits` are type-checked
   and ignored. Real Cal.com honours them on the 2026 versions for an owner's key.
7. **Legacy handlers not mirrored.** Without a known version header, a create body is checked against the five
   verified legacy constraints; when they all pass, the only entry is a sandbox constraint on
   `cal-api-version`. The legacy get, list and cancel exist on Cal.com (the get and cancel without auth); the
   sandbox answers them, after the token check, with that same sandbox entry in the legacy validation envelope.
   The legacy `GET /v2/slots/available` is not mirrored and answers 404.
8. **Availability model.** Working hours, working days, event length, minimum notice, a rolling horizon in days
   and busy time are modelled; nothing else. A create or reschedule may start off the slot grid when the host is
   free for the whole interval (Cal.com checks availability for the interval; not confirmed on hosted). Two
   racing creates are serialised, so the second one gets the 400 taken-slot error; real Cal.com can instead
   answer 409 `{"statusCode":409,"message":"booking_conflict_error"}` in a race.
9. **Durations.** The event type has one length, so there is no multi-duration booking. On slots, `duration` only
   sets the `end` of `format=range` items.
10. **Location.** Without a location, `location` and `meetingUrl` are `"integrations:daily"` and no video URL is
    created; Cal.com returns the Cal Video URL in both. A location object is reduced to its address, link or
    phone.
11. **Who cancelled.** `cancelledByEmail` is the host's email, because the sandbox treats the bearer token as the
    host's key (inferred).
12. **Messages not in the vendor evidence.** A few checks have no counterpart in Cal.com's input classes, or
    one whose message was not read: the type checks on attendee `email` and `phoneNumber` and on `rescheduledBy`
    (Cal.com's format checks there are no-ops), `location must be an object`, the `meetingUrl` URL test (an
    approximation of validator.js `isURL`), the cursor error and the 2026-05-01 `status` and `limit` texts
    (class-validator defaults). The order of several failed constraints on one property follows the decorator
    order in the source and was not observed live.
13. **Slots lookups.** `usernames` is served when there are at least two names, all of them the host's, and an
    `organizationSlug` (which is not checked).
14. **Error `path`.** The query string is echoed exactly as received; the real edge proxy re-encodes some
    characters (`/` as `%2F`, `@` as `%40`).
15. **Unknown routes.** Any unknown path or method under `/v2` answers the NestJS-style 404
    `Cannot <METHOD> <path>` and is logged with the group `unrouted`.
16. **Bodies.** A body that is not valid JSON (including `NaN`, `Infinity` or a number too large for a double) is
    validated as an empty body; Express would reject it earlier with a different 400. An empty cancel body is
    treated as `{}`.
17. **Attendees.** A phone-only attendee gets `email` and `displayEmail` `""`; Cal.com derives a placeholder
    address. `language` is always present (default `en`).
18. **Identifiers.** Booking ids are small integers counted per sandbox; the host id is 1.
19. **Cursor ordering.** The 2026-05-01 walk is modelled on booking start time with the documented anchors;
    explicit `sort*` parameters override it. Cursors are opaque base64 strings of a page offset.
20. **Date range.** A date or date-time whose UTC form falls outside the years 1 to 9998 gets the
    `must be a valid ISO 8601 date string` error. Numeric query parameters follow JavaScript's `parseInt` and
    `Number` rules, which read ASCII digits only; a value beyond a double's range counts as not a number.
21. **List under 2026-02-25.** `GET /v2/bookings` with `cal-api-version: 2026-02-25` is answered like
    2024-08-13 (offset pages). The docs name 2026-05-01 for this operation, and the 2026-02-25 behaviour was
    neither documented nor observed.
22. **Repeated query parameters.** The first value is used. Express would hand an array to the validation
    pipes (for example `status=upcoming&status=past`).
23. **ValidationPipe error entries.** Entries of `details.errors` carry `property`, `children` and
    `constraints`, as in the live legacy capture. The open-source bootstrap also enables `target` and `value`,
    so hosted Cal.com may add `value` when an invalid value was supplied (not observed).

### Fault responses

Fault modes are sandbox features. Their responses use the Cal.com shapes above.

| Mode | Response |
|---|---|
| `error_500` | 500 envelope, `InternalServerErrorException`, message `"Internal server error"` (sandbox text) |
| `timeout` | Holds the request for `hang_s` seconds without doing anything, then 504 `GatewayTimeoutException` `"Gateway Timeout"` (sandbox). Clients with a shorter timeout never see it. |
| `commit_then_timeout` | Performs the call (a write commits), holds for `hang_s`, then returns the normal response |
| `not_found` | Slots and list: 404 `"Event Type not found"`. Create: 404 `"Event type with id <id> not found."`. Get and reschedule: 404 `"Booking with uid=<uid> was not found in the database"`. Cancel: 404 `"Booking with uid=<uid> not found"`. Nothing is written. |
| `malformed` | 200 with an unexpected schema. Slots: `{"status":"success","data":{"busy":[{"start":…,"end":…}]}}`, listing every period of the requested window that is not a free slot (off hours, bookings, notice, horizon), so a client that scrapes ISO timestamps finds busy times. Single bookings: `{"status":"success","data":{"booking":{"startTime","endTime","bookingStatus"}}}`, and writes commit. List: `{"status":"success","data":{"items":[…],"count":n}}`. |
| `slot_taken_after_offer` | On any group except slots: before the call runs, every still-free slot of the most recent slots response the client received with its normal schema is booked by `third-party@example.com`, so a create or reschedule onto an offered slot gets the taken-slot 400. On `slots`: the list is answered normally and then taken. |
| `slow` | Waits `latency_ms`, then answers normally. `latency_ms` also delays any other mode before it applies. |

## Google Calendar API v3

Base URL swap: `https://www.googleapis.com` becomes `http://<sandbox>:8100`, and the service-account token URI
(`BT_GOOGLE_TOKEN_URI`) becomes `http://<sandbox>:8100/token`. The sandbox has one calendar,
`seed.google_calendar_id` (default `primary`). It is the host calendar that the other mirrored APIs share:
`freeBusy` reports seeded blocks, Cal.com bookings, third-party takes and Google events as busy, and Google events
block Cal.com slots. Every caller is treated as a service account without domain-wide delegation;
`seed.google_sa_can_invite: true` simulates delegation.

### Endpoints

| Method and path | Reference | Checked | Evidence |
|---|---|---|---|
| `POST /calendar/v3/freeBusy` | [Freebusy: query](https://developers.google.com/workspace/calendar/api/v3/reference/freebusy/query) | 2026-09-26 | Doc (request and response schema, `notFound` reason, inclusive start and exclusive end), inferred (HTTP 200 with the `notFound` entry and `busy: []`, no `groups` key, time formats: raw captures) |
| `POST /calendar/v3/calendars/{calendarId}/events` | [Events: insert](https://developers.google.com/workspace/calendar/api/v3/reference/events/insert), [Events resource](https://developers.google.com/workspace/calendar/api/v3/reference/events) | 2026-09-26 | Doc (id rules, attendee rule, required `start` and `end`, 409 `duplicate`), inferred (no conflict check: no such error exists; response key order and formats from a 2026 capture; 400 id body, 403 attendee body, `Missing end time.` from raw captures) |
| `GET /calendar/v3/calendars/{calendarId}/events/{eventId}` | [Events: get](https://developers.google.com/workspace/calendar/api/v3/reference/events/get) | 2026-09-26 | Doc (cancelled events are always returned, `timeZone` parameter) |
| `PATCH /calendar/v3/calendars/{calendarId}/events/{eventId}` | [Events: patch](https://developers.google.com/workspace/calendar/api/v3/reference/events/patch), [Extended properties](https://developers.google.com/workspace/calendar/api/guides/extended-properties) | 2026-09-26 | Doc (patch semantics, `null` removes a property key), inferred (revive with `status: confirmed`: several field reports and a 2026 end-to-end test) |
| `DELETE /calendar/v3/calendars/{calendarId}/events/{eventId}` | [Events: delete](https://developers.google.com/workspace/calendar/api/v3/reference/events/delete) | 2026-09-26 | Doc (empty body, 410 `deleted` on a second delete), inferred (204) |
| `GET /calendar/v3/calendars/{calendarId}/events` | [Events: list](https://developers.google.com/workspace/calendar/api/v3/reference/events/list) | 2026-09-26 | Doc (`timeMin`, `timeMax`, `privateExtendedProperty`, `showDeleted`, `singleEvents`, `orderBy`, `maxResults`, response schema), inferred (the ordering error text: client issues) |
| `POST /token` (Google's `https://oauth2.googleapis.com/token`) | [Using OAuth 2.0 for server to server applications](https://developers.google.com/identity/protocols/oauth2/service-account) | 2026-09-26 | Doc (grant type, JWT header and claims, `aud`, one-hour limit, error codes and texts), inferred (response key order and `expires_in: 3599`: a 2026 capture; `Invalid grant_type: `: client issues; the audience error and its HTTP 400: 2022 and 2025 raw captures) |

### Mirrored behaviour

| Behaviour | What the sandbox does | Evidence |
|---|---|---|
| Error envelope | Calendar backend errors: `{"error":{"errors":[{"domain","reason","message"[,"locationType","location"]}],"code":N,"message":…}}`, keys in that order, top-level `message` equal to the item's, **no `status` key** | Doc (errors guide), inferred (a raw 2020 capture) |
| Auth errors | Front-end shape with `status`: `{"error":{"code":401,"message":…,"errors":[{"message","domain","reason","location","locationType"}],"status":"UNAUTHENTICATED"}}`. No `Authorization` header: `"Request is missing required authentication credential. …"` with `"Login Required."` / `required`; a wrong token: `"Request had invalid authentication credentials. …"` with `"Invalid Credentials"` / `authError` | Inferred (raw client dumps, 2019 and 2024) |
| Scopes | Once a token has been exchanged at `/token`, each method needs one of the scopes its reference page lists, checked against the latest accepted grant. `freeBusy`: `calendar.readonly`, `calendar`, `calendar.events.freebusy` or `calendar.freebusy` (not `calendar.events`). Insert, patch and delete: `calendar`, `calendar.events`, `calendar.app.created` or `calendar.events.owned`. Get and list: those four, `calendar.readonly`, `calendar.events.readonly`, `calendar.events.freebusy`, `calendar.events.owned.readonly` or `calendar.events.public.readonly`. Otherwise 403 in the front-end shape: `{"error":{"code":403,"message":"Request had insufficient authentication scopes.","errors":[{"message":"Insufficient Permission","domain":"global","reason":"insufficientPermissions"}],"status":"PERMISSION_DENIED","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"ACCESS_TOKEN_SCOPE_INSUFFICIENT","domain":"googleapis.com","metadata":{"service":"calendar-json.googleapis.com","method":"calendar.v3.Freebusy.Query"}}]}}` (the method name follows the call, e.g. `calendar.v3.Events.Insert`). Checked before any fault rule | Doc (scope lists on each reference page), inferred (the 403 message, `status`, reason, service and method from 2026 captures of a `freeBusy` call with only `calendar.events`; the `errors` entry is what other Google APIs send with it and was not seen in a Calendar capture) |
| Bytes | `application/json; charset=UTF-8`, pretty-printed with a trailing newline: 1-space indent for backend bodies (byte-identical to a raw 409 capture), 2-space for front-end auth errors | Inferred (raw captures); indentation is cosmetic, never parse on it |
| freeBusy request | `timeMin` and `timeMax` are RFC 3339 with an offset; `timeZone` optional (default UTC); `items[].id` | Doc |
| freeBusy response | `{"kind":"calendar#freeBusy","timeMin":"…000Z","timeMax":"…000Z","calendars":{id:{"busy":[{"start","end"}]}}}`; bounds echoed in UTC with milliseconds; busy times `Z` without milliseconds, or the `timeZone`'s offset when one is given; no `groups` key | Doc (schema), inferred (the bounds' format, offset busy times and the missing `groups` from raw captures; `Z` busy times without a `timeZone` are a guess, no capture shows them) |
| freeBusy for another calendar | HTTP 200, entry `{"errors":[{"domain":"global","reason":"notFound"}],"busy":[]}` (errors first, `busy` still present); an item without an id becomes the key `""` | Doc (`notFound` reason), inferred (the 200 and `busy: []` from several raw captures) |
| freeBusy errors | `timeMax <= timeMin`: 400 `calendar`/`timeRangeEmpty` `"The specified time range is empty."` at `parameter`/`timeMax` | Doc (errors guide) |
| Event ids | Client ids must match `[a-v0-9]{5,1024}` (base32hex), else 400 `invalid` `"Invalid resource id value."`; without an id the sandbox generates 26 base32hex characters | Doc (rules), inferred (400 body) |
| Duplicate id | 409 `duplicate` `"The requested identifier already exists."`, also for the id of a deleted event | Doc (errors guide), inferred (reuse after delete: field reports 2016–2026) |
| No conflict checking | Insert and patch accept overlapping events, including overlaps with Cal.com bookings and third-party blocks | Inferred (no conflict error exists in the reference or the errors guide) |
| Event resource | Keys in the reference's order, absent keys omitted: `kind`, `etag` (a quoted number), `id`, `status`, `htmlLink`, `created` (no milliseconds), `updated` (milliseconds), `summary`, `description`, `location`, `creator`, `organizer` (`{email, self: true}`), `start`, `end`, `transparency` (only `transparent`), `visibility` (only when not `default`), `iCalUID` (`<id>@google.com`), `sequence`, `attendees`, `extendedProperties`, `reminders` (`{"useDefault": true}` by default), `eventType` `default` | Doc (fields and order), inferred (formats and defaults from a 2026 capture) |
| Event times | `dateTime` rendered in the calendar's zone (`seed.host_timezone`) with its offset and whole seconds (`+00:00` for UTC); the `timeZone` sent is echoed. A `dateTime` without an offset is read in the object's `timeZone`; without either: 400 `required` `"Missing time zone definition for <start|end> time."`. `get` and `list` render in the `timeZone` query parameter when given | Doc (offset rule, response `timeZone`), inferred (rendering) |
| Missing times | 400 `required` `"Missing end time."` (checked first), `"Missing start time."`; `end` before `start`: 400 `calendar`/`timeRangeEmpty` | Inferred (`Missing end time.` from a raw capture; the others follow it) |
| Service-account attendees | A request with non-empty `attendees`: 403 `calendar`/`forbiddenForServiceAccounts` `"Service accounts cannot invite attendees without Domain-Wide Delegation of Authority."`, whatever `sendUpdates` says, on insert and on a patch that sends attendees; an empty list passes. With `google_sa_can_invite` attendees are stored with `responseStatus: needsAction` | Doc (rule), inferred (body from raw captures; `sendUpdates=none` from a 2026 report; patch assumed like insert) |
| Extended properties | Private and shared string maps echoed; keys over 44 characters dropped, values cut at 1024 characters | Doc (extended-properties guide) |
| Tombstones | Delete answers 204 with an empty body and leaves the event as a `cancelled` tombstone that keeps its details: `get` returns it (200), `list` hides it unless `showDeleted=true` (or `updatedMin` is set), a second delete is 410 `deleted` `"Resource has been deleted"`, insert with its id is 409, and a patch with `status: confirmed` revives it; a patch with `status: cancelled` deletes | Doc (get and list rules, 410), inferred (204, revive) |
| Patch | Objects merge, arrays and scalars replace, `null` removes a key; read-only fields in the body are ignored; `updated` and `etag` change on every write; `sequence` grows when the time or location changes | Doc (semantics), inferred (sequence rule) |
| Preconditions | `If-Match` on patch and delete: 412 `conditionNotMet` `"Precondition Failed"` at `header`/`If-Match` when it differs from the event's `etag` | Doc (version-resources guide, which names update and delete), inferred (patch, a partial update) |
| List | `timeMin` bounds the end and `timeMax` the start, both exclusive; `privateExtendedProperty`/`sharedExtendedProperty` `key=value`, repeated constraints must all match; `orderBy=startTime` needs `singleEvents=true`, else 400 `"The requested ordering is not available for the particular query."`; `maxResults` default 250, capped at 2500; `pageToken`/`nextPageToken`; response keys `kind` (`calendar#events`), `etag`, `summary`, `updated`, `timeZone`, `accessRole` (`writer`), `defaultReminders`, `nextPageToken`, `items` | Doc, inferred (ordering text) |
| `sendUpdates` | Validated (`all`, `externalOnly`, `none`), otherwise ignored: the sandbox sends no notifications | Doc (values) |

### OAuth token endpoint

| Behaviour | What the sandbox does | Evidence |
|---|---|---|
| Request | `POST /token`, form-encoded `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer` and `assertion=<JWT>`. The one vendor route that takes no bearer token | Doc |
| Assertion checks | Compact JWS with header `alg` `RS256` and a non-empty signature; claims `iss` (an email), `aud` exactly `https://oauth2.googleapis.com/token` (what google-auth always sends, even with another token URI), `iat` and `exp` at most 3600 s apart, `exp` after the sandbox clock's now, `iat` at most 300 s ahead of it, a non-empty space-delimited `scope` whose items are `https://` URLs (or `openid`, `email`, `profile`), so a comma-separated list or a bare `calendar.events` is rejected; `sub` (an email) only with `google_sa_can_invite` | Doc (claims, audience, one-hour limit, scope rules), source (google-auth's audience) |
| Success | 200 `{"access_token":"<BT_SANDBOX_TOKEN>","expires_in":3599,"token_type":"Bearer"}` in that order; the grant (issuer, subject, scopes, `iat`, `exp`, `kid`) is kept in `/_state` under `google.token_grants`; later events name the issuer (or subject) as `creator` | Inferred (a 2026 capture) |
| Errors | `{"error","error_description"}` with status 400: `unsupported_grant_type` `"Invalid grant_type: <value>"`; `invalid_grant` `"Invalid JWT Signature."` (wrong `alg`, or not a JWS), `"Invalid JWT: Failed audience check."` (missing or wrong `aud`), `"Invalid email or User ID."` (no `iss`), `"Not a valid email"` (an `iss` or delegated `sub` without `@`), the verbatim `"Invalid JWT: Token must be a short-lived token (60 minutes) and in a reasonable timeframe. …"`; `invalid_scope` `"Invalid OAuth scope or ID token audience provided."`; `unauthorized_client` `"Unauthorized client or scope in request."` for a `sub` without delegation (the guide's text for a service account not authorized for domain-wide delegation) | Doc (codes and texts, and which text answers a `sub` without delegation), inferred (`Invalid grant_type: ` from client issues; the audience text from 2022 and 2025 raw captures); sandbox (which text answers a bad `iss`, see deviations) |
| Faults and the log | Assertion checks run before the fault engine, like the bearer check of the other routes, so a rejected assertion never counts against a rule. The request log keeps the assertion without its signature and replaces access tokens with `[redacted]` | Sandbox |

### Known deviations

1. **Auth and scopes.** Every route requires a bearer, although `freeBusy` is "Authorization optional" for public
   calendars, and only in the `Authorization` header: the `access_token` query parameter, which Google also
   accepts, gets the 401. Any request with the sandbox token is treated as the same service account. Every access
   token is the sandbox token, so scopes are checked against the latest grant, not per token: a client that uses
   the sandbox token without an exchange, or keeps a token it obtained before `POST /_control/reset` (which clears
   the grants), has every scope. `calendar.app.created` and `calendar.events.owned` are accepted for writes,
   although on Google they cover only calendars or events the caller created or owns, which a shared host
   calendar is not.
2. **One calendar.** Only `seed.google_calendar_id` exists; `primary` is not an alias of the host's address (or
   the reverse). With the default seed, `primary` is the host calendar. On Google, `primary` for a service account
   without domain-wide delegation is the service account's own empty calendar, so `freeBusy` on it reports no busy
   time and inserts land where the host never sees them: against a real calendar, configure the shared calendar's
   id. No `calendarList`, ACLs or access roles; the list response always says `writer`. Group ids in `freeBusy`
   get `notFound`; `groupExpansionMax` and `calendarExpansionMax` are not enforced.
3. **Busy time not visible as events.** Seeded blocks, Cal.com bookings and third-party takes are busy in
   `freeBusy` but are not returned by `events.list`; on a real calendar they would be events.
4. **freeBusy details.** Busy blocks are merged when they overlap or touch and clipped to the query window (the
   documentation is silent). There is no range limit, while Google rejects long ranges with 400
   `timeRangeTooLong` at an undocumented limit (reported at about two months in 2015; a current production client
   reports about three months and queries in 90-day chunks), so a client should split long ranges itself. An
   unknown `timeZone` gets a sandbox 400 `invalid` `"Invalid value for: timeZone"`.
5. **Timed events only.** An all-day event (`start.date`) gets a sandbox 400. Recurrence, instances,
   conference data, attachments, working-location and other event types are not mirrored. Unknown or unsupported
   body fields are ignored and not echoed; an `iCalUID` in the insert body is ignored (the sandbox always uses
   `<id>@google.com`).
6. **Identifiers and tombstones.** An id collision is always detected (the reference warns that it may not be).
   Tombstones never expire (Google: they "eventually disappear") and `get` always returns them, although 2026 field
   reports also saw 404 or 410. The 403 that follows an emptied trash is not mirrored. A patch without `status` on
   a tombstone keeps it cancelled (undocumented). `sequence` grows on time and location changes only (inferred).
7. **List details.** No sync tokens: `nextSyncToken` is never returned and any `syncToken` gets 410
   `fullSyncRequired`. `q` is a case-insensitive substring match over summary, description, location and attendee
   emails, not Google's full-text search. Without `orderBy`, events come in insertion order.
8. **Synthetic values.** `etag`, the list `etag`, `htmlLink`, server-assigned ids and `created`/`updated` (sandbox clock)
   are synthetic but shaped like Google's. `organizer` has no `displayName`; `defaultReminders` is empty.
   Attendees keep only `email`, `displayName`, `optional` and `responseStatus`.
9. **Texts not in the vendor evidence.** `"Required"` for a missing `freeBusy` bound; `"Bad Request"`/`badRequest`
   for malformed timestamps, filters and page tokens; `"Parse Error"`/`parseError` for a body that is not a JSON
   object; `"Invalid value for: <field>"`; `"Missing start time."`; the time-zone texts; `"Missing attendee
   email."`; the `invalidParameter` texts; and the `badRequest` reason of the ordering error. The order in which
   the checks run (query, body, calendar, fields, attendees, duplicate id) was not observed.
10. **Headers and limits.** No `Server`, `ETag` or quota headers; no `usageLimits` 403 or 429. `If-None-Match`
    (304) is not mirrored. The extended-property limit of 300 properties and 32 kB per event is not enforced.
11. **Unknown routes.** Any unknown path or method under `/calendar/v3` answers the backend 404 `notFound` and is
    logged as `unrouted`; Google's front end may answer an HTML page instead.
12. **Token endpoint.** The path is `/token` on the sandbox host. The signature is not verified, because the
    sandbox has no public key for the caller's service account. `exp - iat` may be at most 3600 s, five minutes
    stricter than the 65 minutes after which Google documents a rejection. `iat` and `exp` are checked against the
    sandbox clock, so a client must sign with the clock the sandbox uses (the same injected clock when the sandbox
    runs on a fixed or test clock). Every error is status 400; the documentation gives no status, and raw captures
    of the audience error show 400. Only `https://oauth2.googleapis.com/token` is accepted as `aud`, not the older
    Google token URLs that some clients still send. The guide gives `"Not a valid email"` and `"Invalid email or
    User ID."` for a `sub` that names an unknown user; the sandbox also uses them for a missing or malformed `iss`,
    where Google's answer was not verified. `"Missing required parameter: assertion"` is a sandbox text, and
    structural failures reuse `"Invalid JWT Signature."`. Another method on `/token` gets a sandbox 404
    `invalid_request`. The access token is the sandbox token and never expires. A JSON body is accepted as well as
    a form. Responses are pretty-printed with 2 spaces and carry no cache headers. A self-signed JWT sent straight
    as the bearer (google-auth does this when `always_use_jwt_access` is on or no scopes are set) gets the 401
    `"Invalid Credentials"`, so an adapter that skips the token exchange fails here; whether Calendar accepts such
    tokens is not documented.

### Fault responses

| Mode | Response |
|---|---|
| `error_500` | 500 `backendError` `"Backend Error"`; token endpoint: 500 `{"error":"internal_failure","error_description":"Backend Error"}` (sandbox) |
| `timeout` | Holds for `hang_s` without doing anything, then 503 `backendError` `"Backend Error"`; token endpoint: 503 `temporarily_unavailable` (sandbox) |
| `commit_then_timeout` | Performs the call (an insert, patch or delete commits; a token grant is recorded), holds for `hang_s`, then returns the normal response |
| `not_found` | `freebusy`: 200 with a `notFound` entry for every requested calendar and no busy time. `events.*`: 404 `notFound` `"Not Found"`, nothing written. Token endpoint: 400 `invalid_grant` `"Invalid email or User ID."` |
| `malformed` | 200 with an unexpected schema; writes commit. `freebusy`: each calendar entry carries its busy periods under `busyPeriods` instead of `busy`, so a client that reads only `busy` sees a free calendar. Insert, get and patch: `{"event":{"eventId","startTime","endTime","eventStatus"}}`. List: `{"kind":"calendar#events","events":[…],"count":n}`. Delete: `{"deleted":{"eventId":…}}`. Token endpoint: camel-case `accessToken`, `expiresIn`, `tokenType` |
| `slot_taken_after_offer` | On any group except `freebusy`: before the call runs, every free working-hours slot inside the window of the most recent `freeBusy` response that gave the client the host calendar with its normal schema is booked by `third-party@example.com`; then the call proceeds, and an insert still succeeds because Google does no conflict checking. On `freebusy`: the response is sent normally and its window is then taken. Only the `freebusy` form can be caught by a client that re-checks `freeBusy` before inserting: on `events.insert` the take happens after that re-check, and it merges with the client's own event in later `freeBusy` answers |
| `slow` | Waits `latency_ms`, then answers normally |

## HubSpot CRM API v3

Base URL swap: `https://api.hubapi.com` becomes `http://<sandbox>:8100`, and the HubSpot token (`BT_HUBSPOT_TOKEN`)
is the sandbox token. Objects have numeric string ids, string property values sorted by name, and
`createdAt`/`updatedAt`. `/_state` shows contacts and meetings with every stored property, as create and update
return them, and meetings with their contact associations.

### Endpoints

| Method and path | Reference | Checked | Evidence |
|---|---|---|---|
| `POST /crm/v3/objects/contacts/search` | [Search contacts](https://developers.hubspot.com/docs/api-reference/legacy/crm/objects/contacts/search/search-contacts), [Search the CRM](https://developers.hubspot.com/docs/api-reference/legacy/crm/search-the-crm) | 2026-09-26 | Doc (request and response schema, filter semantics and limits, default properties, `paging.next.after`), inferred (alphabetical property order: captures) |
| `POST /crm/v3/objects/contacts` | [Create contact](https://developers.hubspot.com/docs/api-reference/legacy/crm/objects/contacts/create-contact), [Contacts guide](https://developers.hubspot.com/docs/api-reference/legacy/crm/objects/contacts/guide) | 2026-09-26 | Doc (201, `Location`, input and output schema), inferred (409 body from 2021–2026 captures; default properties from a 2024 capture; validation bodies from 2020–2025 captures) |
| `PATCH /crm/v3/objects/contacts/{contactId}` | [Update contact](https://developers.hubspot.com/docs/api-reference/legacy/crm/objects/contacts/update-contact) | 2026-09-26 | Doc (200, `idProperty`, empty string clears, unknown and read-only properties are errors), inferred (404 bodies from captures) |
| `POST /crm/v3/objects/meetings` | [Create meeting](https://developers.hubspot.com/docs/api-reference/legacy/crm/activities/meetings/create-meeting), [Meetings guide](https://developers.hubspot.com/docs/api-reference/legacy/crm/activities/meetings/guide), [Associations guide](https://developers.hubspot.com/docs/api-reference/legacy/crm/associations/associate-records/guide) | 2026-09-26 | Doc (201, properties, `hs_timestamp` default, inline associations, type 200 meeting to contact), inferred (outcome values, `INVALID_OPTION` body, association error bodies) |
| `PATCH /crm/v3/objects/meetings/{meetingId}` | [Update meeting](https://developers.hubspot.com/docs/api-reference/legacy/crm/activities/meetings/update-meeting) | 2026-09-26 | Doc (200, semantics) |
| `GET /crm/v3/objects/meetings/{meetingId}` | [Get meeting](https://developers.hubspot.com/docs/api-reference/legacy/crm/activities/meetings/get-meeting) | 2026-09-26 | Doc (`properties`, `associations`, null for unset, unknown omitted, association shape), inferred (`meeting_event_to_contact`) |

### Mirrored behaviour

| Behaviour | What the sandbox does | Evidence |
|---|---|---|
| Error envelope | Compact JSON, keys `status` (`"error"`), `message`, `correlationId`, then `errors`, `context` and `category` when present; `correlationId` is a UUIDv7 and equals the `x-hubspot-correlation-id` and `x-request-id` headers, which every response carries | Doc ([error handling](https://developers.hubspot.com/docs/api-reference/error-handling): fields), inferred (order, compactness, UUIDv7, headers: captures) |
| Content type | `application/json;charset=utf-8` (no space) | Inferred (a 2024 header dump) |
| Auth | Missing or wrong token: 401 `{"status":"error","message":"Authentication credentials not found. This API supports OAuth 2.0 authentication and you can find more details at https://developers.hubspot.com/docs/methods/auth/oauth-overview","correlationId":…,"category":"INVALID_AUTHENTICATION"}` | Inferred (raw logs, 2025 and 2026) |
| Datetimes | Accepted as ISO 8601 (an offset is converted to UTC) or epoch milliseconds, as a string or a number; emitted like Java `Instant.toString()`: no fraction when the milliseconds are zero, otherwise three digits | Doc (both inputs, ISO output), inferred (trimming from captures) |
| Property values | Strings; numbers and booleans are read as their text; an empty string clears a value on update; response properties sorted by name | Doc (strings, clearing), inferred (order) |
| Contact create | 201 with `Location` and every stored property: those sent plus `createdate`, `hs_all_contact_vids`, `hs_email_domain`, `hs_is_contact`, `hs_is_unworked`, `hs_lifecyclestage_lead_date`, `hs_marketable_status`, `hs_marketable_until_renewal`, `hs_object_id`, `hs_object_source` and `hs_object_source_label` (`INTEGRATION`), `hs_pipeline`, `lastmodifieddate`, `lifecyclestage` (`lead`) | Doc (201, `Location`), inferred (a 2024 capture) |
| Existing email | 409 `{"status":"error","message":"Contact already exists. Existing ID: <id>","correlationId":…,"category":"CONFLICT"}` on a create whose email another contact has. The same 409 answers an update that takes another contact's email, and emails are compared case-insensitively | Inferred (create: captures 2021–2026; integrations parse the id from this text), **sandbox** (the update case and the case-insensitive match: no evidence either way) |
| Validation | 400 `VALIDATION_ERROR`; `message` is `"Property values were not valid: "` plus a JSON array of `{isValid, message, error, name, localizedErrorMessage, propertyValue, portalId}`, and `errors[]` repeats each as `{message, code, context:{propertyName:[…]}}`; every invalid property of the request is listed. Codes: `PROPERTY_DOESNT_EXIST` `Property "<p>" does not exist`, `READ_ONLY_VALUE` `"<p>" is a calculated property; its value cannot be set.`, `INVALID_EMAIL` `Email address <v> is invalid`, `INVALID_OPTION` `<v> was not one of the allowed options: [label: "…"\nvalue: "…"\ndisplay_order: n\nhidden: false\nread_only: false\n, …]` | Inferred (2020–2025 captures; the error-handling page shows the `errors[]` form) |
| Not found | A numeric id that does not exist: 404 `{"status":"error","message":"resource not found","correlationId":…}` with no category. A non-numeric id: 404 `"Object not found.  objectId are usually numeric."` (two spaces) with `context.id` and `OBJECT_NOT_FOUND`. `?idProperty=email` looks a contact up by email | Doc (`idProperty`), inferred (bodies from captures; the `GET` form is assumed equal to `PATCH`) |
| Search | `filterGroups` are OR-ed and filters inside a group AND-ed; a top-level `filters` list counts as one more group; `EQ` and `NEQ` compare case-insensitively, while `IN` and `NOT_IN` lowercase the stored value only, so their `values` must be lowercase to match; at most 5 groups (the guide's limit; the spec's field description says 6), 6 filters per group and 18 filters; `limit` default 10, at most 200; `after` as a string or a number; `sorts` as strings or `{propertyName, direction}`; `query` searches `firstname`, `lastname`, `email`, `phone`, `hs_additional_emails`, `fax`, `mobilephone`, `company` and `hs_marketable_until_renewal`; response `{"total":n,"results":[…],"paging":{"next":{"after":"<n>"}}}` with `paging` only when more remain; default properties `createdate`, `email`, `firstname`, `hs_object_id`, `lastmodifieddate`, `lastname`; with `properties`, those plus `createdate`, `hs_object_id` and `lastmodifieddate`, unset ones as `null` | Doc (guide and spec; a top-level `filters` list appears alone in a guide example), **sandbox** (top-level `filters` combined with `filterGroups`), inferred (`null` for unset properties, as the get guide documents) |
| Meeting create | 201 with `Location`; properties `hs_timestamp`, `hs_meeting_title`, `hs_meeting_body`, `hs_internal_meeting_notes`, `hs_meeting_external_url`, `hs_meeting_location`, `hs_meeting_start_time`, `hs_meeting_end_time`, `hs_meeting_outcome`, `hubspot_owner_id`, `hs_activity_type`, `hs_attachment_ids`; `hs_timestamp` defaults to `hs_meeting_start_time`; the response echoes the properties sent plus `hs_createdate`, `hs_lastmodifieddate` and `hs_object_id`, without associations | Doc (properties, default), inferred (response set) |
| Meeting outcome | `SCHEDULED`, `COMPLETED`, `RESCHEDULED`, `NO_SHOW`, `CANCELED` (one L); anything else is `INVALID_OPTION` | Inferred (the reference lists the labels only; the values match the default property definition and the values integrations send) |
| Associations | Inline on create: `[{"to":{"id":…},"types":[{"associationCategory":"HUBSPOT_DEFINED","associationTypeId":200}]}]` (meeting to contact; `to.id` as a string or a number). `GET …?associations=contacts` adds `"associations":{"contacts":{"results":[{"id":"…","type":"meeting_event_to_contact"}]}}`, and leaves the key out when there are none. Type 199 (contact to meeting) gets 400 `"invalid from object type 0-47 for associations to be created. expected: 0-1"`; a contact that does not exist gets 400 `"One or more associations are invalid"` with `context` `INVALID_OBJECT_IDS`, `objectId`, `objectType` | Doc (type ids and shape), inferred (type label; error bodies extrapolated from captures of neighbouring cases) |
| Meeting get | `properties` as a comma list or repeated; defaults `hs_createdate`, `hs_lastmodifieddate`, `hs_object_id`; requested but unset properties are `null`, unknown ones are left out | Doc (query parameters, `null` and omission rules; the defaults are the meetings row of the search guide's default-property table, assumed equal for `GET`) |

### Known deviations

1. **Auth and scopes.** The token is the sandbox token; its `pat-<region>-<uuid>` format is not checked, and scopes
   are never checked, so no 403 `MISSING_SCOPES`.
2. **Search is immediately consistent.** HubSpot documents that new or updated objects take "a few moments" to
   appear in search (integrators report up to 30 seconds), which is how a real search can miss a contact that
   create then reports as a 409 conflict. The sandbox finds every object at once.
3. **No rate limits.** No 429 (five searches per second per account; 100 requests per 10 seconds for private apps)
   and no `X-HubSpot-RateLimit-*` headers.
4. **Property schema.** The sandbox knows a fixed set of default properties: for contacts `email`, `firstname`,
   `lastname`, `phone`, `mobilephone`, `company`, `jobtitle`, `website`, `address`, `city`, `state`, `zip`,
   `country`, `hs_timezone`, `hs_language`, `lifecyclestage`, `hs_lead_status`, `hubspot_owner_id`, `message`,
   `createdate`; for meetings the properties listed above. Any other name gets `PROPERTY_DOESNT_EXIST`, while a
   real account may have custom properties. Only `hs_meeting_outcome` is checked as an enumeration; custom
   outcomes and the forward-only rule of `lifecyclestage` are not modelled.
5. **Response property sets.** Create and update return every stored property; HubSpot's exact sets are
   undocumented. The contact defaults follow a 2024 capture without the app-specific `hs_object_source_id`. A
   cleared property is removed rather than stored as an empty string.
6. **Datetimes.** An epoch value in seconds is read as milliseconds (a 1970 date). What v3 does with one is not
   established: HubSpot's
   [changelog of 2026-08-11](https://developers.hubspot.com/changelog/crm-api-write-validation-enforcement) says
   such values "could be rejected with errors like INVALID_DATE", so it may answer 400 instead. A value that is
   neither ISO nor a number gets the sandbox text `"<v> was not a valid long."` with code `INVALID_LONG`. A
   meeting with neither `hs_timestamp` nor `hs_meeting_start_time` gets a sandbox `MISSING_REQUIRED_PROPERTY` 400.
7. **Associations.** Only meeting-to-contact (type 200, `HUBSPOT_DEFINED`) is mirrored; other type ids get a sandbox
   400. Associations sent on a contact create are ignored. `GET` never returns `_unlabeled` duplicates.
8. **Search details.** `CONTAINS_TOKEN` is a substring match (a `*` wildcard is dropped); `GT`, `GTE`, `LT`, `LTE`
   and `BETWEEN` compare numbers, datetimes or text; `query` is a case-insensitive substring over the searchable
   properties (the guide: a value "containing the specified string"). Enumeration properties such as
   `lifecyclestage` compare case-insensitively here, although HubSpot documents them as case-sensitive for every
   operator. Without `sorts`, results come by id; only the first sort applies. A `limit` above 200 is clamped, a
   `null` `limit` or `after` means the default, and paging beyond 10,000 results is not rejected (HubSpot answers
   400).
9. **Texts not in the vendor evidence.** Input-class failures (a missing `properties`, a malformed filter or
   association, too many filters) use `"Invalid input JSON: <reason>"` with `VALIDATION_ERROR`; the limit texts
   follow older captured wording. An unparsable body gets `"Invalid input JSON on line L, column C: <reason>"`
   without a category, with the parser's own reason. 500 `"internal error"`, 504 `"gateway timeout"` and the 404
   of an unknown route (`"resource not found"`) are sandbox texts.
10. **Identifiers and headers.** Ids are small integers counted per sandbox (shared with Cal.com booking ids).
    `Location` uses the sandbox's base URL. Of HubSpot's response headers only `x-hubspot-correlation-id` and
    `x-request-id` are sent (no Cloudflare or rate-limit headers). `propertiesWithHistory`, `archived=true` and
    `idProperty` values other than `email` are not mirrored; nothing is ever archived.

### Fault responses

| Mode | Response |
|---|---|
| `error_500` | 500 `{"status":"error","message":"internal error","correlationId":…}` (sandbox) |
| `timeout` | Holds for `hang_s` without doing anything, then 504 `"gateway timeout"` (sandbox) |
| `commit_then_timeout` | Performs the call (a create or update commits), holds for `hang_s`, then returns the normal response; a blind retry of a contact create then gets the 409 conflict |
| `not_found` | 404 `"resource not found"` without a category; nothing is written |
| `malformed` | 200 with an unexpected schema; writes commit. Search: `{"count":n,"objects":[{"objectId":<number>,"props":{…}}]}`. Single objects: `{"object":{"objectId":<number>,"props":{…}}}` |
| `slot_taken_after_offer` | No effect on CRM groups: the call runs normally |
| `slow` | Waits `latency_ms`, then answers normally |

## Control API (sandbox only)

| Method and path | Body | Response |
|---|---|---|
| `POST /_control/reset` | none | `{"ok":true,"seed":{…}}`: default seed, no bookings, events, CRM objects, busy blocks, faults or log |
| `POST /_control/seed` | partial seed, merged over the defaults (not over the current seed); the Google mirror uses `google_calendar_id` (default `primary`) and `google_sa_can_invite` (default `false`, simulates domain-wide delegation) | `{"seed":{…}}`, the effective seed; `existing_bookings` become busy blocks. 422 `invalid_seed` on bad input |
| `POST /_control/faults` | `{"rules":[{"group","mode","times","after_calls","latency_ms","hang_s","id"}]}` | `{"faults":[…]}`; replaces all rules; 422 `invalid_faults` for unknown groups, modes or fields |
| `POST /_control/bookings` | `{"calendar":"calcom"\|"google","lead_email","lead_name","start","title"?,"lead_timezone"?}`; Google also takes `"event_id"?` and `"extended_properties"?` (extra private properties) | 201 with the vendor object, created through the vendor's own create path but not logged and never faulted. Cal.com: the booking of `POST /v2/bookings`. Google: the event of `events.insert` on the seeded calendar, summary `<event title> with <lead name>`, the lead in `description` and in `extendedProperties.private.bt_lead_email`, attendees only with `google_sa_can_invite`; like `events.insert` it does no conflict checking. 409 `booking_rejected` with the vendor error when the vendor rejects it; 501 for a calendar this build does not mirror |
| `GET /_state` | none | `{"now","seed","calcom":{"bookings"},"google":{"events","token_grants"},"hubspot":{"contacts","meetings"},"external_busy","faults","request_log"}`; vendor objects appear exactly as the vendor API returns them (Google tombstones included, meetings with their contact associations); `token_grants` lists each accepted service-account assertion: `iss`, `sub`, `scopes`, `iat`, `exp`, `kid`, `issued_at` |
| `GET /_ui` | none, no token | Read-only HTML: the next 10 business days of the host calendar (Cal.com bookings, Google events with their lead, busy blocks; cancelled ones struck through), later bookings and events, the HubSpot contacts and meetings (outcome and associated contacts), active faults and the last 30 log lines; refreshes every 3 seconds |

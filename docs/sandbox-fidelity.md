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

- **Auth.** Every route except `GET /_ui` requires `Authorization: Bearer <BT_SANDBOX_TOKEN>` (default
  `sandbox`). A vendor route answers a missing or wrong token in that vendor's error shape; `/_control/*` and
  `/_state` answer `401 {"error":"unauthorized"}`.
- **Request log.** Every vendor call is appended to the request log when it arrives, including calls rejected
  for auth or version routing: sequence number, time, method, path, endpoint group, query, JSON body, final
  status, the full JSON response and the fault mode applied. `completed` turns true when the response is
  produced, so a call that is still hanging is visible in `GET /_state`. Logged responses are snapshots; a later
  change to a booking does not rewrite them. A call to an unknown path or method under a vendor prefix is logged
  with the group `unrouted`, which no fault rule can target. If the sandbox itself fails while answering, the
  call gets the vendor's 500 envelope and its entry completes with status 500 and fault `null`, so it can be
  told apart from an injected `error_500`. Control calls are never logged.
- **Faults.** Rules installed with `POST /_control/faults` are counted only for calls that pass auth and version
  routing. A rule targets an endpoint group, skips its first `after_calls` matching calls and then fires `times`
  times (`null` means every call). The first rule that fires decides the mode; the others still count the call.
- **Concurrency.** State changes run one at a time under a lock, so two creates racing for one slot never both
  succeed. Sleeps for `slow`, `timeout` and `commit_then_timeout` happen outside the lock, so a hanging request
  never delays other calls. A call whose handler has not run yet (it is inside a `slow` or `latency_ms` delay)
  when `POST /_control/reset` runs never touches the fresh state: it is skipped and answered as a gateway
  timeout.
- **JSON bodies.** Parsed strictly: `NaN`, `Infinity` and numbers too large for a double make the body invalid,
  as they do for the vendors' own parsers.

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

The Google Calendar v3 subset (`freeBusy.query`, events insert with a client-supplied id, get, patch and delete,
and the fake OAuth token endpoint) is covered in its own section, completed together with that shape in the
sandbox. `POST /_control/bookings` with `"calendar":"google"` answers 501 until then.

## HubSpot CRM API v3

The HubSpot CRM v3 subset (contact search, create and update; meeting create, update and get with a contact
association) is covered in its own section, completed together with that shape in the sandbox.

## Control API (sandbox only)

| Method and path | Body | Response |
|---|---|---|
| `POST /_control/reset` | none | `{"ok":true,"seed":{…}}`: default seed, no bookings, events, CRM objects, busy blocks, faults or log |
| `POST /_control/seed` | partial seed, merged over the defaults (not over the current seed) | `{"seed":{…}}`, the effective seed; `existing_bookings` become busy blocks. 422 `invalid_seed` on bad input |
| `POST /_control/faults` | `{"rules":[{"group","mode","times","after_calls","latency_ms","hang_s","id"}]}` | `{"faults":[…]}`; replaces all rules; 422 `invalid_faults` for unknown groups, modes or fields |
| `POST /_control/bookings` | `{"calendar":"calcom","lead_email","lead_name","start","title"?,"lead_timezone"?}` | 201 with the vendor booking object, created through the same code path as `POST /v2/bookings` but not logged and never faulted; 409 `booking_rejected` with the vendor error when the slot is not bookable; 501 for a calendar this build does not mirror |
| `GET /_state` | none | `{"now","seed","calcom":{"bookings"},"google":{"events"},"hubspot":{"contacts","meetings"},"external_busy","faults","request_log"}`; vendor objects appear exactly as the vendor API returns them |
| `GET /_ui` | none, no token | Read-only HTML: the next 10 business days of the host calendar, CRM tables, active faults and the last 30 log lines; refreshes every 3 seconds |

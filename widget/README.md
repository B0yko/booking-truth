# Embeddable booking widget

`widget.js` is the chat widget of the bundled booking agent: one vanilla JavaScript file, no framework, no build
step, no third-party requests. The agent serves it at `/widget.js`, and `/demo` serves `demo.html`, a sample
landing page for Halvenrook, a fictional builder of garden offices.

## Embed

```html
<script src="https://agent.example.com/widget.js" data-agent="https://agent.example.com" async></script>
```

- `data-agent` is the agent's base URL. A relative value is resolved against the page, so `data-agent="/"` means
  "the origin this page is served from" (that is what `demo.html` uses). Without the attribute, the widget uses
  the origin of its own `src`.
- Add the embedding page's origin to the agent's `BT_ALLOWED_ORIGINS`, or the browser blocks the chat requests.
- `window.BookingTruthWidget.open()` opens the panel, so the page's own "Book a call" buttons can use it.

The widget adds one `<booking-truth-widget>` element to `<body>` and renders everything inside its Shadow DOM,
so page styles do not reach it and its styles do not leak into the page.

## What it does

1. Asks for the prospect's name and email before the first message.
2. Sends the browser's IANA time zone (`Intl.DateTimeFormat().resolvedOptions().timeZone`) as
   `lead.timezone_hint`. The agent treats it only as a hint and states it back; a zone the prospect names wins.
3. Keeps the signed widget session token the agent returns and sends it with every later request. On the
   `widget` channel, reschedule and cancel reach only bookings made in the same widget session.
4. Renders the agent's quick replies as buttons. A quick reply that carries an action (`select_slot`,
   `reschedule`, `cancel`, `confirm_timezone` with its ids) sends that structured action; the button text is
   only shown in the transcript, never sent or matched. A quick reply without an action sends its label as an
   ordinary message, which the agent reads like anything else the prospect types.
5. Shows a booking card for every confirmed booking, reschedule or cancellation, with Reschedule and Cancel
   buttons (structured `reschedule` / `cancel` actions with the booking's `booking_uid`; cancel asks for
   confirmation first).
6. Shows an "offline demo mode" banner when `GET {agent}/v1/version` returns `"offline": true`.

## Protocol

Every request is `POST {agent}/v1/widget/chat` with a JSON body; ids come from `crypto.randomUUID()`:

```json
{
  "session_id": "0b6c…",
  "message_id": "5f1e…",
  "channel": "widget",
  "lead": {"email": "ana.k@example.com", "name": "Ana K.", "timezone_hint": "Europe/Berlin"},
  "message": "Could we talk on Tuesday afternoon?",
  "session_token": "…"
}
```

A button sends `"action": {"type": "select_slot", "slot_id": "s_…"}` instead of `"message"`. The response carries
`reply`, `quick_replies` (`label`, `action`, optional `start_utc`), `booking` (`ref`, `status`, `start_utc`,
`end_utc`, `zone`, `local_label`, `action`) or `null`, `agent_version`, `guard`, `usage` and `session_token`.

| Status | Shown to the prospect | Next step offered |
| --- | --- | --- |
| 409 `lead_busy`, 429 | the `reply` from the agent (a fixed text if there is none) | Try again (new message id) |
| 5xx, network error, timeout (60 s) | a fixed error text | Try again (same message id, so the agent's dedupe returns its stored answer instead of acting twice) |
| 410 (turn limit), 401/403 (session token rejected) | the `reply`, or a fixed text | Start a new conversation |
| 413 / `input_too_long` | a fixed text | the message goes back into the input box |

## Accessibility and phone layout

- Real `<button>`, `<label>` and `<form>` elements; errors are tied to their fields with `aria-describedby` and
  `aria-invalid`.
- The transcript is a `role="log"` live region, so new messages are announced; a `role="status"` line says when
  the assistant is typing.
- Keyboard: Enter sends, Shift+Enter adds a line, Escape closes the panel and returns focus to the launcher.
- At 480 px wide and below (tested at 375 px) the panel fills the screen. Inputs use a 16 px font so phones do
  not zoom in on focus.

## Storage

The conversation (name, email, session id and token, transcript) is kept in the page's `sessionStorage`, so it
survives a reload or a move to another page of the same site and is gone when the tab closes. If storage is
blocked, the widget still works for the current page view.

## Development

The file stays small and checked. From the repository root:

```bash
npm ci
npm run lint        # eslint (flat config in eslint.config.mjs)
npm run typecheck   # tsc --noEmit --allowJs --checkJs -p widget/tsconfig.json (the file starts with // @ts-check)
npm test            # node:test unit tests of the pure functions, tests/widget/*.test.mjs
python3 scripts/check_widget_size.py   # at most 25,600 bytes as served, no other hosts, no HTML parsing
```

The pure functions (request building, response normalisation, card data) are exported through `module.exports`
when the file is loaded by Node, and through `window.BookingTruthWidget` in a browser.

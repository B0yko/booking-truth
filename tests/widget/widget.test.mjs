// Unit tests for the pure functions of widget/widget.js (run with `npm test`).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const widgetPath = new URL("../../widget/widget.js", import.meta.url);
const w = require(fileURLToPath(widgetPath)); // not .pathname: it stays percent-encoded (spaces, non-ASCII)
const source = readFileSync(widgetPath, "utf8");
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

const state = (extra = {}) => ({
  sessionId: "s-1",
  lead: { name: "Ana K.", email: "ana.k@example.com" },
  zone: "Europe/Berlin",
  token: null,
  ...extra,
});

describe("agentBase", () => {
  it("strips trailing slashes and keeps a path prefix", () => {
    assert.equal(w.agentBase("https://agent.example.com/", null, undefined), "https://agent.example.com");
    assert.equal(w.agentBase("https://agent.example.com/bt//", null, undefined), "https://agent.example.com/bt");
  });

  it("resolves a relative value against the page", () => {
    assert.equal(w.agentBase("/", null, "http://localhost:8000/demo"), "http://localhost:8000");
    assert.equal(w.agentBase(" /agent/ ", null, "https://www.example.com/p"), "https://www.example.com/agent");
  });

  it("falls back to the origin of the script src", () => {
    const src = "https://agent.example.com/widget.js?v=1";
    assert.equal(w.agentBase(null, src, "https://www.example.com/"), "https://agent.example.com");
    assert.equal(w.agentBase("", "/widget.js", "http://127.0.0.1:8000/demo"), "http://127.0.0.1:8000");
  });

  it("rejects non-http(s) and missing values", () => {
    assert.equal(w.agentBase("javascript:alert(1)", null, "https://www.example.com/"), null);
    assert.equal(w.agentBase("ftp://agent.example.com", null, undefined), null);
    assert.equal(w.agentBase(null, null, "https://www.example.com/"), null);
    assert.equal(w.agentBase("not a url", null, undefined), null);
  });
});

describe("newId", () => {
  it("uses crypto.randomUUID when present", () => {
    assert.equal(w.newId({ randomUUID: () => "fixed-id" }), "fixed-id");
    assert.match(w.newId(), UUID);
  });

  it("builds a v4 UUID from getRandomValues outside secure contexts", () => {
    const fake = { getRandomValues: (/** @type {Uint8Array} */ a) => globalThis.crypto.getRandomValues(a) };
    const ids = new Set(Array.from({ length: 200 }, () => w.newId(fake)));
    assert.equal(ids.size, 200);
    for (const id of ids) assert.match(id, UUID);
  });
});

describe("browserZone", () => {
  it("returns the runtime's IANA zone", () => {
    assert.equal(w.browserZone(), Intl.DateTimeFormat().resolvedOptions().timeZone);
  });
});

describe("validateLead", () => {
  it("accepts and tidies a name and email", () => {
    const v = w.validateLead("  Ana   K. ", " ana.k@example.com ");
    assert.deepEqual(v, { lead: { name: "Ana K.", email: "ana.k@example.com" }, errors: {} });
  });

  it("reports each invalid field", () => {
    const v = w.validateLead("", "ana.k@");
    assert.equal(v.lead, null);
    assert.ok(v.errors.name);
    assert.ok(v.errors.email);
    assert.ok(w.validateLead("x".repeat(101), "a@example.com").errors.name);
    assert.ok(w.validateLead("Ana", `${"a".repeat(250)}@example.com`).errors.email);
    assert.ok(w.validateLead("Ana", "ana k@example.com").errors.email);
  });
});

describe("normalizeAction", () => {
  it("keeps each action type with its ids", () => {
    assert.deepEqual(w.normalizeAction({ type: "select_slot", slot_id: "s_abc" }), { type: "select_slot", slot_id: "s_abc" });
    assert.deepEqual(w.normalizeAction({ type: "cancel", booking_uid: "b1" }), { type: "cancel", booking_uid: "b1" });
    assert.deepEqual(w.normalizeAction({ type: "confirm_timezone", zone: "Asia/Kolkata" }), {
      type: "confirm_timezone",
      zone: "Asia/Kolkata",
    });
    assert.deepEqual(w.normalizeAction({ type: "reschedule", booking_uid: "b1", slot_id: "s_2", label: "x" }), {
      type: "reschedule",
      booking_uid: "b1",
      slot_id: "s_2",
    });
  });

  it("drops unknown types, missing ids and non-string ids", () => {
    for (const raw of [
      null,
      "select_slot",
      { type: "book", slot_id: "s" },
      { type: "select_slot" },
      { type: "select_slot", slot_id: 7 },
      { type: "cancel", booking_uid: "" },
      { type: "__proto__", booking_uid: "b" },
      { type: "constructor", booking_uid: "b" },
      { type: "toString" },
    ]) {
      assert.equal(w.normalizeAction(raw), null, JSON.stringify(raw));
    }
  });
});

describe("normalizeQuickReplies", () => {
  it("keeps structured slot replies with their start time", () => {
    const qr = w.normalizeQuickReplies([
      { label: "Tue 6 Oct, 3:00 PM", action: { type: "select_slot", slot_id: "s_1" }, start_utc: "2026-10-06T13:00:00Z" },
    ]);
    assert.deepEqual(qr, [
      { label: "Tue 6 Oct, 3:00 PM", action: { type: "select_slot", slot_id: "s_1" }, start_utc: "2026-10-06T13:00:00Z" },
    ]);
  });

  it("keeps a reply without an action as text and drops invalid ones", () => {
    const qr = w.normalizeQuickReplies([
      { label: "Keep my current time", action: null },
      { label: "Bad", action: { type: "select_slot" } },
      { action: { type: "select_slot", slot_id: "s" } },
      "text",
      { label: "", action: null },
    ]);
    assert.deepEqual(qr, [{ label: "Keep my current time", action: null, start_utc: null }]);
  });

  it("caps the count and the label length", () => {
    const many = Array.from({ length: 20 }, (_, i) => ({ label: `${i} ${"x".repeat(200)}`, action: null }));
    const qr = w.normalizeQuickReplies(many);
    assert.equal(qr.length, 12);
    assert.equal(qr[0].label.length, 120);
    assert.deepEqual(w.normalizeQuickReplies(undefined), []);
  });
});

describe("normalizeBooking", () => {
  it("requires a ref and defaults the action", () => {
    assert.equal(w.normalizeBooking({ status: "active" }), null);
    assert.equal(w.normalizeBooking(null), null);
    assert.deepEqual(w.normalizeBooking({ ref: "abc", start_utc: "2026-10-06T13:00:00Z", zone: 5 }), {
      ref: "abc",
      status: "",
      start_utc: "2026-10-06T13:00:00Z",
      end_utc: "",
      zone: "",
      local_label: "",
      action: "booked",
    });
  });
});

describe("normalizeResponse", () => {
  const booking = {
    ref: "1a2b3c4d5e",
    status: "active",
    start_utc: "2026-10-06T13:00:00Z",
    end_utc: "2026-10-06T13:30:00Z",
    zone: "Europe/Berlin",
    local_label: "Tuesday, 6 October 2026, 3:00 PM",
    action: "booked",
  };

  it("maps a successful reply", () => {
    const r = w.normalizeResponse(200, {
      reply: "Booked: Tuesday ...",
      quick_replies: [{ label: "Reschedule", action: { type: "reschedule", booking_uid: "1a2b3c4d5e" } }],
      booking,
      agent_version: "0123456789ab",
      guard: { blocked: false, repaired: false, events: [] },
      usage: { usd: 0 },
      session_token: "tok-1",
    });
    assert.equal(r.kind, "ok");
    assert.equal(r.reply, "Booked: Tuesday ...");
    assert.deepEqual(r.booking, booking);
    assert.equal(r.token, "tok-1");
    assert.equal(r.version, "0123456789ab");
    assert.equal(r.retry, false);
    assert.deepEqual(r.quickReplies[0].action, { type: "reschedule", booking_uid: "1a2b3c4d5e" });
  });

  it("shows the agent's reply for 409 lead_busy and 429, and offers a retry", () => {
    const busy = w.normalizeResponse(409, { error: "lead_busy", reply: "Still on your last message." });
    assert.deepEqual([busy.kind, busy.reply, busy.retry, busy.booking], ["busy", "Still on your last message.", true, null]);
    const limited = w.normalizeResponse(429, { reply: "Slow down a little." });
    assert.deepEqual([limited.kind, limited.reply, limited.retry], ["limited", "Slow down a little.", true]);
  });

  it("falls back to a fixed text when an error has no reply", () => {
    for (const [status, body, kind, retry] of [
      [409, null, "busy", true],
      [429, {}, "limited", true],
      [410, { error: "session_ended" }, "ended", false],
      [403, { error: "forbidden" }, "expired", false],
      [401, {}, "expired", false],
      [413, {}, "too_long", false],
      [422, { error: "input_too_long" }, "too_long", false],
      [422, { detail: [] }, "error", true],
      [500, null, "error", true],
      [502, "<html>", "error", true],
      [0, null, "network", true],
      [200, { quick_replies: [] }, "error", true],
    ]) {
      const r = w.normalizeResponse(status, body);
      assert.equal(r.kind, kind, `${status} ${JSON.stringify(body)}`);
      assert.equal(r.retry, retry, `${status}`);
      assert.ok(r.reply.length > 10, `${status} has a fallback text`);
      assert.deepEqual(r.quickReplies, []);
    }
  });

  it("keeps a session token from an error body", () => {
    assert.equal(w.normalizeResponse(409, { reply: "x", session_token: "tok-2" }).token, "tok-2");
  });
});

describe("retryMessageId", () => {
  const make = () => "fresh";
  it("keeps the id after network and server errors, so dedupe can answer", () => {
    assert.equal(w.retryMessageId("network", "m-1", make), "m-1");
    assert.equal(w.retryMessageId("error", "m-1", make), "m-1");
  });
  it("uses a new id after 409 and 429, which the agent did not take", () => {
    assert.equal(w.retryMessageId("busy", "m-1", make), "fresh");
    assert.equal(w.retryMessageId("limited", "m-1", make), "fresh");
  });
});

describe("buildRequest", () => {
  it("builds a text message request on the widget channel", () => {
    assert.deepEqual(w.buildRequest(state(), { messageId: "m-1", message: "Can we talk Tuesday?" }), {
      session_id: "s-1",
      message_id: "m-1",
      channel: "widget",
      lead: { email: "ana.k@example.com", name: "Ana K.", timezone_hint: "Europe/Berlin" },
      message: "Can we talk Tuesday?",
    });
  });

  it("sends an action instead of text, with the session token", () => {
    const body = w.buildRequest(state({ token: "tok-1", zone: null }), {
      messageId: "m-2",
      message: "ignored",
      action: { type: "select_slot", slot_id: "s_1" },
    });
    assert.deepEqual(body.action, { type: "select_slot", slot_id: "s_1" });
    assert.equal("message" in body, false);
    assert.equal(body.session_token, "tok-1");
    assert.deepEqual(body.lead, { email: "ana.k@example.com", name: "Ana K." });
  });

  it("omits the token until the agent issues one and caps the text", () => {
    const body = w.buildRequest(state(), { messageId: "m-3", message: "x".repeat(5000) });
    assert.equal("session_token" in body, false);
    assert.equal(body.message.length, w.MAX_INPUT);
  });
});

describe("bookingCard", () => {
  const base = { ref: "1a2b3c4d5e6f", status: "active", start_utc: "2026-10-06T13:00:00Z", end_utc: "", zone: "Europe/Berlin" };

  it("offers structured reschedule and cancel buttons for an active booking", () => {
    const card = w.bookingCard({ ...base, local_label: "Tue 6 Oct, 3:00 PM", action: "booked" });
    assert.equal(card.title, "Booked");
    assert.equal(card.when, "Tue 6 Oct, 3:00 PM");
    assert.equal(card.meta, "Europe/Berlin · ref 1a2b3c4d");
    assert.equal(card.active, true);
    assert.deepEqual(
      card.buttons.map((b) => b.action),
      [
        { type: "reschedule", booking_uid: "1a2b3c4d5e6f" },
        { type: "cancel", booking_uid: "1a2b3c4d5e6f" },
      ],
    );
    assert.ok(card.buttons[1].confirm, "cancel asks for confirmation");
  });

  it("titles a reschedule and shows no buttons once cancelled", () => {
    assert.equal(w.bookingCard({ ...base, local_label: "", action: "rescheduled" }).title, "Rescheduled");
    const cancelled = w.bookingCard({ ...base, local_label: "", status: "cancelled", action: "cancelled" });
    assert.equal(cancelled.title, "Cancelled");
    assert.equal(cancelled.when, "2026-10-06T13:00:00Z");
    assert.deepEqual(cancelled.buttons, []);
  });
});

describe("isOffline", () => {
  it("is true only for offline: true", () => {
    assert.equal(w.isOffline({ offline: true, agent_version: "x" }), true);
    assert.equal(w.isOffline({ offline: "true" }), false);
    assert.equal(w.isOffline({}), false);
    assert.equal(w.isOffline(null), false);
  });
});

describe("widget.js source", () => {
  it("starts with // @ts-check and fits the size budget", () => {
    assert.equal(source.split("\n")[0], "// @ts-check");
    assert.ok(Buffer.byteLength(source) <= 25600, `${Buffer.byteLength(source)} bytes`);
  });

  it("only requests the agent base URL", () => {
    const calls = source.match(/fetch\(/g) || [];
    const toAgent = source.match(/fetch\(base \+ "\/v1\/(widget\/chat|version)"/g) || [];
    assert.equal(toAgent.length, calls.length);
    assert.equal(/XMLHttpRequest|sendBeacon|WebSocket|EventSource|import\(/.test(source), false);
    const hosts = [...source.matchAll(/https?:\/\/([a-z0-9.-]+)/gi)].map((m) => m[1]);
    assert.deepEqual([...new Set(hosts)], ["agent.example.com"]);
  });

  it("never parses HTML", () => {
    assert.equal(/innerHTML|outerHTML|insertAdjacentHTML|document\.write/.test(source), false);
  });
});

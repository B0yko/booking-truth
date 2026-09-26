// @ts-check
/*
 * booking-truth chat widget (Apache-2.0): one file, no framework, no third-party requests.
 * Embed: <script src="https://agent.example.com/widget.js" data-agent="https://agent.example.com" async></script>
 * data-agent is the agent base URL, resolved against the page ("/" = page origin; default: the script's
 * origin). Requests go only there: GET /v1/version, POST /v1/widget/chat. Buttons send structured actions.
 * window.BookingTruthWidget.open() opens the panel from the page's own button.
 */
(function (/** @type {any} */ root, /** @type {() => any} */ factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else if (root && root.document) {
    root.BookingTruthWidget = api;
    api.boot(root.document);
  }
})(globalThis, function () {
  "use strict";

  /**
   * @typedef {{type: string, slot_id?: string, booking_uid?: string, zone?: string}} Action
   * @typedef {{name: string, email: string}} Lead
   * @typedef {{label: string, action: Action | null, start_utc: string | null}} QuickReply
   * @typedef {{ref: string, status: string, start_utc: string, end_utc: string, zone: string,
   *   local_label: string, action: string}} Booking
   * @typedef {{kind: string, reply: string, quickReplies: QuickReply[], booking: Booking | null,
   *   token: string | null, version: string | null, retry: boolean}} Result
   * @typedef {{label: string, action: Action, echo: string, confirm?: string, yes?: string}} CardButton
   * @typedef {{role: string, text?: string, booking?: Booking}} Item
   */

  const MAX_INPUT = 2000;
  /** @type {Record<string, string[]>} required ids per action type */
  const ACTIONS = { select_slot: ["slot_id"], reschedule: ["booking_uid"], cancel: ["booking_uid"], confirm_timezone: ["zone"] };
  const AGAIN = "Please try again in a moment.";
  const NEW = "Start a new conversation to continue.";
  /** @type {Record<string, string>} */
  const FALLBACK = {
    busy: "I'm still working on your previous message. " + AGAIN,
    limited: "You're sending messages quickly. " + AGAIN,
    ended: "This conversation has reached its limit. " + NEW,
    expired: "This chat session is no longer valid. " + NEW,
    too_long: "That message is too long. Please shorten it.",
    error: "Something went wrong on our side. " + AGAIN,
    network: "The booking assistant can't be reached. " + AGAIN,
  };

  /** @param {unknown} v @returns {string | null} */
  const str = (v) => (typeof v === "string" && v ? v : null);
  /** @param {unknown} v @returns {Record<string, any>} */
  const obj = (v) => (v && typeof v === "object" ? /** @type {any} */ (v) : {});

  /**
   * Agent base URL without a trailing slash; null unless http(s).
   * @param {string | null | undefined} dataAgent @param {string | null | undefined} scriptSrc @param {string} [pageUrl]
   */
  function agentBase(dataAgent, scriptSrc, pageUrl) {
    let url;
    try {
      const raw = (dataAgent || "").trim();
      if (raw) url = new URL(raw, pageUrl);
      else if (scriptSrc) url = new URL(new URL(scriptSrc, pageUrl).origin);
      else return null;
    } catch {
      return null;
    }
    return /^https?:$/.test(url.protocol) ? (url.origin + url.pathname).replace(/\/+$/, "") : null;
  }

  /** crypto.randomUUID, else a v4 UUID from getRandomValues (insecure contexts). @param {any} [c] @returns {string} */
  function newId(c = globalThis.crypto) {
    if (typeof c.randomUUID === "function") return c.randomUUID();
    const hex = (/** @type {string} */ d) => (+d ^ (c.getRandomValues(new Uint8Array(1))[0] & (15 >> (+d / 4)))).toString(16);
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, hex);
  }

  /** The browser's IANA zone, sent only as a hint. @returns {string | null} */
  function browserZone() {
    try {
      return Intl.DateTimeFormat().resolvedOptions().timeZone || null;
    } catch {
      return null;
    }
  }

  /** @param {string} name @param {string} email @returns {{lead: Lead | null, errors: Record<string, string>}} */
  function validateLead(name, email) {
    const n = String(name || "").trim().replace(/\s+/g, " ");
    const e = String(email || "").trim();
    /** @type {Record<string, string>} */
    const errors = {};
    if (!n || n.length > 100) errors.name = "Please enter your name (up to 100 characters).";
    if (e.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e)) errors.email = "Please enter a valid email address.";
    return { lead: errors.name || errors.email ? null : { name: n, email: e }, errors };
  }

  /** A known action type with its required ids, else null. @param {unknown} raw @returns {Action | null} */
  function normalizeAction(raw) {
    const a = obj(raw);
    const need = ACTIONS[a.type];
    if (!Array.isArray(need)) return null;
    /** @type {Record<string, string>} */
    const out = { type: a.type };
    for (const k of ["slot_id", "booking_uid", "zone"]) {
      const v = str(a[k]);
      if (v) out[k] = v;
      else if (need.includes(k)) return null;
    }
    return /** @type {Action} */ (out);
  }

  /**
   * Valid action: sent as the action. No action: the label is sent as a message. Invalid action: dropped.
   * @param {unknown} list @returns {QuickReply[]}
   */
  function normalizeQuickReplies(list) {
    /** @type {QuickReply[]} */
    const out = [];
    for (const q of Array.isArray(list) ? list.map(obj) : []) {
      const label = str(q.label);
      const action = q.action == null ? null : normalizeAction(q.action);
      if (label && (action || q.action == null)) out.push({ label: label.slice(0, 120), action, start_utc: str(q.start_utc) });
    }
    return out.slice(0, 12);
  }

  /** @param {unknown} raw @returns {Booking | null} */
  function normalizeBooking(raw) {
    const b = obj(raw);
    /** @type {Record<string, string>} */
    const out = {};
    for (const k of ["ref", "status", "start_utc", "end_utc", "zone", "local_label", "action"]) out[k] = str(b[k]) || "";
    out.action ||= "booked";
    return out.ref ? /** @type {Booking} */ (out) : null;
  }

  /** @type {Record<number, string>} */
  const KINDS = { 0: "network", 401: "expired", 403: "expired", 409: "busy", 410: "ended", 413: "too_long", 429: "limited" };

  /** What to show for a status (0 = no response) and JSON body. @param {number} status @param {unknown} data @returns {Result} */
  function normalizeResponse(status, data) {
    const d = obj(data);
    const reply = str(d.reply) || "";
    const booking = normalizeBooking(d.booking);
    const base = { reply, quickReplies: [], booking, token: str(d.session_token), version: str(d.agent_version) };
    if (status >= 200 && status < 300 && (reply || booking)) {
      return { ...base, kind: "ok", quickReplies: normalizeQuickReplies(d.quick_replies), retry: false };
    }
    const kind = KINDS[status] || (d.error === "input_too_long" ? "too_long" : "error");
    const retry = ["busy", "limited", "error", "network"].includes(kind);
    return { ...base, booking: null, kind, reply: reply || FALLBACK[kind], retry };
  }

  /**
   * A retry keeps the message id so the agent's dedupe answers instead of acting twice; after 409 or 429 the
   * agent did not take the message, so a new id is used. @param {string} kind @param {string} messageId @param {() => string} make
   */
  const retryMessageId = (kind, messageId, make) => (kind === "busy" || kind === "limited" ? make() : messageId);

  /**
   * The POST /v1/widget/chat body.
   * @param {{sessionId: string, lead: Lead, zone: string | null, token: string | null}} s
   * @param {{messageId: string, message?: string, action?: Action | null}} input
   */
  function buildRequest(s, input) {
    /** @type {Record<string, any>} */
    const body = { session_id: s.sessionId, message_id: input.messageId, channel: "widget" };
    body.lead = { email: s.lead.email, name: s.lead.name };
    if (s.zone) body.lead.timezone_hint = s.zone;
    if (input.action) body.action = input.action;
    else body.message = String(input.message || "").slice(0, MAX_INPUT);
    if (s.token) body.session_token = s.token;
    return body;
  }

  /** Plain data for the booking card; its buttons carry structured actions. @param {Booking} b */
  function bookingCard(b) {
    const off = b.status === "cancelled" || b.action === "cancelled";
    const uid = { booking_uid: b.ref };
    /** @type {CardButton[]} */
    const buttons = [
      { label: "Reschedule", action: { type: "reschedule", ...uid }, echo: "I'd like to reschedule." },
      { label: "Cancel", action: { type: "cancel", ...uid }, echo: "Please cancel my call.", confirm: "Cancel this call?", yes: "Yes, cancel" },
    ];
    return {
      title: off ? "Cancelled" : b.action === "rescheduled" ? "Rescheduled" : "Booked",
      when: b.local_label || b.start_utc,
      meta: [b.zone, "ref " + b.ref.slice(0, 8)].filter(Boolean).join(" · "),
      active: !off,
      buttons: off ? [] : buttons,
    };
  }

  /** True when GET /v1/version reports the scripted offline policy. @param {unknown} data */
  const isOffline = (data) => obj(data).offline === true;

  /** Set by mount; `BookingTruthWidget.open()` lets a page's own "Book a call" button open the panel. */
  let opener = () => {};

  // DOM -------------------------------------------------------------------------------------------------

  const CSS = `:host{all:initial}*{box-sizing:border-box}
.w{--a:#1f6f5c;--on:#fff;--bg:#fff;--fg:#1d2522;--mu:#5b6a64;--bd:#d9e1dd;--bot:#eef3f0;font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--fg)}
button,input,textarea{font:inherit;color:inherit}button{cursor:pointer}:disabled{opacity:.55}
:focus-visible{outline:2px solid var(--a);outline-offset:2px}[hidden]{display:none!important}
.l{position:fixed;right:16px;bottom:16px;z-index:2147483000;min-height:48px;padding:0 22px;border:0;border-radius:999px;background:var(--a);color:var(--on);font-weight:600;box-shadow:0 6px 20px #0004}
.p{position:fixed;right:16px;bottom:16px;z-index:2147483001;width:380px;height:min(640px,calc(100vh - 32px));display:flex;flex-direction:column;background:var(--bg);border:1px solid var(--bd);border-radius:16px;overflow:hidden;box-shadow:0 12px 40px #0005}
.hd{display:flex;align-items:center;gap:8px;padding:10px 12px 10px 16px;background:var(--a);color:var(--on)}
.hd div{flex:1;min-width:0}.hd h2{margin:0;font-size:16px}.hd p{margin:0;font-size:12px;opacity:.85}
.hd button{min-height:36px;padding:0 10px;border:1px solid #fff8;border-radius:8px;background:none;color:var(--on)}.hd :focus-visible{outline-color:var(--on)}
.bn{margin:0;padding:8px 16px;background:#fff3cf;color:#553f00;font-size:13px}
.lg{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:8px}
.m{max-width:85%;padding:9px 12px;border-radius:14px;white-space:pre-wrap;overflow-wrap:anywhere}
.u{align-self:flex-end;background:var(--a);color:var(--on);border-bottom-right-radius:4px}
.b{align-self:flex-start;background:var(--bot);border-bottom-left-radius:4px}
.s{align-self:center;font-size:13px;color:var(--mu);text-align:center}
.c{display:grid;gap:2px;padding:10px 12px;border:1px solid var(--bd);border-left:4px solid var(--a);border-radius:12px}
.c.x{border-left-color:var(--mu)}.c small{color:var(--mu)}.c .q{padding:6px 0 0;align-items:center}
.t{margin:0;padding:0 16px 6px;font-size:13px;color:var(--mu)}.t:empty,.q:empty,.e:empty{display:none}
.q{display:flex;flex-wrap:wrap;gap:6px;padding:0 14px 10px}
.qb{min-height:40px;padding:6px 14px;border:1px solid var(--a);border-radius:999px;background:var(--bg);text-align:left}
.f{display:grid;gap:6px;padding:14px 16px 16px;border-top:1px solid var(--bd)}.f p{margin:0}.f label{font-size:13px;font-weight:600}
input,textarea{width:100%;min-height:44px;padding:10px 12px;border:1px solid var(--bd);border-radius:10px;background:var(--bg);font-size:16px}
.e{color:#b3261e;font-size:13px}.n{font-size:12px;color:var(--mu)}
.go{min-height:44px;padding:0 16px;border:0;border-radius:10px;background:var(--a);color:var(--on);font-weight:600}
.cm{display:flex;gap:8px;padding:10px 12px 12px;border-top:1px solid var(--bd)}.cm textarea{flex:1;resize:none}
.p:focus{outline:0}.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
@media (max-width:480px){.p{inset:0;width:auto;height:100%;border:0;border-radius:0}}`;

  /** Build the widget in a Shadow DOM, wired to the agent at `base`. @param {Document} doc @param {string} base */
  function mount(doc, base) {
    const title = "Book a call";
    const key = "bt-widget:" + base;
    /** @type {{lead: Lead | null, sessionId: string, token: string | null, items: Item[], qr: QuickReply[], open: boolean}} */
    const st = { lead: null, sessionId: newId(), token: null, items: [], qr: [], open: false };
    /** @type {Storage | null} */
    let store = null;
    try {
      store = doc.defaultView && doc.defaultView.sessionStorage;
      const saved = obj(JSON.parse((store && store.getItem(key)) || "{}"));
      if (saved.lead && Array.isArray(saved.items) && Array.isArray(saved.qr)) Object.assign(st, saved);
    } catch {
      /* storage blocked or corrupt: start fresh */
    }
    const save = () => {
      try {
        if (store) store.setItem(key, JSON.stringify({ ...st, items: st.items.slice(-60) }));
      } catch {
        /* the chat works without storage */
      }
    };
    const zone = browserZone();
    let busy = false;

    /** An element; props are attributes except "text" and "on<event>". @type {(tag: string, props?: Record<string, any>, kids?: any[]) => any} */
    const E = (tag, props = {}, kids = []) => {
      const el = doc.createElement(tag);
      for (const [k, v] of Object.entries(props)) {
        if (v == null || v === false) continue;
        if (k === "text") el.textContent = v;
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? "" : String(v));
      }
      for (const kid of kids) if (kid) el.append(kid);
      return el;
    };
    const button = (/** @type {string} */ text, /** @type {() => void} */ onclick) =>
      E("button", { type: "button", class: "qb", text, onclick });

    const launcher = E("button", { type: "button", class: "l", "aria-controls": "bt-p", "aria-expanded": "false", text: title });
    const banner = E("p", { class: "bn", hidden: true, text: "Offline demo mode: replies come from a scripted assistant." });
    const log = E("div", { class: "lg", role: "log", "aria-live": "polite", "aria-label": "Conversation" });
    const typing = E("p", { class: "t", role: "status" });
    const qrBox = E("div", { class: "q", role: "group", "aria-label": "Suggested replies" });
    const nameIn = E("input", { id: "bt-n", autocomplete: "name", maxlength: 100, "aria-describedby": "bt-ne" });
    const mailIn = E("input", { id: "bt-m", type: "email", autocomplete: "email", maxlength: 254, "aria-describedby": "bt-me" });
    const nameErr = E("p", { id: "bt-ne", class: "e" });
    const mailErr = E("p", { id: "bt-me", class: "e" });
    const hint = zone && `Your browser time zone (${zone}) is shared as a hint. You can name another.`;
    const startForm = E("form", { class: "f", novalidate: true, "aria-label": "Your details" }, [
      E("p", { text: "Tell us who you are and we'll find a time that suits you." }),
      E("label", { for: "bt-n", text: "Name" }),
      nameIn,
      nameErr,
      E("label", { for: "bt-m", text: "Email" }),
      mailIn,
      mailErr,
      E("button", { type: "submit", class: "go", text: "Start chat" }),
      hint && E("p", { class: "n", text: hint }),
    ]);
    const input = E("textarea", { id: "bt-i", rows: 1, maxlength: MAX_INPUT, placeholder: "Type a message" });
    const sendBtn = E("button", { type: "submit", class: "go", text: "Send" });
    const composer = E("form", { class: "cm", hidden: true }, [E("label", { class: "sr", for: "bt-i", text: "Message" }), input, sendBtn]);
    const closeBtn = E("button", { type: "button", text: "Close", "aria-label": "Close chat" });
    const head = E("header", { class: "hd" }, [E("div", {}, [E("h2", { text: title }), E("p", { text: "Booking assistant" })]), closeBtn]);
    const parts = [head, banner, log, typing, qrBox, startForm, composer];
    const panel = E("section", { id: "bt-p", class: "p", role: "dialog", "aria-label": title, tabindex: -1, hidden: true }, parts);
    const wrap = E("div", { class: "w" }, [E("style", { text: CSS }), launcher, panel]);
    const host = doc.createElement("booking-truth-widget");
    host.attachShadow({ mode: "open" }).append(wrap);
    doc.body.append(host);

    const scrollDown = () => (log.scrollTop = log.scrollHeight);

    /** The booking card; only the latest card (`live`) keeps its buttons. @param {Booking} b @param {boolean} live */
    function cardEl(b, live) {
      const card = bookingCard(b);
      const lines = [E("strong", { text: card.title }), E("span", { text: card.when }), E("small", { text: card.meta })];
      const el = E("div", { class: card.active ? "c" : "c x", role: "group", "aria-label": "Booking: " + card.title }, lines);
      if (!live || !card.active) return el;
      const acts = E("div", { class: "q" });
      const act = (/** @type {CardButton} */ c) => {
        if (busy) return;
        acts.remove();
        press({ action: c.action }, c.echo);
      };
      const ask = (/** @type {CardButton} */ c) => {
        acts.replaceChildren(E("span", { text: c.confirm }), button(c.yes || c.label, () => act(c)), button("Keep it", plain));
        acts.children[1].focus();
      };
      const plain = () => acts.replaceChildren(...card.buttons.map((c) => button(c.label, () => (c.confirm ? ask(c) : act(c)))));
      plain();
      el.append(acts);
      return el;
    }

    /** @param {Item} item @param {boolean} live */
    function render(item, live) {
      if (item.role === "card" && item.booking) {
        if (live) log.querySelectorAll(".c .q").forEach((/** @type {Element} */ n) => n.remove());
        log.append(cardEl(item.booking, live));
      } else {
        const who = { user: "You: ", bot: "Assistant: " }[item.role] || "";
        const cls = { user: "m u", bot: "m b" }[item.role] || "m s";
        log.append(E("div", { class: cls }, [who && E("span", { class: "sr", text: who }), item.text || ""]));
      }
      scrollDown();
    }

    /** @param {Item} item */
    const add = (item) => {
      st.items.push(item);
      render(item, true);
      save();
    };

    /** @param {{label: string, run: () => void}[]} list */
    const setButtons = (list) => {
      qrBox.replaceChildren(...list.map((b) => button(b.label, b.run)));
      scrollDown();
    };

    /** @param {QuickReply[]} qr */
    function setQuickReplies(qr) {
      st.qr = qr;
      save();
      setButtons(qr.map((q) => ({ label: q.label, run: () => press(q.action ? { action: q.action } : { message: q.label }, q.label) })));
    }

    /** @param {boolean} on */
    function setBusy(on) {
      busy = on;
      sendBtn.disabled = on;
      log.setAttribute("aria-busy", String(on));
      typing.textContent = on ? "Assistant is typing…" : "";
    }

    function restart() {
      st.sessionId = newId();
      st.token = null;
      st.items = [];
      log.replaceChildren();
      const first = st.lead ? st.lead.name.split(" ")[0] : "";
      add({ role: "bot", text: `Hi ${first}! When would you like to talk? Name a day or a time of day and I'll find open slots.` });
      setQuickReplies([{ label: "Find me a time this week", action: null, start_utc: null }]);
      input.focus();
    }

    /**
     * Send a message or an action. `echo` is shown as the prospect's message; for an action it is never sent.
     * @param {{message?: string, action?: Action | null, messageId?: string}} what
     * @param {string} echo
     */
    async function send(what, echo) {
      if (busy || !st.lead) return;
      const messageId = what.messageId || newId();
      if (echo) add({ role: "user", text: echo });
      setQuickReplies([]);
      setBusy(true);
      const body = buildRequest({ sessionId: st.sessionId, lead: st.lead, zone, token: st.token }, { ...what, messageId });
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 60000);
      /** @type {Result} */
      let res;
      try {
        const headers = { "Content-Type": "application/json" };
        const r = await fetch(base + "/v1/widget/chat", { method: "POST", headers, body: JSON.stringify(body), credentials: "omit", signal: ctl.signal });
        res = normalizeResponse(r.status, await r.json().catch(() => null));
      } catch {
        res = normalizeResponse(0, null);
      }
      clearTimeout(timer);
      setBusy(false);
      if (res.token) st.token = res.token;
      if (res.kind === "ok") {
        if (res.reply) add({ role: "bot", text: res.reply });
        if (res.booking) add({ role: "card", booking: res.booking });
        return setQuickReplies(res.quickReplies);
      }
      add({ role: "sys", text: res.reply });
      if (res.retry) {
        const again = { ...what, messageId: retryMessageId(res.kind, messageId, newId) };
        setButtons([{ label: "Try again", run: () => press(again, "") }]);
      } else if (res.kind === "ended" || res.kind === "expired") {
        setButtons([{ label: "Start a new conversation", run: restart }]);
      } else if (what.message) input.value = what.message;
    }

    /** A button press: its button disappears, so focus moves to the panel. @type {typeof send} */
    const press = (what, echo) => {
      panel.focus();
      return send(what, echo);
    };

    /** @param {boolean} open @param {boolean} [focus] */
    function setOpen(open, focus = true) {
      st.open = open;
      panel.hidden = !open;
      launcher.hidden = open;
      launcher.setAttribute("aria-expanded", String(open));
      save();
      scrollDown();
      if (focus) (open ? (st.lead ? input : nameIn) : launcher).focus();
    }

    const showChat = () => {
      startForm.hidden = true;
      composer.hidden = false;
    };

    launcher.addEventListener("click", () => setOpen(true));
    closeBtn.addEventListener("click", () => setOpen(false));
    panel.addEventListener("keydown", (/** @type {KeyboardEvent} */ ev) => ev.key === "Escape" && setOpen(false));
    startForm.addEventListener("submit", (/** @type {Event} */ ev) => {
      ev.preventDefault();
      const { lead, errors } = validateLead(nameIn.value, mailIn.value);
      nameErr.textContent = errors.name || "";
      mailErr.textContent = errors.email || "";
      nameIn.setAttribute("aria-invalid", String(!!errors.name));
      mailIn.setAttribute("aria-invalid", String(!!errors.email));
      if (!lead) return (errors.name ? nameIn : mailIn).focus();
      st.lead = lead;
      showChat();
      restart();
    });
    composer.addEventListener("submit", (/** @type {Event} */ ev) => {
      ev.preventDefault();
      const text = input.value.trim();
      if (!text || busy) return;
      input.value = "";
      send({ message: text }, text);
    });
    input.addEventListener("keydown", (/** @type {KeyboardEvent} */ ev) => {
      if (ev.key !== "Enter" || ev.shiftKey || ev.isComposing) return;
      ev.preventDefault();
      composer.requestSubmit();
    });

    if (st.lead) {
      showChat();
      const last = st.items.map((i) => i.role).lastIndexOf("card");
      st.items.forEach((item, i) => render(item, i === last));
      setQuickReplies(st.qr);
    }
    if (st.open) setOpen(true, false);
    opener = () => setOpen(true);

    fetch(base + "/v1/version", { credentials: "omit" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => (banner.hidden = !isOffline(data)))
      .catch(() => {});
  }

  /** Mount once, when the DOM is ready. @param {Document} doc */
  function boot(doc) {
    const script = /** @type {HTMLScriptElement | null} */ (doc.currentScript || doc.querySelector("script[data-agent]"));
    const base = agentBase(script && script.getAttribute("data-agent"), script && script.src, doc.baseURI);
    if (!base) return console.warn("booking-truth widget: set data-agent to the agent's http(s) base URL");
    const start = () => doc.querySelector("booking-truth-widget") || mount(doc, base);
    if (doc.body) start();
    else doc.addEventListener("DOMContentLoaded", start, { once: true });
  }

  return {
    ...{ MAX_INPUT, agentBase, newId, browserZone, validateLead, normalizeAction, normalizeQuickReplies },
    ...{ normalizeBooking, normalizeResponse, retryMessageId, buildRequest, bookingCard, isOffline, boot, open: () => opener() },
  };
});

"""The channel-agnostic turn pipeline: ``AgentCore.handle_turn``.

One turn, in order (``docs/adr`` and the guard list in the README):

1. input validation: message length (413), widget session token (403), turn cap (410);
2. ``dedupe``: a repeated ``(session_id, message_id)`` returns the stored response;
3. ``lead_lock``: one turn per lead at a time, across channels (409 ``lead_busy``);
4. load the lead's zone state, the session history and the context for the model;
5. ``tz_resolver`` pre-scan of the prospect's text;
6. a structured action runs a code path; text runs the LLM tool loop (at most 8 model calls);
7. ``claim_ledger`` claim check of the reply (with offer grounding under ``fail_closed``), one repair, then a
   safe template; ``fail_closed``: a turn that ends with the calendar unavailable hands the conversation off;
8. ``rendered_confirmation``: code renders the confirmation line of every verified write of the turn from its
   ledger entry and puts it above the reply;
9. quick replies from state; 10. persist history, trace steps and the response;
11. CRM: ``crm_outbox`` queues verified writes; without it, the naive rule writes a meeting when the reply
    says "booked".

Every guard is evaluated from configuration at its hook point (the ``_hook_*`` methods here and in
:class:`~booking_truth.agent.tools.ToolExecutor`). A voice or other gateway calls
:meth:`AgentCore.handle_turn` once per utterance.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from booking_truth import __version__
from booking_truth.agent import render
from booking_truth.agent.guards import GuardConfig
from booking_truth.agent.guards.claim_check import CheckResult, LedgerFact, check_reply, guard_note
from booking_truth.agent.guards.fail_closed import (
    HANDOFF_SUMMARY,
    last_read_failed,
    offer_reference,
    unavailable_now,
)
from booking_truth.agent.guards.tz.resolver import get_resolver
from booking_truth.agent.loop import (
    AGENT_TEMPERATURE,
    MAX_TOKENS,
    DeclaredClaim,
    FinalAnswer,
    Usage,
    parse_final_answer,
    run_tool_loop,
)
from booking_truth.agent.models import (
    ActionIn,
    BookingView,
    ChatRequest,
    ChatResponse,
    ErrorBody,
    GuardInfo,
    QuickReply,
    UsageInfo,
)
from booking_truth.agent.naive import reply_says_booked
from booking_truth.agent.session_token import sign_session, token_hash, verify_session
from booking_truth.agent.tools import (
    HandoffNotifier,
    ToolExecutor,
    TurnContext,
    WriteRecord,
    business_days,
    valid_zone,
)
from booking_truth.agent.version import load_prompt
from booking_truth.calendars.base import CalendarAdapter
from booking_truth.config import Settings
from booking_truth.crm.base import ContactPayload, CrmAdapter, CrmOk, CrmSyncPayload, MeetingPayload
from booking_truth.llm.types import LLM, ChatMessage, LLMError, ToolCall
from booking_truth.store import Lead, LedgerEntry, OutboxItem, Store, normalize_email
from booking_truth.timeutil import Clock, iso_ms_z, iso_z, parse_iso
from booking_truth.trace.validate import trace_errors

QUICK_REPLY_SLOTS = 6
TRACE_SOURCE = f"booking-truth-agent/{__version__}"
#: How often the lead's lease is renewed while a turn holds it (the lease itself is
#: :data:`~booking_truth.store.repos.LEAD_LOCK_LEASE_S`, comfortably longer than this).
LEAD_LOCK_RENEW_S = 10.0


@dataclass
class AgentDeps:
    settings: Settings
    llm: LLM
    calendar: CalendarAdapter
    crm: CrmAdapter
    store: Store
    clock: Clock
    guards: GuardConfig
    version: str
    model_id: str
    handoffs: HandoffNotifier
    offline: bool
    crm_note: str | None = None
    prompt: str = field(default_factory=load_prompt)


@dataclass(frozen=True)
class TurnOutcome:
    status: int
    body: dict[str, Any]


class LeadBusy(Exception):
    """Another turn holds the lead's lock."""


@dataclass(frozen=True)
class Confirmation:
    """A verified write of the turn with its code-rendered confirmation (``rendered_confirmation``).
    ``when`` is the long label the line states (date, local time, IANA zone and UTC offset)."""

    write: WriteRecord
    entry: LedgerEntry
    when: str
    line: str


@dataclass
class TurnResult:
    reply: str
    claims: tuple[DeclaredClaim, ...] = ()
    messages: list[ChatMessage] = field(default_factory=list)
    user_text: str = ""
    blocked: bool = False
    repaired: bool = False
    #: The code path offered times to move ``booking_uid`` to (its quick replies are reschedule actions).
    reschedule_for: str | None = None
    #: The reply is the model's final answer (a code-rendered reply is never sent back for repair).
    from_model: bool = False

    def set_final(self, reply: str, claims: Sequence[DeclaredClaim] = ()) -> None:
        """Replace the reply, and the final answer in the turn's history, with ``reply``."""
        self.reply = reply
        self.claims = tuple(claims)
        body = {"reply": reply, "claims": [c.to_json() for c in claims]}
        final = ChatMessage.assistant(json.dumps(body, ensure_ascii=False))
        if self._has_final():
            self.messages[-1] = final
        else:
            self.messages.append(final)

    def _has_final(self) -> bool:
        last = self.messages[-1] if self.messages else None
        return last is not None and last.role == "assistant" and not last.tool_calls

    def add_call(self, name: str, args: dict[str, Any], output: Any) -> None:
        """Record a tool call that code made after the final answer, before that answer in the turn's
        history, so the model sees next turn that it happened."""
        call_id = f"code_{len(self.messages)}_{name}"
        content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        pair = [
            ChatMessage.assistant(None, [ToolCall(call_id, name, json.dumps(args, ensure_ascii=False))]),
            ChatMessage.tool(call_id, content, name=name),
        ]
        at = len(self.messages) - 1 if self._has_final() else len(self.messages)
        self.messages[at:at] = pair


def history_entry(message: ChatMessage) -> tuple[str, dict[str, Any]]:
    content: dict[str, Any] = {"content": message.content}
    if message.tool_calls:
        content["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls
        ]
    if message.tool_call_id is not None:
        content["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        content["name"] = message.name
    return message.role, content


def history_message(role: str, content: Any) -> ChatMessage | None:
    if not isinstance(content, dict) or role not in ("user", "assistant", "tool"):
        return None
    calls = [
        ToolCall(str(c["id"]), str(c["name"]), str(c.get("arguments") or "{}"))
        for c in content.get("tool_calls") or []
        if isinstance(c, dict) and c.get("id") and c.get("name")
    ]
    try:
        return ChatMessage(
            role=role,  # type: ignore[arg-type]
            content=content.get("content"),
            tool_calls=calls,
            tool_call_id=content.get("tool_call_id"),
            name=content.get("name"),
        )
    except ValueError:
        return None


class AgentCore:
    """Runs turns. Construct once per process with :class:`AgentDeps`."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    @property
    def store(self) -> Store:
        return self.deps.store

    @property
    def settings(self) -> Settings:
        return self.deps.settings

    def on(self, guard: str) -> bool:
        return self.deps.guards.on(guard)

    # Entry point ------------------------------------------------------------------------------------------

    async def handle_turn(self, req: ChatRequest, *, widget: bool = False) -> TurnOutcome:
        """One inbound message or action. ``widget``: the request came through the widget endpoint."""
        if widget and req.channel != "widget":
            req = req.model_copy(update={"channel": "widget"})
        settings = self.settings
        if req.message is not None and len(req.message) > settings.max_input_chars:
            detail = f"messages are limited to {settings.max_input_chars} characters"
            return TurnOutcome(413, ErrorBody(error="input_too_long", detail=detail).to_json())
        email = normalize_email(req.lead.email)
        session = self.store.sessions.get(req.session_id)
        if session is not None and session.lead_email != email:
            return TurnOutcome(
                403, ErrorBody(error="forbidden", detail="the session belongs to another lead").to_json()
            )
        token: str | None = None
        if req.channel == "widget":
            secret = settings.session_secret.get_secret_value()
            token = sign_session(secret, req.session_id, email)
            needs_token = session is not None or req.session_token is not None
            if needs_token and not verify_session(secret, req.session_token, req.session_id, email):
                return TurnOutcome(403, ErrorBody(error="invalid_session_token").to_json())
        if session is not None and (session.ended or session.turns >= settings.max_turns_per_session):
            return TurnOutcome(410, ErrorBody(error="session_ended", reply=render.SESSION_ENDED).to_json())

        stored = await self._hook_dedupe_begin(req)
        if stored is not None:
            return TurnOutcome(200, stored)
        try:
            async with self._hook_lead_lock(email):
                body = await self._run_turn(req, email, token)
        except LeadBusy:
            await self._hook_dedupe_abort(req)
            return TurnOutcome(409, ErrorBody(error="lead_busy", reply=render.LEAD_BUSY).to_json())
        except BaseException:
            await self._hook_dedupe_abort(req)
            raise
        await self._hook_dedupe_finish(req, body)
        return TurnOutcome(200, body)

    # Guard hooks around the turn ----------------------------------------------------------------------------

    async def _hook_dedupe_begin(self, req: ChatRequest) -> dict[str, Any] | None:
        """Hook for ``dedupe``: claim ``(session_id, message_id)``; return the stored response of a repeat
        (waiting for an in-flight twin). ``None``: run the turn.

        A twin still in flight is waited for (:meth:`~booking_truth.store.repos.MessagesRepo.await_done`,
        up to 30 s, polling every 50 ms); if it never finishes — it failed and discarded its row, or the
        wait timed out — the row is claimed again, which runs the turn fresh rather than waiting forever.
        The stored response is returned exactly as it was written, so every repeat of a message gets a
        byte-identical reply, in and out of an in-flight race, with no separate guard event: there is
        nothing to add to a reply that already is what it was the first time."""
        if not self.on("dedupe"):
            return None
        messages = self.store.messages
        claim = messages.claim_pending(req.session_id, req.message_id)
        if claim == "pending":
            stored = await messages.await_done(req.session_id, req.message_id)
            if stored is not None:
                return stored
            claim = messages.claim_pending(req.session_id, req.message_id)
        if claim == "done":
            row = messages.get(req.session_id, req.message_id)
            return row.response if row is not None else None
        return None

    async def _hook_dedupe_finish(self, req: ChatRequest, body: dict[str, Any]) -> None:
        """Hook for ``dedupe``: store the response for repeats of this message."""
        if self.on("dedupe"):
            self.store.messages.complete(req.session_id, req.message_id, body)

    async def _hook_dedupe_abort(self, req: ChatRequest) -> None:
        """Hook for ``dedupe``: drop the pending row of a turn that failed, so a retry runs again."""
        if self.on("dedupe"):
            self.store.messages.discard(req.session_id, req.message_id)

    @asynccontextmanager
    async def _hook_lead_lock(self, email: str) -> AsyncIterator[None]:
        """Hook for ``lead_lock``: hold the lead's lease for the turn (renewed while it runs); raise
        :class:`LeadBusy` when it cannot be taken within the wait.

        The lease is keyed by the normalised email (:meth:`~booking_truth.store.repos.LocksRepo.lead_key`),
        so a prospect writing on two channels at once (the widget and a webhook, two tabs) is serialised
        to one turn at a time; a turn that cannot take the lease within
        :data:`~booking_truth.store.repos.LEAD_LOCK_WAIT_S` gets ``409 lead_busy`` instead of running
        concurrently with the one that holds it. A background task renews the lease every
        :data:`LEAD_LOCK_RENEW_S` for as long as the turn runs, comfortably inside the lease itself, and is
        cancelled and awaited before the lease is released, win or fail."""
        if not self.on("lead_lock"):
            yield
            return
        locks = self.store.locks
        key = locks.lead_key(email)
        owner = uuid.uuid4().hex
        if not await locks.await_acquire(key, owner):
            raise LeadBusy
        renewal = asyncio.create_task(self._renew_lead_lock(key, owner))
        try:
            yield
        finally:
            renewal.cancel()
            with suppress(asyncio.CancelledError):
                await renewal
            locks.release(key, owner)

    async def _renew_lead_lock(self, key: str, owner: str) -> None:
        """Keeps ``lead_lock``'s lease alive while the turn runs; cancelled from
        :meth:`_hook_lead_lock`'s ``finally`` once it is done. Stops on its own if the lease was ever lost
        (it cannot have been, short of a bug, since only its owner extends it, but a lost lease is not
        worth renewing forever)."""
        while True:
            await asyncio.sleep(LEAD_LOCK_RENEW_S)
            if not self.store.locks.renew(key, owner):
                return

    def _hook_tz_prescan(self, text: str, ctx: TurnContext) -> str | None:
        """Hook for ``tz_resolver``: resolve a zone the prospect states in ``text`` before the model
        runs (design-agent.md SSB.5). A resolved statement updates the lead's zone (source ``stated``)
        and is returned to state back; an ambiguous one offers ``confirm_timezone`` quick replies
        (``ctx.state.tz_candidates``, rendered by :meth:`_quick_replies`) and states nothing back yet,
        so the model's own ``resolve_timezone`` call still asks the question. With no statement in
        ``text`` at all, a browser hint already in use as ``ctx.zone`` is stated back instead, every
        turn it is still the only zone known, until a stated or confirmed zone replaces it."""
        detected = get_resolver().prescan(text, now=ctx.now, horizon_days=self.settings.horizon_days)
        if detected is not None:
            phrase, resolution = detected
            if resolution.status == "resolved" and resolution.zone is not None:
                self._set_lead_zone(ctx, resolution.zone, source="stated")
                return render.zone_statement(resolution.zone, ctx.now)
            if resolution.status == "ambiguous":
                ctx.state.tz_candidates = list(resolution.candidates)
                ctx.state.event("tz_resolver", "prescan_ambiguous", phrase)
                return None
        if ctx.zone_source == "browser_hint":
            return render.zone_statement(ctx.zone, ctx.now, browser=True)
        return None

    def _set_lead_zone(self, ctx: TurnContext, zone: str, *, source: str) -> None:
        """Store a zone the pre-scan resolved and update ``ctx`` for the rest of the turn; a zone the
        prospect already confirmed stays confirmed when the pre-scan finds the same one again (the
        same rule :meth:`~booking_truth.agent.tools.ToolExecutor._set_zone` uses for the tool path)."""
        if zone == ctx.zone and ctx.zone_source == "confirmed":
            return
        self.store.leads.set_zone(ctx.lead_email, zone, source=source, confirmed=False)
        ctx.zone, ctx.zone_source = zone, source

    async def _hook_claim_check(
        self,
        ctx: TurnContext,
        result: TurnResult,
        *,
        executor: ToolExecutor,
        system: str,
        history: Sequence[ChatMessage],
        usage: Usage,
    ) -> TurnResult:
        """``claim_ledger``: check the reply's declared and detected claims against the ledger (and, with an
        offer reference, every offered time); a model reply that fails gets one repair call with a guard note,
        and a reply that still fails (or a code-rendered one) becomes the safe template."""
        facts = self._ledger_facts(ctx)
        offers = self._hook_offer_reference(ctx, [*history, *result.messages])
        check = self._check(ctx, result.reply, result.claims, facts, offers)
        if check.ok:
            return result
        ctx.state.event("claim_ledger", "claim_blocked", check.summary())
        if result.from_model:
            answer = await self._repair(
                ctx, result, check, facts, executor=executor, system=system, history=history, usage=usage
            )
            if answer is not None and answer.reply.strip():
                recheck = self._check(ctx, answer.reply, answer.claims, facts, offers)
                if recheck.ok:
                    ctx.state.event("claim_ledger", "repaired", answer.reply)
                    result.set_final(answer.reply, answer.claims)
                    result.repaired = True
                    return result
                ctx.state.event("claim_ledger", "repair_blocked", recheck.summary())
        reply = await self._safe_reply(ctx, executor, check, facts, result, history)
        ctx.state.event("claim_ledger", "safe_template", reply)
        result.set_final(reply)
        result.blocked = True
        return result

    def _hook_offer_reference(
        self, ctx: TurnContext, conversation: Sequence[ChatMessage]
    ) -> Sequence[datetime] | None:
        """``fail_closed``'s offer grounding inside the claim check: the starts every specific time in a reply
        must come from. They are the lead's latest successful slot list while it is fresh (the list a slot
        id can still be booked from) and the lead's bookings the calendar reported in ``conversation`` (the
        call being moved or cancelled); the claim check adds the ledger's verified starts. With no fresh
        list the reference is only those bookings, so a reply offers no times until a lookup succeeds.
        ``None`` (guard off): offers are not checked."""
        if not self.on("fail_closed"):
            return None
        ttl = float(self.settings.slot_ttl_seconds)
        return offer_reference(self.store.slot_lists.latest(ctx.lead_email, ttl_s=ttl), conversation)

    def _ledger_facts(self, ctx: TurnContext) -> list[LedgerFact]:
        claims, key = self.store.claims, self.deps.calendar.event_key
        facts = [
            LedgerFact(e.action, e.booking_ref, e.start_utc)
            for e in claims.current_bookings(ctx.lead_email, key)
        ]
        cancelled = claims.entries(ctx.lead_email, event_key=key, action="cancelled", status="verified")
        facts += [LedgerFact("cancelled", e.booking_ref, e.start_utc) for e in cancelled]
        return facts

    def _check(
        self,
        ctx: TurnContext,
        reply: str,
        claims: Sequence[DeclaredClaim],
        facts: Sequence[LedgerFact],
        offers: Sequence[datetime] | None,
    ) -> CheckResult:
        result = check_reply(
            reply,
            [(c.type, c.time) for c in claims],
            facts,
            zone=ctx.zone,
            now=ctx.now,
            host_zone=self.settings.host_timezone,
            offer_reference=offers,
        )
        ungrounded = [v.detail for v in result.violations if v.problem == "not_offered"]
        if ungrounded:
            ctx.state.event("fail_closed", "offer_blocked", "; ".join(ungrounded))
        return result

    async def _repair(
        self,
        ctx: TurnContext,
        result: TurnResult,
        check: CheckResult,
        facts: Sequence[LedgerFact],
        *,
        executor: ToolExecutor,
        system: str,
        history: Sequence[ChatMessage],
        usage: Usage,
    ) -> FinalAnswer | None:
        """One more model call: the turn so far without the rejected answer, then a guard note that says what
        was wrong, what the ledger shows and what the rejected answer was."""
        shown = [c.line for c in self._hook_confirmations(ctx)]
        note = guard_note(check, facts, zone=ctx.zone, draft=result.reply, shown=shown)
        turn = list(result.messages)
        if turn and turn[-1].role == "assistant" and not turn[-1].tool_calls:
            turn.pop()
        messages = [ChatMessage.system(system), *history, *turn, ChatMessage.system(note)]
        try:
            response = await self.deps.llm.chat(
                messages=messages,
                tools=list(executor.specs),
                temperature=AGENT_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                component="agent",
            )
        except LLMError as exc:
            ctx.state.event("claim_ledger", "repair_failed", f"{exc.kind}: {exc}")
            return None
        usage.add(response)
        if response.tool_calls:
            ctx.state.event("claim_ledger", "repair_failed", "the repair answer called a tool")
            return None
        return parse_final_answer(response.content)

    async def _safe_reply(
        self,
        ctx: TurnContext,
        executor: ToolExecutor,
        check: CheckResult,
        facts: Sequence[LedgerFact],
        result: TurnResult,
        history: Sequence[ChatMessage],
    ) -> str:
        """The safe template: nothing was booked (or changed, for a reschedule or cancel claim, or when the
        lead already had a booking), plus the next step: a hand-off while the calendar is unavailable (its
        last read in this conversation failed), else an offer to look for times.

        With ``rendered_confirmation`` and a verified write in this turn, the code-rendered confirmation line
        goes above the reply and says what happened, so the safe reply is only the sentence that follows
        it: saying that nothing was booked would contradict the line."""
        confirmed = self._hook_confirmations(ctx)
        if confirmed:
            return render.confirmation_follow_up(confirmed[-1].entry.action)
        written = {w.booking.ref for w in ctx.state.writes}
        had_booking = any(f.current and f.action != "cancelled" and f.ref not in written for f in facts)
        changed = bool(check.kinds & {"rescheduled", "cancelled"}) or had_booking
        first = render.SAFE_NOT_CHANGED if changed else render.SAFE_NOT_BOOKED
        if not (ctx.state.calendar_unavailable or last_read_failed([*history, *result.messages])):
            return f"{first} {render.NEXT_STEP_LOOK}"
        await self._handoff_once(ctx, executor, result)
        return f"{first} {render.NEXT_STEP_HANDOFF}"

    async def _handoff_once(self, ctx: TurnContext, executor: ToolExecutor, result: TurnResult) -> str | None:
        """Hand the conversation off because the calendar is unavailable, unless it already was handed off
        (by the model or by code, in any turn of this session). The call goes into the turn's history before
        the final answer, so the model sees next turn that it happened. Returns the hand-off's reference, or
        ``None`` when there already was one."""
        if self.store.handoffs.for_session(ctx.session_id):
            return None
        args = {"summary": HANDOFF_SUMMARY, "preferred_times_text": result.user_text[:300]}
        output = await executor.run("handoff_to_human", args)
        result.add_call("handoff_to_human", args, output)
        return str(output.get("reference") or "") if isinstance(output, dict) else ""

    async def _hook_unavailable_handoff(
        self, ctx: TurnContext, executor: ToolExecutor, result: TurnResult
    ) -> None:
        """``fail_closed``: while the calendar is unavailable (the turn's last calendar call was a read that
        failed even after its retry), the prospect is told so and a colleague follows up by email. When
        nobody has handed this conversation off yet, code makes the hand-off the model did not make, records
        it in the turn's history and adds the hand-off sentence to the reply."""
        if not unavailable_now(result.messages):
            return
        reference = await self._handoff_once(ctx, executor, result)
        if reference is None:
            return
        ctx.state.event("fail_closed", "handoff", reference)
        if render.NEXT_STEP_HANDOFF not in result.reply:
            result.set_final(f"{result.reply}\n\n{render.NEXT_STEP_HANDOFF}", result.claims)

    async def _hook_crm_outbox(self, ctx: TurnContext) -> None:
        """Hook for ``crm_outbox``: queue a validated CRM payload for each write of the turn that the
        calendar confirmed, never from the model's reply text. The HubSpot adapter and the worker that
        drains this queue land in a later milestone; this hook only writes the outbox row.

        Independent of ``claim_ledger`` (``agent.guards.REQUIRES`` ties only ``rendered_confirmation`` to
        it): a ``verified`` write's fields come from its read-back ledger entry, and a ``trusted`` write
        (``claim_ledger`` off, so there is no ledger row) is queued straight from the write itself, which
        already carries what the calendar returned. Only an ``unverified`` write (its read-back failed) is
        skipped: nothing reaches the CRM through this hook without a calendar-confirmed result. The
        payload is validated again by :meth:`~booking_truth.store.repos.OutboxRepo.enqueue`, so a bad one
        is refused rather than queued; that should not happen from this data, and a guard event marks it
        if it ever does. This runs after the turn's response is already built (design-agent.md SSB.12),
        so — like the naive rule's own CRM calls — each queued item gets its own trace step instead of a
        guard event on this turn's reply."""
        steps: list[dict[str, Any]] = []
        for write in ctx.state.writes:
            fields = self._crm_fields(ctx, write)
            if fields is None:
                continue
            booking_ref, start_utc, end_utc, zone = fields
            try:
                payload = CrmSyncPayload(
                    action=write.action,
                    lead_email=ctx.lead_email,
                    lead_name=ctx.lead_name,
                    booking_ref=booking_ref,
                    previous_ref=write.previous_ref if write.action == "rescheduled" else None,
                    zone=zone,
                    start_utc=start_utc,
                    end_utc=end_utc,
                )
            except ValidationError as exc:
                ctx.state.event("crm_outbox", "invalid_payload", f"{write.action} {booking_ref}: {exc}")
                continue
            item = self.store.outbox.enqueue(ctx.lead_email, "crm_sync", payload)
            ctx.state.event("crm_outbox", "enqueued", f"{write.action} {booking_ref}")
            steps += self._outbox_step(item, payload)
        if steps:
            self.store.trace_steps.extend(ctx.session_id, steps)

    def _crm_fields(self, ctx: TurnContext, write: WriteRecord) -> tuple[str, datetime, datetime, str] | None:
        """``(booking_ref, start_utc, end_utc, zone)`` for ``crm_outbox`` to sync ``write``, or ``None``
        when it is not a calendar-confirmed result yet. A ``verified`` write is read from its ledger
        entry (``None`` if that row is somehow missing: defensive, since a verified write always writes
        one); a ``trusted`` write (no ledger, ``claim_ledger`` off) is read from the write itself."""
        if write.status == "verified":
            entry = self._ledger_entry(ctx, write)
            if entry is None:
                return None
            return entry.booking_ref, entry.start_utc, entry.end_utc, valid_zone(entry.zone) or write.zone
        if write.status == "trusted":
            return write.booking.ref, write.booking.start, write.booking.end, write.zone
        return None

    def _outbox_step(self, item: OutboxItem, payload: CrmSyncPayload) -> list[dict[str, Any]]:
        ts = iso_ms_z(self.deps.clock.now())
        args = {"action": payload.action, "booking_ref": payload.booking_ref}
        return [
            {"ts": ts, "kind": "tool_call", "role": "agent", "name": "crm.outbox.enqueue", "args": args},
            {
                "ts": ts,
                "kind": "tool_result",
                "role": "tool",
                "name": "crm.outbox.enqueue",
                "ok": True,
                "output": {"item_id": item.id, "status": item.status},
                "error": None,
            },
        ]

    def _hook_confirmations(self, ctx: TurnContext) -> list[Confirmation]:
        """``rendered_confirmation``: the turn's verified writes in the order they happened, each with the
        confirmation line rendered from its ledger entry (date, local time, IANA zone with its UTC offset,
        reference). The zone is the one the entry was written in: the lead's zone, else the host's. Empty
        with the guard off, and for a write with no verified entry."""
        if not self.on("rendered_confirmation"):
            return []
        found: list[Confirmation] = []
        for write in ctx.state.writes:
            if write.status != "verified":
                continue
            entry = self._ledger_entry(ctx, write)
            if entry is None:
                continue
            zone = valid_zone(entry.zone) or write.zone
            when = render.long_label(entry.start_utc, zone)
            line = render.confirmation_line(entry.action, entry.start_utc, zone, entry.booking_ref)
            found.append(Confirmation(write, entry, when, line))
        return found

    def _ledger_entry(self, ctx: TurnContext, write: WriteRecord) -> LedgerEntry | None:
        """The verified ledger entry this turn's read-back recorded for ``write``."""
        entries = self.store.claims.entries(
            ctx.lead_email,
            event_key=self.deps.calendar.event_key,
            action=write.action,
            status="verified",
            session_id=ctx.session_id,
        )
        matching = [e for e in entries if e.booking_ref == write.booking.ref]
        return matching[-1] if matching else None

    def _add_confirmations(self, ctx: TurnContext, result: TurnResult) -> None:
        """Put the confirmation lines above the reply (in the response and in the history the model sees
        next turn). The reply's claims start with the lines' claims, which replace the reply's own claims of
        the same kinds."""
        confirmed = self._hook_confirmations(ctx)
        if not confirmed:
            return
        lines = "\n".join(c.line for c in confirmed)
        kinds = {c.entry.action for c in confirmed}
        claims = [DeclaredClaim(c.entry.action, c.when) for c in confirmed]
        claims += [c for c in result.claims if c.type not in kinds]
        result.set_final(f"{lines}\n\n{result.reply}", claims)
        for confirmation in confirmed:
            ctx.state.event("rendered_confirmation", "rendered", confirmation.line)

    # The turn ---------------------------------------------------------------------------------------------

    def _zone(self, lead: Lead, hint: str | None) -> tuple[str, str]:
        stored = valid_zone(lead.tz_zone)
        if stored is not None:
            return stored, "confirmed" if lead.tz_confirmed else (lead.tz_source or "stated")
        hinted = valid_zone(hint)
        if hinted is not None:
            return hinted, "browser_hint"
        return self.settings.host_timezone, "host_default"

    async def _run_turn(self, req: ChatRequest, email: str, token: str | None) -> dict[str, Any]:
        store = self.store
        store.sessions.get_or_create(
            req.session_id,
            lead_email=email,
            channel=req.channel,
            token_hash=token_hash(token) if token else None,
        )
        store.sessions.increment_turns(req.session_id)
        lead = store.leads.upsert(email, name=req.lead.name)
        zone, source = self._zone(lead, req.lead.timezone_hint)
        ctx = TurnContext(
            session_id=req.session_id,
            message_id=req.message_id,
            channel=req.channel,
            lead_email=email,
            lead_name=lead.name,
            zone=zone,
            zone_source=source,
            now=self.deps.clock.now(),
        )
        executor = ToolExecutor(self.deps, ctx)
        usage = Usage()
        history = self._history(req.session_id)
        system = ""
        statement: str | None = None
        if req.action is not None:
            result = await self._action(req.action, executor, usage, history)
        else:
            text = req.message or ""
            statement = self._hook_tz_prescan(text, ctx) if self.on("tz_resolver") else None
            system = self.system_prompt(ctx)
            result = await self._llm_turn(text, executor, usage, history, system)
        if self.on("claim_ledger"):
            result = await self._claim_ledger(
                ctx, executor, result, system=system or self.system_prompt(ctx), history=history, usage=usage
            )
        if self.on("fail_closed"):
            await self._hook_unavailable_handoff(ctx, executor, result)
        if statement and ctx.zone not in result.reply:
            result.reply = f"{result.reply}\n\n{statement}"
        self._add_confirmations(ctx, result)
        quick = self._quick_replies(ctx, result)
        booking = self._booking_view(ctx)
        response = ChatResponse(
            reply=result.reply,
            quick_replies=quick,
            booking=booking,
            agent_version=self.deps.version,
            guard=GuardInfo(blocked=result.blocked, repaired=result.repaired, events=list(ctx.state.events)),
            usage=UsageInfo(**usage.to_json()),
            session_token=token,
        )
        body = response.to_json()
        self._persist(req, ctx, result, body)
        if self.on("crm_outbox"):
            await self._hook_crm_outbox(ctx)
        else:
            await self._naive_crm(ctx, result.reply)
        return body

    async def _claim_ledger(
        self,
        ctx: TurnContext,
        executor: ToolExecutor,
        result: TurnResult,
        *,
        system: str,
        history: Sequence[ChatMessage],
        usage: Usage,
    ) -> TurnResult:
        """``claim_ledger`` on the turn's reply. A write whose read-back failed makes the reply the
        unconfirmed template, with a hand-off to a colleague; otherwise the reply goes through the claim
        check."""
        unverified = [w for w in ctx.state.writes if w.status == "unverified"]
        if not unverified:
            return await self._hook_claim_check(
                ctx, result, executor=executor, system=system, history=history, usage=usage
            )
        if not ctx.state.handoffs:
            write = unverified[0]
            when = render.long_label(write.booking.start, write.zone)
            await executor.handoff(
                f"A calendar write ({write.action}, {when}, reference {write.booking.ref}) could not be "
                "confirmed by a read-back; please check the calendar and confirm it with the prospect.",
                result.user_text[:300],
            )
        result.set_final(render.UNCONFIRMED)
        return result

    # Model path ---------------------------------------------------------------------------------------------

    def context_block(self, ctx: TurnContext) -> dict[str, Any]:
        local = ctx.now.astimezone(ZoneInfo(ctx.zone))
        active: list[dict[str, str]] = []
        if self.on("claim_ledger"):
            refs = set(self.store.widget_bookings.refs(ctx.session_id)) if ctx.channel == "widget" else None
            for entry in self.store.claims.current_bookings(ctx.lead_email, self.deps.calendar.event_key):
                if refs is not None and entry.booking_ref not in refs:
                    continue
                active.append(
                    {
                        "booking_uid": entry.booking_ref,
                        "label": render.slot_label(entry.start_utc, ctx.zone, now=ctx.now),
                        "start_utc": iso_z(entry.start_utc),
                    }
                )
        return {
            "today": local.date().isoformat(),
            "weekday": local.strftime("%A"),
            "zone": ctx.zone,
            "zone_source": ctx.zone_source,
            "host_zone": self.settings.host_timezone,
            "meeting_minutes": self.settings.slot_minutes,
            "lead_name": ctx.lead_name,
            "active_bookings": active,
            "channel": ctx.channel,
        }

    def system_prompt(self, ctx: TurnContext) -> str:
        block = json.dumps(self.context_block(ctx), ensure_ascii=False)
        return f"{self.deps.prompt.rstrip()}\n\n<context>\n{block}\n</context>\n"

    async def _llm_turn(
        self, text: str, executor: ToolExecutor, usage: Usage, history: list[ChatMessage], system: str
    ) -> TurnResult:
        user = ChatMessage.user(text)
        try:
            loop = await run_tool_loop(
                self.deps.llm, system=system, history=[*history, user], executor=executor, usage=usage
            )
        except LLMError as exc:
            executor.state.event("agent", "llm_error", f"{exc.kind}: {exc}")
            reply = self._describe_writes(executor.ctx) or render.LLM_UNAVAILABLE
            return TurnResult(reply=reply, messages=[user, ChatMessage.assistant(reply)], user_text=text)
        if loop.answer is None:
            reply = self._describe_writes(executor.ctx) or render.TOOL_LOOP_EXHAUSTED
            executor.state.event("agent", "tool_loop_exhausted", str(usage.calls))
            return TurnResult(
                reply=reply, messages=[user, *loop.messages, ChatMessage.assistant(reply)], user_text=text
            )
        if loop.answer.reply:
            return TurnResult(
                reply=loop.answer.reply,
                claims=loop.answer.claims,
                messages=[user, *loop.messages],
                user_text=text,
                from_model=True,
            )
        reply = self._describe_writes(executor.ctx) or render.TOOL_LOOP_EXHAUSTED
        return TurnResult(
            reply=reply, claims=loop.answer.claims, messages=[user, *loop.messages], user_text=text
        )

    def _describe_writes(self, ctx: TurnContext) -> str | None:
        """A code-rendered reply for the turn's last successful write, when the model gave none."""
        if not ctx.state.writes:
            return None
        return self._write_reply(ctx, ctx.state.writes[-1])

    def _write_reply(self, ctx: TurnContext, write: WriteRecord) -> str:
        """The code-rendered reply about a write of this turn. When the write's confirmation line is rendered
        above the reply (``rendered_confirmation``), only the sentence that follows the line, which already
        states the date, time, zone and reference."""
        if write.status == "unverified":
            return render.UNCONFIRMED
        if any(c.write is write for c in self._hook_confirmations(ctx)):
            return render.confirmation_follow_up(write.action)
        start, ref = write.booking.start, write.booking.ref
        if write.action == "booked":
            return render.booked_text(start, write.zone, ref)
        if write.action == "rescheduled":
            return render.rescheduled_text(start, write.zone, ref)
        return render.cancelled_text(start, write.zone)

    # Code paths for structured actions ------------------------------------------------------------------

    async def _action(
        self, action: ActionIn, executor: ToolExecutor, usage: Usage, history: list[ChatMessage]
    ) -> TurnResult:
        ctx = executor.ctx
        if executor.mode != "guarded" and action.type == "reschedule":
            system = self.system_prompt(ctx)
            return await self._llm_turn(
                "I'd like to reschedule my booking.", executor, usage, history, system
            )
        if action.type == "confirm_timezone":
            zone = valid_zone(action.zone)
            if zone is None:
                text = f"I don't know the time zone {action.zone!r}. Which city are you in?"
                return self._coded(f"[{action.describe()}]", text)
            self.store.leads.set_zone(ctx.lead_email, zone, source="confirmed", confirmed=True)
            ctx.zone, ctx.zone_source = zone, "confirmed"
            system = self.system_prompt(ctx)
            return await self._llm_turn(f"My time zone is {zone}.", executor, usage, history, system)
        if executor.mode != "guarded" and action.type == "select_slot":
            return self._coded(f"[{action.describe()}]", render.unsupported_action_text())
        if action.type == "select_slot":
            return await self._select_slot(str(action.slot_id), executor)
        if action.type == "reschedule":
            uid = str(action.booking_uid)
            if action.slot_id and executor.mode == "guarded":
                return await self._reschedule_action(uid, action.slot_id, executor)
            return await self._offer_action(
                executor,
                user_text="I'd like to reschedule my booking.",
                lead_in="Sure, here are some open times to move your call to",
                reschedule_for=uid,
            )
        return await self._cancel_action(str(action.booking_uid), executor)

    def _coded(
        self,
        user_text: str,
        reply: str,
        *,
        calls: Sequence[tuple[str, dict[str, Any], Any]] = (),
        claims: Sequence[DeclaredClaim] = (),
        reschedule_for: str | None = None,
    ) -> TurnResult:
        """A code-path turn, written to history the way the tool loop would write it."""
        messages = [ChatMessage.user(user_text)]
        for index, (name, args, result) in enumerate(calls):
            call_id = f"code_{index}_{name}"
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            messages.append(ChatMessage.assistant(None, [ToolCall(call_id, name, json.dumps(args))]))
            messages.append(ChatMessage.tool(call_id, content, name=name))
        final = {"reply": reply, "claims": [c.to_json() for c in claims]}
        messages.append(ChatMessage.assistant(json.dumps(final, ensure_ascii=False)))
        return TurnResult(
            reply=reply,
            claims=tuple(claims),
            messages=messages,
            user_text=user_text,
            reschedule_for=reschedule_for,
        )

    def _label(self, ctx: TurnContext, start: Any) -> str:
        return render.slot_label(start, ctx.zone, now=ctx.now)

    async def _select_slot(self, slot_id: str, executor: ToolExecutor) -> TurnResult:
        ctx = executor.ctx
        slot = self.store.slot_lists.find_slot(
            ctx.lead_email, slot_id, ttl_s=float(self.settings.slot_ttl_seconds)
        )
        user_text = f"I'll take {slot['label']}." if slot else f"[select_slot slot_id={slot_id}]"
        args = {"slot_id": slot_id}
        result = await executor.run("book_slot", args)
        calls: list[tuple[str, dict[str, Any], Any]] = [("book_slot", args, result)]
        if isinstance(result, dict) and result.get("booked") is True:
            write = ctx.state.writes[-1]
            reply = self._write_reply(ctx, write)
            claim = DeclaredClaim("booked", self._label(ctx, write.booking.start))
            return self._coded(user_text, reply, calls=calls, claims=[claim])
        if isinstance(result, dict) and result.get("reason") == "already_booked":
            existing = result.get("existing") or {}
            if slot is not None:
                reply = (
                    f"You already have a call booked for {existing.get('label')} ({ctx.zone}). "
                    f"Would you like me to move it to {slot['label']} instead?"
                )
            else:
                reply = f"You already have a call booked for {existing.get('label')} ({ctx.zone})."
            return self._coded(user_text, reply, calls=calls)
        if isinstance(result, dict) and result.get("booked") == "unconfirmed":
            return self._coded(user_text, render.UNCONFIRMED, calls=calls)
        if isinstance(result, dict) and result.get("reason") in ("slot_taken", "unknown_or_expired_slot"):
            lead_in = (
                render.slot_taken_text() if result["reason"] == "slot_taken" else render.expired_slot_text()
            )
            return await self._offer_action(
                executor,
                user_text=user_text,
                lead_in=lead_in + " Here are other open times",
                calls=calls,
                after_list=True,
            )
        return self._coded(user_text, render.calendar_error_text(), calls=calls)

    async def _offer_action(
        self,
        executor: ToolExecutor,
        *,
        user_text: str,
        lead_in: str,
        calls: list[tuple[str, dict[str, Any], Any]] | None = None,
        reschedule_for: str | None = None,
        after_list: bool = False,
    ) -> TurnResult:
        """Look up and offer slots by code: the dates of the lead's last slot list (``after_list``), else the
        next 5 business days; an empty range moves on to the next one, up to three times."""
        ctx = executor.ctx
        calls = list(calls or [])
        local_today = ctx.now.astimezone(ZoneInfo(ctx.zone)).date()
        days = business_days(local_today, 5)
        first, last = days[0], days[-1]
        if after_list:
            previous = self.store.slot_lists.latest(ctx.lead_email)
            if previous is not None and previous.slots:
                dates = sorted({str(s.get("local_date")) for s in previous.slots if s.get("local_date")})
                try:
                    first = max(date.fromisoformat(dates[0]), local_today)
                    last = max(date.fromisoformat(dates[-1]), first)
                except ValueError:
                    pass
        span = (last - first).days + 1
        for _ in range(4):
            args = {"from_date": first.isoformat(), "to_date": last.isoformat()}
            result = await executor.run("find_slots", args)
            calls.append(("find_slots", args, result))
            slots = result.get("slots") if isinstance(result, dict) else None
            if slots:
                labels = [str(s["label"]) for s in slots[:4]]
                reply = render.offer_text(labels, ctx.zone, lead_in=lead_in)
                claims = [DeclaredClaim("offered", label) for label in labels]
                return self._coded(
                    user_text, reply, calls=calls, claims=claims, reschedule_for=reschedule_for
                )
            if not isinstance(result, dict) or result.get("unavailable") or "error" in result:
                if not self.store.handoffs.for_session(ctx.session_id):
                    handoff = await executor.handoff(
                        "The prospect asked for times, but the calendar is unavailable.", ""
                    )
                    calls.append(
                        (
                            "handoff_to_human",
                            {"summary": handoff.summary, "preferred_times_text": ""},
                            {"handoff": "created", "reference": f"H{handoff.id}"},
                        )
                    )
                return self._coded(user_text, render.unavailable_text(), calls=calls)
            first, last = first + timedelta(days=span), last + timedelta(days=span)
        return self._coded(user_text, render.no_slots_text(ctx.zone), calls=calls)

    async def _reschedule_action(self, uid: str, slot_id: str, executor: ToolExecutor) -> TurnResult:
        ctx = executor.ctx
        args = {"booking_uid": uid, "slot_id": slot_id}
        result = await executor.run("reschedule_booking", args)
        calls: list[tuple[str, dict[str, Any], Any]] = [("reschedule_booking", args, result)]
        user_text = "Please move my booking to the time I picked."
        if isinstance(result, dict) and result.get("rescheduled") is True:
            write = ctx.state.writes[-1]
            reply = self._write_reply(ctx, write)
            claim = DeclaredClaim("rescheduled", self._label(ctx, write.booking.start))
            return self._coded(user_text, reply, calls=calls, claims=[claim])
        if isinstance(result, dict) and result.get("rescheduled") == "unconfirmed":
            return self._coded(user_text, render.UNCONFIRMED, calls=calls)
        reason = result.get("reason") if isinstance(result, dict) else None
        if reason == "not_allowed":
            return self._coded(user_text, render.not_allowed_text(), calls=calls)
        if reason in ("slot_taken", "unknown_or_expired_slot"):
            lead_in = render.slot_taken_text() if reason == "slot_taken" else render.expired_slot_text()
            return await self._offer_action(
                executor,
                user_text=user_text,
                lead_in=lead_in.replace("nothing is booked", "nothing was changed")
                + " Here are other open times",
                calls=calls,
                reschedule_for=uid,
                after_list=True,
            )
        return self._coded(user_text, render.calendar_error_text(changed=True), calls=calls)

    async def _cancel_action(self, uid: str, executor: ToolExecutor) -> TurnResult:
        ctx = executor.ctx
        args = {"booking_uid": uid, "reason": "Cancelled by the prospect"}
        result = await executor.run("cancel_booking", args)
        calls: list[tuple[str, dict[str, Any], Any]] = [("cancel_booking", args, result)]
        user_text = "Please cancel my booking."
        if isinstance(result, dict) and result.get("cancelled") is True:
            write = ctx.state.writes[-1]
            reply = self._write_reply(ctx, write)
            claim = DeclaredClaim("cancelled", self._label(ctx, write.booking.start))
            return self._coded(user_text, reply, calls=calls, claims=[claim])
        if isinstance(result, dict) and result.get("cancelled") == "unconfirmed":
            return self._coded(user_text, render.UNCONFIRMED, calls=calls)
        reason = result.get("reason") if isinstance(result, dict) else None
        if reason == "not_allowed":
            return self._coded(user_text, render.not_allowed_text(), calls=calls)
        if reason == "already_cancelled":
            return self._coded(
                user_text, "That call was already cancelled. Nothing is booked for it now.", calls=calls
            )
        return self._coded(
            user_text,
            "I'm sorry, the calendar didn't accept the cancellation, so your call is still booked.",
            calls=calls,
        )

    # Response parts -----------------------------------------------------------------------------------------

    def _quick_replies(self, ctx: TurnContext, result: TurnResult) -> list[QuickReply]:
        state = ctx.state
        replies: list[QuickReply] = []
        if state.reschedule_offer is not None:
            offer = state.reschedule_offer
            slot = self.store.slot_lists.find_slot(
                ctx.lead_email, offer["slot_id"], ttl_s=float(self.settings.slot_ttl_seconds)
            )
            label = slot["label"] if slot else "the new time"
            replies.append(
                QuickReply(
                    label=f"Move it to {label}",
                    action={
                        "type": "reschedule",
                        "booking_uid": offer["booking_uid"],
                        "slot_id": offer["slot_id"],
                    },
                )
            )
            replies.append(QuickReply(label="Keep my current time", action=None))
            return replies
        shown = state.shown
        if shown is not None and self.on("slot_ids"):
            mentioned = [s for s in shown.slots if s["label"] in result.reply]
            slots = mentioned or ([] if state.writes else shown.slots)
            for slot in slots[:QUICK_REPLY_SLOTS]:
                if result.reschedule_for:
                    action = {
                        "type": "reschedule",
                        "booking_uid": result.reschedule_for,
                        "slot_id": slot["slot_id"],
                    }
                else:
                    action = {"type": "select_slot", "slot_id": slot["slot_id"]}
                replies.append(QuickReply(label=slot["label"], action=action, start_utc=slot["start_utc"]))
        for zone in state.tz_candidates:
            label = f"{zone} ({render.utc_offset(ctx.now, zone)})"
            replies.append(QuickReply(label=label, action={"type": "confirm_timezone", "zone": zone}))
        return replies

    def _hook_booking_write(self, ctx: TurnContext) -> WriteRecord | None:
        """The write the ``booking`` field shows: the turn's last successful write; with ``claim_ledger``, a
        write whose read-back failed (``unverified``) is not shown."""
        writes = ctx.state.writes
        if self.on("claim_ledger"):
            writes = [w for w in writes if w.status in ("verified", "trusted")]
        return writes[-1] if writes else None

    def _booking_view(self, ctx: TurnContext) -> BookingView | None:
        write = self._hook_booking_write(ctx)
        if write is None:
            return None
        booking = write.booking
        return BookingView(
            ref=booking.ref,
            status=booking.status,
            start_utc=iso_z(booking.start),
            end_utc=iso_z(booking.end),
            zone=write.zone,
            local_label=render.long_label(booking.start, write.zone),
            action=write.action,
        )

    # Persistence ------------------------------------------------------------------------------------------

    def _history(self, session_id: str) -> list[ChatMessage]:
        messages = []
        for entry in self.store.history.for_session(session_id):
            message = history_message(entry.role, entry.content)
            if message is not None:
                messages.append(message)
        return messages

    def _persist(self, req: ChatRequest, ctx: TurnContext, result: TurnResult, body: dict[str, Any]) -> None:
        store = self.store
        store.history.extend(req.session_id, [history_entry(m) for m in result.messages])
        now = iso_ms_z(self.deps.clock.now())
        user_step: dict[str, Any] = {
            "ts": iso_ms_z(ctx.now),
            "kind": "message",
            "role": "user",
            "name": None,
            "content": req.message if req.message is not None else result.user_text,
            "args": {"message_id": req.message_id, "channel": req.channel},
        }
        if req.action is not None:
            user_step["args"]["action"] = req.action.model_dump(exclude_none=True)
        agent_step = {
            "ts": now,
            "kind": "message",
            "role": "agent",
            "name": None,
            "content": result.reply,
            "args": {"message_id": req.message_id},
            "ok": True,
            "output": {
                "claims": [c.to_json() for c in result.claims],
                "quick_replies": body.get("quick_replies") or [],
                "booking": body.get("booking"),
                "guard": body.get("guard"),
                "usage": body.get("usage"),
            },
        }
        store.trace_steps.extend(req.session_id, [user_step, *ctx.state.steps, agent_step])

    # Naive CRM rule ---------------------------------------------------------------------------------------

    def _last_book_start(self, ctx: TurnContext) -> Any:
        if ctx.state.last_book_start is not None:
            return ctx.state.last_book_start
        for step in reversed(self.store.trace_steps.for_session(ctx.session_id)):
            if step.get("kind") == "tool_call" and step.get("name") == "book":
                raw = (step.get("args") or {}).get("start_iso")
                try:
                    return parse_iso(str(raw).replace(" ", "T")) if raw else None
                except ValueError:
                    return None
        return None

    async def _naive_crm(self, ctx: TurnContext, reply: str) -> None:
        """Without ``crm_outbox``: write a meeting when the reply says "booked", at the last book call's
        time."""
        if not reply_says_booked(reply):
            return
        start = self._last_book_start(ctx)
        if start is None:
            return
        crm = self.deps.crm
        ref = ctx.state.writes[-1].booking.ref if ctx.state.writes else "unverified"
        steps: list[list[dict[str, Any]]] = []
        try:
            contact_payload = ContactPayload(email=ctx.lead_email, name=ctx.lead_name)
            contact = await crm.upsert_contact(contact_payload)
            steps.append(self._crm_step("crm.upsert_contact", {"email": contact_payload.email}, contact))
            if isinstance(contact, CrmOk):
                meeting_payload = MeetingPayload(
                    contact_id=contact.id,
                    booking_ref=ref,
                    title="Intro call",
                    start_utc=start,
                    end_utc=start + timedelta(minutes=self.settings.slot_minutes),
                    outcome="SCHEDULED",
                )
                meeting = await crm.create_meeting(meeting_payload)
                steps.append(
                    self._crm_step(
                        "crm.create_meeting", {"start_utc": iso_z(start), "booking_ref": ref}, meeting
                    )
                )
        except ValidationError:
            return
        if steps:
            flat = [s for pair in steps for s in pair]
            self.store.trace_steps.extend(ctx.session_id, flat)

    def _crm_step(self, name: str, args: dict[str, Any], result: Any) -> list[dict[str, Any]]:
        ts = iso_ms_z(self.deps.clock.now())
        ok = isinstance(result, CrmOk)
        output = (
            {"id": result.id}
            if isinstance(result, CrmOk)
            else {"reason": result.reason, "detail": result.detail}
        )
        return [
            {"ts": ts, "kind": "tool_call", "role": "agent", "name": name, "args": args},
            {
                "ts": ts,
                "kind": "tool_result",
                "role": "tool",
                "name": name,
                "ok": ok,
                "output": output,
                "error": None if ok else output.get("reason"),
            },
        ]

    # Trace ------------------------------------------------------------------------------------------------

    def session_trace(self, session_id: str) -> dict[str, Any] | None:
        """The session's ``agent-trace/v1`` record, validated; ``None`` for an unknown session."""
        session = self.store.sessions.get(session_id)
        if session is None:
            return None
        steps = self.store.trace_steps.for_session(session_id)
        final_text: str | None = None
        claims: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for step in steps:
            if step.get("kind") == "message" and step.get("role") == "agent":
                final_text = step.get("content")
                output = step.get("output") or {}
                claims = [
                    {
                        "type": "offered_slots" if c.get("type") == "offered" else str(c.get("type")),
                        "subject": {"time": c.get("time")},
                    }
                    for c in output.get("claims") or []
                    if isinstance(c, dict) and c.get("type")
                ]
                guard = output.get("guard") or {}
                events += [
                    {"message_id": (step.get("args") or {}).get("message_id"), **e}
                    for e in guard.get("events") or []
                    if isinstance(e, dict)
                ]
        trace = {
            "schema": "agent-trace/v1",
            "trace_id": f"agent-session/{session_id}",
            "source": TRACE_SOURCE,
            "task": {
                "id": session_id,
                "domain": "booking",
                "instruction": "Conversation with the booking agent",
            },
            "steps": steps,
            "final_claim": {"text": final_text, "claims": claims},
            "ground_truth": {"outcome": "unknown", "checked_by": "none"},
            "meta": {
                "agent_version": self.deps.version,
                "model": self.deps.model_id,
                "guards": self.deps.guards.label,
                "channel": session.channel,
                "turns": session.turns,
                "guard_events": events,
                "claims_source": "agent_declared",
            },
        }
        errors = trace_errors(trace)
        if errors:
            raise ValueError("session trace is invalid: " + "; ".join(errors[:5]))
        return trace

"""The channel-agnostic turn pipeline: ``AgentCore.handle_turn``.

One turn, in order (``docs/adr`` and the guard list in the README):

1. input validation: message length (413), widget session token (403), turn cap (410);
2. ``dedupe``: a repeated ``(session_id, message_id)`` returns the stored response;
3. ``lead_lock``: one turn per lead at a time, across channels (409 ``lead_busy``);
4. load the lead's zone state, the session history and the context for the model;
5. ``tz_resolver`` pre-scan of the prospect's text;
6. a structured action runs a code path; text runs the LLM tool loop (at most 8 model calls);
7. ``claim_ledger`` claim check of the reply (with offer grounding under ``fail_closed``), one repair, then a
   safe template;
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

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from booking_truth import __version__
from booking_truth.agent import render
from booking_truth.agent.guards import GuardConfig
from booking_truth.agent.guards.claim_check import CheckResult, LedgerFact, check_reply, guard_note
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
from booking_truth.crm.base import ContactPayload, CrmAdapter, CrmOk, MeetingPayload
from booking_truth.llm.types import LLM, ChatMessage, LLMError, ToolCall
from booking_truth.store import Lead, LedgerEntry, Store, normalize_email
from booking_truth.timeutil import Clock, iso_ms_z, iso_z, parse_iso
from booking_truth.trace.validate import trace_errors

QUICK_REPLY_SLOTS = 6
TRACE_SOURCE = f"booking-truth-agent/{__version__}"


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
        if self.messages and self.messages[-1].role == "assistant" and not self.messages[-1].tool_calls:
            self.messages[-1] = final
        else:
            self.messages.append(final)


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


CALENDAR_READS = frozenset({"find_slots", "list_my_bookings"})


def last_read_failed(messages: Sequence[ChatMessage]) -> bool:
    """Whether the last calendar read (``find_slots`` or ``list_my_bookings``) in ``messages`` failed."""
    for message in reversed(messages):
        if message.role != "tool" or message.name not in CALENDAR_READS:
            continue
        content = message.content or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return content.startswith("Error: calendar")
        return isinstance(data, dict) and bool(data.get("unavailable"))
    return False


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
        (waiting for an in-flight twin). ``None``: run the turn."""
        return None

    async def _hook_dedupe_finish(self, req: ChatRequest, body: dict[str, Any]) -> None:
        """Hook for ``dedupe``: store the response for repeats of this message."""
        return None

    async def _hook_dedupe_abort(self, req: ChatRequest) -> None:
        """Hook for ``dedupe``: drop the pending row of a turn that failed, so a retry runs again."""
        return None

    @asynccontextmanager
    async def _hook_lead_lock(self, email: str) -> AsyncIterator[None]:
        """Hook for ``lead_lock``: hold the lead's lease for the turn (renewed while it runs); raise
        :class:`LeadBusy` when it cannot be taken within the wait."""
        yield

    def _hook_tz_prescan(self, text: str, ctx: TurnContext) -> str | None:
        """Hook for ``tz_resolver``: resolve a zone the prospect states in ``text`` (or state back the
        browser hint) before the model runs; returns the statement-back line to add to the reply."""
        return None

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
        check = self._check(ctx, result.reply, result.claims, facts)
        if check.ok:
            return result
        ctx.state.event("claim_ledger", "claim_blocked", check.summary())
        if result.from_model:
            answer = await self._repair(
                ctx, result, check, facts, executor=executor, system=system, history=history, usage=usage
            )
            if answer is not None and answer.reply.strip():
                recheck = self._check(ctx, answer.reply, answer.claims, facts)
                if recheck.ok:
                    ctx.state.event("claim_ledger", "repaired", answer.reply)
                    result.set_final(answer.reply, answer.claims)
                    result.repaired = True
                    return result
                ctx.state.event("claim_ledger", "repair_blocked", recheck.summary())
        reply = await self._safe_reply(ctx, executor, check, facts, [*history, *result.messages])
        ctx.state.event("claim_ledger", "safe_template", reply)
        result.set_final(reply)
        result.blocked = True
        return result

    def _hook_offer_reference(self, ctx: TurnContext) -> Sequence[datetime] | None:
        """Hook for ``fail_closed``'s offer grounding inside the claim check: the starts every offered time
        must come from (the lead's latest successful slot list). ``None``: offers are not checked."""
        return None

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
        self, ctx: TurnContext, reply: str, claims: Sequence[DeclaredClaim], facts: Sequence[LedgerFact]
    ) -> CheckResult:
        return check_reply(
            reply,
            [(c.type, c.time) for c in claims],
            facts,
            zone=ctx.zone,
            now=ctx.now,
            host_zone=self.settings.host_timezone,
            offer_reference=self._hook_offer_reference(ctx),
        )

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
        conversation: Sequence[ChatMessage],
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
        if not (ctx.state.calendar_unavailable or last_read_failed(conversation)):
            return f"{first} {render.NEXT_STEP_LOOK}"
        handed_off = any(m.role == "tool" and m.name == "handoff_to_human" for m in conversation)
        if not (ctx.state.handoffs or handed_off):
            user = next((m.content or "" for m in reversed(conversation) if m.role == "user"), "")
            await executor.handoff(
                "The prospect asked for a call, but the calendar is unavailable.", user[:300]
            )
        return f"{first} {render.NEXT_STEP_HANDOFF}"

    async def _hook_crm_outbox(self, ctx: TurnContext) -> None:
        """Hook for ``crm_outbox``: queue a validated CRM payload for each verified write of the turn."""
        return None

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
                if not ctx.state.handoffs:
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

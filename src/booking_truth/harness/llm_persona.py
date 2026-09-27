"""An LLM-driven simulated prospect (``booking-truth test`` with an LLM key configured).

The persona card (given name plus initial, style, goal, how it states its zone, its clarification text,
and its hidden acceptable window, rendered as local dates and hours in its TRUE zone) goes into the system
prompt, built once per trial from :mod:`booking_truth.harness.prompts.persona`. The persona sees only the
conversation: each of the agent's replies, and its own earlier messages. It answers with one structured
JSON object, ``{"message": str, "accepts": null | {"offered_index": int} | {"time_text": str}, "end": bool}``
(``response_format`` ``json_schema``, strict), at temperature 0.7.

This wrapper, not the model, decides what the persona actually accepted, in UTC: a quick reply's own
``start_utc`` when the accepted index names one (:func:`booking_truth.harness.personas.reply_offers`), else
the harness's own time reader (:mod:`booking_truth.harness.timeparse`) over the agent's text, read in the
persona's true zone. It then checks that instant against the hidden window exactly as
:class:`~booking_truth.harness.personas.ScriptedPersona` does, raising
:class:`~booking_truth.harness.personas.PersonaError` when it falls outside.

Implements the :class:`~booking_truth.harness.personas.Persona` protocol, so
:mod:`booking_truth.harness.runner` drives it exactly like a scripted persona: the 14-turn cap and the end
token are both handled there and by the ``end`` field of the structured answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from booking_truth.harness.adapters import AgentReply
from booking_truth.harness.personas import AgentView, Offer, PersonaError, PersonaTurn, reply_offers
from booking_truth.harness.scenarios import ResolvedScenario, date_text, dates_text
from booking_truth.harness.timeparse import TimeSpan, find_times
from booking_truth.llm.types import LLM, ChatMessage, LLMError, LLMResponse, content_looks_truncated
from booking_truth.timeutil import iso_z

PERSONA_TEMPERATURE = 0.7
PERSONA_MAX_TOKENS = 400
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "persona.md"
_OPENING_CUE = "(The call is connecting. Send your opening message now.)"

_TIME_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "persona_turn",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "accepts": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "properties": {"offered_index": {"type": "integer"}},
                            "required": ["offered_index"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {"time_text": {"type": "string"}},
                            "required": ["time_text"],
                            "additionalProperties": False,
                        },
                    ]
                },
                "end": {"type": "boolean"},
            },
            "required": ["message", "accepts", "end"],
            "additionalProperties": False,
        },
    },
}

_WS = re.compile(r"\s+")


def _hour12(hour: int, minute: int) -> str:
    twelve = hour % 12 or 12
    suffix = "am" if hour < 12 else "pm"
    return f"{twelve} {suffix}" if minute == 0 else f"{twelve}:{minute:02d} {suffix}"


def _persona_card(resolved: ResolvedScenario) -> dict[str, Any]:
    """The persona's own ground truth, as JSON: never sent to the agent under test."""
    persona = resolved.scenario.persona
    window = resolved.window
    year = resolved.persona_today.year
    card: dict[str, Any] = {
        "given_name": persona.given_name,
        "initial": persona.initial,
        "display_name": persona.display_name,
        "style": persona.style,
        "goal": persona.goal,
        "timezone_statement": persona.timezone_statement,
        "clarification": persona.clarification,
        "true_zone": persona.true_zone,
        "host_zone": resolved.host_zone,
        "today": date_text(resolved.persona_today, reference_year=year),
        "hidden_window": {
            "zone": window.zone,
            "dates": [date_text(d, reference_year=year) for d in window.dates],
            "summary": f"{dates_text(window.dates, reference_year=year)}, between "
            f"{_hour12(window.start.hour, window.start.minute)} and "
            f"{_hour12(window.end.hour, window.end.minute)}, {window.zone} time",
        },
    }
    if resolved.setup_start_utc is not None:
        local = resolved.setup_start_utc.astimezone(ZoneInfo(persona.true_zone))
        card["existing_booking"] = {
            "date": date_text(local.date(), reference_year=year),
            "time_local": _hour12(local.hour, local.minute),
            "zone": persona.true_zone,
        }
    return card


def _system_prompt(resolved: ResolvedScenario) -> str:
    base = PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    block = json.dumps(_persona_card(resolved), ensure_ascii=False, indent=2)
    return f"{base}\n\n<persona>\n{block}\n</persona>\n"


def _annotate(text: str, offers: Sequence[Offer]) -> str:
    """The agent's text, with its offers enumerated, so ``accepts.offered_index`` has something to refer to
    (the same slots a real prospect would see as labelled quick-reply buttons)."""
    if not offers:
        return text
    listed = "\n".join(f"[{index}] {offer.label}" for index, offer in enumerate(offers))
    return f"{text}\n\nOptions in this message:\n{listed}"


def _normalize(text: str) -> str:
    return _WS.sub(" ", re.sub(r"[^a-z0-9:]+", " ", text.lower())).strip()


def _matching_span(spans: Sequence[TimeSpan], needle: str) -> TimeSpan | None:
    """The span among ``spans`` whose own text best matches ``needle`` (the persona's copied phrase)."""
    target = _normalize(needle)
    if not target:
        return None
    best: TimeSpan | None = None
    best_score = 0
    for span in spans:
        hay = _normalize(span.text)
        if hay and (target in hay or hay in target):
            score = len(hay) if hay in target else len(target)
            if score > best_score:
                best, best_score = span, score
    return best


def _resolve_time_text(
    time_text: str, last_text: str | None, *, prospect_zone: str, host_zone: str, reference: datetime
) -> datetime | None:
    """The instant a copied phrase names: matched against the times found in the agent's own last message
    first (so its stated zone and date context apply), else parsed from the phrase alone."""
    if last_text:
        spans = find_times(last_text, prospect_zone=prospect_zone, host_zone=host_zone, reference=reference)
        match = _matching_span(spans, time_text)
        if match is not None:
            return match.utc
    alone = find_times(time_text, prospect_zone=prospect_zone, host_zone=host_zone, reference=reference)
    return alone[0].utc if alone else None


def _parse(content: str | None) -> dict[str, Any]:
    if not content:
        raise LLMError("the persona model returned an empty response", kind="malformed")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"the persona model returned invalid JSON: {exc}", kind="malformed") from None
    if not isinstance(data, dict) or not isinstance(data.get("message"), str):
        raise LLMError("the persona model's response is missing 'message'", kind="malformed")
    return data


class LLMPersona:
    """A prospect played by an LLM; see the module docstring. Construct one per trial."""

    def __init__(
        self, resolved: ResolvedScenario, llm: LLM, *, model: str, supports_actions: bool = True
    ) -> None:
        self.resolved = resolved
        self.llm = llm
        self.model = model
        self.supports_actions = supports_actions
        self.turns = 0
        self.usage_usd = 0.0
        self._system = _system_prompt(resolved)
        self._history: list[ChatMessage] = []
        self._ended = False

    @property
    def prospect_zone(self) -> str:
        return self.resolved.scenario.persona.true_zone

    @property
    def host_zone(self) -> str:
        return self.resolved.host_zone

    async def next_turn(self, view: AgentView) -> PersonaTurn | None:
        if self._ended:
            return None
        offers: list[Offer] = []
        if view.last_reply is not None:
            offers = reply_offers(
                view.last_reply,
                prospect_zone=self.prospect_zone,
                host_zone=self.host_zone,
                reference=view.now,
            )
            self._history.append(ChatMessage.user(_annotate(view.last_reply.reply or "", offers)))
        else:
            self._history.append(ChatMessage.user(_OPENING_CUE))
        messages = [ChatMessage.system(self._system), *self._history]
        response = await self._chat(messages, PERSONA_MAX_TOKENS)
        try:
            payload = _parse(response.content)
        except LLMError as exc:
            if exc.kind != "malformed" or not content_looks_truncated(response):
                raise
            # A truncated structured turn is a token-budget problem, not a bad model: resending the
            # identical request would very likely fail again identically (persona replies run at
            # temperature 0.7, so this is not guaranteed, but doubling the budget fixes the actual cause
            # either way), so retry once with a larger budget instead of repeating it as is.
            response = await self._chat(messages, PERSONA_MAX_TOKENS * 2)
            payload = _parse(response.content)
        message = payload["message"].strip()
        self._history.append(ChatMessage.assistant(message))
        self.turns += 1
        end = bool(payload.get("end", False))
        accepted = self._accepted_instant(payload.get("accepts"), offers, view.last_reply, view.now)
        if accepted is None:
            self._ended = end
            return PersonaTurn("say", message, end=end)
        self._check_window(accepted)
        chosen = next((o for o in offers if o.start_utc == accepted), None) or Offer(accepted, message)
        action = None
        if self.supports_actions and chosen.slot_id is not None:
            action = {"type": "select_slot", "slot_id": chosen.slot_id}
        self._ended = end
        return PersonaTurn("pick", message, action=action, offer=chosen, end=end, choices=tuple(offers))

    async def _chat(self, messages: Sequence[ChatMessage], max_tokens: int) -> LLMResponse:
        response = await self.llm.chat(
            messages=messages,
            temperature=PERSONA_TEMPERATURE,
            model=self.model,
            max_tokens=max_tokens,
            response_format=_TIME_SCHEMA,
            component="persona",
        )
        self.usage_usd += response.usage.usd
        return response

    def _accepted_instant(
        self,
        accepts: Any,
        offers: Sequence[Offer],
        last_reply: AgentReply | None,
        now: datetime,
    ) -> datetime | None:
        if not isinstance(accepts, dict):
            return None
        index = accepts.get("offered_index")
        if isinstance(index, int) and not isinstance(index, bool):
            return offers[index].start_utc if 0 <= index < len(offers) else None
        time_text = accepts.get("time_text")
        if isinstance(time_text, str) and time_text.strip():
            last_text = last_reply.reply if last_reply is not None else None
            return _resolve_time_text(
                time_text,
                last_text,
                prospect_zone=self.prospect_zone,
                host_zone=self.host_zone,
                reference=now,
            )
        return None

    def _check_window(self, instant: datetime) -> None:
        if not self.resolved.window_contains(instant):
            raise PersonaError(f"the persona accepted {iso_z(instant)}, which is outside its hidden window")


__all__ = ["PERSONA_TEMPERATURE", "LLMPersona"]

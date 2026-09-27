"""The LLM prospect-belief extractor (``docs/adr/0008``).

Structured output keeps times as local wall clock plus an IANA zone, never a UTC instant computed by the
model: ``{"status": ..., "time": {"local": "YYYY-MM-DDTHH:MM", "zone": "..."} | null, "offered": [...],
"evidence": "..."}`` (``response_format`` ``json_schema``, strict, temperature 0). This module converts each
local time to UTC itself, with :mod:`zoneinfo`: a local time a daylight-saving gap skips over is rejected
(dropped), and a time a fold makes ambiguous takes the earlier instant.

A response the token budget cuts off mid-answer is retried once, at double the budget, instead of being
retried identically at the harness's attempt level: temperature 0 means an identical request would fail
again identically, wasting spend and budget headroom for no gain (see :func:`content_looks_truncated
<booking_truth.llm.types.content_looks_truncated>`).

Implements :class:`~booking_truth.harness.beliefs.BeliefExtractor`, independent of the agent's own claim
guard (``booking_truth.agent.guards``): it shares no code with it, only the plain :class:`Belief` value type
both sides of the harness use.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from booking_truth.harness.beliefs import BELIEF_STATUSES, Belief, BeliefSource
from booking_truth.llm.types import LLM, ChatMessage, LLMError, LLMResponse, content_looks_truncated
from booking_truth.timeutil import ensure_utc

EXTRACTOR_TEMPERATURE = 0.0
EXTRACTOR_MAX_TOKENS = 1200
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "extractor.md"

_LOCAL_TIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"local": {"type": "string"}, "zone": {"type": "string"}},
    "required": ["local", "zone"],
    "additionalProperties": False,
}
_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "prospect_belief",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": list(BELIEF_STATUSES)},
                "time": {"anyOf": [{"type": "null"}, _LOCAL_TIME_SCHEMA]},
                "offered": {"type": "array", "items": _LOCAL_TIME_SCHEMA},
                "evidence": {"type": "string"},
            },
            "required": ["status", "time", "offered", "evidence"],
            "additionalProperties": False,
        },
    },
}


def local_to_utc(local: str, zone: str) -> datetime | None:
    """Local wall time (``YYYY-MM-DDTHH:MM``) plus an IANA zone (or ``UTC``) to a UTC instant.

    ``None`` for a local time that does not exist in ``zone`` (a DST gap) or names no zone this process
    knows; a local time a DST fold makes ambiguous resolves to the earlier of its two instants.
    """
    try:
        naive = datetime.strptime(local.strip(), "%Y-%m-%dT%H:%M")  # noqa: DTZ007 - tzinfo attached below
    except ValueError:
        return None
    if zone.strip() == "UTC":
        return naive.replace(tzinfo=UTC)
    try:
        tz = ZoneInfo(zone.strip())
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None
    early = naive.replace(tzinfo=tz, fold=0)
    late = naive.replace(tzinfo=tz, fold=1)
    if early.utcoffset() == late.utcoffset():
        return ensure_utc(early)
    early_utc, late_utc = ensure_utc(early), ensure_utc(late)
    return early_utc if early_utc < late_utc else None  # equal-or-later means a gap: nonexistent, rejected


def _time_utc(value: Any) -> datetime | None:
    if not isinstance(value, dict):
        return None
    local, zone = value.get("local"), value.get("zone")
    if not isinstance(local, str) or not isinstance(zone, str):
        return None
    return local_to_utc(local, zone)


def _system_prompt(*, prospect_zone: str, host_zone: str, reference: datetime) -> str:
    base = PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    context = {
        "prospect_zone": prospect_zone,
        "host_zone": host_zone,
        "reference_utc": reference.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    block = json.dumps(context, ensure_ascii=False, indent=2)
    return f"{base}\n\n<context>\n{block}\n</context>\n"


def _user_message(agent_messages: Sequence[str]) -> str:
    if not agent_messages:
        return "(The agent sent no messages in this conversation.)"
    return "\n\n".join(f"Message {index + 1}: {text}" for index, text in enumerate(agent_messages))


def _parse(content: str | None) -> dict[str, Any]:
    if not content:
        raise LLMError("the extractor model returned an empty response", kind="malformed")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"the extractor model returned invalid JSON: {exc}", kind="malformed") from None
    if not isinstance(data, dict) or data.get("status") not in BELIEF_STATUSES:
        raise LLMError("the extractor model's response is missing a valid 'status'", kind="malformed")
    return data


class LLMExtractor:
    """The LLM belief extractor behind :class:`~booking_truth.harness.beliefs.BeliefExtractor`."""

    source: BeliefSource = "llm"

    def __init__(self, llm: LLM, *, model: str) -> None:
        self.llm = llm
        self.model = model
        self.usage_usd = 0.0
        #: Every call's returned (model, provider), in order. This instance is shared across trials
        #: (unlike ``LLMPersona``, fresh per trial), so the runner reads a diffed slice of this list -
        #: the calls made since it last checked - into the manifest's ``CallRecorder``, under the same
        #: lock it already uses to isolate this trial's own cost (``Runner._extractor_cost_lock``).
        self.calls_made: list[tuple[str, str | None]] = []

    async def extract(
        self,
        agent_messages: Sequence[str],
        *,
        prospect_zone: str,
        host_zone: str,
        reference: datetime,
    ) -> Belief:
        reference = ensure_utc(reference)
        system = _system_prompt(prospect_zone=prospect_zone, host_zone=host_zone, reference=reference)
        messages = [ChatMessage.system(system), ChatMessage.user(_user_message(agent_messages))]
        response = await self._chat(messages, EXTRACTOR_MAX_TOKENS)
        try:
            data = _parse(response.content)
        except LLMError as exc:
            if exc.kind != "malformed" or not content_looks_truncated(response):
                raise
            # A truncated structured answer is a token-budget problem, not a bad model: resending the
            # identical request at temperature 0 would fail again identically, so retry once with a
            # larger budget instead of repeating it.
            response = await self._chat(messages, EXTRACTOR_MAX_TOKENS * 2)
            data = _parse(response.content)
        offered = {t for item in data.get("offered") or () if (t := _time_utc(item)) is not None}
        return Belief(
            status=data["status"],
            time_utc=_time_utc(data.get("time")),
            offered_utc=tuple(offered),
            source="llm",
            evidence=str(data.get("evidence") or ""),
        )

    async def _chat(self, messages: Sequence[ChatMessage], max_tokens: int) -> LLMResponse:
        response = await self.llm.chat(
            messages=messages,
            temperature=EXTRACTOR_TEMPERATURE,
            model=self.model,
            max_tokens=max_tokens,
            response_format=_SCHEMA,
            component="extractor",
        )
        self.usage_usd += response.usage.usd
        self.calls_made.append((response.model_returned, response.provider))
        return response


__all__ = ["LLMExtractor", "local_to_utc"]

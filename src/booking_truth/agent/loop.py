"""The LLM tool loop: at most 8 model calls per turn, then a final answer.

The final answer is the assistant's content, which the base prompt asks to be a JSON object
``{"reply": str, "claims": [{"type": "booked" | "rescheduled" | "cancelled" | "offered", "time": str}]}``. It
is parsed leniently: code fences are stripped; a model that adds a preamble or a sign-off around the object,
instead of nothing else as the prompt asks, still gets that object read out of the surrounding prose, so the
prose is never shown to the prospect as raw JSON. Content with no such object anywhere is the whole reply
with no declared claims.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from booking_truth.agent.tools import ToolExecutor
from booking_truth.llm.types import LLM, ChatMessage, LLMResponse, ToolSpec

MAX_MODEL_CALLS = 8
AGENT_TEMPERATURE = 0.2
MAX_TOKENS = 1024
CLAIM_TYPES = frozenset({"booked", "rescheduled", "cancelled", "offered"})

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass(frozen=True)
class DeclaredClaim:
    type: str
    time: str

    def to_json(self) -> dict[str, str]:
        return {"type": self.type, "time": self.time}


@dataclass(frozen=True)
class FinalAnswer:
    reply: str
    claims: tuple[DeclaredClaim, ...] = ()
    structured: bool = False


def _loads(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def _embedded_json_object(text: str) -> str | None:
    """The first balanced ``{...}`` in ``text``, ignoring braces inside JSON string values, or ``None``.

    A fallback for a model that wraps its final answer in a preamble or a sign-off ("Sure! {...}", "{...}
    Let me know!") instead of answering with nothing else, as the base prompt asks: the object is still
    findable even though the whole content is not itself valid JSON.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_final_answer(content: str | None) -> FinalAnswer:
    """``{"reply": ..., "claims": [...]}`` from the model's content; anything else is a plain reply."""
    text = (content or "").strip()
    fenced = _FENCE.match(text)
    candidate = fenced.group(1) if fenced else text
    data = _loads(candidate)
    if data is None and fenced is None:
        embedded = _embedded_json_object(candidate)
        if embedded is not None:
            data = _loads(embedded)
    if not isinstance(data, dict) or not isinstance(data.get("reply"), str):
        return FinalAnswer(reply=text)
    claims: list[DeclaredClaim] = []
    raw_claims = data.get("claims")
    if isinstance(raw_claims, list):
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            when = item.get("time")
            if isinstance(kind, str) and kind in CLAIM_TYPES:
                claims.append(DeclaredClaim(kind, when if isinstance(when, str) else ""))
    return FinalAnswer(reply=data["reply"].strip(), claims=tuple(claims), structured=True)


@dataclass
class Usage:
    """Token and cost totals of one turn, and every model id and provider that answered."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    models: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    calls: int = 0

    def add(self, response: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        self.usd += response.usage.usd
        if response.model_returned and response.model_returned not in self.models:
            self.models.append(response.model_returned)
        if response.provider and response.provider not in self.providers:
            self.providers.append(response.provider)

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usd": round(self.usd, 8),
            "models": list(self.models),
            "providers": list(self.providers),
        }


@dataclass
class LoopResult:
    answer: FinalAnswer | None
    #: The assistant and tool messages this turn added after the user message.
    messages: list[ChatMessage]
    exhausted: bool = False


async def run_tool_loop(
    llm: LLM,
    *,
    system: str,
    history: Sequence[ChatMessage],
    executor: ToolExecutor,
    usage: Usage,
    model: str | None = None,
    max_calls: int = MAX_MODEL_CALLS,
    temperature: float = AGENT_TEMPERATURE,
) -> LoopResult:
    """Call the model until it answers without tool calls, or ``max_calls`` is reached (``exhausted``).

    ``history`` ends with the current user message. Tool calls of one response run in order.
    """
    added: list[ChatMessage] = []
    tools: list[ToolSpec] = list(executor.specs)
    for _ in range(max_calls):
        messages = [ChatMessage.system(system), *history, *added]
        response = await llm.chat(
            messages=messages,
            tools=tools,
            temperature=temperature,
            model=model,
            max_tokens=MAX_TOKENS,
            component="agent",
        )
        usage.add(response)
        if not response.tool_calls:
            answer = parse_final_answer(response.content)
            added.append(ChatMessage.assistant(response.content or answer.reply))
            return LoopResult(answer=answer, messages=added)
        added.append(response.to_message())
        for call in response.tool_calls:
            content = await executor.execute(call)
            added.append(ChatMessage.tool(call.id, content, name=call.name))
    return LoopResult(answer=None, messages=added, exhausted=True)

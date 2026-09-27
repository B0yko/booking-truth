"""Provider-neutral chat types and the ``LLM`` protocol every model backend implements.

The agent, the harness personas, the belief extractor and the offline ``FakeLLM`` all speak
these types. Only :mod:`booking_truth.llm.client` knows about the OpenAI wire format beyond
the ``to_openai`` helpers below.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]

_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class LLMError(RuntimeError):
    """A model call failed or was refused. The message is safe to show and log: it never holds a key.

    ``kind`` is a short machine-readable reason, for example ``timeout``, ``connection``, ``auth``,
    ``rate_limit``, ``payment_required``, ``bad_request``, ``not_found``, ``server``, ``provider``,
    ``malformed``, ``offline``, ``budget`` or ``pricing``.
    """

    def __init__(self, message: str, *, kind: str = "error", status_code: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


class BudgetExceeded(LLMError):
    """The cost ledger total plus this call's estimate would pass the configured cap."""

    def __init__(self, message: str, *, total_usd: float, estimate_usd: float, cap_usd: float) -> None:
        super().__init__(message, kind="budget")
        self.total_usd = total_usd
        self.estimate_usd = estimate_usd
        self.cap_usd = cap_usd


class PricingError(LLMError):
    """The model is missing from the price table, or the table itself is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="pricing")


@dataclass(frozen=True)
class ToolCall:
    """One function call requested by the model. ``arguments`` is the raw JSON string it produced."""

    id: str
    name: str
    arguments: str

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ChatMessage:
    """One chat message. Use the ``system``/``user``/``assistant``/``tool`` constructors."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unknown chat role {self.role!r}")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages carry tool calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a tool message needs the tool_call_id it answers")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid on tool messages")
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError(f"a {self.role} message needs content")

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str, *, name: str | None = None) -> ChatMessage:
        return cls(role="user", content=content, name=name)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: Sequence[ToolCall] = ()) -> ChatMessage:
        return cls(role="assistant", content=content, tool_calls=list(tool_calls))

    @classmethod
    def tool(cls, tool_call_id: str, content: str, *, name: str | None = None) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    def to_openai(self) -> dict[str, Any]:
        """The OpenAI chat-completions wire dict.

        ``name`` is sent for system, user and assistant messages. On tool messages it is kept for
        traces only, because the tool-message schema has no ``name`` field.
        """
        wire: dict[str, Any] = {"role": self.role}
        if self.role == "tool":
            wire["tool_call_id"] = self.tool_call_id
            wire["content"] = self.content
            return wire
        wire["content"] = self.content
        if self.tool_calls:
            wire["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        if self.name is not None:
            wire["name"] = self.name
        return wire


@dataclass(frozen=True)
class ToolSpec:
    """A function the model may call. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        if not _TOOL_NAME.match(self.name):
            raise ValueError(f"tool name {self.name!r} must match {_TOOL_NAME.pattern}")
        if self.parameters.get("type") != "object":
            raise ValueError(f"tool {self.name!r}: parameters must be a JSON Schema with type 'object'")

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass(frozen=True)
class Usage:
    """Token counts and cost of one call.

    ``usd`` is what the ledger and the budget count: tokens times the price table, raised to
    ``provider_reported_cost`` (the ``usage.cost`` the endpoint returned, if any) when that is higher,
    so spend is never counted below what the endpoint says it charged.
    ``cached_tokens`` is the part of ``prompt_tokens`` served from the provider's prompt cache.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    usd: float = 0.0
    provider_reported_cost: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMResponse:
    """The parsed result of one chat call.

    ``model_returned`` is the ``model`` field of the response, which can differ from the requested id
    (a dated slug, or a fallback model). ``provider`` is the upstream provider the router reports,
    or ``None`` when the endpoint does not say.
    """

    content: str | None
    tool_calls: list[ToolCall]
    usage: Usage
    model_requested: str
    model_returned: str
    provider: str | None
    response_id: str
    latency_s: float
    finish_reason: str | None = None

    def to_message(self) -> ChatMessage:
        """The assistant message to append to the conversation before sending tool results."""
        content = self.content if self.content is not None or self.tool_calls else ""
        return ChatMessage.assistant(content, self.tool_calls)


def content_looks_truncated(response: LLMResponse) -> bool:
    """Whether ``response`` looks cut off by the token budget rather than a complete, merely wrong,
    answer: a caller expecting a JSON object can use this to decide whether a parse failure is worth
    retrying with a larger ``max_tokens`` (resending an identical request at temperature 0 would just
    fail again identically). ``finish_reason: "length"`` is the definitive signal; failing that, a
    non-empty response that does not end in a closing brace is a reasonable proxy for a response cut
    off mid-value, since a complete JSON object always ends with one.
    """
    if response.finish_reason == "length":
        return True
    content = (response.content or "").rstrip()
    return bool(content) and not content.endswith("}")


class LLM(Protocol):
    """A chat model with tool calling. Implemented by ``OpenAICompatClient`` and the offline ``FakeLLM``."""

    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        temperature: float,
        model: str | None = None,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        component: str = "agent",
        run_id: str | None = None,
    ) -> LLMResponse: ...

"""The tool loop and the lenient parse of the final answer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agent_env import AgentEnv, executor

from booking_truth.agent.loop import MAX_MODEL_CALLS, Usage, parse_final_answer, run_tool_loop
from booking_truth.llm.types import ChatMessage, LLMResponse, ToolCall, ToolSpec
from booking_truth.llm.types import Usage as CallUsage


def test_a_json_final_answer_gives_the_reply_and_its_claims() -> None:
    content = json.dumps(
        {
            "reply": " You're booked. ",
            "claims": [
                {"type": "booked", "time": "Tuesday 6 October, 3:00 PM"},
                {"type": "offered"},
                {"type": "teleported", "time": "now"},
                "not a claim",
            ],
        }
    )
    answer = parse_final_answer(content)
    assert answer.structured
    assert answer.reply == "You're booked."
    assert [(c.type, c.time) for c in answer.claims] == [
        ("booked", "Tuesday 6 October, 3:00 PM"),
        ("offered", ""),
    ]


def test_code_fences_are_stripped() -> None:
    answer = parse_final_answer('```json\n{"reply": "Hi", "claims": []}\n```')
    assert (answer.reply, answer.structured) == ("Hi", True)


def test_anything_else_is_a_plain_reply_without_claims() -> None:
    for content in ("You're booked for Tuesday.", '{"text": "Hi"}', "[1, 2]", "", None):
        answer = parse_final_answer(content)
        assert answer.claims == ()
        assert not answer.structured
        assert answer.reply == (content or "").strip()


class Looping:
    """A model that always calls a tool, to hit the call cap."""

    def __init__(self) -> None:
        self.calls = 0

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
    ) -> LLMResponse:
        self.calls += 1
        provider = "p1" if self.calls == 1 else "p2"
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(f"c{self.calls}", "list_my_bookings", "{}")],
            usage=CallUsage(prompt_tokens=10, completion_tokens=2, usd=0.001),
            model_requested="m",
            model_returned="vendor/model-a",
            provider=provider,
            response_id=str(self.calls),
            latency_s=0.0,
        )


async def test_the_loop_stops_after_eight_model_calls(guarded: AgentEnv) -> None:
    model = Looping()
    usage = Usage()
    tools = executor(guarded)
    result = await run_tool_loop(
        model, system="s", history=[ChatMessage.user("hi")], executor=tools, usage=usage
    )
    assert result.exhausted
    assert result.answer is None
    assert model.calls == MAX_MODEL_CALLS == 8
    assert usage.calls == 8
    assert usage.prompt_tokens == 80
    assert usage.completion_tokens == 16
    assert usage.to_json()["models"] == ["vendor/model-a"]
    assert usage.to_json()["providers"] == ["p1", "p2"]
    assert usage.to_json()["usd"] == 0.008
    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert len(tool_messages) == 8
    assert [s["name"] for s in tools.state.steps].count("list_my_bookings") == 16

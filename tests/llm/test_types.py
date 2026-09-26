import pytest

from booking_truth.llm.types import (
    BudgetExceeded,
    ChatMessage,
    LLMError,
    LLMResponse,
    PricingError,
    ToolCall,
    ToolSpec,
    Usage,
)


def test_messages_convert_to_the_openai_wire_format() -> None:
    call = ToolCall(id="call_1", name="find_slots", arguments='{"from_date": "2026-10-05"}')
    assert ChatMessage.system("rules").to_openai() == {"role": "system", "content": "rules"}
    assert ChatMessage.user("hi", name="prospect").to_openai() == {
        "role": "user",
        "content": "hi",
        "name": "prospect",
    }
    assert ChatMessage.assistant(None, [call]).to_openai() == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "find_slots", "arguments": '{"from_date": "2026-10-05"}'},
            }
        ],
    }
    # A tool message keeps `name` locally but does not send it: the wire schema has no such field.
    assert ChatMessage.tool("call_1", '{"slots": []}', name="find_slots").to_openai() == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"slots": []}',
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"role": "tool", "content": "x"}, "tool_call_id"),
        ({"role": "user", "content": None}, "needs content"),
        ({"role": "user", "content": "x", "tool_call_id": "c"}, "only valid on tool"),
        ({"role": "user", "content": "x", "tool_calls": [ToolCall("c", "f", "{}")]}, "only assistant"),
        ({"role": "robot", "content": "x"}, "unknown chat role"),
    ],
)
def test_invalid_messages_are_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ChatMessage(**kwargs)  # type: ignore[arg-type]


def test_tool_spec_to_openai_and_validation() -> None:
    spec = ToolSpec(
        name="book_slot",
        description="Book one offered slot.",
        parameters={"type": "object", "properties": {"slot_id": {"type": "string"}}, "required": ["slot_id"]},
    )
    assert spec.to_openai() == {
        "type": "function",
        "function": {
            "name": "book_slot",
            "description": "Book one offered slot.",
            "parameters": spec.parameters,
        },
    }
    with pytest.raises(ValueError, match="must match"):
        ToolSpec(name="book slot", description="", parameters={"type": "object"})
    with pytest.raises(ValueError, match="type 'object'"):
        ToolSpec(name="book_slot", description="", parameters={"type": "string"})


def test_response_to_message_round_trips_tool_calls() -> None:
    call = ToolCall(id="call_9", name="list_my_bookings", arguments="{}")
    response = LLMResponse(
        content=None,
        tool_calls=[call],
        usage=Usage(prompt_tokens=3, completion_tokens=2),
        model_requested="vendor/flash",
        model_returned="vendor/flash-20260910",
        provider="DeepInfra",
        response_id="gen-1",
        latency_s=0.1,
    )
    assert response.usage.total_tokens == 5
    assert response.to_message() == ChatMessage(role="assistant", content=None, tool_calls=[call])
    empty = LLMResponse(
        content=None,
        tool_calls=[],
        usage=Usage(),
        model_requested="m",
        model_returned="m",
        provider=None,
        response_id="",
        latency_s=0.0,
    )
    assert empty.to_message().to_openai() == {"role": "assistant", "content": ""}


def test_error_hierarchy() -> None:
    budget = BudgetExceeded("stop", total_usd=1.0, estimate_usd=0.1, cap_usd=1.0)
    assert isinstance(budget, LLMError)
    assert budget.kind == "budget"
    assert isinstance(PricingError("missing"), LLMError)
    assert PricingError("missing").kind == "pricing"

"""Request and response models of the chat endpoints."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from booking_truth.agent.models import ActionIn, ChatRequest, ChatResponse, ErrorBody, GuardInfo, UsageInfo


def request(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "session_id": "s-1",
        "message_id": "m-1",
        "lead": {"email": "maya@example.com"},
        "message": "Hi",
    }
    body.update(overrides)
    return body


def test_a_text_request_defaults_to_the_api_channel() -> None:
    parsed = ChatRequest.model_validate(request())
    assert parsed.channel == "api"
    assert parsed.message == "Hi"
    assert parsed.action is None
    assert parsed.lead.name is None
    assert parsed.lead.timezone_hint is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"message": None},
        {"message": "   "},
        {"action": {"type": "cancel", "booking_uid": "b1"}},
    ],
    ids=["neither", "blank-message", "both"],
)
def test_exactly_one_of_message_or_action(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="exactly one of message or action"):
        ChatRequest.model_validate(request(**overrides))


def test_an_action_alone_is_accepted() -> None:
    parsed = ChatRequest.model_validate(
        request(message=None, action={"type": "select_slot", "slot_id": "s_x"})
    )
    assert parsed.action is not None
    assert parsed.action.describe() == "select_slot slot_id=s_x"


@pytest.mark.parametrize(
    ("action", "missing"),
    [
        ({"type": "select_slot"}, "slot_id"),
        ({"type": "reschedule"}, "booking_uid"),
        ({"type": "cancel"}, "booking_uid"),
        ({"type": "confirm_timezone"}, "zone"),
    ],
)
def test_actions_need_their_ids(action: dict[str, Any], missing: str) -> None:
    with pytest.raises(ValidationError, match=missing):
        ActionIn.model_validate(action)


def test_unknown_action_types_and_channels_are_refused() -> None:
    with pytest.raises(ValidationError):
        ActionIn.model_validate({"type": "delete_everything"})
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(request(channel="sms"))


@pytest.mark.parametrize("email", ["not-an-email", "a b@example.com", "@example.com"])
def test_the_lead_email_must_look_like_an_address(email: str) -> None:
    with pytest.raises(ValidationError, match="email"):
        ChatRequest.model_validate(request(lead={"email": email}))


def test_ids_must_be_printable_without_spaces() -> None:
    with pytest.raises(ValidationError, match="printable"):
        ChatRequest.model_validate(request(session_id="has space"))
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(request(message_id="x" * 129))


def test_blank_optional_lead_fields_become_none() -> None:
    parsed = ChatRequest.model_validate(
        request(lead={"email": "maya@example.com", "name": "", "timezone_hint": " "})
    )
    assert parsed.lead.name is None
    assert parsed.lead.timezone_hint is None


def test_the_response_drops_a_missing_session_token() -> None:
    response = ChatResponse(reply="Hi", agent_version="abc123def456")
    data = response.to_json()
    assert set(data) == {"reply", "quick_replies", "booking", "agent_version", "guard", "usage"}
    assert data["guard"] == {"blocked": False, "repaired": False, "events": []}
    assert data["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "usd": 0.0,
        "models": [],
        "providers": [],
    }
    with_token = ChatResponse(
        reply="Hi", agent_version="x", session_token="wst1.t", guard=GuardInfo(), usage=UsageInfo()
    ).to_json()
    assert with_token["session_token"] == "wst1.t"


def test_error_bodies_leave_out_empty_fields() -> None:
    assert ErrorBody(error="unauthorized").to_json() == {"error": "unauthorized"}
    assert ErrorBody(error="lead_busy", reply="wait").to_json() == {"error": "lead_busy", "reply": "wait"}

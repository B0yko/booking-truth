"""Request and response models of ``POST /v1/chat`` and ``POST /v1/widget/chat``.

A request carries exactly one of ``message`` (text) or ``action`` (a structured action from a quick reply or
a widget button). A response always carries ``reply``, ``quick_replies``, ``booking``, ``agent_version``,
``guard`` and ``usage``; widget responses also carry ``session_token``.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Channel = Literal["api", "widget", "webhook"]
ActionType = Literal["select_slot", "reschedule", "cancel", "confirm_timezone"]
BookingAction = Literal["booked", "rescheduled", "cancelled"]

#: Hard ceiling on a message before the configurable ``BT_MAX_INPUT_CHARS`` check (which answers 413).
MAX_BODY_MESSAGE_CHARS = 100_000
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ID = re.compile(r"^[\x21-\x7e]+$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class LeadIn(_Model):
    email: str = Field(min_length=3, max_length=254)
    name: str | None = Field(default=None, max_length=200)
    timezone_hint: str | None = Field(default=None, max_length=64)

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        if not _EMAIL.fullmatch(value):
            raise ValueError("lead.email must be an email address")
        return value

    @field_validator("name", "timezone_hint")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value or None


class ActionIn(_Model):
    type: ActionType
    slot_id: str | None = Field(default=None, max_length=64)
    booking_uid: str | None = Field(default=None, max_length=128)
    zone: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _ids(self) -> ActionIn:
        needed = {
            "select_slot": ("slot_id",),
            "reschedule": ("booking_uid",),
            "cancel": ("booking_uid",),
            "confirm_timezone": ("zone",),
        }[self.type]
        missing = [name for name in needed if not getattr(self, name)]
        if missing:
            raise ValueError(f"action {self.type} needs {', '.join(missing)}")
        return self

    def describe(self) -> str:
        """A short text form, for history and traces."""
        parts: list[str] = [self.type]
        for name in ("slot_id", "booking_uid", "zone"):
            value = getattr(self, name)
            if value:
                parts.append(f"{name}={value}")
        return " ".join(parts)


class ChatRequest(_Model):
    session_id: str = Field(min_length=1, max_length=128)
    message_id: str = Field(min_length=1, max_length=128)
    channel: Channel = "api"
    lead: LeadIn
    message: str | None = Field(default=None, max_length=MAX_BODY_MESSAGE_CHARS)
    action: ActionIn | None = None
    session_token: str | None = Field(default=None, max_length=256)

    @field_validator("session_id", "message_id")
    @classmethod
    def _printable(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("ids must be printable ASCII without spaces")
        return value

    @model_validator(mode="after")
    def _one_input(self) -> ChatRequest:
        has_message = self.message is not None and self.message != ""
        if has_message == (self.action is not None):
            raise ValueError("send exactly one of message or action")
        return self


class QuickReply(BaseModel):
    """A structured action offered to the prospect. ``action: null`` means "send the label as text"."""

    label: str
    action: dict[str, str] | None
    start_utc: str | None = None


class BookingView(BaseModel):
    ref: str
    status: Literal["active", "cancelled"]
    start_utc: str
    end_utc: str
    zone: str
    local_label: str
    action: BookingAction


class GuardEvent(BaseModel):
    guard: str
    event: str
    detail: str = ""


class GuardInfo(BaseModel):
    blocked: bool = False
    repaired: bool = False
    events: list[GuardEvent] = Field(default_factory=list)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    models: list[str] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    quick_replies: list[QuickReply] = Field(default_factory=list)
    booking: BookingView | None = None
    agent_version: str
    guard: GuardInfo = Field(default_factory=GuardInfo)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    session_token: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        if data.get("session_token") is None:
            data.pop("session_token", None)
        return data


class ErrorBody(BaseModel):
    error: str
    reply: str | None = None
    detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

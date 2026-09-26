"""Pydantic models for ``agent-trace/v1``.

The JSON Schema in ``schemas/agent-trace-v1.json`` is normative; these models are a typed way to
build traces. Every trace is still validated against the schema before it is written.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_ID = "agent-trace/v1"

Domain = Literal["booking", "crm", "coding", "browser", "other"]
StepKind = Literal["message", "tool_call", "tool_result", "state_probe"]
StepRole = Literal["user", "agent", "tool", "environment"]
Outcome = Literal["success", "failure", "unknown"]
CheckedBy = Literal["state_probe", "human", "none"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Task(_Strict):
    id: str
    domain: Domain
    instruction: str


class Step(_Strict):
    i: int = Field(ge=0)
    ts: str
    kind: StepKind
    role: StepRole
    name: str | None = None
    content: str | None = None
    args: dict[str, Any] | None = None
    ok: bool | None = None
    output: Any = None
    error: str | dict[str, Any] | None = None


class Claim(_Strict):
    type: str
    subject: dict[str, Any] = Field(default_factory=dict)


class FinalClaim(_Strict):
    text: str | None
    claims: list[Claim] = Field(default_factory=list)


class GroundTruth(_Strict):
    outcome: Outcome
    checked_by: CheckedBy
    details: dict[str, Any] | None = None


class Trace(_Strict):
    schema_: Literal["agent-trace/v1"] = Field(default="agent-trace/v1", alias="schema")
    trace_id: str
    source: str
    task: Task
    steps: list[Step] = Field(default_factory=list)
    final_claim: FinalClaim
    ground_truth: GroundTruth
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_json_dict(self) -> dict[str, Any]:
        data = self.model_dump(by_alias=True, mode="json")
        if data["ground_truth"].get("details") is None:
            data["ground_truth"].pop("details", None)
        return data

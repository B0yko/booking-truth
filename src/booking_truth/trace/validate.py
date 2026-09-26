"""Validate ``agent-trace/v1`` records: JSON Schema plus the ordering rules the schema cannot express."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from booking_truth.resources import data_path


class TraceValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@cache
def _validator() -> Draft202012Validator:
    schema = json.loads((data_path("schemas") / "agent-trace-v1.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def trace_errors(trace: Any) -> list[str]:
    """Return every problem with ``trace``; an empty list means it is valid."""
    errors = [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(_validator().iter_errors(trace), key=lambda e: list(map(str, e.absolute_path)))
    ]
    if errors or not isinstance(trace, dict):
        return errors
    last_i = -1
    seen_calls: set[str] = set()
    for pos, step in enumerate(trace["steps"]):
        if step["i"] <= last_i:
            errors.append(f"steps/{pos}: i={step['i']} is not greater than the previous step's i={last_i}")
        last_i = step["i"]
        kind = step["kind"]
        name = step.get("name")
        if kind == "message":
            if not isinstance(step.get("content"), str):
                errors.append(f"steps/{pos}: a message step needs string content")
        elif not name:
            errors.append(f"steps/{pos}: a {kind} step needs a name")
        if kind == "tool_call" and name:
            seen_calls.add(name)
        if kind == "tool_result" and name and name not in seen_calls:
            errors.append(f"steps/{pos}: tool_result '{name}' has no preceding tool_call with that name")
    return errors


def validate_trace(trace: Any) -> None:
    errors = trace_errors(trace)
    if errors:
        raise TraceValidationError(errors)


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    """Yield ``(line_number, parsed_json)`` for each non-empty line."""
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.strip():
                yield lineno, json.loads(line)


def write_jsonl(path: Path, traces: Iterable[dict[str, Any]]) -> int:
    """Validate and write traces as JSON Lines. Returns the number written."""
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for trace in traces:
            validate_trace(trace)
            fh.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count

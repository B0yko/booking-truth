"""``agent_version``: what goes into the hash, and that every input changes it."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from booking_truth.agent.version import (
    PACKAGE_ROOT,
    SOURCE_DIRS,
    compute_agent_version,
    prompt_files,
    schemas_of,
    source_hash,
)
from booking_truth.llm.types import ToolSpec

GUARDED_TOOLS = [
    ToolSpec(
        "find_slots", "Find slots.", {"type": "object", "properties": {"from_date": {"type": "string"}}}
    ),
    ToolSpec("book_slot", "Book a slot.", {"type": "object", "properties": {"slot_id": {"type": "string"}}}),
]
NAIVE_TOOLS = [
    GUARDED_TOOLS[0],
    ToolSpec("book", "Book a time.", {"type": "object", "properties": {"start_iso": {"type": "string"}}}),
]


def base(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "prompts": {"base.md": "You are the booking assistant."},
        "tool_schemas": schemas_of(GUARDED_TOOLS),
        "model_id": "vendor/model-2026-01-01",
        "guards_label": "all",
        "package_version": "0.1.0",
        "source_digest": "a" * 64,
    }
    values.update(overrides)
    return values


def test_the_version_is_twelve_hex_characters_and_deterministic() -> None:
    first = compute_agent_version(**base())
    assert len(first) == 12
    int(first, 16)
    assert first == compute_agent_version(**base())


def test_tool_order_does_not_matter() -> None:
    schemas = schemas_of(GUARDED_TOOLS)
    assert compute_agent_version(**base(tool_schemas=schemas)) == compute_agent_version(
        **base(tool_schemas=list(reversed(schemas)))
    )


def test_every_input_changes_the_version() -> None:
    reference = compute_agent_version(**base())
    variants = [
        base(prompts={"base.md": "You are the booking assistant!"}),
        base(prompts={"base.md": "You are the booking assistant.", "guarded.md": "x"}),
        base(tool_schemas=schemas_of(NAIVE_TOOLS)),
        base(model_id="vendor/model-2026-02-01"),
        base(guards_label="off"),
        base(package_version="0.1.1"),
        base(source_digest="b" * 64),
    ]
    versions = {compute_agent_version(**v) for v in variants}
    assert reference not in versions
    assert len(versions) == len(variants)


def test_the_source_hash_covers_agent_calendar_and_crm_sources(tmp_path: Path) -> None:
    for name in SOURCE_DIRS:
        shutil.copytree(PACKAGE_ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    before = source_hash(tmp_path)
    assert before == source_hash(tmp_path)
    for name in SOURCE_DIRS:
        target = next((tmp_path / name).rglob("*.py"))
        original = target.read_bytes()
        target.write_bytes(original + b"\n# changed\n")
        assert source_hash(tmp_path) != before, name
        target.write_bytes(original)
    assert source_hash(tmp_path) == before
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "other.py").write_text("x = 1\n")
    assert source_hash(tmp_path) == before  # other packages are not part of the agent's source
    (tmp_path / "agent" / "notes.txt").write_text("not a source file\n")
    assert source_hash(tmp_path) == before


def test_the_bundled_prompt_is_found() -> None:
    prompts = prompt_files()
    assert "base.md" in prompts
    assert "<context>" not in prompts["base.md"]  # the context block is added per turn
    assert '{"reply":' in prompts["base.md"]


def test_the_claim_type_is_not_shown_as_a_pipe_joined_placeholder() -> None:
    """ "type": "booked|rescheduled|cancelled|offered" reads, to a real model, like an example value to
    copy rather than an enum to pick one from; a model that copies it verbatim produces a "type" the code
    silently drops (it is not one of the four kinds), losing the claim instead of checking it. The example
    must show one concrete kind, with the four kinds spelled out separately in prose."""
    base = prompt_files()["base.md"]
    assert "booked|rescheduled" not in base
    assert re.search(r'"type":\s*"booked"', base)
    for kind in ("booked", "rescheduled", "cancelled", "offered"):
        assert kind in base

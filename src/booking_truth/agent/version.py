"""``agent_version``: one hash over everything that decides the agent's behaviour.

``agent_version = sha256(prompt files + tool schemas + exact model id + guard config + package version +
source hash)[:12]``. The source hash covers the files under ``src/booking_truth/{agent,calendars,crm}``
(Python, Markdown and JSON), because the package version stays the same during development and would
otherwise hide code changes from the harness's drift check.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import booking_truth

PACKAGE_ROOT = Path(booking_truth.__file__).resolve().parent
SOURCE_DIRS: tuple[str, ...] = ("agent", "calendars", "crm")
SOURCE_SUFFIXES: frozenset[str] = frozenset({".py", ".md", ".json"})
PROMPTS_DIR = PACKAGE_ROOT / "agent" / "prompts"
VERSION_CHARS = 12


def _source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in SOURCE_DIRS:
        base = root / name
        if base.is_dir():
            files += [
                p
                for p in base.rglob("*")
                if p.is_file() and p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts
            ]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def source_hash(root: Path | None = None) -> str:
    """sha256 over the sorted relative paths and bytes of the agent, calendar and CRM sources."""
    base = root or PACKAGE_ROOT
    digest = hashlib.sha256()
    for path in _source_files(base):
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@cache
def installed_source_hash() -> str:
    return source_hash()


def prompt_files(directory: Path | None = None) -> dict[str, str]:
    """Every prompt file by name."""
    base = directory or PROMPTS_DIR
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(base.glob("*.md"))}


def load_prompt(name: str = "base.md") -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def compute_agent_version(
    *,
    prompts: Mapping[str, str],
    tool_schemas: Sequence[Mapping[str, Any]],
    model_id: str,
    guards_label: str,
    package_version: str,
    source_digest: str,
) -> str:
    """The 12-character version of one agent configuration."""
    tools = sorted(
        (json.dumps(schema, sort_keys=True, separators=(",", ":")) for schema in tool_schemas),
    )
    material = {
        "prompts": {name: prompts[name] for name in sorted(prompts)},
        "tools": tools,
        "model": model_id,
        "guards": guards_label,
        "package": package_version,
        "source": source_digest,
    }
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:VERSION_CHARS]


def schemas_of(specs: Iterable[Any]) -> list[dict[str, Any]]:
    """OpenAI tool dicts of ``ToolSpec`` objects."""
    return [spec.to_openai() for spec in specs]

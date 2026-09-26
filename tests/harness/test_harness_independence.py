"""The harness measures the agent; it must not reuse the agent's own claim guard.

Every module under ``booking_truth.harness`` is parsed (never imported) and checked for imports of
``booking_truth.agent.guards``: absolute, relative and dynamic (a module path in a string). The belief
extractor, the time reader, grading, metrics and redaction must not touch ``booking_truth.agent`` at all.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[2] / "src" / "booking_truth" / "harness"
PACKAGE = "booking_truth.harness"
FORBIDDEN = "booking_truth.agent.guards"
#: Pure-logic modules that must not depend on the agent package at all.
AGENT_FREE = ("beliefs", "timeparse", "lexicon_extractor", "grading", "metrics", "redact")


def _module_name(path: Path) -> str:
    parts = path.relative_to(HARNESS).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE, *parts))


def _resolve(module: str | None, level: int, current: str, is_package: bool) -> str:
    if level == 0:
        return module or ""
    base = current.split(".") if is_package else current.split(".")[:-1]
    base = base[: len(base) - (level - 1)] if level > 1 else base
    return ".".join([*base, module] if module else base)


def imported_modules(source: str, current: str, *, is_package: bool = False) -> Iterator[str]:
    """Every module a source file imports, including ``from x import y`` as ``x.y`` and module paths in
    string constants (``importlib.import_module("...")``)."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node.module, node.level, current, is_package)
            yield base
            for alias in node.names:
                yield f"{base}.{alias.name}"
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and "booking_truth." in node.value
        ):
            yield node.value.strip()


def _within(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def harness_files() -> list[Path]:
    return sorted(HARNESS.rglob("*.py"))


def test_harness_package_is_found() -> None:
    names = {p.stem for p in harness_files()}
    assert {"lexicon_extractor", "timeparse", "grading"} <= names


@pytest.mark.parametrize("path", harness_files(), ids=lambda p: str(p.relative_to(HARNESS)))
def test_no_harness_module_imports_the_agent_guards(path: Path) -> None:
    current = _module_name(path)
    found = [
        m
        for m in imported_modules(
            path.read_text(encoding="utf-8"), current, is_package=path.stem == "__init__"
        )
        if _within(m, FORBIDDEN)
    ]
    assert found == [], f"{current} imports {found}"


@pytest.mark.parametrize("name", AGENT_FREE)
def test_pure_logic_modules_do_not_touch_the_agent(name: str) -> None:
    path = HARNESS / f"{name}.py"
    found = [
        m
        for m in imported_modules(path.read_text(encoding="utf-8"), f"{PACKAGE}.{name}")
        if _within(m, "booking_truth.agent")
    ]
    assert found == [], f"{name} imports {found}"


@pytest.mark.parametrize(
    "source",
    [
        "import booking_truth.agent.guards.lexicon",
        "from booking_truth.agent.guards import lexicon",
        "from booking_truth.agent import guards",
        "from ..agent.guards.lexicon import detect",
        "from ..agent import guards",
        "import importlib\nimportlib.import_module('booking_truth.agent.guards.lexicon')",
    ],
)
def test_the_check_catches_every_import_form(source: str) -> None:
    found = [m for m in imported_modules(source, f"{PACKAGE}.example") if _within(m, FORBIDDEN)]
    assert found, source


def test_the_check_allows_other_modules() -> None:
    source = (
        "from booking_truth.harness.timeparse import find_times\nfrom ..agent.api import create_agent_app"
    )
    modules = list(imported_modules(source, f"{PACKAGE}.example"))
    assert "booking_truth.agent.api" in modules
    assert not any(_within(m, FORBIDDEN) for m in modules)

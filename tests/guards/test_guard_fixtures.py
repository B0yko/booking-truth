"""Every guard fixture in ``tests/fixtures/guards/``: the scenario meets ``expect_on`` with all guards on, and
``expect_off`` when only that guard (and what depends on it) is off.

The fixtures are regression tests by construction, not a benchmark: each one pairs a scripted misbehaviour
with the guard that exists to catch it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fixture_runner import Mode, describe, failures, fixture_files, load_fixture, run_fixture

FIXTURES = fixture_files()


def test_there_are_guard_fixtures() -> None:
    assert FIXTURES, "tests/fixtures/guards/ has no fixtures"


@pytest.mark.parametrize("mode", ["on", "off"])
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
async def test_guard_fixture(path: Path, mode: Mode) -> None:
    fixture = load_fixture(path)
    if fixture.calendar == "google":
        pytest.skip("the Google Calendar adapter is not part of this build yet")
    expectation = fixture.expect_on if mode == "on" else fixture.expect_off
    observation = await run_fixture(fixture, mode)
    problems = failures(expectation, observation)
    assert not problems, f"{path.name} ({mode}): {problems}\n{describe(observation)}"

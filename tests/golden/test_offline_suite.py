"""The full 24-scenario suite offline, for guarded and naive mode on both calendar shapes.

Each (mode, calendar) combination gets its own sandbox and bundled agent, both pinned to the same fixed
instant (:data:`offline_suite_lib.FIXED_INSTANT`), so a scenario's outcome is exactly reproducible from
one run to the next: nothing here reads the wall clock or talks to the network. This is a regression
suite "by construction, not a benchmark" (``docs/metrics.md``): it does not judge whether an outcome is
good, only whether it is the one already recorded in ``tests/golden/offline_suite.json``.

A mismatch means something changed the agent's, the sandbox's or the scenario suite's behaviour.
Explain the change, then regenerate the snapshot deliberately with
``uv run python scripts/update_offline_snapshot.py``, never by hand.
"""

from __future__ import annotations

import pytest
from offline_suite_lib import (
    CALENDARS,
    MODES,
    BuiltinCalendar,
    BuiltinMode,
    combo_label,
    integrity_violations,
    load_snapshot,
    outcomes_of,
    run_combo,
)

pytestmark = pytest.mark.slow

COMBOS = [(mode, calendar) for mode in MODES for calendar in CALENDARS]
COMBO_IDS = [combo_label(mode, calendar) for mode, calendar in COMBOS]


@pytest.mark.parametrize(("mode", "calendar"), COMBOS, ids=COMBO_IDS)
async def test_offline_suite(mode: BuiltinMode, calendar: BuiltinCalendar) -> None:
    snapshot = load_snapshot()
    label = combo_label(mode, calendar)
    missing = [sid for sid, cols in snapshot["outcomes"].items() if label not in cols]
    assert not missing, f"{label}: no snapshot column for {missing}"
    expected = {sid: cols[label] for sid, cols in snapshot["outcomes"].items()}

    result = await run_combo(mode, calendar, grade_crm=bool(snapshot.get("grade_crm", True)))

    assert result.status == "complete", f"{label}: run did not complete ({result.status}: {result.detail})"
    actual = outcomes_of(result)
    assert actual == expected, (
        f"{label}: outcome drifted from the pinned snapshot; if this is deliberate, explain why and "
        "regenerate it with scripts/update_offline_snapshot.py"
    )
    if mode == "guarded":
        violations = integrity_violations(result)
        assert not violations, f"{label}: the guarded agent had an integrity violation: {violations}"

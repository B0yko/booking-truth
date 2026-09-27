"""README tables: rendering from a results directory, "not run" rows, and marker updates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from fake_run import standard_slots, write_fake_run

from booking_truth.harness.readme_tables import (
    EVAL_EXTRACTOR_FILE,
    EVAL_TZ_FILE,
    TABLE_IDS,
    ReadmeTablesError,
    apply_tables,
    render_tables,
    update_readme,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "update_readme_tables.py"


@pytest.fixture
def results(tmp_path: Path) -> Path:
    out = tmp_path / "results" / "20261001-bench"
    write_fake_run(out, standard_slots(), grading="llm", run_id="20261001-bench")
    return out


def readme_with_markers(ids: tuple[str, ...] = TABLE_IDS) -> str:
    blocks = "\n\n".join(f"<!-- bt:{i} -->\nold\n<!-- /bt:{i} -->" for i in ids)
    return f"# Project\n\nIntro.\n\n{blocks}\n\nThe end.\n"


def test_every_table_renders(results: Path) -> None:
    tables = render_tables(results)
    assert list(tables) == list(TABLE_IDS)
    headline = tables["headline"]
    assert headline.splitlines()[0] == "| Metric | naive | guarded |"
    assert (
        "| False-success rate (trial level) | 33.3% [6.1, 79.2] (1/3) | 33.3% [6.1, 79.2] (1/3) |" in headline
    )
    assert "| Wrong-time rate | 33.3% [6.1, 79.2] (1/3) | 0.0% [0.0, 56.2] (0/3) |" in headline
    assert headline.count("| pass^1 |") == 1
    assert "vendor/model-a" in headline
    assert "MacBook Air M5, 24 GB" in headline
    assert "naive `naive-v1`, guarded `guarded-v1`" in headline
    assert "reproduce: `booking-truth test --pool pool.yaml --k 1`" in headline
    assert "regenerate: `booking-truth report results/20261001-bench`" in headline
    assert "| fault-slots-500-once | 0/1 | 0/1 | 1/1 | 1/1 |" in tables["faults"]
    assert "| tz-ist |" in tables["tz"]
    assert "Total spend of the benchmark run: $0.0060." in tables["cost"]


def test_missing_eval_and_ci_inputs_render_not_run_rows(results: Path) -> None:
    tables = render_tables(results)
    assert "| not run | - | - | - |" in tables["tz-eval"]
    assert "| not run | - | - | - |" in tables["extractor-eval"]
    assert "| not run | - | - |" in tables["ci"]
    assert "By construction, not a benchmark" in tables["ci"]


def test_eval_and_ci_inputs_fill_their_tables(results: Path) -> None:
    """The eval CLI's actual output names and schema (``booking-truth eval tz``/``eval extractor``), not
    the file names or the invented shape an earlier version of this module expected."""
    (results / EVAL_TZ_FILE).write_text(
        json.dumps(
            {
                "n_items": 100,
                "model": "vendor/model-a",
                "deterministic": {
                    "n": 100,
                    "counts": {
                        "correct": 90,
                        "correctly_flagged": 8,
                        "missed_ambiguity": 0,
                        "over_cautious": 0,
                        "silent_wrong_resolution": 2,
                    },
                    "rates": {
                        "correct": 0.9,
                        "correctly_flagged": 0.08,
                        "missed_ambiguity": 0.0,
                        "over_cautious": 0.0,
                        "silent_wrong_resolution": 0.02,
                    },
                },
                "llm": {
                    "n": 100,
                    "counts": {
                        "correct": 80,
                        "correctly_flagged": 3,
                        "missed_ambiguity": 0,
                        "over_cautious": 0,
                        "silent_wrong_resolution": 17,
                    },
                    "rates": {
                        "correct": 0.8,
                        "correctly_flagged": 0.03,
                        "missed_ambiguity": 0.0,
                        "over_cautious": 0.0,
                        "silent_wrong_resolution": 0.17,
                    },
                },
                "llm_skipped_reason": None,
            }
        )
    )
    (results / EVAL_EXTRACTOR_FILE).write_text(
        json.dumps(
            {
                "n_items": 80,
                "model": "vendor/model-a",
                "lexicon": {
                    "n": 80,
                    "status_accuracy": 0.9,
                    "time_match_accuracy": 0.85,
                    "success_recall": 0.97,
                },
                "llm": {"n": 80, "status_accuracy": 0.95, "time_match_accuracy": 0.9, "success_recall": None},
                "llm_skipped_reason": None,
            }
        )
    )
    (results / "ci_fixtures.json").write_text(
        json.dumps(
            {"fixtures": [{"name": "claim_ledger", "status": "pass"}, {"name": "dedupe", "status": "fail"}]}
        )
    )
    tables = render_tables(results)
    tz = tables["tz-eval"]
    expected_row = (
        "| Deterministic | 90.0% (90/100) | 8.0% (8/100) | 0.0% (0/100) | 0.0% (0/100) | 2.0% (2/100) |"
    )
    assert expected_row in tz
    assert "| LLM-only (same model) |" in tz
    assert "| Lexicon | 90.0% | 85.0% | 97.0% |" in tables["extractor-eval"]
    assert "| LLM | 95.0% | 90.0% | - |" in tables["extractor-eval"]
    assert "| naive | 100.0% [20.7, 100.0] (1/1) |" in tables["extractor-eval"]
    assert "`20261001-bench/naive/fault-slots-500-once/0/1`" in tables["extractor-eval"]
    assert "| 2 | 1 | 1 |" in tables["ci"]


def test_eval_tables_render_a_real_run_s_output_byte_for_byte_faithfully(results: Path) -> None:
    """The eval CLI's real output (a trimmed copy of an actual ``booking-truth eval tz``/``eval
    extractor`` run), not a hand-written stand-in for its schema."""
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "eval_outputs"
    (results / EVAL_TZ_FILE).write_text((fixtures / "tz-eval.json").read_text(encoding="utf-8"))
    extractor_fixture = (fixtures / "extractor-eval.json").read_text(encoding="utf-8")
    (results / EVAL_EXTRACTOR_FILE).write_text(extractor_fixture)
    tables = render_tables(results)
    tz = tables["tz-eval"]
    expected_row = (
        "| Deterministic | 61.0% (61/100) | 15.0% (15/100) | 5.0% (5/100) | 19.0% (19/100) | 0.0% (0/100) |"
    )
    assert expected_row in tz
    assert "| LLM-only (same model) | 74.0% (74/100) |" in tz
    assert "deepseek/deepseek-v4-flash" in tz
    extractor = tables["extractor-eval"]
    assert "| Lexicon | 92.5% | 88.8% | 100.0% |" in extractor
    assert "| LLM | 93.8% | 98.8% | 100.0% |" in extractor


def test_failures_lists_guarded_violations_with_their_root_cause(results: Path) -> None:
    trace = "20261001-bench/guarded/tz-ist/0/1"
    (results / "failure_notes.yaml").write_text(
        f"{trace}: >\n  The stated time was rendered\n  in the host zone.\n"
    )
    table = render_tables(results)["failures"]
    assert f"| `{trace}` | time_mismatch | The stated time was rendered in the host zone. |" in table
    assert "naive" not in table.split("<sub>")[0]


def test_update_readme_rewrites_blocks_and_check_mode_only_reports(results: Path, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(readme_with_markers(), encoding="utf-8")
    checked = update_readme(readme, results, check=True)
    assert checked.changed == list(TABLE_IDS)
    assert readme.read_text() == readme_with_markers()
    written = update_readme(readme, results)
    assert written.changed == list(TABLE_IDS)
    text = readme.read_text()
    assert text.startswith("# Project\n\nIntro.\n\n<!-- bt:headline -->\n| Metric |")
    assert text.endswith("<!-- /bt:ci -->\n\nThe end.\n")
    again = update_readme(readme, results, check=True)
    assert again.in_sync
    assert again.changed == []


def test_missing_markers_are_reported_not_invented() -> None:
    update = apply_tables(readme_with_markers(("headline",)), {i: f"table {i}" for i in TABLE_IDS})
    assert update.changed == ["headline"]
    assert update.missing == [i for i in TABLE_IDS if i != "headline"]
    assert "table faults" not in update.text


def test_results_without_a_summary_are_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ReadmeTablesError, match=r"summary\.json is missing"):
        render_tables(tmp_path / "empty")


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("update_readme_tables", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ci_script_checks_and_updates(
    results: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    readme = tmp_path / "README.md"
    readme.write_text(readme_with_markers(), encoding="utf-8")
    assert script.main([str(results), "--readme", str(readme), "--check"]) == 1
    assert "out of date: headline" in capsys.readouterr().err
    assert script.main([str(results), "--readme", str(readme)]) == 0
    assert script.main([str(results), "--readme", str(readme), "--check"]) == 0
    assert script.main([str(tmp_path / "nope"), "--readme", str(readme)]) == 2

"""README tables: rendering from a results directory, "not run" rows, and marker updates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from fake_run import standard_slots, write_fake_run

from booking_truth.harness.readme_tables import (
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
    (results / "eval_tz.json").write_text(
        json.dumps(
            {
                "n": 100,
                "methods": {
                    "resolver": {"correct": 90, "ambiguous_flagged": 8, "silent_wrong": 2},
                    "llm only": {"correct": 80, "ambiguous_flagged": 3, "silent_wrong": 17},
                },
            }
        )
    )
    (results / "eval_extractor.json").write_text(
        json.dumps(
            {
                "n": 80,
                "methods": {
                    "llm": {"status_accuracy": 0.95, "time_accuracy": 0.9, "success_recall": None},
                    "lexicon": {"status_accuracy": 0.9, "time_accuracy": 0.85, "success_recall": 0.97},
                },
            }
        )
    )
    (results / "ci_fixtures.json").write_text(
        json.dumps(
            {"fixtures": [{"name": "claim_ledger", "status": "pass"}, {"name": "dedupe", "status": "fail"}]}
        )
    )
    tables = render_tables(results)
    assert "| resolver | 90/100 | 8/100 | 2/100 |" in tables["tz-eval"]
    assert "| lexicon | 90.0% | 85.0% | 97.0% |" in tables["extractor-eval"]
    assert "| llm | 95.0% | 90.0% | - |" in tables["extractor-eval"]
    assert "| naive | 100.0% [20.7, 100.0] (1/1) |" in tables["extractor-eval"]
    assert "`20261001-bench/naive/fault-slots-500-once/0/1`" in tables["extractor-eval"]
    assert "| 2 | 1 | 1 |" in tables["ci"]


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

"""``RunConfig.llm_extractor``: the runner switches to LLM grading, records both beliefs and labels the
trace's ``meta.claims_source`` accordingly, exactly the wiring ``booking-truth test`` performs once an LLM
key is configured. The extractor here is a stand-in for :class:`~booking_truth.harness.llm_extractor.
LLMExtractor`; that class's own request and parsing behaviour is covered by ``test_llm_extractor.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from stub_agent import SANDBOX_TOKEN, bundled_agent, running_stub

from booking_truth.harness.beliefs import Belief, BeliefSource
from booking_truth.harness.lexicon_extractor import LexiconBeliefExtractor
from booking_truth.harness.report import MANIFEST_FILE, TRACES_FILE
from booking_truth.harness.runner import RunConfig, run
from booking_truth.harness.scenarios import load_suite, select_scenarios
from booking_truth.trace.validate import iter_jsonl

SUITE = load_suite()


class FakeExtractor:
    """Delegates to the lexicon extractor but reports as the LLM source and spends money, so the wiring
    (grading mode, cost split, claims_source) can be checked without a real model."""

    source: BeliefSource = "llm"

    def __init__(self) -> None:
        self.usage_usd = 0.0
        self._lexicon = LexiconBeliefExtractor()

    async def extract(
        self, agent_messages: Sequence[str], *, prospect_zone: str, host_zone: str, reference: datetime
    ) -> Belief:
        self.usage_usd += 0.0021
        belief = await self._lexicon.extract(
            agent_messages, prospect_zone=prospect_zone, host_zone=host_zone, reference=reference
        )
        return Belief(
            status=belief.status,
            time_utc=belief.time_utc,
            offered_utc=belief.offered_utc,
            source="llm",
            evidence=belief.evidence,
        )


async def test_the_runner_switches_to_llm_grading_when_an_extractor_is_configured(
    sandbox_url: str, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    with running_stub(sandbox_url) as (_, base):
        agent = bundled_agent("stub", base, sandbox_url)
        extractor = FakeExtractor()
        config = RunConfig(
            agents=[agent],
            scenarios=select_scenarios(SUITE, ["happy-book-host-zone"]),
            suite_scenarios=SUITE,
            k=1,
            run_id="llm-wiring-run",
            out_dir=out,
            settle_s=3,
            stable_s=0.15,
            hardware="test machine, 1 GB",
            sandbox_token=SANDBOX_TOKEN,
            command="booking-truth test --agent http://localhost:8000/v1/chat --sandbox http://localhost:8100",
            llm_extractor=extractor,
            extractor_model="fake/extractor-model",
        )
        result = await run(config)

    assert result.status == "complete"
    assert result.results[0].outcome == "pass"
    assert result.results[0].llm_belief_status == "booked"
    assert result.results[0].lexicon_belief_status == "booked"
    assert result.results[0].extractor_usd > 0

    manifest = json.loads((out / MANIFEST_FILE).read_text())
    assert manifest["grading"]["mode"] == "llm"
    assert manifest["grading"]["extractor"] == "llm"
    assert manifest["grading"]["persona"] == "scripted"  # no persona_factory: still the scripted persona
    assert manifest["models"]["extractor"] == "fake/extractor-model"

    traces = [record for _, record in iter_jsonl(out / TRACES_FILE)]
    final = next(t for t in traces if t["meta"]["final_attempt"])
    assert final["meta"]["claims_source"] == "llm"

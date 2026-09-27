"""``RunConfig.llm_extractor``: the runner switches to LLM grading, records both beliefs and labels the
trace's ``meta.claims_source`` accordingly, exactly the wiring ``booking-truth test`` performs once an LLM
key is configured. The extractor here is a stand-in for :class:`~booking_truth.harness.llm_extractor.
LLMExtractor`; that class's own request and parsing behaviour is covered by ``test_llm_extractor.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest
from stub_agent import SANDBOX_TOKEN, bundled_agent, running_stub

from booking_truth.harness.beliefs import Belief, BeliefSource
from booking_truth.harness.lexicon_extractor import LexiconBeliefExtractor
from booking_truth.harness.personas import AgentView, ScriptedPersona
from booking_truth.harness.report import MANIFEST_FILE, TRACES_FILE
from booking_truth.harness.runner import AgentUnderTest, RunConfig, run
from booking_truth.harness.scenarios import ResolvedScenario, load_suite, select_scenarios
from booking_truth.llm.types import LLMError
from booking_truth.sandbox.app import create_sandbox_app
from booking_truth.serve import BackgroundServer
from booking_truth.trace.validate import iter_jsonl

SUITE = load_suite()


class FakeExtractor:
    """Delegates to the lexicon extractor but reports as the LLM source and spends money, so the wiring
    (grading mode, cost split, claims_source) can be checked without a real model."""

    source: BeliefSource = "llm"

    def __init__(self) -> None:
        self.usage_usd = 0.0
        self.calls_made: list[tuple[str, str | None]] = []
        self._lexicon = LexiconBeliefExtractor()

    async def extract(
        self, agent_messages: Sequence[str], *, prospect_zone: str, host_zone: str, reference: datetime
    ) -> Belief:
        self.usage_usd += 0.0021
        self.calls_made.append(("fake/extractor-model", "fake-provider"))
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
    # The extractor's own calls (run-1 defect 6: only the agent's turns were ever attributed) land in the
    # manifest's llm_calls under "extractor", not folded into "agent:stub" or dropped altogether.
    assert manifest["llm_calls"]["extractor"]["calls"] == 1
    assert manifest["llm_calls"]["extractor"]["models_returned"] == ["fake/extractor-model"]
    assert manifest["llm_calls"]["extractor"]["providers"] == ["fake-provider"]
    assert "persona" not in manifest["llm_calls"]  # no persona_factory: nothing to attribute

    traces = [record for _, record in iter_jsonl(out / TRACES_FILE)]
    final = next(t for t in traces if t["meta"]["final_attempt"])
    assert final["meta"]["claims_source"] == "llm"


class RecordingPersona:
    """Wraps a scripted persona, recording a fake (model, provider) pair per turn as ``LLMPersona`` would
    from a real call - a stand-in that drives the same conversation without any HTTP, so the runner's
    manifest wiring for the ``persona`` component can be checked in isolation."""

    def __init__(self, resolved: ResolvedScenario, supports_actions: bool) -> None:
        self._inner = ScriptedPersona(resolved, supports_actions=supports_actions)
        self.calls_made: list[tuple[str, str | None]] = []

    @property
    def turns(self) -> int:
        return self._inner.turns

    @property
    def usage_usd(self) -> float:
        return self._inner.usage_usd

    async def next_turn(self, view: AgentView) -> object:
        turn = await self._inner.next_turn(view)
        if turn is not None:
            self.calls_made.append(("fake/persona-model", "fake-provider"))
        return turn


async def test_persona_calls_are_attributed_in_the_manifest_too(sandbox_url: str, tmp_path: Path) -> None:
    out = tmp_path / "run"
    with running_stub(sandbox_url) as (_, base):
        agent = bundled_agent("stub", base, sandbox_url)
        config = RunConfig(
            agents=[agent],
            scenarios=select_scenarios(SUITE, ["happy-book-host-zone"]),
            suite_scenarios=SUITE,
            k=1,
            run_id="persona-wiring-run",
            out_dir=out,
            settle_s=3,
            stable_s=0.15,
            hardware="test machine, 1 GB",
            sandbox_token=SANDBOX_TOKEN,
            command="booking-truth test --agent http://localhost:8000/v1/chat --sandbox http://localhost:8100",
            persona_factory=RecordingPersona,
            persona_model="fake/persona-model",
        )
        result = await run(config)

    assert result.status == "complete"
    assert result.results[0].outcome == "pass"
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    assert manifest["grading"]["persona"] == "llm"
    persona_calls = manifest["llm_calls"]["persona"]
    assert persona_calls["calls"] > 0
    assert persona_calls["models_returned"] == ["fake/persona-model"]
    assert persona_calls["providers"] == ["fake-provider"]
    assert persona_calls["unknown_attribution_calls"] == 0
    assert manifest["llm_attribution_incomplete"] is False


class ConcurrentCostExtractor:
    """One shared extractor instance used by two trials that run at the same time, on separate sandbox
    pairs, exactly as ``booking-truth test`` wires ``LLMExtractor`` (docs/adr, "Pool"). The first call to
    reach ``extract`` is slow and finishes after the second, so a trial's own extractor spend can only be
    read correctly if it is measured without depending on the shared instance's cumulative total at the
    moment the *other* trial happens to finish.
    """

    source: BeliefSource = "llm"

    SLOW_COST = 0.011
    FAST_COST = 0.003

    def __init__(self) -> None:
        self.usage_usd = 0.0
        self._lexicon = LexiconBeliefExtractor()
        self._started = asyncio.Event()
        self._claimed_slow = False

    async def extract(
        self, agent_messages: Sequence[str], *, prospect_zone: str, host_zone: str, reference: datetime
    ) -> Belief:
        if not self._claimed_slow:
            self._claimed_slow = True
            self._started.set()
            await asyncio.sleep(0.15)
            cost = self.SLOW_COST
        else:
            await self._started.wait()
            cost = self.FAST_COST
        self.usage_usd += cost
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


class ExtractorThatFailsAfterBilling:
    """Bills the model call and then fails to parse its response (a malformed or truncated structured
    output, which a real provider can still return under ``response_format: json_schema``), exactly the
    ``LLMExtractor.extract`` failure mode: the underlying ``llm.chat`` call already succeeded and was
    billed (``usage_usd`` reflects that) before ``_parse`` raises. The harness's own per-trial
    ``extractor_usd`` must still reflect that real spend, not silently drop it because the call that
    incurred it did not return a belief.
    """

    source: BeliefSource = "llm"

    COST = 0.0042

    def __init__(self) -> None:
        self.usage_usd = 0.0

    async def extract(
        self, agent_messages: Sequence[str], *, prospect_zone: str, host_zone: str, reference: datetime
    ) -> Belief:
        self.usage_usd += self.COST
        raise LLMError("the extractor model returned invalid JSON", kind="malformed")


async def test_extractor_spend_is_kept_even_when_the_call_errors_after_billing(
    sandbox_url: str, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    with running_stub(sandbox_url) as (_, base):
        agent = bundled_agent("stub", base, sandbox_url)
        extractor = ExtractorThatFailsAfterBilling()
        config = RunConfig(
            agents=[agent],
            scenarios=select_scenarios(SUITE, ["happy-book-host-zone"]),
            suite_scenarios=SUITE,
            k=1,
            run_id="extractor-error-run",
            out_dir=out,
            settle_s=3,
            stable_s=0.15,
            hardware="test machine, 1 GB",
            sandbox_token=SANDBOX_TOKEN,
            command="booking-truth test --agent http://localhost:8000/v1/chat --sandbox http://localhost:8100",
            llm_extractor=extractor,
            extractor_model="fake/extractor-model",
            max_attempts=1,
        )
        result = await run(config)

    assert result.status == "complete"
    assert result.results[0].outcome == "harness_error"
    # The call really happened and really cost money (the extractor's own cumulative counter proves it);
    # the trial's recorded extractor_usd, which feeds the manifest's total spend and the report's
    # cost-per-conversation table, must account for it too.
    assert extractor.usage_usd == pytest.approx(ExtractorThatFailsAfterBilling.COST)
    assert result.results[0].extractor_usd == pytest.approx(ExtractorThatFailsAfterBilling.COST)


async def test_a_trial_extractor_cost_excludes_a_concurrent_trials_spend(tmp_path: Path) -> None:
    first = BackgroundServer(create_sandbox_app(SANDBOX_TOKEN)).start()
    second = BackgroundServer(create_sandbox_app(SANDBOX_TOKEN)).start()
    try:
        with running_stub(first.url) as (_, base_a), running_stub(second.url) as (_, base_b):
            one = bundled_agent("pool", base_a, first.url).endpoints[0]
            two = bundled_agent("pool", base_b, second.url).endpoints[0]
            agent = AgentUnderTest(label="pool", endpoints=[one, two], target="stub")
            extractor = ConcurrentCostExtractor()
            config = RunConfig(
                agents=[agent],
                scenarios=select_scenarios(SUITE, ["smoke"]),
                suite_scenarios=SUITE,
                k=1,
                run_id="extractor-race-run",
                out_dir=tmp_path / "run",
                settle_s=3,
                stable_s=0.15,
                hardware="test machine, 1 GB",
                sandbox_token=SANDBOX_TOKEN,
                command=(
                    "booking-truth test --agent http://localhost:8000/v1/chat --sandbox http://localhost:8100"
                ),
                llm_extractor=extractor,
                extractor_model="fake/extractor-model",
            )
            result = await run(config)
    finally:
        first.stop()
        second.stop()

    assert result.status == "complete"
    costs = sorted(r.extractor_usd for r in result.results)
    assert costs == pytest.approx([ConcurrentCostExtractor.FAST_COST, ConcurrentCostExtractor.SLOW_COST])
    assert sum(r.extractor_usd for r in result.results) == pytest.approx(
        ConcurrentCostExtractor.FAST_COST + ConcurrentCostExtractor.SLOW_COST
    )

"""``booking-truth test``, ``report`` and ``compare``.

Exit codes of ``test``: 0 when every result slot passed, 1 when the run completed but some trial did not pass,
2 when the run was aborted (wiring preflight, version drift, budget stop), is invalid, or the command line is
wrong. A ``--dry-run`` exits 0 unless it was aborted.
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from booking_truth.config import load_settings
from booking_truth.harness.adapters import (
    DEFAULT_API_KEY,
    AgentClient,
    AgentConfig,
    AgentConfigError,
    BundledAgentClient,
    BundledEndpoints,
    HttpAgentClient,
    load_agent_config,
)
from booking_truth.harness.beliefs import BeliefExtractor
from booking_truth.harness.builtin import (
    BUILTIN_TARGETS,
    BuiltinAgent,
    BuiltinUnavailable,
    start_builtin_agent,
    start_sandbox,
)
from booking_truth.harness.llm_extractor import LLMExtractor
from booking_truth.harness.llm_persona import LLMPersona
from booking_truth.harness.personas import Persona
from booking_truth.harness.report import (
    MANIFEST_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    TRACES_FILE,
    CompareRefused,
    RunDirError,
    compare_runs,
    fmt_pass_hat,
    fmt_rate,
    regenerate,
)
from booking_truth.harness.runner import (
    AgentUnderTest,
    Endpoint,
    PersonaFactory,
    RunAborted,
    RunConfig,
    RunResult,
    run,
)
from booking_truth.harness.sandbox_client import DEFAULT_TOKEN
from booking_truth.harness.scenarios import ResolvedScenario, ScenarioError, load_suite, select_scenarios
from booking_truth.llm.client import OpenAICompatClient
from booking_truth.llm.types import LLMError
from booking_truth.serve import BackgroundServer

EXIT_PASS, EXIT_FAILURES, EXIT_ABORTED = 0, 1, 2
_LABEL = re.compile(r"^([a-z0-9][a-z0-9_-]{0,39})=(.+)$")


class UsageError(ValueError):
    """A wrong combination of command-line options."""


# Pool file --------------------------------------------------------------------------------------------------


class PoolPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1)
    sandbox: str = Field(min_length=1)


class PoolAgent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,39}$")
    mode: Literal["guarded", "naive"] | None = None
    agent_config: str | None = None
    pairs: list[PoolPair] = Field(min_length=1)


class PoolFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: list[PoolAgent] = Field(min_length=1)


def load_pool(path: Path) -> PoolFile:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return PoolFile.model_validate(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise UsageError(f"{path.name}: cannot read the pool file: {exc}") from None
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors(include_url=False)
        )
        raise UsageError(f"{path.name}: invalid pool file: {problems}") from None


# Agents -----------------------------------------------------------------------------------------------------


@dataclass
class Resources:
    """Servers started for this run (in-process sandboxes and builtin agents), stopped at the end."""

    servers: list[BackgroundServer] = field(default_factory=list)
    agents: list[BuiltinAgent] = field(default_factory=list)

    def close(self) -> None:
        for agent in self.agents:
            agent.stop()
        for server in self.servers:
            server.stop()


def http_endpoint(
    name: str, url: str, sandbox_url: str, config: AgentConfig | None, api_key: str | None
) -> tuple[Endpoint, Literal["bundled", "agent.yaml"]]:
    """An endpoint driven with the bundled protocol (bearer ``api_key``, else ``BT_API_KEY``, else
    ``dev-local-key``) or, with ``config``, with the generic HTTP adapter."""
    if config is None:
        key = api_key or os.environ.get("BT_API_KEY") or DEFAULT_API_KEY

        def bundled() -> AgentClient:
            return BundledAgentClient(url, api_key=key)

        side = BundledEndpoints(url, key)
        return Endpoint(name=name, sandbox_url=sandbox_url, make_client=bundled, side=side), "bundled"
    configured = config.model_copy(update={"url": url})

    def generic() -> AgentClient:
        return HttpAgentClient(configured)

    return Endpoint(name=name, sandbox_url=sandbox_url, make_client=generic), "agent.yaml"


def _split_label(raw: str) -> tuple[str | None, str]:
    match = _LABEL.match(raw)
    if match and not match[1].startswith(("http", "builtin")):
        return match[1], match[2]
    return None, raw


def build_agents(
    specs: Sequence[str],
    *,
    sandbox: str | None,
    agent_config: AgentConfig | None,
    pool: PoolFile | None,
    pool_dir: Path | None,
    sandbox_token: str,
    resources: Resources,
    agent_factory: Callable[..., Any] | None = None,
    api_key: str | None = None,
) -> tuple[list[AgentUnderTest], str]:
    """The agents under test and the sandbox token the harness uses. ``api_key`` is the bundled protocol's
    bearer (``BT_API_KEY`` from the environment or ``.env``)."""
    if pool is not None:
        if specs or sandbox is not None:
            raise UsageError("--pool replaces --agent and --sandbox; do not combine them")
        pooled: list[AgentUnderTest] = []
        for entry in pool.agents:
            config = agent_config
            if entry.agent_config is not None:
                config_path = (pool_dir or Path.cwd()) / entry.agent_config
                config = load_agent_config(config_path)
            endpoints: list[Endpoint] = []
            protocol: Literal["bundled", "agent.yaml"] = "bundled"
            for index, pair in enumerate(entry.pairs, start=1):
                endpoint, protocol = http_endpoint(
                    f"{entry.label}#{index}", pair.agent, pair.sandbox, config, api_key
                )
                endpoints.append(endpoint)
            pooled.append(
                AgentUnderTest(
                    label=entry.label,
                    endpoints=endpoints,
                    kind="http",
                    protocol=protocol,
                    target=entry.pairs[0].agent,
                    mode=entry.mode,
                )
            )
        return pooled, sandbox_token

    if not specs:
        if agent_config is None:
            raise UsageError(
                "pass --agent <url|builtin|builtin:naive> (or --pool, or --agent-config with a url)"
            )
        specs = [agent_config.url]
    parsed = [_split_label(raw) for raw in specs]
    for _, target in parsed:
        if target not in BUILTIN_TARGETS and not target.startswith(("http://", "https://")):
            raise UsageError(f"--agent {target!r}: expected an http(s) URL, builtin or builtin:naive")
    builtin = [target in BUILTIN_TARGETS for _, target in parsed]
    if sandbox == "auto" and not all(builtin):
        raise UsageError(
            "--sandbox auto is valid only with --agent builtin or builtin:naive; with an agent URL, "
            "--sandbox must be the sandbox URL that agent is configured to use"
        )
    if sandbox is None and not all(builtin):
        raise UsageError(
            "--sandbox <url> is required with an agent URL: the sandbox that agent is configured to use"
        )
    agents: list[AgentUnderTest] = []
    used: set[str] = set()
    token = sandbox_token
    if sandbox in (None, "auto") and sandbox_token == DEFAULT_TOKEN:
        token = secrets.token_urlsafe(16)  # in-process sandboxes get a fresh token
    for index, (label, target) in enumerate(parsed, start=1):
        if target in BUILTIN_TARGETS:
            mode = BUILTIN_TARGETS[target]
            name = label or mode
            if sandbox in (None, "auto"):
                server, _ = start_sandbox(token)
                resources.servers.append(server)
                sandbox_url = server.url
            else:
                assert sandbox is not None
                sandbox_url = sandbox
            started = start_builtin_agent(
                mode, sandbox_url=sandbox_url, sandbox_token=token, api_key=api_key, factory=agent_factory
            )
            resources.agents.append(started)
            endpoint, _ = http_endpoint(name, started.url, sandbox_url, None, started.api_key)
            agent = AgentUnderTest(
                label=name, endpoints=[endpoint], kind="builtin", protocol="bundled", target=target, mode=mode
            )
        else:
            if not target.startswith(("http://", "https://")):
                raise UsageError(f"--agent {target!r}: expected an http(s) URL, builtin or builtin:naive")
            name = label or ("agent" if len(parsed) == 1 else f"agent-{index}")
            assert sandbox is not None
            endpoint, protocol = http_endpoint(name, target, sandbox, agent_config, api_key)
            agent = AgentUnderTest(
                label=name, endpoints=[endpoint], kind="http", protocol=protocol, target=target
            )
        if agent.label in used:
            raise UsageError(f"two agents are labelled {agent.label!r}; name them with --agent <label>=<url>")
        used.add(agent.label)
        agents.append(agent)
    return agents, token


# Command line -----------------------------------------------------------------------------------------------


def _parse_as_of(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(
            f"expected a date as YYYY-MM-DD, got {value!r}", param_hint="--as-of"
        ) from None


def _only(values: Sequence[str] | None) -> list[str]:
    found: list[str] = []
    for value in values or []:
        found += [part.strip() for part in value.split(",") if part.strip()]
    return found


def _display_path(path: Path) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path.name


def reproduce_command(
    *,
    specs: Sequence[str],
    agent_config: Path | None,
    sandbox: str | None,
    suite: Path | None,
    only: Sequence[str],
    k: int,
    pool: Path | None,
    grade_crm: bool,
    settle_s: float,
    persona_model: str | None,
    extractor_model: str | None,
    as_of: date | None,
    hardware: str | None,
    budget_usd: float | None,
    dry_run: bool,
) -> str:
    """The command line of this run, with file paths reduced to their names."""
    parts = ["booking-truth", "test"]
    for spec in specs:
        parts += ["--agent", spec]
    if agent_config is not None:
        parts += ["--agent-config", agent_config.name]
    if sandbox is not None:
        parts += ["--sandbox", sandbox]
    if pool is not None:
        parts += ["--pool", pool.name]
    if suite is not None:
        parts += ["--suite", suite.name]
    for value in only:
        parts += ["--only", value]
    parts += ["--k", str(k)]
    if grade_crm:
        parts.append("--grade-crm")
    if settle_s != 15.0:
        parts += ["--settle-s", f"{settle_s:g}"]
    if persona_model:
        parts += ["--persona-model", persona_model]
    if extractor_model:
        parts += ["--extractor-model", extractor_model]
    if as_of is not None:
        parts += ["--as-of", as_of.isoformat()]
    if hardware:
        parts += ["--hardware", hardware]
    if budget_usd is not None:
        parts += ["--budget-usd", f"{budget_usd:g}"]
    if dry_run:
        parts.append("--dry-run")
    return shlex.join(parts)


def _echo_result(result: RunResult) -> None:
    summary = result.summary or {}
    typer.echo(f"run {result.run_id}: {result.status}" + (f" ({result.detail})" if result.detail else ""))
    for agent, entry in (summary.get("by_agent") or {}).items():
        passed = entry["outcomes"]["pass"]
        typer.echo(
            f"  {agent}: {passed}/{entry['slots']} pass; pass^1 {fmt_pass_hat(entry['pass_hat_1'])}; "
            f"false-success rate {fmt_rate(entry['false_success_rate'])}"
        )
    grading = (result.manifest.get("grading") or {}).get("label")
    if grading:
        typer.echo(f"  grading: {grading}")
    accounting = summary.get("accounting") or {}
    if accounting and not accounting.get("ok"):
        typer.echo(
            f"  accounting: {accounting.get('slots')}/{accounting.get('expected_slots')} slots, "
            f"{accounting.get('harness_error_slots')} harness_error"
            + (" - the run is INVALID (over 2% harness_error)" if accounting.get("invalid") else ""),
            err=True,
        )
    if result.projection is not None:
        projection = result.projection
        typer.echo(
            f"projected cost of the full run: ${projection['projected_usd']:.4f}; with the "
            f"{projection['safety_factor']}x safety factor ${projection['projected_usd_with_safety']:.4f}"
        )
    if result.out_dir is not None:
        files = ", ".join((SUMMARY_FILE, REPORT_FILE, TRACES_FILE, MANIFEST_FILE))
        typer.echo(f"outputs in {_display_path(result.out_dir)}: {files}")


def exit_code(result: RunResult) -> int:
    summary = result.summary or {}
    if result.status != "complete":
        return EXIT_ABORTED
    accounting = summary.get("accounting") or {}
    if accounting and accounting.get("invalid"):
        return EXIT_ABORTED
    if result.projection is not None:
        return EXIT_PASS
    if all(r.outcome == "pass" for r in result.results) and result.results:
        return EXIT_PASS
    return EXIT_FAILURES


def cmd_test(
    agent: Annotated[
        list[str] | None,
        typer.Option(
            "--agent",
            help="Agent under test: an URL (bundled /v1/chat protocol unless --agent-config), builtin or "
            "builtin:naive. Repeat to test several agents; name one with <label>=<url>.",
        ),
    ] = None,
    agent_config: Annotated[
        Path | None,
        typer.Option(
            "--agent-config", exists=True, dir_okay=False, help="agent.yaml for the generic HTTP adapter."
        ),
    ] = None,
    sandbox: Annotated[
        str | None,
        typer.Option("--sandbox", help="The sandbox URL the agent uses, or auto (builtin agents only)."),
    ] = None,
    suite: Annotated[
        Path | None,
        typer.Option("--suite", file_okay=False, exists=True, help="Scenario directory (default: bundled)."),
    ] = None,
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Scenario id or tag; repeat or comma-separate.")
    ] = None,
    k: Annotated[int, typer.Option("--k", min=1, max=50, help="Trials per scenario.")] = 5,
    pool: Annotated[
        Path | None,
        typer.Option(
            "--pool", exists=True, dir_okay=False, help="pool.yaml: labelled (agent, sandbox) pairs."
        ),
    ] = None,
    grade_crm: Annotated[bool, typer.Option("--grade-crm", help="Also grade the HubSpot CRM state.")] = False,
    settle_s: Annotated[
        float,
        typer.Option("--settle-s", min=0, max=600, help="Max seconds to wait for the end state to settle."),
    ] = 15.0,
    persona_model: Annotated[
        str | None, typer.Option("--persona-model", help="Model id for LLM personas.")
    ] = None,
    extractor_model: Annotated[
        str | None, typer.Option("--extractor-model", help="Model id for the LLM belief extractor.")
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", metavar="YYYY-MM-DD", help="Compute scenario dates from this date."),
    ] = None,
    hardware: Annotated[
        str | None, typer.Option("--hardware", help="Hardware text for the manifest.")
    ] = None,
    budget_usd: Annotated[
        float | None, typer.Option("--budget-usd", min=0, help="Stop before the run's spend would pass this.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Run one happy-path and one fault scenario and project the cost."),
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output directory (default: runs/<run id>).")
    ] = None,
) -> None:
    """Test appointment-setting agents against the sandbox and grade the calendar's end state."""
    specs = list(agent or [])
    only_list = _only(only)
    as_of_date = _parse_as_of(as_of)
    try:
        settings = load_settings()
    except ValueError as exc:
        typer.echo(f"error: invalid BT_* settings: {exc}", err=True)
        raise typer.Exit(EXIT_ABORTED) from None
    try:
        scenarios = load_suite(suite)
        selected = select_scenarios(scenarios, only_list)
    except ScenarioError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ABORTED) from None
    offline = settings.offline
    reason = "no LLM key configured"
    persona_model_id = persona_model or settings.persona_model_id
    extractor_model_id = extractor_model or settings.extractor_model_id
    if offline:
        typer.echo(
            f"note: {reason}; grading offline with scripted personas and the lexicon extractor", err=True
        )
    else:
        typer.echo(
            f"note: LLM personas ({persona_model_id}) and the LLM belief extractor "
            f"({extractor_model_id}) are active",
            err=True,
        )
    started = datetime.now(UTC)
    run_id = out.name if out is not None else started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = out if out is not None else Path("runs") / run_id
    resources = Resources()
    persona_llm: OpenAICompatClient | None = None
    extractor_llm: OpenAICompatClient | None = None
    persona_factory: PersonaFactory | None = None
    llm_extractor: BeliefExtractor | None = None
    try:
        if not offline:
            persona_llm = OpenAICompatClient.from_settings(
                settings, component="persona", model=persona_model_id
            )
            extractor_llm = OpenAICompatClient.from_settings(
                settings, component="extractor", model=extractor_model_id
            )

            def _make_persona(resolved: ResolvedScenario, supports_actions: bool) -> Persona:
                assert persona_llm is not None
                return LLMPersona(
                    resolved, persona_llm, model=persona_model_id, supports_actions=supports_actions
                )

            persona_factory = _make_persona
            llm_extractor = LLMExtractor(extractor_llm, model=extractor_model_id)
        config_model = load_agent_config(agent_config) if agent_config is not None else None
        pool_model = load_pool(pool) if pool is not None else None
        agents, token = build_agents(
            specs,
            sandbox=sandbox,
            agent_config=config_model,
            pool=pool_model,
            pool_dir=pool.parent if pool is not None else None,
            sandbox_token=settings.sandbox_token.get_secret_value(),
            resources=resources,
            api_key=settings.api_key.get_secret_value() if settings.api_key is not None else None,
        )
        config = RunConfig(
            agents=agents,
            scenarios=selected,
            suite_scenarios=scenarios,
            k=k,
            suite_dir=suite,
            run_id=run_id,
            out_dir=out_dir,
            grade_crm=grade_crm,
            settle_s=settle_s,
            as_of=as_of_date,
            hardware=hardware,
            budget_usd=budget_usd,
            ledger_cap_usd=settings.budget_usd,
            ledger_dir=settings.resolved_ledger_dir if settings.budget_usd is not None else None,
            dry_run=dry_run,
            offline_reason=reason,
            persona_model=persona_model_id if persona_factory is not None else persona_model,
            extractor_model=extractor_model_id if llm_extractor is not None else extractor_model,
            sandbox_token=token,
            only=only_list,
            persona_factory=persona_factory,
            llm_extractor=llm_extractor,
            command=reproduce_command(
                specs=specs,
                agent_config=agent_config,
                sandbox=sandbox,
                suite=suite,
                only=only_list,
                k=k,
                pool=pool,
                grade_crm=grade_crm,
                settle_s=settle_s,
                persona_model=persona_model,
                extractor_model=extractor_model,
                as_of=as_of_date,
                hardware=hardware,
                budget_usd=budget_usd,
                dry_run=dry_run,
            ),
            progress=typer.echo,
        )

        async def _execute() -> RunResult:
            try:
                return await run(config)
            finally:
                if persona_llm is not None:
                    await persona_llm.aclose()
                if extractor_llm is not None:
                    await extractor_llm.aclose()

        result = asyncio.run(_execute())
    except (UsageError, AgentConfigError, BuiltinUnavailable, ScenarioError, LLMError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ABORTED) from None
    except RunAborted as exc:
        typer.echo(f"aborted: {exc}", err=True)
        raise typer.Exit(EXIT_ABORTED) from None
    finally:
        resources.close()
    _echo_result(result)
    raise typer.Exit(exit_code(result))


def cmd_report(
    run_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="A run directory (runs/<id>).")
    ],
) -> None:
    """Recompute summary.json and report.md from traces.jsonl and manifest.json (no network)."""
    try:
        summary = regenerate(run_dir)
    except RunDirError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ABORTED) from None
    accounting = summary.get("accounting") or {}
    typer.echo(
        f"wrote {SUMMARY_FILE} and {REPORT_FILE} in {_display_path(run_dir)} "
        f"({accounting.get('slots')}/{accounting.get('expected_slots')} result slots)"
    )


def cmd_compare(
    run_a: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="The baseline run directory.")],
    run_b: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="The run to compare with it.")],
    allow_version_drift: Annotated[
        bool, typer.Option("--allow-version-drift", help="Compare even when an agent's version changed.")
    ] = False,
) -> None:
    """Compare two runs' headline metrics; refuses when an agent's version changed."""
    try:
        text = compare_runs(run_a, run_b, allow_version_drift=allow_version_drift)
    except (RunDirError, CompareRefused) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ABORTED) from None
    typer.echo(text)


def register(app: typer.Typer) -> None:
    """Add ``test``, ``report`` and ``compare`` to the root command."""
    app.command("test")(cmd_test)
    app.command("report")(cmd_report)
    app.command("compare")(cmd_compare)

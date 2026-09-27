"""The deployment files stay consistent with the code: .env.example, the Docker build context, both compose
files, the benchmark pool and the widget size check."""

from __future__ import annotations

import importlib.util
import os
import re
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from booking_truth import __version__
from booking_truth.config import Settings
from booking_truth.harness.cli_test import load_pool

ROOT = Path(__file__).resolve().parents[2]
IMAGE = f"ghcr.io/b0yko/booking-truth:{__version__}"
LLM_KEY = "${BT_LLM_API_KEY:-}"


def compose(name: str) -> dict[str, Any]:
    data = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)", line)
        if match:
            assert match.group(1) not in values, f"{match.group(1)} is listed twice"
            values[match.group(1)] = match.group(2)
    return values


def mib(limit: str) -> int:
    match = re.fullmatch(r"(\d+)([mg])", limit.lower())
    assert match, limit
    return int(match.group(1)) * (1024 if match.group(2) == "g" else 1)


def port(url_or_mapping: str) -> int:
    return int(re.findall(r":(\d+)", url_or_mapping)[0])


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("BT_") or name == "OPENROUTER_API_KEY":
            monkeypatch.delenv(name)


# .env.example ----------------------------------------------------------------------------------------------


def test_env_example_lists_every_setting_once() -> None:
    expected = {f"BT_{name.upper()}" for name in Settings.model_fields}
    assert set(env_example()) == expected


@pytest.mark.usefixtures("clean_env")
def test_copying_env_example_changes_only_the_api_key() -> None:
    defaults = Settings(_env_file=None)  # type: ignore[call-arg]
    copied = Settings(_env_file=ROOT / ".env.example")  # type: ignore[call-arg]
    assert copied.offline
    assert copied.api_key is not None
    assert copied.api_key.get_secret_value() == "dev-local-key"
    skip = {"api_key", "session_secret"}  # the session secret is random per process when unset
    assert copied.model_dump(exclude=skip) == defaults.model_dump(exclude=skip)


# Docker build context --------------------------------------------------------------------------------------


def test_every_packaged_data_file_reaches_the_docker_build() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sources = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    allowed = {line[1:].rstrip("/") for line in ignore if line.startswith("!")}
    copied: set[str] = set()
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY ") and "--from" not in line:
            copied.update(p.rstrip("/").removeprefix("./") for p in line.split()[1:-1])
    for source in [*sources, "pyproject.toml", "uv.lock", "README.md", "LICENSE"]:
        top = source.split("/")[0]
        assert source in allowed or top in allowed, f".dockerignore drops {source}"
        assert source in copied or top in copied, f"the Dockerfile does not copy {source}"


# docker-compose.yml ----------------------------------------------------------------------------------------


def test_demo_stack_builds_or_pulls_the_release_image() -> None:
    services = compose("docker-compose.yml")["services"]
    assert set(services) == {"sandbox", "agent"}
    for service in services.values():
        assert service["image"] == IMAGE
        assert service["build"] == "."
        assert service["ports"]
        assert all(p.startswith("127.0.0.1:") for p in service["ports"])


def test_demo_agent_uses_the_real_adapters_against_the_sandbox() -> None:
    data = compose("docker-compose.yml")
    agent, sandbox = data["services"]["agent"], data["services"]["sandbox"]
    env = agent["environment"]
    token = sandbox["environment"]["BT_SANDBOX_TOKEN"]
    assert token == "${BT_SANDBOX_TOKEN:-sandbox}"
    assert env["BT_CALENDAR"] == "calcom"
    assert env["BT_CALCOM_BASE_URL"] == "http://sandbox:8100"
    assert env["BT_CRM"] == "hubspot"
    assert env["BT_HUBSPOT_BASE_URL"] == "http://sandbox:8100"
    assert env["BT_CALCOM_API_KEY"] == token
    assert env["BT_HUBSPOT_TOKEN"] == token
    assert env["BT_API_KEY"] == "${BT_API_KEY:-dev-local-key}"
    assert env["BT_GUARDS"] == "${BT_GUARDS:-all}"
    assert agent["depends_on"]["sandbox"]["condition"] == "service_healthy"


def test_demo_agent_state_is_on_a_named_volume() -> None:
    data = compose("docker-compose.yml")
    agent = data["services"]["agent"]
    db_dir = str(Path(agent["environment"]["BT_DB_PATH"]).parent)
    mounts = dict(v.split(":", 1)[::-1] for v in agent["volumes"])
    assert db_dir in mounts
    assert mounts[db_dir] in (data.get("volumes") or {}), "BT_DB_PATH must live on a named volume"


def test_env_file_is_optional_and_the_llm_key_never_comes_from_openrouter() -> None:
    data = compose("docker-compose.yml")
    agent, sandbox = data["services"]["agent"], data["services"]["sandbox"]
    assert agent["env_file"] == [{"path": ".env", "required": False}]
    assert "env_file" not in sandbox  # the sandbox needs no secrets from .env
    assert agent["environment"]["BT_LLM_API_KEY"] == LLM_KEY
    assert agent["environment"]["OPENROUTER_API_KEY"] == ""
    for name in ("docker-compose.yml", "docker-compose.bench.yml"):
        assert "${OPENROUTER_API_KEY" not in (ROOT / name).read_text(encoding="utf-8")


# docker-compose.bench.yml and scripts/bench.sh -------------------------------------------------------------


def bench_services() -> tuple[dict[str, Any], dict[str, Any]]:
    services = compose("docker-compose.bench.yml")["services"]
    agents = {k: v for k, v in services.items() if k.startswith("agent-")}
    sandboxes = {k: v for k, v in services.items() if k.startswith("sandbox-")}
    assert len(agents) == 6
    assert len(sandboxes) == 6
    assert len(services) == 12
    return agents, sandboxes


def test_bench_pool_fits_the_colima_vm() -> None:
    agents, sandboxes = bench_services()
    services = {**agents, **sandboxes}
    total = sum(mib(s["mem_limit"]) for s in services.values())
    # 4 CPU / 6 GiB VM: leave at least 2 GiB for the VM's kernel, dockerd, containerd and image builds.
    assert total <= 4 * 1024, f"{total} MiB of limits"
    for service in services.values():
        assert service["image"] == IMAGE
        assert service["build"] == "."
        assert all(p.startswith("127.0.0.1:") for p in service["ports"])
        assert "env_file" not in service


def test_each_bench_agent_has_its_own_sandbox_and_the_key_only_from_the_script() -> None:
    agents, sandboxes = bench_services()
    used: set[str] = set()
    for agent in agents.values():
        env = agent["environment"]
        target = env["BT_CALCOM_BASE_URL"].removeprefix("http://").split(":")[0]
        assert target in sandboxes
        assert target not in used
        assert env["BT_HUBSPOT_BASE_URL"] == env["BT_CALCOM_BASE_URL"]
        assert set(agent["depends_on"]) == {target}
        assert env["BT_LLM_API_KEY"] == LLM_KEY
        assert any(v.startswith("${BT_LEDGER_DIR:?") for v in agent["volumes"])
        used.add(target)
    for sandbox in sandboxes.values():
        assert set(sandbox["environment"]) == {"BT_SANDBOX_TOKEN"}


def test_bench_agents_pass_through_the_model_and_provider_settings() -> None:
    """scripts/bench.sh pins BT_LLM_MODEL/BT_LLM_PROVIDER and passes BT_PERSONA_MODEL/BT_EXTRACTOR_MODEL
    through its own environment; docker-compose.bench.yml must forward all four to every agent."""
    agents, _sandboxes = bench_services()
    for agent in agents.values():
        env = agent["environment"]
        assert env["BT_LLM_MODEL"] == "${BT_LLM_MODEL:-}"
        assert env["BT_LLM_PROVIDER"] == "${BT_LLM_PROVIDER:-}"
        assert env["BT_PERSONA_MODEL"] == "${BT_PERSONA_MODEL:-}"
        assert env["BT_EXTRACTOR_MODEL"] == "${BT_EXTRACTOR_MODEL:-}"


def test_bench_pool_file_matches_the_compose_ports() -> None:
    agents, sandboxes = bench_services()
    by_port = {port(a["ports"][0]): a for a in agents.values()}
    sandbox_port = {name: port(s["ports"][0]) for name, s in sandboxes.items()}
    guards = {"guarded": "all", "naive": "off"}
    pool = load_pool(ROOT / "scripts" / "bench-pool.yaml")
    seen: list[int] = []
    for entry in pool.agents:
        assert entry.mode is not None
        assert entry.label == entry.mode
        assert len(entry.pairs) == 3
        for pair in entry.pairs:
            agent = by_port[port(pair.agent)]
            assert agent["environment"]["BT_GUARDS"] == guards[entry.mode]
            sandbox = agent["environment"]["BT_CALCOM_BASE_URL"].removeprefix("http://").split(":")[0]
            assert port(pair.sandbox) == sandbox_port[sandbox]
            assert pair.agent.endswith("/v1/chat")
            seen.append(port(pair.agent))
    assert sorted(seen) == sorted(by_port)


def test_bench_script_defaults_the_pinned_model_and_provider() -> None:
    script = (ROOT / "scripts" / "bench.sh").read_text(encoding="utf-8")
    assert 'export BT_LLM_MODEL="${BT_LLM_MODEL:-deepseek/deepseek-v4-flash}"' in script
    assert 'export BT_LLM_PROVIDER="${BT_LLM_PROVIDER:-deepinfra/fp8}"' in script
    # Still overridable from the environment (the default is only the fallback of a parameter expansion),
    # and BT_PERSONA_MODEL/BT_EXTRACTOR_MODEL reach the harness (this script's own children) the same way.
    assert 'export BT_PERSONA_MODEL="${BT_PERSONA_MODEL:-}"' in script
    assert 'export BT_EXTRACTOR_MODEL="${BT_EXTRACTOR_MODEL:-}"' in script
    table = yaml.safe_load((ROOT / "pricing.yaml").read_text(encoding="utf-8"))
    assert "deepinfra/fp8" in table["models"]["deepseek/deepseek-v4-flash"]["providers"]


def test_bench_script_never_writes_the_key() -> None:
    script = (ROOT / "scripts" / "bench.sh").read_text(encoding="utf-8")
    assert '--pool "$pool"' in script
    assert 'pool="scripts/bench-pool.yaml"' in script
    assert not re.search(r"^\s*set\s+-[a-z]*x", script, re.MULTILINE), "xtrace would print the key"
    for line in script.splitlines():
        if re.search(r"OPENROUTER_API_KEY|BT_LLM_API_KEY", line) and not line.lstrip().startswith("#"):
            assert not re.search(r">>?\s*[^&\s]|\btee\b", line), line


# scripts/check_widget_size.py ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def size_check() -> ModuleType:
    path = ROOT / "scripts" / "check_widget_size.py"
    spec = importlib.util.spec_from_file_location("check_widget_size", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_widget_passes_the_size_check(size_check: ModuleType) -> None:
    assert size_check.problems((ROOT / "widget" / "widget.js").read_bytes()) == []
    assert size_check.main(["check_widget_size.py"]) == 0


def test_size_check_rejects_bad_widgets(size_check: ModuleType, tmp_path: Path) -> None:
    good = "// @ts-check\nconst a = 1;\n"
    assert size_check.problems(good.encode()) == []
    assert any("budget" in p for p in size_check.problems((good + "x" * 25_600).encode()))
    assert any("@ts-check" in p for p in size_check.problems(b"const a = 1;\n"))
    assert any(
        "cdn.example.net" in p
        for p in size_check.problems(b'// @ts-check\nfetch("https://cdn.example.net/x");\n')
    )
    assert any("innerHTML" in p for p in size_check.problems(b"// @ts-check\nel.innerHTML = reply;\n"))
    assert size_check.main(["check_widget_size.py", str(tmp_path / "missing.js")]) == 1

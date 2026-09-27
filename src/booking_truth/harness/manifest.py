"""``manifest.json``: what a run was, so its numbers can be reproduced and compared.

It records the date, the ``--as-of`` date, the hardware, the harness version and git SHA, the
scenario-suite hash, each agent's version and source hash, the model ids, the models and providers the calls
returned (flagged when they varied), temperatures and the total spend.

The hardware comes from ``sysctl`` (model and memory) or ``--hardware``; the machine's host name is never
read. No absolute path, email address or URL host other than ``localhost`` is ever written.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import booking_truth
from booking_truth import __version__
from booking_truth.harness.redact import redact, scrub

MANIFEST_SCHEMA = "booking-truth/manifest/v1"
_PACKAGE_ROOT = Path(booking_truth.__file__).resolve().parents[2]


def _run(args: Sequence[str], *, cwd: Path | None = None) -> str | None:
    executable = shutil.which(args[0])
    if executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument lists, no shell
            [executable, *args[1:]], capture_output=True, text=True, timeout=5, check=False, cwd=cwd
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _gib(total_bytes: int) -> str:
    return f"{round(total_bytes / 1024**3)} GB"


def detect_hardware() -> str:
    """``<model>, <memory>`` from ``sysctl`` on macOS, or from DMI and ``/proc/meminfo`` on Linux."""
    model = _run(["sysctl", "-n", "hw.model"])
    memsize = _run(["sysctl", "-n", "hw.memsize"])
    if model and memsize and memsize.isdigit():
        return f"{model}, {_gib(int(memsize))}"
    product = Path("/sys/devices/virtual/dmi/id/product_name")
    try:
        model = product.read_text(encoding="utf-8").strip() if product.exists() else None
    except OSError:
        model = None
    memory = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory = _gib(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        memory = None
    return f"{model or platform.machine() or 'unknown machine'}, {memory or 'unknown memory'}"


def hardware_description(override: str | None) -> str:
    return override.strip() if override and override.strip() else detect_hardware()


def git_info(root: Path | None = None) -> dict[str, Any]:
    """The checkout's commit and whether tracked files differ from it; ``unknown`` outside a checkout."""
    root = root or _PACKAGE_ROOT
    if not (root / ".git").exists():
        return {"sha": "unknown", "dirty": None}
    sha = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if sha is None:
        return {"sha": "unknown", "dirty": None}
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    return {"sha": sha, "dirty": bool(status)}


def suite_hash(directory: Path) -> str:
    """``sha256:<hex>`` over every ``*.yaml`` file's name and bytes, in name order."""
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.yaml")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


@dataclass
class CallRecorder:
    """Returned model ids and upstream providers per component (``persona``, ``extractor``,
    ``agent:<label>``), with the number of calls that returned each (model, provider) pair."""

    models: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    providers: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pairs: dict[str, dict[tuple[str, str], int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    def record(self, component: str, *, model: str | None, provider: str | None) -> None:
        self.calls[component] += 1
        if model:
            self.models[component].add(model)
        if provider:
            self.providers[component].add(provider)
        self.pairs[component][(model or "", provider or "")] += 1

    def record_usage(self, component: str, usage: Mapping[str, Any] | None) -> None:
        """Read every model id and upstream provider an agent turn's ``usage`` object reports.

        The bundled protocol's ``usage.models``/``usage.providers`` (plural: see
        ``booking_truth.agent.loop.Usage.to_json``) are the distinct ids and providers every internal
        model call of that turn returned, already deduplicated by the agent - never a singular
        ``usage.model``/``usage.provider``. A turn is counted once; its models and providers each join
        the distinct sets (so ``varied`` is exact), and the ``per_call`` breakdown pairs them
        positionally, which is exact for the common case of one model and one provider throughout the
        turn and best-effort otherwise, since the wire format does not preserve the true per-call
        pairing.
        """
        if not isinstance(usage, Mapping):
            return
        models = [m for m in usage.get("models") or () if isinstance(m, str) and m]
        providers = [p for p in usage.get("providers") or () if isinstance(p, str) and p]
        if not models and not providers:
            return
        self.calls[component] += 1
        for model in models:
            self.models[component].add(model)
        for provider in providers:
            self.providers[component].add(provider)
        left, right = models or [""], providers or [""]
        for index in range(max(len(left), len(right))):
            model = left[index] if index < len(left) else ""
            provider = right[index] if index < len(right) else ""
            self.pairs[component][(model, provider)] += 1

    def to_json(self) -> dict[str, Any]:
        components = sorted(set(self.calls) | set(self.models) | set(self.providers))
        out: dict[str, Any] = {}
        for name in components:
            models = sorted(self.models.get(name, set()))
            providers = sorted(self.providers.get(name, set()))
            out[name] = {
                "calls": self.calls.get(name, 0),
                "models_returned": models,
                "providers": providers,
                "varied": len(models) > 1 or len(providers) > 1,
                "per_call": [
                    {"model": model or None, "provider": provider or None, "calls": count}
                    for (model, provider), count in sorted(self.pairs.get(name, {}).items())
                ],
            }
        return out


@dataclass
class AgentManifest:
    label: str
    kind: str  # builtin | http
    mode: str | None
    protocol: str  # bundled | agent.yaml
    target: str
    calendar: str | None = None
    agent_version: str | None = None
    versions_seen: list[str] = field(default_factory=list)
    version_info: dict[str, Any] | None = None
    endpoints: int = 1

    def to_json(self) -> dict[str, Any]:
        info = self.version_info or {}
        return {
            "label": self.label,
            "kind": self.kind,
            "mode": self.mode,
            "protocol": self.protocol,
            "target": self.target,
            "calendar": self.calendar,
            "agent_version": self.agent_version,
            "versions_seen": sorted(set(self.versions_seen)),
            "source_hash": info.get("source_hash"),
            "model": info.get("model"),
            "temperature": info.get("temperature"),
            "guards": info.get("guards"),
            "version_info": info or None,
            "endpoints": self.endpoints,
        }


def _spend(values: Iterable[float]) -> float:
    return round(sum(values), 6)


def build_manifest(
    *,
    run_id: str,
    date: str,
    as_of: str | None,
    hardware: str,
    suite: str,
    suite_digest: str,
    scenarios: Sequence[str],
    k: int,
    agents: Sequence[AgentManifest],
    grading: Mapping[str, Any],
    models: Mapping[str, Any],
    temperatures: Mapping[str, Any],
    calls: CallRecorder,
    spend: Mapping[str, float],
    status: str,
    status_detail: str | None,
    options: Mapping[str, Any],
    command: str,
    dry_run: bool,
    projection: Mapping[str, Any] | None = None,
    git: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "date": date,
        "as_of": as_of,
        "hardware": hardware,
        "harness_version": __version__,
        "git": dict(git) if git is not None else git_info(),
        "suite": suite,
        "scenario_suite_hash": suite_digest,
        "scenarios": list(scenarios),
        "k": k,
        "agents": [a.to_json() for a in agents],
        "grading": dict(grading),
        "models": dict(models),
        "temperatures": dict(temperatures),
        "llm_calls": calls.to_json(),
        "spend": {name: _spend([value]) for name, value in spend.items()},
        "status": status,
        "status_detail": status_detail,
        "options": dict(options),
        "command": command,
        "dry_run": dry_run,
        "projection": dict(projection) if projection is not None else None,
    }
    # URL hosts become ``localhost`` or ``[host]``: a committed manifest names no machine or address.
    cleaned: dict[str, Any] = scrub(redact(manifest, None))
    return cleaned

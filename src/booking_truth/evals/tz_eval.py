"""``booking-truth eval tz``: the deterministic timezone resolver against LLM-only resolution, on the
held-out test split of ``datasets/tz_phrases.jsonl``.

Both sides answer with one of ``resolved`` (a single zone), ``ambiguous`` (candidate zones) or ``unknown``.
Scored against the gold label exactly as ``datasets/README.md`` defines it ("Scoring"), against an answer
that names a single zone equivalent to gold (or, for an ambiguous gold, to one of its candidates), one that
names a single zone equivalent to nothing in gold, and one that answers ambiguous or unknown:

- gold ``resolved``: ``correct``, ``silent_wrong_resolution``, ``over_cautious`` (asked needlessly).
- gold ``ambiguous``: ``missed_ambiguity``, ``silent_wrong_resolution``, ``correctly_flagged``.
- gold ``unknown``: n/a, ``silent_wrong_resolution``, ``correctly_flagged``.

Two zones are *equivalent* when their UTC offset is identical at every instant sampled daily from
2026-01-01T00:00Z to 2028-01-01T00:00Z, the same window ``datasets/README.md`` fixes for the dataset's own
labels; this is computed fresh here with :mod:`zoneinfo`, independent of the resolver's own internals, so
the score does not depend on how the resolver happens to implement equivalence.

This is the one module in ``harness/`` that imports ``booking_truth.agent.guards`` (the deterministic
resolver under test): it evaluates that guard rather than reusing it to grade a trial, so it is not the
circularity ``docs/adr/0008`` guards against (see ``tests/harness/test_harness_independence.py``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any, Literal, get_args
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from booking_truth.agent.guards.tz.resolver import Resolution, TimezoneResolver
from booking_truth.harness.report import table
from booking_truth.llm.types import LLM, BudgetExceeded, ChatMessage, LLMError
from booking_truth.resources import data_path

Status = Literal["resolved", "ambiguous", "unknown"]
Category = Literal[
    "correct", "missed_ambiguity", "silent_wrong_resolution", "over_cautious", "correctly_flagged"
]
CATEGORIES: tuple[Category, ...] = get_args(Category)
#: The category that matters most (spec: "the number that matters").
HEADLINE_CATEGORY: Category = "silent_wrong_resolution"

EVAL_TEMPERATURE = 0.0
EVAL_MAX_TOKENS = 300
RESOLVER_NOW = datetime(2026, 1, 1, tzinfo=UTC)
RESOLVER_HORIZON_DAYS = 730  # 2026-01-01 .. 2028-01-01, the equivalence window datasets/README.md fixes
_EQUIV_STEP = timedelta(days=1)

PROMPT = (
    "You resolve a short phrase a prospect used to say where they are or which time zone they use, to "
    "an IANA zoneinfo time zone identifier (for example Europe/Berlin, America/Chicago, Asia/Kolkata).\n\n"
    "- If the phrase names exactly one time zone unambiguously, answer resolved with that zone.\n"
    "- If the phrase could reasonably name more than one materially different time zone (a genuinely "
    "ambiguous abbreviation, region or city name), answer ambiguous and list the candidate zones.\n"
    "- If the phrase names no place, zone or offset at all, answer unknown.\n\n"
    'Reply with exactly one JSON object: {"status": "resolved" | "ambiguous" | "unknown", '
    '"zone": "<IANA zone>" or null, "candidates": ["<IANA zone>", ...]}. `zone` is set only when `status` '
    "is `resolved`. `candidates` lists every reasonable reading only when `status` is `ambiguous`; leave "
    "it empty otherwise."
)
_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "tz_resolution",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": list(get_args(Status))},
                "zone": {"anyOf": [{"type": "null"}, {"type": "string"}]},
                "candidates": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status", "zone", "candidates"],
            "additionalProperties": False,
        },
    },
}


# Dataset -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TzItem:
    id: str
    text: str
    status: Status
    zone: str | None
    candidates: tuple[str, ...]


def load_test_items(path: Path | str | None = None) -> list[TzItem]:
    """The ``split == "test"`` rows of ``datasets/tz_phrases.jsonl``, sorted by id."""
    file = Path(path) if path is not None else data_path("datasets") / "tz_phrases.jsonl"
    items: list[TzItem] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "test":
            continue
        label = row["label"]
        items.append(
            TzItem(
                id=str(row["id"]),
                text=str(row["text"]),
                status=label["status"],
                zone=label.get("zone"),
                candidates=tuple(label.get("candidates") or ()),
            )
        )
    if not items:
        raise ValueError(f"{file}: no test-split rows found")
    return sorted(items, key=lambda item: item.id)


# Equivalence, computed fresh (not from the resolver's own internals) ------------------------------------


@cache
def _signature(zone: str) -> tuple[int, ...] | None:
    try:
        tz = ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None
    samples: list[int] = []
    instant = RESOLVER_NOW
    end = RESOLVER_NOW + timedelta(days=RESOLVER_HORIZON_DAYS)
    while True:
        offset = instant.astimezone(tz).utcoffset() or timedelta(0)
        samples.append(int(offset.total_seconds()))
        if instant >= end:
            break
        instant = min(instant + _EQUIV_STEP, end)
    return tuple(samples)


def zones_equivalent(a: str, b: str) -> bool:
    """Whether ``a`` and ``b`` carry the same UTC offset at every sampled instant of the window."""
    if a == b:
        return True
    sig_a, sig_b = _signature(a), _signature(b)
    return sig_a is not None and sig_a == sig_b


# Scoring -------------------------------------------------------------------------------------------------


def score(
    *, gold_status: Status, gold_zone: str | None, gold_candidates: Sequence[str], answer: Resolution
) -> Category:
    """One item's score against ``datasets/README.md``'s scoring table."""
    if answer.status == "resolved":
        assert answer.zone is not None
        if gold_status == "resolved":
            assert gold_zone is not None
            return "correct" if zones_equivalent(answer.zone, gold_zone) else "silent_wrong_resolution"
        if gold_status == "ambiguous":
            if any(zones_equivalent(answer.zone, candidate) for candidate in gold_candidates):
                return "missed_ambiguity"
            return "silent_wrong_resolution"
        return "silent_wrong_resolution"  # gold unknown
    return "over_cautious" if gold_status == "resolved" else "correctly_flagged"


# The LLM-only side ----------------------------------------------------------------------------------------


def _parse(content: str | None) -> Resolution:
    if not content:
        raise LLMError("the eval model returned an empty response", kind="malformed")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"the eval model returned invalid JSON: {exc}", kind="malformed") from None
    if not isinstance(data, dict) or data.get("status") not in get_args(Status):
        raise LLMError("the eval model's response is missing a valid 'status'", kind="malformed")
    zone = data.get("zone")
    candidates = tuple(z for z in data.get("candidates") or () if isinstance(z, str))
    return Resolution(
        status=data["status"], zone=zone if isinstance(zone, str) else None, candidates=candidates
    )


async def llm_resolve(llm: LLM, model: str, text: str) -> Resolution:
    response = await llm.chat(
        messages=[ChatMessage.system(PROMPT), ChatMessage.user(text)],
        temperature=EVAL_TEMPERATURE,
        model=model,
        max_tokens=EVAL_MAX_TOKENS,
        response_format=_SCHEMA,
        component="eval_tz",
    )
    return _parse(response.content)


# Aggregation and the report --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemResult:
    id: str
    text: str
    gold_status: Status
    deterministic: Category
    llm: Category | None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "gold_status": self.gold_status,
            "deterministic": self.deterministic,
            "llm": self.llm,
        }


def _counts(categories: Sequence[Category]) -> dict[str, Any]:
    n = len(categories)
    counted = {name: sum(c == name for c in categories) for name in CATEGORIES}
    return {
        "n": n,
        "counts": counted,
        "rates": {name: (round(count / n, 6) if n else None) for name, count in counted.items()},
        "headline": HEADLINE_CATEGORY,
    }


@dataclass
class TzEvalResult:
    model: str | None
    items: list[ItemResult] = field(default_factory=list)
    llm_skipped_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        deterministic = [item.deterministic for item in self.items]
        llm = [item.llm for item in self.items if item.llm is not None]
        return {
            "eval": "tz",
            "dataset": "datasets/tz_phrases.jsonl",
            "split": "test",
            "model": self.model,
            "equivalence_window": {
                "start": RESOLVER_NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "horizon_days": RESOLVER_HORIZON_DAYS,
            },
            "n_items": len(self.items),
            "deterministic": _counts(deterministic),
            "llm": _counts(llm) if llm else None,
            "llm_skipped_reason": self.llm_skipped_reason,
            "items": [item.to_json() for item in self.items],
        }


async def run_tz_eval(
    *, llm: LLM | None, model: str | None, dataset_path: Path | str | None = None
) -> TzEvalResult:
    """Score the deterministic resolver, and the LLM (when ``llm`` is given), against every test item.

    A live-call failure (a budget stop or any other model error) stops the LLM side for every remaining
    item; the deterministic side always covers all of them. ``model`` names the model used (or requested).
    """
    items = load_test_items(dataset_path)
    resolver = TimezoneResolver()
    result = TzEvalResult(model=model)
    active = llm
    for item in items:
        deterministic = resolver.resolve(item.text, now=RESOLVER_NOW, horizon_days=RESOLVER_HORIZON_DAYS)
        det_score = score(
            gold_status=item.status,
            gold_zone=item.zone,
            gold_candidates=item.candidates,
            answer=deterministic,
        )
        llm_score: Category | None = None
        if active is not None:
            assert model is not None
            try:
                answer = await llm_resolve(active, model, item.text)
                llm_score = score(
                    gold_status=item.status,
                    gold_zone=item.zone,
                    gold_candidates=item.candidates,
                    answer=answer,
                )
            except BudgetExceeded as exc:
                result.llm_skipped_reason = f"budget stop after {len(result.items)} item(s): {exc}"
                active = None
            except LLMError as exc:
                result.llm_skipped_reason = f"LLM error after {len(result.items)} item(s): {exc}"
                active = None
        result.items.append(ItemResult(item.id, item.text, item.status, det_score, llm_score))
    return result


# Markdown --------------------------------------------------------------------------------------------------

CATEGORY_LABELS: dict[Category, str] = {
    "correct": "Correct",
    "missed_ambiguity": "Missed ambiguity",
    "silent_wrong_resolution": "Silent wrong resolution",
    "over_cautious": "Over-cautious (asked needlessly)",
    "correctly_flagged": "Correctly flagged (ambiguous or unknown)",
}


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _side_rows(side: Mapping[str, Any] | None) -> list[list[str]]:
    if side is None:
        return [[CATEGORY_LABELS[c], "n/a", "n/a"] for c in CATEGORIES]
    counts, rates = side["counts"], side["rates"]
    return [[CATEGORY_LABELS[c], str(counts[c]), _pct(rates[c])] for c in CATEGORIES]


def render_tz_eval_md(data: Mapping[str, Any]) -> str:
    det, llm = data["deterministic"], data.get("llm")
    lines = ["# Timezone resolver eval", ""]
    lines += table(
        ["Field", "Value"],
        [
            ["Dataset", f"`{data['dataset']}` (held-out test split, n = {data['n_items']})"],
            [
                "Equivalence window",
                f"{data['equivalence_window']['start']} + {data['equivalence_window']['horizon_days']} days",
            ],
            ["Model (LLM-only side)", data["model"] or "n/a"],
        ],
    )
    lines += ["", "## Deterministic resolver", ""]
    lines += table(["Category", "Count", "Rate"], _side_rows(det))
    lines += ["", "## LLM-only resolution", ""]
    if llm is None:
        reason = data.get("llm_skipped_reason") or "no LLM key configured"
        lines.append(f"Not run: {reason}. Offline: only the deterministic resolver is scored.")
    else:
        lines += table(["Category", "Count", "Rate"], _side_rows(llm))
        if data.get("llm_skipped_reason"):
            lines += [
                "",
                f"Stopped early: {data['llm_skipped_reason']}. The table above covers only the "
                f"{llm['n']} item(s) scored before that.",
            ]
    lines += [
        "",
        "## Notes",
        "",
        f"- **{CATEGORY_LABELS[HEADLINE_CATEGORY]}** is the number that matters: a resolver that answers "
        "with a single zone that is not what the prospect meant, without asking, silently mis-schedules "
        "the meeting.",
        "- Candidate lists are not scored; only the resolved/ambiguous/unknown status and, when resolved, "
        "the zone.",
        "- Reproduce with `booking-truth eval tz`.",
        "",
    ]
    return "\n".join(lines)

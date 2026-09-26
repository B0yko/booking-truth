#!/usr/bin/env python3
"""Refresh ``pricing.yaml`` from OpenRouter's public model and endpoint listings.

Both endpoints used here are public and need no API key:

* ``GET /api/v1/models`` gives each model's ``canonical_slug`` and its model-level ``pricing``
  (the top provider's price, in USD per token, as decimal strings);
* ``GET /api/v1/models/<canonical_slug>/endpoints`` gives each provider endpoint's ``tag`` and
  ``pricing``. Listed endpoint prices are already net of any ``discount``.

Prices are converted to USD per million tokens with ``Decimal`` and written in a fixed order,
so the same listings always produce the same file.

Examples::

    uv run python scripts/update_pricing.py \\
        --models deepseek/deepseek-v4.1-flash qwen/qwen3-235b-a22b-2507 \\
        --providers deepseek/deepseek-v4.1-flash=deepinfra/fp8,fireworks \\
        --out pricing.yaml

    # Refresh the models and pinned providers already listed in pricing.yaml:
    uv run python scripts/update_pricing.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx
import yaml

from booking_truth.llm.pricing import PriceTable

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
PER_MILLION = Decimal(1_000_000)
TIMEOUT_S = 30.0
_PLAIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:-]*$")


class PricingUpdateError(RuntimeError):
    """The listings could not be fetched or do not contain what was asked for."""


@dataclass(frozen=True)
class Prices:
    prompt: Decimal
    completion: Decimal
    cache_read: Decimal | None = None

    def worst(self, other: Prices) -> Prices:
        """The higher of each price. Without a cache price, cached tokens cost the prompt price."""
        cache: Decimal | None = None
        if self.cache_read is not None or other.cache_read is not None:
            cache = max(
                self.prompt if self.cache_read is None else self.cache_read,
                other.prompt if other.cache_read is None else other.cache_read,
            )
        return Prices(max(self.prompt, other.prompt), max(self.completion, other.completion), cache)


@dataclass
class ModelRow:
    model_id: str
    canonical_slug: str
    prices: Prices
    expires: str | None = None
    providers: dict[str, Prices] = field(default_factory=dict)


# Conversion ------------------------------------------------------------------------------------


def per_million(value: Any, where: str) -> Decimal:
    """Convert a per-token USD decimal string to USD per million tokens."""
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise PricingUpdateError(f"{where}: expected a decimal price string, got {value!r}")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise PricingUpdateError(f"{where}: not a number: {value!r}") from exc
    if not number.is_finite() or number < 0:
        # OpenRouter uses "-1" for router models whose price is only known after routing.
        raise PricingUpdateError(f"{where}: no fixed price ({value!r}); refusing to guess")
    return number * PER_MILLION


def prices_from(pricing: Any, where: str) -> Prices:
    if not isinstance(pricing, Mapping):
        raise PricingUpdateError(f"{where}: missing 'pricing' object")
    for key in ("prompt", "completion"):
        if key not in pricing:
            raise PricingUpdateError(f"{where}: pricing has no '{key}'")
    cache = pricing.get("input_cache_read")
    return Prices(
        prompt=per_million(pricing["prompt"], f"{where}.prompt"),
        completion=per_million(pricing["completion"], f"{where}.completion"),
        cache_read=None if cache is None else per_million(cache, f"{where}.input_cache_read"),
    )


def tag_matches(endpoint_tag: Any, pinned: str) -> bool:
    """Whether pinning ``pinned`` can route to the endpoint tagged ``endpoint_tag``.

    A full tag (``deepinfra/fp8``) matches only itself. A base slug (``deepinfra``) matches every
    endpoint of that provider, as OpenRouter's routing does, so it is priced at the worst of them.
    """
    if not isinstance(endpoint_tag, str):
        return False
    if endpoint_tag == pinned:
        return True
    return "/" not in pinned and endpoint_tag.split("/", 1)[0] == pinned


def build_rows(
    models_payload: Mapping[str, Any],
    endpoints_by_model: Mapping[str, Mapping[str, Any]],
    wanted: Mapping[str, Sequence[str]],
    *,
    warn: Callable[[str], None] | None = None,
) -> list[ModelRow]:
    """Turn the two listings into one row per wanted model, with the wanted provider tags."""
    say: Callable[[str], None] = warn or (lambda message: print(message, file=sys.stderr))
    data = models_payload.get("data")
    if not isinstance(data, list):
        raise PricingUpdateError("models listing has no 'data' list")
    by_id = {m.get("id"): m for m in data if isinstance(m, Mapping)}
    rows: list[ModelRow] = []
    for model_id in sorted(wanted):
        model = by_id.get(model_id)
        if model is None:
            raise PricingUpdateError(f"model {model_id!r} is not in the OpenRouter models listing")
        pricing = model.get("pricing")
        if isinstance(pricing, Mapping) and pricing.get("overrides"):
            say(f"note: {model_id} has conditional price overrides; only the base price is recorded")
        row = ModelRow(
            model_id=model_id,
            canonical_slug=str(model.get("canonical_slug") or model_id),
            prices=prices_from(pricing, model_id),
            expires=str(model["expiration_date"]) if model.get("expiration_date") else None,
        )
        tags = wanted[model_id]
        if tags:
            endpoints_payload = endpoints_by_model.get(model_id)
            endpoints = (
                endpoints_payload.get("data", {}).get("endpoints")
                if isinstance(endpoints_payload, Mapping)
                else None
            )
            if not isinstance(endpoints, list):
                raise PricingUpdateError(f"endpoint listing for {model_id} has no 'data.endpoints' list")
            for tag in tags:
                matches = [e for e in endpoints if isinstance(e, Mapping) and tag_matches(e.get("tag"), tag)]
                if not matches:
                    known = sorted({str(e.get("tag")) for e in endpoints if isinstance(e, Mapping)})
                    raise PricingUpdateError(
                        f"provider tag {tag!r} does not serve {model_id}; listed tags: {', '.join(known)}"
                    )
                if len(matches) > 1:
                    say(f"note: {model_id} has {len(matches)} endpoints matching {tag}; using the highest")
                price = prices_from(matches[0].get("pricing"), f"{model_id} [{tag}]")
                for extra in matches[1:]:
                    price = price.worst(prices_from(extra.get("pricing"), f"{model_id} [{tag}]"))
                if not any("tools" in (e.get("supported_parameters") or []) for e in matches):
                    say(f"note: {tag} does not list 'tools' for {model_id}")
                row.providers[tag] = price
        rows.append(row)
    return rows


# Rendering -------------------------------------------------------------------------------------


def number(value: Decimal) -> str:
    """Plain decimal text with no exponent and no trailing zeros."""
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def scalar(text: str) -> str:
    return text if _PLAIN.match(text) and not text.endswith(":") else _quoted(text)


def _quoted(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _price_fields(prices: Prices) -> list[tuple[str, str]]:
    fields = [("prompt", number(prices.prompt)), ("completion", number(prices.completion))]
    if prices.cache_read is not None:
        fields.append(("cache_read", number(prices.cache_read)))
    return fields


def render(rows: Iterable[ModelRow], *, source: str, checked: str) -> str:
    lines = [
        "# LLM prices in USD per million tokens, used by the cost ledger and the budget stop.",
        "# Model prices are OpenRouter's model-level prices (its top provider). Provider prices are per",
        "# endpoint tag (a base slug such as `deepinfra` is priced at its most expensive endpoint). A",
        "# provider pinned with BT_LLM_PROVIDER must be listed for the model, or live calls are refused.",
        "# Refresh with: uv run python scripts/update_pricing.py",
        f"source: {source}",
        f"checked: {_quoted(checked)}",
        "currency: USD",
        "unit: per_million_tokens",
    ]
    rows = sorted(rows, key=lambda r: r.model_id)
    if not rows:
        lines.append("models: {}")
        return "\n".join(lines) + "\n"
    lines.append("models:")
    for row in rows:
        lines.append(f"  {scalar(row.model_id)}:")
        lines.append(f"    canonical_slug: {scalar(row.canonical_slug)}")
        if row.expires:
            lines.append(f"    expires: {_quoted(row.expires)}")
        lines.extend(f"    {key}: {value}" for key, value in _price_fields(row.prices))
        if row.providers:
            lines.append("    providers:")
            for tag in sorted(row.providers):
                inner = ", ".join(f"{key}: {value}" for key, value in _price_fields(row.providers[tag]))
                lines.append(f"      {scalar(tag)}: {{{inner}}}")
    return "\n".join(lines) + "\n"


def verify(text: str, rows: Sequence[ModelRow]) -> None:
    """Load the rendered text with the product's own loader and compare every price."""
    table = PriceTable.from_dict(yaml.safe_load(text), origin="rendered pricing.yaml")
    for row in rows:
        loaded = table.price_for(row.model_id)
        if (loaded.prompt, loaded.completion, loaded.cache_read) != (
            row.prices.prompt,
            row.prices.completion,
            row.prices.cache_read,
        ):
            raise PricingUpdateError(f"round trip changed the price of {row.model_id}")
        for tag, prices in row.providers.items():
            pinned = table.price_for(row.model_id, tag)
            if (pinned.prompt, pinned.completion, pinned.cache_read) != (
                prices.prompt,
                prices.completion,
                prices.cache_read,
            ):
                raise PricingUpdateError(f"round trip changed the price of {row.model_id} [{tag}]")


# Fetching --------------------------------------------------------------------------------------


def fetch_json(client: httpx.Client, url: str) -> dict[str, Any]:
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        raise PricingUpdateError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise PricingUpdateError(f"GET {url} returned HTTP {response.status_code}: {response.text[:200]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise PricingUpdateError(f"GET {url} did not return JSON") from exc
    if not isinstance(payload, dict):
        raise PricingUpdateError(f"GET {url} returned a non-object JSON body")
    return payload


def fetch_listings(
    client: httpx.Client, base_url: str, wanted: Mapping[str, Sequence[str]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    base = base_url.rstrip("/")
    models_payload = fetch_json(client, f"{base}/models")
    by_id = {
        m.get("id"): m for m in models_payload.get("data") or [] if isinstance(m, Mapping) and m.get("id")
    }
    endpoints: dict[str, dict[str, Any]] = {}
    for model_id, tags in sorted(wanted.items()):
        if not tags or model_id not in by_id:
            continue
        # The endpoints path takes the canonical (dated) slug, as in the model's `links.details`.
        slug = by_id[model_id].get("canonical_slug") or model_id
        endpoints[model_id] = fetch_json(client, f"{base}/models/{slug}/endpoints")
    return models_payload, endpoints


# Command line ----------------------------------------------------------------------------------


def parse_providers(items: Sequence[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in items:
        model_id, sep, tags = item.partition("=")
        if not sep or not model_id.strip() or not tags.strip():
            raise PricingUpdateError(f"--providers expects model=tag[,tag...], got {item!r}")
        bucket = result.setdefault(model_id.strip(), [])
        for tag in tags.split(","):
            if tag.strip() and tag.strip() not in bucket:
                bucket.append(tag.strip())
    return result


def wanted_from_file(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = data.get("models") or {}
    return {str(m): sorted((spec or {}).get("providers") or {}) for m, spec in models.items()}


def main(argv: Sequence[str] | None = None, *, client: httpx.Client | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", default=[], metavar="SLUG", help="model ids to price")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=[],
        metavar="MODEL=TAG[,TAG]",
        help="provider endpoint tags to price per model (for example deepinfra/fp8)",
    )
    parser.add_argument("--out", type=Path, default=Path("pricing.yaml"), help="file to write")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenRouter API base URL")
    parser.add_argument("--checked", default=None, help="date to record (default: today, UTC)")
    args = parser.parse_args(argv)

    try:
        providers = parse_providers(args.providers)
        if args.models:
            wanted: dict[str, list[str]] = {m: list(providers.get(m, [])) for m in args.models}
            extra = sorted(set(providers) - set(wanted))
            if extra:
                raise PricingUpdateError(f"--providers names models not in --models: {', '.join(extra)}")
        else:
            wanted = wanted_from_file(args.out)
            for model_id, tags in providers.items():
                wanted.setdefault(model_id, [])
                wanted[model_id] = sorted(set(wanted[model_id]) | set(tags))
        if not wanted:
            raise PricingUpdateError("no models given and none listed in the output file")

        own_client = client is None
        http = client or httpx.Client(timeout=TIMEOUT_S, follow_redirects=True)
        try:
            models_payload, endpoints = fetch_listings(http, args.base_url, wanted)
        finally:
            if own_client:
                http.close()
        rows = build_rows(models_payload, endpoints, wanted)
        checked = args.checked or datetime.now(UTC).date().isoformat()
        text = render(rows, source=f"{args.base_url.rstrip('/')}/models", checked=checked)
        verify(text, rows)
    except PricingUpdateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} models, checked {checked})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

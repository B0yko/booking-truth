"""Price table: USD per million tokens, per model and optionally per pinned provider.

File format (``pricing.yaml``)::

    source: https://openrouter.ai/api/v1/models
    checked: "2026-09-26"
    currency: USD
    unit: per_million_tokens
    models:
      vendor/model-id:
        canonical_slug: vendor/model-id-20260910   # optional; the dated slug a router may return
        expires: "2026-10-09"                      # optional; informational
        prompt: 0.30
        completion: 1.20
        cache_read: 0.006                          # optional; defaults to the prompt price
        providers:                                 # required for every provider pinned with this model
          deepinfra/fp8: {prompt: 0.14, completion: 0.42, cache_read: 0.0042}

A pinned provider (``BT_LLM_PROVIDER``) must be listed under the model's ``providers`` under the exact
slug that is pinned; otherwise live calls are refused, because the model-level price can be lower
than what the pinned provider charges. A base slug such as ``deepinfra`` matches every variant the
router knows, so ``scripts/update_pricing.py`` prices it at the most expensive of them.

All arithmetic is done in ``Decimal`` and results are rounded to 1e-8 USD.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from booking_truth.llm.types import PricingError
from booking_truth.resources import data_path

PER_MILLION = Decimal(1_000_000)
USD_QUANTUM = Decimal("0.00000001")
CHARS_PER_TOKEN_ESTIMATE = 3

_PRICE_KEYS = frozenset({"prompt", "completion", "cache_read"})
_MODEL_KEYS = _PRICE_KEYS | {"canonical_slug", "expires", "providers"}


@dataclass(frozen=True)
class Price:
    """USD per million tokens. ``cache_read`` is ``None`` when cached input is billed at the prompt price."""

    prompt: Decimal
    completion: Decimal
    cache_read: Decimal | None = None

    @property
    def cached_prompt(self) -> Decimal:
        return self.prompt if self.cache_read is None else self.cache_read


@dataclass(frozen=True)
class ModelPrice:
    default: Price
    providers: Mapping[str, Price] = field(default_factory=dict)
    canonical_slug: str | None = None
    expires: str | None = None


def usd(value: Decimal) -> float:
    """Round a USD amount to 1e-8 and return it as a float."""
    return float(value.quantize(USD_QUANTUM, rounding=ROUND_HALF_UP))


def _decimal(value: Any, where: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise PricingError(f"{where}: expected a number, got {value!r}")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise PricingError(f"{where}: expected a number, got {value!r}") from exc
    if not number.is_finite() or number < 0:
        raise PricingError(f"{where}: price must be a finite number >= 0, got {value!r}")
    return number


def _price(raw: Any, where: str, allowed: frozenset[str]) -> Price:
    if not isinstance(raw, Mapping):
        raise PricingError(f"{where}: expected a mapping")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise PricingError(f"{where}: unknown key(s) {', '.join(map(str, unknown))}")
    for key in ("prompt", "completion"):
        if key not in raw:
            raise PricingError(f"{where}: missing '{key}'")
    cache_raw = raw.get("cache_read")
    return Price(
        prompt=_decimal(raw["prompt"], f"{where}.prompt"),
        completion=_decimal(raw["completion"], f"{where}.completion"),
        cache_read=None if cache_raw is None else _decimal(cache_raw, f"{where}.cache_read"),
    )


class PriceTable:
    """Looks up prices and turns token counts into USD. A model missing from the table has no price."""

    def __init__(
        self,
        models: Mapping[str, ModelPrice],
        *,
        source: str = "",
        checked: str = "",
        origin: str = "<memory>",
    ) -> None:
        self._models = dict(models)
        self.source = source
        self.checked = checked
        self.origin = origin
        self._aliases: dict[str, str] = {}
        for model_id, entry in self._models.items():
            if entry.canonical_slug and entry.canonical_slug not in self._models:
                self._aliases[entry.canonical_slug] = model_id

    # Loading ---------------------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None = None) -> PriceTable:
        """Load a table from ``path``, or the bundled ``pricing.yaml`` when ``path`` is ``None``."""
        try:
            file = Path(path).expanduser() if path is not None else data_path("pricing.yaml")
        except FileNotFoundError as exc:
            raise PricingError(f"the bundled price table is missing: {exc}") from exc
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            raise PricingError(f"cannot read price table {file}: {exc.strerror or exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PricingError(f"price table {file} is not valid YAML: {exc}") from exc
        origin = "bundled pricing.yaml" if path is None else str(file)
        return cls.from_dict(data, origin=origin)

    @classmethod
    def from_dict(cls, data: Any, *, origin: str = "<memory>") -> PriceTable:
        if not isinstance(data, Mapping):
            raise PricingError(f"price table {origin}: expected a mapping at the top level")
        currency = data.get("currency", "USD")
        unit = data.get("unit", "per_million_tokens")
        if currency != "USD":
            raise PricingError(f"price table {origin}: currency must be USD, got {currency!r}")
        if unit != "per_million_tokens":
            raise PricingError(f"price table {origin}: unit must be per_million_tokens, got {unit!r}")
        raw_models = data.get("models")
        if raw_models is None:
            raw_models = {}
        if not isinstance(raw_models, Mapping):
            raise PricingError(f"price table {origin}: 'models' must be a mapping")
        models: dict[str, ModelPrice] = {}
        for model_id, raw in raw_models.items():
            where = f"price table {origin}: models.{model_id}"
            if not isinstance(model_id, str) or not model_id:
                raise PricingError(f"price table {origin}: model ids must be non-empty strings")
            default = _price(raw, where, _MODEL_KEYS)
            raw_providers = raw.get("providers") or {}
            if not isinstance(raw_providers, Mapping):
                raise PricingError(f"{where}.providers: expected a mapping")
            providers = {
                str(tag): _price(p, f"{where}.providers.{tag}", _PRICE_KEYS)
                for tag, p in raw_providers.items()
            }
            canonical = raw.get("canonical_slug")
            expires = raw.get("expires")
            models[model_id] = ModelPrice(
                default=default,
                providers=providers,
                canonical_slug=str(canonical) if canonical else None,
                expires=str(expires) if expires else None,
            )
        return cls(
            models,
            source=str(data.get("source", "")),
            checked=str(data.get("checked", "")),
            origin=origin,
        )

    # Lookup ----------------------------------------------------------------------------------

    @property
    def models(self) -> list[str]:
        return sorted(self._models)

    def entry(self, model: str) -> ModelPrice:
        key = self.resolve(model)
        if key is None:
            raise PricingError(
                f"model {model!r} is not in the price table ({self.origin}); live calls with it are refused. "
                "Add it with scripts/update_pricing.py or point BT_PRICING_PATH at a table that lists it"
            )
        return self._models[key]

    def resolve(self, model: str) -> str | None:
        """The table key for a model id or its canonical (dated) slug, or ``None``."""
        if model in self._models:
            return model
        return self._aliases.get(model)

    def has(self, model: str) -> bool:
        return self.resolve(model) is not None

    def price_for(self, model: str, provider: str | None = None) -> Price:
        """The model-level price, or the pinned ``provider``'s own row.

        Raises :class:`PricingError` when a pinned provider has no row for the model: the model-level
        price is only one provider's price and can undercount what the pinned one charges.
        """
        entry = self.entry(model)
        if not provider:
            return entry.default
        pinned = entry.providers.get(provider)
        if pinned is None:
            listed = ", ".join(sorted(entry.providers)) or "none"
            raise PricingError(
                f"provider {provider!r} has no price for model {model!r} in the price table ({self.origin}; "
                f"priced providers: {listed}); live calls pinned to it are refused. Add it with "
                f"scripts/update_pricing.py --providers {model}={provider}"
            )
        return pinned

    # Arithmetic ------------------------------------------------------------------------------

    def cost_decimal(
        self,
        model: str,
        provider: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> Decimal:
        if min(prompt_tokens, completion_tokens, cached_tokens) < 0:
            raise ValueError("token counts must be >= 0")
        price = self.price_for(model, provider)
        cached = min(cached_tokens, prompt_tokens)
        total = (
            Decimal(prompt_tokens - cached) * price.prompt
            + Decimal(cached) * price.cached_prompt
            + Decimal(completion_tokens) * price.completion
        ) / PER_MILLION
        return total.quantize(USD_QUANTUM, rounding=ROUND_HALF_UP)

    def cost(
        self,
        model: str,
        provider: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """USD for one call. ``cached_tokens`` is the cached part of ``prompt_tokens``."""
        return usd(self.cost_decimal(model, provider, prompt_tokens, completion_tokens, cached_tokens))

    def estimate(self, model: str, provider: str | None, prompt_chars: int, max_tokens: int) -> float:
        """Pre-call estimate: about 3 characters per prompt token, plus the full ``max_tokens``."""
        if prompt_chars < 0 or max_tokens < 0:
            raise ValueError("prompt_chars and max_tokens must be >= 0")
        prompt_tokens = math.ceil(prompt_chars / CHARS_PER_TOKEN_ESTIMATE)
        return self.cost(model, provider, prompt_tokens, max_tokens)

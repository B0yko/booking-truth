from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from booking_truth.llm.pricing import Price, PriceTable
from booking_truth.llm.types import PricingError

REQUIRED_PINS = {
    "deepseek/deepseek-v4.1-flash": {"deepinfra/fp8", "fireworks", "together"},
    "qwen/qwen3-235b-a22b-2507": {"gmicloud/fp8", "parasail/fp8", "deepinfra/fp8"},
    "deepseek/deepseek-v4-flash": {"deepinfra/fp8", "nextbit/fp8"},
    "qwen/qwen3-next-80b-a3b-instruct": set(),
    "deepseek/deepseek-v3.2": set(),
}


def test_pinned_provider_price_wins_when_listed(table: PriceTable) -> None:
    assert table.price_for("vendor/flash") == Price(Decimal("0.3"), Decimal("1.2"), Decimal("0.006"))
    assert table.price_for("vendor/flash", "deepinfra/fp8") == Price(
        Decimal("0.14"), Decimal("0.42"), Decimal("0.0042")
    )
    assert table.price_for("vendor/flash", "fireworks").completion == Decimal("0.66")
    assert table.price_for("vendor/flash", "") == table.price_for("vendor/flash")


def test_pinned_provider_without_its_own_row_is_refused(table: PriceTable) -> None:
    # The model-level price is one provider's price; a pinned provider may charge more.
    with pytest.raises(PricingError, match=r"'novita/fp8' has no price for model 'vendor/flash'"):
        table.price_for("vendor/flash", "novita/fp8")
    with pytest.raises(PricingError, match=r"update_pricing\.py --providers vendor/flash=novita/fp8"):
        table.estimate("vendor/flash", "novita/fp8", 100, 100)
    # A base slug routes to any variant of the provider, listed or not, so it needs its own row too.
    with pytest.raises(PricingError, match="priced providers: deepinfra/fp8, deepinfra/turbo, fireworks"):
        table.cost("vendor/flash", "deepinfra", 10, 10)
    with pytest.raises(PricingError, match="priced providers: none"):
        table.price_for("vendor/big", "fireworks")


def test_unknown_model_is_refused_and_canonical_slug_resolves(table: PriceTable) -> None:
    with pytest.raises(PricingError, match="not in the price table"):
        table.price_for("vendor/unknown")
    with pytest.raises(PricingError):
        table.cost("vendor/flash:free", None, 10, 10)
    assert not table.has("vendor/unknown")
    assert table.resolve("vendor/flash-20260910") == "vendor/flash"
    assert table.price_for("vendor/flash-20260910") == table.price_for("vendor/flash")


def test_cost_arithmetic_with_cached_tokens(table: PriceTable) -> None:
    # (800 x 0.30 + 200 x 0.006 + 500 x 1.20) / 1e6
    assert table.cost("vendor/flash", None, 1000, 500, 200) == pytest.approx(0.0008412, abs=1e-12)
    assert table.cost_decimal("vendor/flash", None, 1000, 500, 200) == Decimal("0.00084120")
    # Pinned provider: (1000 x 0.14 + 500 x 0.42) / 1e6
    assert table.cost("vendor/flash", "deepinfra/fp8", 1000, 500) == pytest.approx(0.00035, abs=1e-12)
    # No cache price: cached tokens are billed at the prompt price.
    assert table.cost("vendor/big", None, 1000, 0, 400) == table.cost("vendor/big", None, 1000, 0)
    # Cached tokens can never exceed the prompt tokens they are part of.
    assert table.cost("vendor/flash", None, 10, 0, 50) == table.cost("vendor/flash", None, 10, 0, 10)
    with pytest.raises(ValueError, match=">= 0"):
        table.cost("vendor/flash", None, -1, 0)


def test_cost_is_rounded_to_one_hundred_millionth_of_a_dollar(table: PriceTable) -> None:
    assert table.cost("vendor/big", None, 1, 0) == 0.00000009  # 8.75e-8 rounds half up
    assert table.cost("vendor/big", None, 0, 0) == 0.0


def test_estimate_uses_three_chars_per_token_and_full_max_tokens(table: PriceTable) -> None:
    # ceil(3001 / 3) = 1001 prompt tokens, 1024 completion tokens at the model price.
    expected = (1001 * 0.3 + 1024 * 1.2) / 1_000_000
    assert table.estimate("vendor/flash", None, 3001, 1024) == pytest.approx(expected, abs=1e-8)
    assert table.estimate("vendor/flash", "deepinfra/fp8", 0, 100) == pytest.approx(0.000042, abs=1e-12)
    with pytest.raises(PricingError):
        table.estimate("vendor/nope", None, 10, 10)


def test_load_from_path_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "currency": "USD",
                "unit": "per_million_tokens",
                "models": {"a/b": {"prompt": 1, "completion": 2}},
            }
        )
    )
    table = PriceTable.load(path)
    assert table.models == ["a/b"]
    assert table.origin == str(path)
    with pytest.raises(PricingError, match="cannot read"):
        PriceTable.load(tmp_path / "missing.yaml")
    path.write_text("models: [unclosed")
    with pytest.raises(PricingError, match="not valid YAML"):
        PriceTable.load(path)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"currency": "EUR", "models": {}}, "currency must be USD"),
        ({"unit": "per_token", "models": {}}, "per_million_tokens"),
        ({"models": {"a/b": {"prompt": 1}}}, "missing 'completion'"),
        ({"models": {"a/b": {"prompt": -1, "completion": 1}}}, ">= 0"),
        ({"models": {"a/b": {"prompt": "cheap", "completion": 1}}}, "expected a number"),
        ({"models": {"a/b": {"prompt": 1, "completion": 1, "promt": 1}}}, "unknown key"),
        ({"models": {"a/b": {"prompt": 1, "completion": 1, "providers": {"x": {"prompt": 1}}}}}, "missing"),
        ({"models": ["a/b"]}, "must be a mapping"),
        (["not", "a", "mapping"], "top level"),
    ],
)
def test_invalid_tables_are_rejected(data: Any, match: str) -> None:
    with pytest.raises(PricingError, match=match):
        PriceTable.from_dict(data)


def test_bundled_pricing_yaml_loads_with_positive_prices() -> None:
    table = PriceTable.load()
    assert table.origin == "bundled pricing.yaml"
    assert table.source.startswith("https://openrouter.ai/api/v1/")
    assert table.checked
    assert table.models, "the bundled table must not be empty"
    for model in table.models:
        entry = table.entry(model)
        assert entry.default.prompt > 0, model
        assert entry.default.completion > 0, model
        for tag, price in entry.providers.items():
            assert price.prompt > 0, f"{model} [{tag}]"
            assert price.completion > 0, f"{model} [{tag}]"


def test_bundled_pricing_covers_the_candidate_models_and_pins() -> None:
    table = PriceTable.load()
    for model, pins in REQUIRED_PINS.items():
        assert table.has(model), model
        assert pins <= set(table.entry(model).providers), model

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
import respx
import yaml

from booking_truth.llm.pricing import PriceTable

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "update_pricing.py"
BASE = "https://openrouter.ai/api/v1"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("update_pricing", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["update_pricing"] = module
    spec.loader.exec_module(module)
    return module


MODELS: dict[str, Any] = {
    "data": [
        {
            "id": "vendor/flash",
            "canonical_slug": "vendor/flash-20260910",
            "pricing": {"prompt": "0.0000003", "completion": "0.0000012", "input_cache_read": "0.000000006"},
            "expiration_date": None,
            "links": {"details": "/api/v1/models/vendor/flash-20260910/endpoints"},
        },
        {
            "id": "vendor/old",
            "canonical_slug": "vendor/old-20251201",
            "pricing": {"prompt": "0.000000269", "completion": "0.0000004"},
            "expiration_date": "2026-09-28",
        },
        {
            "id": "openrouter/auto",
            "canonical_slug": "openrouter/auto",
            "pricing": {"prompt": "-1", "completion": "-1"},
        },
    ],
    "total_count": 3,
    "links": {"next": None},
}

ENDPOINTS: dict[str, Any] = {
    "data": {
        "id": "vendor/flash",
        "endpoints": [
            {
                "provider_name": "DeepInfra",
                "tag": "deepinfra/fp8",
                "pricing": {
                    "prompt": "0.00000014",
                    "completion": "0.00000042",
                    "input_cache_read": "0.0000000042",
                    "discount": 0.3,
                },
                "supported_parameters": ["tools", "max_tokens"],
            },
            {
                "provider_name": "BaseTen",
                "tag": "baseten/fp8",
                "pricing": {
                    "prompt": "0.0000003",
                    "completion": "0.0000012",
                    "input_cache_read": "0.00000003",
                },
                "supported_parameters": ["tools"],
            },
            {
                "provider_name": "BaseTen",
                "tag": "baseten/fp8",
                "pricing": {
                    "prompt": "0.0000003",
                    "completion": "0.0000012",
                    "input_cache_read": "0.000000007",
                },
                "supported_parameters": ["tools"],
            },
        ],
    }
}


def test_per_token_strings_become_exact_per_million_decimals(script: ModuleType) -> None:
    assert script.per_million("0.00000004704", "x") == Decimal("0.04704")
    assert script.number(script.per_million("0.0000000875", "x")) == "0.0875"
    assert script.number(Decimal("10.000")) == "10"
    assert script.number(Decimal("0.0000001")) == "0.0000001"
    with pytest.raises(script.PricingUpdateError, match="no fixed price"):
        script.per_million("-1", "openrouter/auto.prompt")


def test_rows_render_deterministically_and_load_back(script: ModuleType) -> None:
    notes: list[str] = []
    wanted = {"vendor/old": [], "vendor/flash": ["deepinfra/fp8", "baseten/fp8"]}
    rows = script.build_rows(MODELS, {"vendor/flash": ENDPOINTS}, wanted, warn=notes.append)
    text = script.render(rows, source=f"{BASE}/models", checked="2026-09-26")
    assert text == script.render(list(reversed(rows)), source=f"{BASE}/models", checked="2026-09-26")
    assert "e-" not in text
    assert any("baseten/fp8" in note for note in notes)  # duplicate tag noted

    data = yaml.safe_load(text)
    assert data["checked"] == "2026-09-26"
    assert list(data["models"]) == ["vendor/flash", "vendor/old"]
    assert data["models"]["vendor/old"]["expires"] == "2026-09-28"
    table = PriceTable.from_dict(data)
    assert table.price_for("vendor/flash").prompt == Decimal("0.3")
    assert table.price_for("vendor/flash", "deepinfra/fp8").cache_read == Decimal("0.0042")
    # Two endpoints share the tag: the higher cache price is kept.
    assert table.price_for("vendor/flash", "baseten/fp8").cache_read == Decimal("0.03")
    assert table.resolve("vendor/flash-20260910") == "vendor/flash"
    script.verify(text, rows)


def test_unknown_model_or_tag_and_router_sentinels_are_refused(script: ModuleType) -> None:
    with pytest.raises(script.PricingUpdateError, match="not in the OpenRouter models listing"):
        script.build_rows(MODELS, {}, {"vendor/missing": []})
    with pytest.raises(script.PricingUpdateError, match="does not serve vendor/flash"):
        script.build_rows(MODELS, {"vendor/flash": ENDPOINTS}, {"vendor/flash": ["together"]})
    with pytest.raises(script.PricingUpdateError, match="no fixed price"):
        script.build_rows(MODELS, {}, {"openrouter/auto": []})
    with pytest.raises(script.PricingUpdateError, match="model=tag"):
        script.parse_providers(["vendor/flash"])
    assert script.parse_providers(["a/b=x,y", "a/b=y,z"]) == {"a/b": ["x", "y", "z"]}


def test_main_fetches_public_listings_and_writes_the_table(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "pricing.yaml"
    with respx.mock(base_url=BASE) as mock:
        models = mock.get("/models").mock(return_value=httpx.Response(200, json=MODELS))
        endpoints = mock.get("/models/vendor/flash-20260910/endpoints").mock(
            return_value=httpx.Response(200, json=ENDPOINTS)
        )
        with httpx.Client() as client:
            code = script.main(
                [
                    "--models",
                    "vendor/flash",
                    "vendor/old",
                    "--providers",
                    "vendor/flash=deepinfra/fp8",
                    "--out",
                    str(out),
                    "--checked",
                    "2026-09-26",
                ],
                client=client,
            )
            assert code == 0
            first = out.read_text()
            # Without --models the models and pins already in the file are refreshed.
            assert script.main(["--out", str(out), "--checked", "2026-09-26"], client=client) == 0
    assert out.read_text() == first
    assert models.call_count == 2
    assert endpoints.call_count == 2
    assert "authorization" not in models.calls[0].request.headers
    table = PriceTable.load(out)
    assert table.models == ["vendor/flash", "vendor/old"]
    assert set(table.entry("vendor/flash").providers) == {"deepinfra/fp8"}
    assert "wrote" in capsys.readouterr().err


def test_main_stops_without_writing_when_a_fetch_fails(script: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "pricing.yaml"
    with respx.mock(base_url=BASE) as mock:
        mock.get("/models").mock(return_value=httpx.Response(503, text="unavailable"))
        with httpx.Client() as client:
            assert script.main(["--models", "vendor/flash", "--out", str(out)], client=client) == 1
    assert not out.exists()


def test_a_base_slug_is_priced_at_the_worst_endpoint_it_can_route_to(script: ModuleType) -> None:
    def endpoint(tag: str, prompt: str, completion: str, cache: str | None = None) -> dict[str, Any]:
        pricing = {"prompt": prompt, "completion": completion}
        if cache is not None:
            pricing["input_cache_read"] = cache
        return {"tag": tag, "pricing": pricing, "supported_parameters": ["tools"]}

    listing = {
        "data": {
            "endpoints": [
                endpoint("deepinfra/fp8", "0.00000014", "0.00000042", "0.0000000042"),
                endpoint("deepinfra/turbo", "0.0000002", "0.0000004"),
                endpoint("deepinfra-ish", "0.000009", "0.000009"),
                endpoint("fireworks", "0.00000022", "0.00000066"),
            ]
        }
    }
    wanted = {"vendor/flash": ["deepinfra", "deepinfra/fp8", "fireworks"]}
    [row] = script.build_rows(MODELS, {"vendor/flash": listing}, wanted, warn=lambda _: None)
    # Base slug: the higher of each price over fp8 and turbo; turbo has no cache price, so its cached
    # tokens cost its prompt price. `deepinfra-ish` is another provider and does not match.
    assert row.providers["deepinfra"] == script.Prices(Decimal("0.2"), Decimal("0.42"), Decimal("0.2"))
    assert row.providers["deepinfra/fp8"] == script.Prices(
        Decimal("0.14"), Decimal("0.42"), Decimal("0.0042")
    )
    assert row.providers["fireworks"].completion == Decimal("0.66")
    text = script.render([row], source=f"{BASE}/models", checked="2026-09-26")
    script.verify(text, [row])
    assert PriceTable.from_dict(yaml.safe_load(text)).price_for(
        "vendor/flash", "deepinfra"
    ).prompt == Decimal("0.2")

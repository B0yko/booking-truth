import pytest
from pydantic import SecretStr, ValidationError

from booking_truth.config import (
    GUARD_NAMES,
    ConfigError,
    Settings,
    guards_label,
    is_local_url,
    parse_guards,
    parse_work_days,
    parse_work_hours,
)


def make(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def test_parse_guards_all_off_and_list() -> None:
    assert parse_guards("all") == frozenset(GUARD_NAMES)
    assert parse_guards("off") == frozenset()
    assert parse_guards("claim_ledger, fail_closed") == {"claim_ledger", "fail_closed"}
    assert guards_label(parse_guards("all")) == "all"
    assert guards_label(frozenset()) == "off"


def test_rendered_confirmation_requires_claim_ledger() -> None:
    with pytest.raises(ConfigError, match="requires 'claim_ledger'"):
        parse_guards("rendered_confirmation")
    assert "rendered_confirmation" in parse_guards("claim_ledger,rendered_confirmation")


def test_unknown_guard_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown guard"):
        parse_guards("claim_ledger,telepathy")


def test_work_hours_and_days() -> None:
    assert [t.isoformat() for t in parse_work_hours("09:00-17:00")] == ["09:00:00", "17:00:00"]
    assert parse_work_days("mon-fri") == (1, 2, 3, 4, 5)
    assert parse_work_days("1,3,5") == (1, 3, 5)
    with pytest.raises(ConfigError):
        parse_work_hours("17:00-09:00")


def test_floating_model_rejected_only_with_pinned_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    local = {"calcom_base_url": "http://sandbox:8100"}
    with pytest.raises(ConfigError, match="floating alias"):
        make(llm_model="vendor/model:latest", **local).validate_for_agent()
    make(llm_model="vendor/model:latest", guards="off", **local).validate_for_agent()


def test_api_key_required_for_non_local_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="BT_API_KEY"):
        make(calcom_base_url="https://api.cal.com", calcom_event_type_id=1).validate_for_agent()
    make(calcom_base_url="http://127.0.0.1:8100").validate_for_agent()
    make(
        api_key=SecretStr("k"), calcom_base_url="https://api.cal.com", calcom_event_type_id=1
    ).validate_for_agent()


def test_local_url_detection() -> None:
    assert is_local_url("http://localhost:8100")
    assert is_local_url("http://127.0.0.1:8100")
    assert is_local_url("http://sandbox:8100")
    assert not is_local_url("https://api.cal.com")


def test_empty_key_means_offline_and_openrouter_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("BT_LLM_API_KEY", "")
    assert make().offline
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    assert not make().offline


def test_invalid_zone_rejected() -> None:
    with pytest.raises(ValidationError):
        make(host_timezone="Mars/Olympus")

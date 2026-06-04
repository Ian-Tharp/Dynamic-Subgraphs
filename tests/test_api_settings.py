# tests/test_api_settings.py
from __future__ import annotations

from app.api.settings import ApiSettings


def test_defaults_are_mock_and_free() -> None:
    settings = ApiSettings.from_env({})
    assert settings.planner == "mock"
    assert settings.provider == "openai"
    assert settings.model == "gpt-5.4-nano"
    assert "gpt-5.4-nano" in settings.model_allowlist
    assert settings.api_key is None
    assert settings.auto_sync_seconds == 25
    assert settings.max_sync_seconds == 120


def test_env_overrides_are_parsed() -> None:
    env = {
        "DS_PLANNER": "openai",
        "DS_MODEL": "gpt-5.4-nano",
        "DS_MODEL_ALLOWLIST": "gpt-5.4-nano, gpt-5.4-mini",
        "DS_RUNS_DIR": "/tmp/runs",
        "DS_API_KEY": "secret",
        "DS_AUTO_SYNC_SECONDS": "5",
        "DS_MAX_SYNC_SECONDS": "30",
    }
    settings = ApiSettings.from_env(env)
    assert settings.planner == "llm"
    assert settings.provider == "openai"
    assert settings.model_allowlist == ("gpt-5.4-nano", "gpt-5.4-mini")
    assert settings.runs_dir == "/tmp/runs"
    assert settings.api_key == "secret"
    assert settings.auto_sync_seconds == 5
    assert settings.max_sync_seconds == 30


def test_model_allowed_check() -> None:
    settings = ApiSettings.from_env({"DS_MODEL_ALLOWLIST": "a,b"})
    assert settings.is_model_allowed("a") is True
    assert settings.is_model_allowed("zzz") is False


def test_model_allowed_accepts_provider_qualified_entries() -> None:
    settings = ApiSettings.from_env({"DS_MODEL_ALLOWLIST": "anthropic:claude-test"})
    assert settings.is_model_allowed("claude-test", provider="anthropic") is True
    assert settings.is_model_allowed("claude-test", provider="openai") is False

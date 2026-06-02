# tests/test_api_deps.py
from __future__ import annotations

import pytest

from app.api.deps import AppContext, resolve_chain_decider, resolve_run_config
from app.api.errors import BadRequest, ServiceUnavailable
from app.api.settings import ApiSettings
from app.supervisor import IterationContext, IterationDecision, StatusIterationDecider


def _ctx(**env) -> AppContext:
    return AppContext.build(ApiSettings.from_env(env))


def test_resolve_defaults_to_settings() -> None:
    ctx = _ctx()
    config = resolve_run_config(ctx, planner=None, model=None)
    assert config.planner == "mock"
    assert config.model == "gpt-5.4-nano"
    assert config.strict_runners is False


def test_resolve_rejects_model_outside_allowlist() -> None:
    ctx = _ctx(DS_MODEL_ALLOWLIST="gpt-5.4-nano")
    with pytest.raises(BadRequest):
        resolve_run_config(ctx, planner=None, model="evil-model")


def test_resolve_openai_without_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ctx = _ctx()
    with pytest.raises(ServiceUnavailable):
        resolve_run_config(ctx, planner="openai", model=None)


def test_resolve_openai_with_key_ok(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    ctx = _ctx()
    config = resolve_run_config(ctx, planner="openai", model=None)
    assert config.planner == "openai"
    assert config.strict_runners is True


def test_resolve_chain_decider_status_is_token_free(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ctx = _ctx()
    config = resolve_run_config(ctx, planner="mock", model=None)

    decider = resolve_chain_decider(
        ctx,
        config=config,
        decider="status",
        success_criteria=None,
        judge_failed_runs=False,
    )

    assert isinstance(decider, StatusIterationDecider)


def test_resolve_chain_decider_llm_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ctx = _ctx()
    config = resolve_run_config(ctx, planner="mock", model=None)

    with pytest.raises(ServiceUnavailable):
        resolve_chain_decider(
            ctx,
            config=config,
            decider="llm",
            success_criteria="must be useful",
            judge_failed_runs=False,
        )


def test_resolve_chain_decider_llm_builds_structured_judge(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    ctx = _ctx()
    config = resolve_run_config(ctx, planner="mock", model=None)
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)

        def fake_decider(context: IterationContext) -> IterationDecision:
            del context
            return IterationDecision(
                action="stop",
                reason="good enough",
                success_criteria_met=True,
            )

        return fake_decider

    monkeypatch.setattr("app.api.deps.build_openai_iteration_decider", fake_builder)

    decider = resolve_chain_decider(
        ctx,
        config=config,
        decider="llm",
        success_criteria="must mention CORE orchestration",
        judge_failed_runs=True,
    )

    assert callable(decider)
    assert captured == {
        "model": "gpt-5.4-nano",
        "success_criteria": "must mention CORE orchestration",
        "judge_failed_runs": True,
    }

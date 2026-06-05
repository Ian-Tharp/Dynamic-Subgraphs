# tests/test_assembly.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.assembly import RunConfig, build_supervisor
from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.recording import FileRecorder
from app.runtime import ModelRef, ProviderRegistry
from app.supervisor import StaticPlanner, Supervisor


def test_build_supervisor_mock_returns_supervisor(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)

    supervisor = build_supervisor(config, recorder=recorder)

    assert isinstance(supervisor, Supervisor)


def test_build_supervisor_mock_runs_end_to_end(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)
    supervisor = build_supervisor(config, recorder=recorder)

    result = supervisor.run("compare two things", run_id="assembly-001")

    assert result.status == "ok"
    assert (tmp_path / "assembly-001" / "spec.json").exists()


def test_build_supervisor_mock_uses_static_planner(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)

    supervisor = build_supervisor(config, recorder=recorder)

    assert isinstance(supervisor._planner, StaticPlanner)


def test_run_config_rejects_the_legacy_openai_planner_alias() -> None:
    # The "openai" alias is normalized to ("llm", provider="openai") at the outer
    # adapters (the API's resolve_run_config, the CLI); RunConfig itself only
    # accepts the two real modes, so the alias can't drift in two places.
    with pytest.raises(ValueError, match="mock"):
        RunConfig(planner="openai", model="gpt-5.4-nano", strict_runners=True)


def test_run_config_roles_default_to_base_ref() -> None:
    config = RunConfig(planner="llm", provider="openai", model="m", strict_runners=True)
    base = ModelRef(provider="openai", model="m")

    # Every role collapses to the base ref when nothing is overridden.
    assert config.worker_ref == base
    assert config.planner_ref == base
    assert config.reducer_ref == base
    assert config.subagent_ref == base
    assert config.judge_ref == base
    assert config.providers_in_use() == ("openai",)


def test_run_config_unset_roles_fall_back_to_worker() -> None:
    worker = ModelRef(provider="anthropic", model="claude-haiku-4-5")
    config = RunConfig(
        planner="llm",
        provider="openai",
        model="gpt-5.4-nano",
        strict_runners=True,
        worker_model=worker,
    )

    # planner/reducer/subagent inherit the worker override, not the base ref.
    assert config.worker_ref == worker
    assert config.reducer_ref == worker
    assert config.subagent_ref == worker
    assert config.planner_ref == worker
    # base ref is still the explicit provider+model.
    assert config.base_ref == ModelRef(provider="openai", model="gpt-5.4-nano")


def test_run_config_mixed_providers_reported_in_use() -> None:
    config = RunConfig(
        planner="llm",
        provider="openai",
        model="gpt-5.4-nano",
        strict_runners=True,
        planner_model=ModelRef(provider="openai", model="gpt-5.4-nano"),
        worker_model=ModelRef(provider="anthropic", model="claude-haiku-4-5"),
        subagent_model=ModelRef(provider="ollama", model="llama3.1"),
    )

    assert config.providers_in_use() == ("anthropic", "ollama", "openai")


def test_run_config_mock_planner_uses_no_providers() -> None:
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)

    assert config.providers_in_use() == ()


class _FakeChatModel:
    def invoke(self, messages: list, /, **kwargs: Any) -> Any:
        del messages, kwargs
        return AIMessage(content="fake provider response")


class _FakeStructuredPlanner:
    def invoke(self, messages: list, /, **kwargs: Any) -> GraphSpec:
        del messages, kwargs
        return GraphSpec(
            graph_id="fake-provider-plan",
            goal="prove provider-neutral assembly",
            nodes=[
                NodeSpec(
                    id="answer",
                    kind=NodeKind.LLM_CALL,
                    outputs=["final"],
                    params={"instruction": "answer through the fake provider"},
                )
            ],
            edges=[
                EdgeSpec.model_validate({"from": "START", "to": "answer"}),
                EdgeSpec.model_validate({"from": "answer", "to": "END"}),
            ],
        )


class _FakeProvider:
    name = "fake"

    def required_env_vars(self) -> tuple[str, ...]:
        return ()

    def build_chat(self, ref: ModelRef) -> _FakeChatModel:
        assert ref == ModelRef(provider="fake", model="fake-model")
        return _FakeChatModel()

    def build_structured_output(self, ref: ModelRef, schema: type) -> Any:
        assert ref == ModelRef(provider="fake", model="fake-model")
        assert schema is GraphSpec
        return _FakeStructuredPlanner()


def test_build_supervisor_llm_uses_registered_provider(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    registry = ProviderRegistry()
    registry.register(_FakeProvider())
    config = RunConfig(
        planner="llm",
        provider="fake",
        model="fake-model",
        strict_runners=True,
    )

    supervisor = build_supervisor(
        config,
        recorder=recorder,
        model_providers=registry,
    )
    result = supervisor.run("use the fake provider", run_id="assembly-provider")

    assert result.status == "ok"
    assert result.result is not None
    assert result.result.state["values"]["final"] == "fake provider response"


class _TrackingProvider:
    """Records which refs each role asked it to build."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.chat_refs: list[ModelRef] = []
        self.structured_refs: list[ModelRef] = []

    def required_env_vars(self) -> tuple[str, ...]:
        return ()

    def build_chat(self, ref: ModelRef) -> _FakeChatModel:
        self.chat_refs.append(ref)
        return _FakeChatModel()

    def build_structured_output(self, ref: ModelRef, schema: type) -> Any:
        self.structured_refs.append(ref)
        return _FakeStructuredPlanner()


def test_build_supervisor_routes_each_role_to_its_provider(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    alpha = _TrackingProvider("alpha")
    beta = _TrackingProvider("beta")
    registry = ProviderRegistry()
    registry.register(alpha)
    registry.register(beta)

    # Planner explicitly on alpha; workers/reducers/subagents on beta.
    # (planner_model must be set explicitly — unset roles fall back to worker,
    # so overriding only worker would route the planner to beta too.)
    config = RunConfig(
        planner="llm",
        provider="alpha",
        model="alpha-model",
        strict_runners=True,
        planner_model=ModelRef(provider="alpha", model="alpha-model"),
        worker_model=ModelRef(provider="beta", model="beta-model"),
    )

    build_supervisor(config, recorder=recorder, model_providers=registry)

    # Planner used alpha for structured output; chat roles used beta.
    assert alpha.structured_refs == [ModelRef(provider="alpha", model="alpha-model")]
    assert alpha.chat_refs == []
    beta_ref = ModelRef(provider="beta", model="beta-model")
    # worker, reducer, subagent all resolve to the same beta ref -> built once.
    assert beta.chat_refs == [beta_ref]
    assert config.providers_in_use() == ("alpha", "beta")

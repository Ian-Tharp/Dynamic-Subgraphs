# tests/test_assembly.py
from __future__ import annotations

from pathlib import Path
from typing import Any

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


def test_run_config_normalizes_legacy_openai_planner() -> None:
    config = RunConfig(
        planner="openai",
        model="gpt-5.4-nano",
        strict_runners=True,
    )

    assert config.planner == "llm"
    assert config.provider == "openai"
    assert config.model_ref == ModelRef(provider="openai", model="gpt-5.4-nano")


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

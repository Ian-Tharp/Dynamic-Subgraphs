"""Tests for the public `dynamic_subgraphs` SDK facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.runtime import ModelRef, ProviderRegistry
from dynamic_subgraphs import (
    Artifact,
    DynamicSubgraphs,
    EngineConfig,
    Model,
    ModelSelection,
    Recording,
    RunResult,
    TokenUsage,
)
from dynamic_subgraphs.engine import _compute_cost

# ---------- Model construction ----------


def test_model_is_modelref_alias() -> None:
    # Identity matters: Model values must compare/cache equal to ModelRefs.
    assert Model is ModelRef
    assert Model("openai", "gpt-5.4-nano") == ModelRef(
        provider="openai", model="gpt-5.4-nano"
    )


def test_model_lmstudio_defaults() -> None:
    ref = Model.lmstudio("openai/gpt-oss-20b")

    assert ref.provider == "openai"
    assert ref.model == "openai/gpt-oss-20b"
    assert ref.base_url == "http://localhost:1234/v1"
    assert ref.api_key == "lm-studio"
    # Local servers reject forced tool_choice -> json_schema by default.
    assert ref.structured_method == "json_schema"


def test_model_openai_compatible_and_ollama() -> None:
    compat = Model.openai_compatible("m", base_url="http://h/v1", api_key="k")
    assert compat.provider == "openai"
    assert compat.base_url == "http://h/v1"
    assert compat.api_key == "k"

    oll = Model.ollama("llama3.1", base_url="http://localhost:11434")
    assert oll.provider == "ollama"
    assert oll.base_url == "http://localhost:11434"


def test_modelref_client_kwargs_promotes_first_class_fields() -> None:
    ref = ModelRef(
        provider="openai",
        model="m",
        temperature=0.2,
        base_url="http://h/v1",
        api_key="k",
    )
    kwargs = ref.client_kwargs()

    assert kwargs == {
        "model": "m",
        "temperature": 0.2,
        "base_url": "http://h/v1",
        "api_key": "k",
    }


# ---------- EngineConfig ----------


def test_engine_config_resolves_selection_and_recording() -> None:
    cfg = EngineConfig(
        model=Model("openai", "m"),
        worker_model=Model("anthropic", "w"),
        recording=Recording.visual_only(),
    )
    sel = cfg.model_selection()
    assert sel.model == Model("openai", "m")
    assert sel.worker_model == Model("anthropic", "w")
    assert cfg.recording_policy() == Recording.visual_only()


def test_engine_exposes_its_config(tmp_path: Path) -> None:
    cfg = EngineConfig(model=Model("openai", "x"), planner="mock", runs_dir=tmp_path)
    engine = DynamicSubgraphs(cfg)
    assert engine.config is cfg


def test_engine_config_coerces_bool_recording() -> None:
    assert EngineConfig(recording=True).recording_policy() == Recording.all()
    assert EngineConfig().recording_policy() == Recording.none()


# ---------- ModelSelection resolution ----------


def test_selection_unset_roles_fall_back_to_worker_then_base() -> None:
    base = Model("openai", "base")
    worker = Model("anthropic", "worker")
    sel = ModelSelection(model=base, worker_model=worker)

    cfg = sel.to_run_config("llm")
    assert cfg.worker_ref == worker
    assert cfg.planner_ref == worker  # falls back to worker, not base
    assert cfg.reducer_ref == worker
    # base ref is still reported for provider/model strings
    assert cfg.base_ref.model == "worker"


def test_selection_merge_applies_overrides_only() -> None:
    base = ModelSelection(model=Model("openai", "a"), worker_model=Model("openai", "w"))
    override = ModelSelection(worker_model=Model("ollama", "local"))

    merged = base.merge(override)
    assert merged.model == Model("openai", "a")  # unchanged
    assert merged.worker_model == Model("ollama", "local")  # overridden


def test_selection_llm_without_model_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="needs a model"):
        ModelSelection().to_run_config("llm")


# ---------- end-to-end via the facade (token-free mock planner) ----------


def test_engine_does_not_write_files_by_default(tmp_path: Path) -> None:
    engine = DynamicSubgraphs(
        EngineConfig(model=Model("openai", "x"), planner="mock", runs_dir=tmp_path)
    )
    result = engine.run("compare two things", run_id="sdk-nofiles-001")

    # The run still works and returns full data...
    assert result.ok and result.status == "ok"
    assert result.plan is not None and result.values
    # ...but nothing is written to disk by default.
    assert result.artifacts == {}
    assert not (tmp_path / "sdk-nofiles-001").exists()
    assert list(tmp_path.iterdir()) == []


# ---------- Artifact / Recording vocabulary ----------


def test_artifact_values_are_filenames() -> None:
    assert Artifact.MERMAID == "graph.mmd"
    assert Artifact.SPEC == "spec.json"


def test_recording_coerce_table() -> None:
    assert Recording.coerce(False) == Recording.none()
    assert Recording.coerce(None) == Recording.none()
    assert Recording.coerce(True) == Recording.all()
    assert Recording.coerce(Artifact.MERMAID) == Recording({Artifact.MERMAID})
    assert Recording.coerce({Artifact.MERMAID, Artifact.TRACE}).kinds == {
        Artifact.MERMAID,
        Artifact.TRACE,
    }
    # strings map by filename or member name
    assert Recording.coerce(["graph.mmd", "trace"]).kinds == {
        Artifact.MERMAID,
        Artifact.TRACE,
    }


def test_recording_unknown_artifact_raises_with_valid_list() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown artifact 'bogus'"):
        Recording.coerce(["bogus"])


def test_recording_set_ops_and_presets() -> None:
    minus_spec = Recording.all() - {Artifact.SPEC}
    assert Artifact.SPEC not in minus_spec
    assert Artifact.MERMAID in minus_spec

    assert Recording.visual_only().recorder_filenames() == {"graph.mmd"}
    assert Recording.visual_only().records_anything() is True
    assert Recording.none().records_anything() is False
    # all() includes the EMITTED sink toggle; none() does not.
    assert Recording.all().emits() is True
    assert Recording.none().emits() is False


def test_engine_selective_record_writes_only_chosen(tmp_path: Path) -> None:
    engine = DynamicSubgraphs(
        EngineConfig(
            model=Model("openai", "x"),
            planner="mock",
            recording=Recording.visual_only(),
            runs_dir=tmp_path,
        )
    )
    result = engine.run("compare two things", run_id="sdk-visual-001")

    assert result.ok
    # Only the mermaid diagram is written — the motivating use case.
    assert set(result.artifacts) == {"graph.mmd"}
    assert (tmp_path / "sdk-visual-001" / "graph.mmd").exists()
    assert not (tmp_path / "sdk-visual-001" / "spec.json").exists()


def test_engine_record_all_minus_spec(tmp_path: Path) -> None:
    engine = DynamicSubgraphs(
        EngineConfig(
            model=Model("openai", "x"),
            planner="mock",
            recording=Recording.all() - {Artifact.SPEC},
            runs_dir=tmp_path,
        )
    )
    result = engine.run("compare two things", run_id="sdk-nospec-001")

    assert "graph.mmd" in result.artifacts
    assert "spec.json" not in result.artifacts


def test_engine_per_run_record_overrides_default(tmp_path: Path) -> None:
    # Engine default writes nothing; one run opts into full recording.
    engine = DynamicSubgraphs(
        EngineConfig(model=Model("openai", "x"), planner="mock", runs_dir=tmp_path)
    )

    quiet = engine.run("one", run_id="sdk-quiet-001")
    loud = engine.run("two", run_id="sdk-loud-001", record=True)

    assert quiet.artifacts == {}
    assert "spec.json" in loud.artifacts


def test_engine_record_true_writes_files(tmp_path: Path) -> None:
    engine = DynamicSubgraphs(
        EngineConfig(
            model=Model("openai", "gpt-5.4-nano"),
            planner="mock",
            recording=True,
            runs_dir=tmp_path,
        )
    )
    result = engine.run("compare two things", run_id="sdk-mock-001")

    assert isinstance(result, RunResult)
    assert result.ok and result.status == "ok"
    assert result.plan is not None and isinstance(result.plan, GraphSpec)
    assert result.values  # produced at least one output
    # Artifacts are surfaced as filename -> path under runs_dir/run_id.
    assert "spec.json" in result.artifacts
    assert result.artifacts["spec.json"].exists()


def test_engine_auto_generates_unique_run_ids(tmp_path: Path) -> None:
    engine = DynamicSubgraphs(
        EngineConfig(
            model=Model("openai", "x"),
            planner="mock",
            recording=True,
            runs_dir=tmp_path,
        )
    )
    a = engine.run("one")
    b = engine.run("two")

    assert a.run_id != b.run_id
    assert (tmp_path / a.run_id).is_dir()


# ---------- docs / drift guards ----------


def test_artifact_values_match_recorder_filenames() -> None:
    from app.recording import DEFAULT_PRODUCERS

    producer_files = {p.filename for p in DEFAULT_PRODUCERS}
    artifact_files = {a.value for a in Artifact if a is not Artifact.EMITTED}
    assert artifact_files == producer_files


def test_every_artifact_is_documented_in_recipes() -> None:
    recipes = (Path(__file__).resolve().parents[1] / "docs" / "recipes.md").read_text(
        encoding="utf-8"
    )
    for artifact in Artifact:
        assert artifact.name in recipes or artifact.value in recipes, (
            f"{artifact!r} not mentioned in docs/recipes.md"
        )


# ---------- token usage + cost ----------


def test_token_usage_from_handler_aggregates_by_model() -> None:
    from types import SimpleNamespace

    handler = SimpleNamespace(
        usage_metadata={
            "gpt-5.4-nano": {
                "input_tokens": 100,
                "output_tokens": 40,
                "total_tokens": 140,
            },
            "claude-haiku-4-5": {
                "input_tokens": 60,
                "output_tokens": 20,
                "total_tokens": 80,
            },
        }
    )
    usage = TokenUsage.from_handler(handler)

    assert usage.input_tokens == 160
    assert usage.output_tokens == 60
    assert usage.total_tokens == 220
    assert set(usage.by_model) == {"gpt-5.4-nano", "claude-haiku-4-5"}


def test_compute_cost_from_price_book() -> None:
    usage = TokenUsage(
        by_model={
            "gpt-5.4-nano": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        }
    )
    pricing = {"gpt-5.4-nano": {"input_per_1m": 0.2, "output_per_1m": 1.25}}
    assert _compute_cost(usage, pricing) == 1.45


def test_compute_cost_none_when_nothing_priceable(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "litellm", None)  # force auto-cost off
    usage = TokenUsage(by_model={"m": {"input_tokens": 1000, "output_tokens": 500}})
    # No price source resolves the model -> None (not a misleading $0.00).
    assert _compute_cost(usage, None) is None
    assert _compute_cost(usage, {"other": {"input_per_1m": 1.0}}) is None


def test_litellm_auto_cost_when_available(monkeypatch) -> None:
    import sys
    import types

    fake = types.SimpleNamespace(
        suppress_debug_info=False,
        cost_per_token=lambda model, prompt_tokens, completion_tokens: (
            prompt_tokens * 1e-6,
            completion_tokens * 2e-6,
        ),
    )
    monkeypatch.setitem(sys.modules, "litellm", fake)
    from dynamic_subgraphs.engine import _litellm_cost

    assert _litellm_cost("any-model", 1_000_000, 1_000_000) == 3.0
    # _compute_cost falls back to LiteLLM when no manual pricing is given.
    usage = TokenUsage(
        by_model={"any": {"input_tokens": 1_000_000, "output_tokens": 0}}
    )
    assert _compute_cost(usage, None) == 1.0


def test_manual_pricing_overrides_litellm(monkeypatch) -> None:
    import sys
    import types

    # LiteLLM would return a huge price; the manual entry must win.
    fake = types.SimpleNamespace(
        suppress_debug_info=False,
        cost_per_token=lambda **k: (999.0, 999.0),
    )
    monkeypatch.setitem(sys.modules, "litellm", fake)
    usage = TokenUsage(by_model={"m": {"input_tokens": 1_000_000, "output_tokens": 0}})
    assert (
        _compute_cost(usage, {"m": {"input_per_1m": 0.5, "output_per_1m": 1.0}}) == 0.5
    )


def test_compute_cost_matches_dated_snapshot_by_alias() -> None:
    # Providers report a dated snapshot; pricing is keyed by the alias.
    usage = TokenUsage(
        by_model={
            "gpt-5.4-nano-2026-03-17": {
                "input_tokens": 1_000_000,
                "output_tokens": 0,
            }
        }
    )
    pricing = {"gpt-5.4-nano": {"input_per_1m": 0.2, "output_per_1m": 1.25}}
    assert _compute_cost(usage, pricing) == 0.2


def test_engine_mock_run_has_zero_usage_and_no_cost(tmp_path: Path) -> None:
    engine = DynamicSubgraphs(
        EngineConfig(model=Model("openai", "x"), planner="mock", runs_dir=tmp_path)
    )
    result = engine.run("compare two things")

    assert isinstance(result.usage, TokenUsage)
    assert result.usage.total_tokens == 0  # mock planner makes no LLM calls
    assert result.cost is None  # no pricing configured
    # usage + cost surface in the JSON-safe view too.
    assert "usage" in result.to_dict() and "cost" in result.to_dict()


# ---------- agent-consumability ----------


def test_capabilities_is_machine_readable_and_complete() -> None:
    import json

    caps = DynamicSubgraphs.capabilities()
    # JSON-safe, no custom encoder needed.
    json.dumps(caps)

    assert "anthropic" in caps["providers"] and "openai" in caps["providers"]
    assert set(caps["planners"]) == {"llm", "mock"}
    assert "graph.mmd" in caps["artifacts"]
    assert "ok" in caps["statuses"]
    assert "json_schema" in caps["structured_methods"]
    assert "lmstudio" in caps["model_constructors"]
    assert "worker" in caps["model_roles"]

    # recording_presets and model_constructors must mirror the real classmethods,
    # not a hand-maintained literal that silently drifts (the never-drift contract
    # the other capability keys already keep).
    from dynamic_subgraphs import Model, Recording

    for preset in caps["recording_presets"]:
        made = getattr(Recording, preset)()
        assert isinstance(made, Recording), f"{preset} is not a Recording preset"
    for ctor in caps["model_constructors"]:
        assert callable(getattr(Model, ctor)), f"{ctor} is not a Model constructor"

    # node_kinds must mirror the registry's NodeKind enum exactly, so the
    # agent-facing surface (and docs that read from it) can't drift from the
    # vocabulary the compiler actually accepts.
    from app.models import NodeKind

    assert set(caps["node_kinds"]) == {k.value for k in NodeKind}
    assert "branch" in caps["node_kinds"]


def test_readme_enumerates_every_node_kind() -> None:
    """The README's vocabulary list must name every NodeKind.

    This exact drift shipped once: the registry had nine kinds but the README
    listed eight (`branch` was missing). This guard fails if a kind is added to
    the enum but not documented.
    """
    from app.models import NodeKind

    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    missing = [k.value for k in NodeKind if f"`{k.value}`" not in text]
    assert not missing, f"README does not document node kinds: {missing}"


def test_unknown_planner_raises_with_valid_list(tmp_path: Path) -> None:
    import pytest

    engine = DynamicSubgraphs(
        EngineConfig(model=Model("openai", "x"), planner="auto", runs_dir=tmp_path)
    )
    with pytest.raises(ValueError, match="Unknown planner 'auto'.*llm, mock"):
        engine.run("anything", run_id="bad-planner")


def test_run_result_to_dict_is_json_safe(tmp_path: Path) -> None:
    import json

    engine = DynamicSubgraphs(
        EngineConfig(
            model=Model("openai", "x"),
            planner="mock",
            recording=True,
            runs_dir=tmp_path,
        )
    )
    result = engine.run("compare two things", run_id="sdk-todict-001")
    payload = result.to_dict()

    # Round-trips through json with no custom encoder.
    json.dumps(payload)
    assert payload["status"] == "ok" and payload["ok"] is True
    assert isinstance(payload["plan"], dict)  # GraphSpec serialized
    assert all(isinstance(p, str) for p in payload["artifacts"].values())


# ---------- per-run model override routing (no real network) ----------


class _FakeChatModel:
    def invoke(self, messages: list, /, **kwargs: Any) -> Any:
        del messages, kwargs
        return AIMessage(content="fake response")


class _FakeStructuredPlanner:
    def invoke(self, messages: list, /, **kwargs: Any) -> GraphSpec:
        del messages, kwargs
        return GraphSpec(
            graph_id="sdk-fake-plan",
            goal="prove SDK provider routing",
            nodes=[
                NodeSpec(
                    id="answer",
                    kind=NodeKind.LLM_CALL,
                    outputs=["final"],
                    params={"instruction": "answer"},
                )
            ],
            edges=[
                EdgeSpec.model_validate({"from": "START", "to": "answer"}),
                EdgeSpec.model_validate({"from": "answer", "to": "END"}),
            ],
        )


class _TrackingProvider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.chat_refs: list[ModelRef] = []

    def required_env_vars(self) -> tuple[str, ...]:
        return ()

    def build_chat(self, ref: ModelRef) -> _FakeChatModel:
        self.chat_refs.append(ref)
        return _FakeChatModel()

    def build_structured_output(self, ref: ModelRef, schema: type) -> Any:
        del ref, schema
        return _FakeStructuredPlanner()


def test_engine_per_run_override_routes_to_named_provider(tmp_path: Path) -> None:
    alpha = _TrackingProvider("alpha")
    beta = _TrackingProvider("beta")
    registry = ProviderRegistry()
    registry.register(alpha)
    registry.register(beta)

    engine = DynamicSubgraphs(
        EngineConfig(
            model=Model("alpha", "alpha-model"),
            providers=registry,
            runs_dir=tmp_path,
        )
    )

    # Per-run override sends the workers to beta for this call only.
    result = engine.run(
        "use beta this time",
        run_id="sdk-route-001",
        worker_model=Model("beta", "beta-model"),
    )

    # Compare by provider+model: the engine attaches a usage callback to each
    # chat model's ref via extra_kwargs, so exact ModelRef equality won't hold.
    assert result.ok
    assert [(r.provider, r.model) for r in beta.chat_refs] == [("beta", "beta-model")]
    assert alpha.chat_refs == []  # alpha was only the (overridden) default

    # A subsequent run with no override falls back to the engine default.
    alpha.chat_refs.clear()
    result2 = engine.run("use the default", run_id="sdk-route-002")
    assert result2.ok
    assert [(r.provider, r.model) for r in alpha.chat_refs] == [
        ("alpha", "alpha-model")
    ]

"""RunConfig.fixed_spec: hand-authored specs on the REAL runner path (PR7a)."""

from app.assembly import RunConfig, _build_planner
from app.runtime import default_model_providers
from app.supervisor import StaticPlanner, build_demo_spec


def test_fixed_spec_short_circuits_llm_planner() -> None:
    spec = build_demo_spec()
    config = RunConfig(
        planner="llm", model="gpt-5.4-nano", strict_runners=True, fixed_spec=spec
    )
    planner = _build_planner(config, model_providers=default_model_providers())
    assert isinstance(planner, StaticPlanner)
    assert planner("any prompt") is spec  # prompt ignored, exact spec returned


def test_fixed_spec_with_mock_planner_still_static() -> None:
    # mock + fixed_spec: static plan, mock token-free runners — offline benchmarking.
    spec = build_demo_spec()
    config = RunConfig(planner="mock", model="m", strict_runners=False, fixed_spec=spec)
    planner = _build_planner(config, model_providers=default_model_providers())
    assert planner("x") is spec


def test_engine_threads_fixed_spec(tmp_path) -> None:
    from dynamic_subgraphs import DynamicSubgraphs, EngineConfig

    spec = build_demo_spec()
    r = DynamicSubgraphs(EngineConfig(planner="mock", runs_dir=tmp_path)).run(
        "ignored", fixed_spec=spec, record=False
    )
    assert r.ok
    assert r.plan is not None and r.plan.graph_id == spec.graph_id


def test_fixed_spec_invalid_still_validated(tmp_path) -> None:
    """Invalid fixed_spec is still rejected by the validation stage.

    Demonstrates that fixed_spec bypasses *planning* but NOT *validation* —
    registry allowlists/budgets are still enforced (governance layer is intact).
    """
    from app.models.graph_spec import EdgeSpec
    from dynamic_subgraphs import DynamicSubgraphs, EngineConfig

    # Dangling edge: synthesize -> ghost (but ghost doesn't exist)
    bad_spec = build_demo_spec()
    bad_spec.edges.append(
        EdgeSpec.model_validate({"from": "synthesize", "to": "ghost"})
    )

    r = DynamicSubgraphs(EngineConfig(planner="mock", runs_dir=tmp_path)).run(
        "ignored", fixed_spec=bad_spec, record=False
    )
    assert not r.ok
    assert r.status == "validation_failed"
    assert r.plan is None

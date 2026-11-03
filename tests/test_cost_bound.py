"""Pre-flight cost bound (app.governance.cost) and the cost gate."""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from app.governance import CostModel, ModelPrice, TokenCap, check_plan_policy
from app.governance.cost import estimate_plan_cost
from app.models import GraphSpec, NodeKind
from app.models.graph_spec import GraphBudget
from app.policy import ExecutionPolicy
from app.recording import NullRecorder
from app.runtime import LangGraphExecutor
from app.supervisor import StaticPlanner, Supervisor
from app.supervisor.planner import build_demo_spec
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig

# $1 / $4 per 1M tokens; default cap 8k in + 2k out => $0.016 per call.
PRICES = {"cheap": ModelPrice(1.0, 4.0), "pricey": ModelPrice(10.0, 40.0)}
PER_CALL = 8_000 / 1e6 * 1.0 + 2_000 / 1e6 * 4.0


def _cost_model(**kw: Any) -> CostModel:
    return CostModel(prices=PRICES, default_model="cheap", **kw)


def _bound(spec: GraphSpec, cm: CostModel, *, calls: int = 8, fanout: int = 64):
    return estimate_plan_cost(spec, cm, max_llm_calls=calls, max_fanout=fanout)


@pytest.fixture
def chain(make_edge):
    def _chain(*nodes, budget: GraphBudget | None = None) -> GraphSpec:
        ids = ["START", *[n.id for n in nodes], "END"]
        return GraphSpec(
            graph_id="g",
            goal="t",
            budget=budget or GraphBudget(),
            nodes=list(nodes),
            edges=[make_edge(a, b) for a, b in itertools.pairwise(ids)],
        )

    return _chain


# ---------- the estimate ----------


def test_demo_plan_bound_counts_each_llm_node_once() -> None:
    bound = _bound(build_demo_spec(), _cost_model())

    assert bound.llm_calls == 3  # two gathers + synthesize; concat reduce is free
    assert bound.tokens == 3 * 10_000
    assert bound.usd == pytest.approx(3 * PER_CALL)
    assert not bound.open_ended


def test_node_model_override_and_kind_caps_are_used(make_node, chain) -> None:
    spec = chain(
        make_node("a", params={"instruction": "x", "model": "pricey"}),
        make_node("b"),
    )
    cm = _cost_model(kind_caps={NodeKind.LLM_CALL: TokenCap(1_000, 1_000)})

    bound = _bound(spec, cm)

    assert bound.tokens == 4_000
    assert bound.usd == pytest.approx((1_000 * 10 + 1_000 * 40 + 1_000 + 4_000) / 1e6)


def test_prefix_price_match_covers_dated_snapshots(make_node, chain) -> None:
    spec = chain(make_node("a", params={"instruction": "x", "model": "cheap-2026-01"}))

    assert _bound(spec, _cost_model()).usd == pytest.approx(PER_CALL)


def test_parallel_map_llm_child_counts_up_to_fanout(make_node, chain) -> None:
    spec = chain(
        make_node(
            "seed",
            NodeKind.TOOL_CALL,
            outputs=["items"],
            params={"tool_name": "web_search"},
        ),
        make_node(
            "fan",
            NodeKind.PARALLEL_MAP,
            inputs=["items"],
            params={
                "over": "items",
                "child_kind": "llm_call",
                "child_params": {"instruction": "x"},
            },
        ),
    )

    bound = _bound(spec, _cost_model(), calls=100, fanout=5)

    assert bound.llm_calls == 5


def test_bound_never_exceeds_the_runtime_llm_call_ceiling(make_node, chain) -> None:
    spec = chain(*[make_node(f"n{i}") for i in range(6)])

    bound = _bound(spec, _cost_model(), calls=2)

    assert bound.llm_calls == 2
    assert bound.usd == pytest.approx(2 * PER_CALL)


def test_open_ended_spawn_is_priced_at_the_most_expensive_model(
    make_node, chain
) -> None:
    spec = chain(
        make_node(
            "child", NodeKind.SPAWN_SUBGRAPH, params={"sub_goal": "dig", "name": "dig"}
        )
    )
    cm = _cost_model(extra_models=frozenset({"pricey"}))

    bound = _bound(spec, cm, calls=3)

    assert bound.open_ended
    assert bound.llm_calls == 3
    assert bound.usd == pytest.approx(3 * (8_000 * 10 + 2_000 * 40) / 1e6)


def test_unpriced_model_gives_no_dollar_figure(make_node, chain) -> None:
    spec = chain(make_node("a", params={"instruction": "x", "model": "mystery"}))

    bound = _bound(spec, _cost_model())

    assert bound.usd is None
    assert bound.unpriced_models == ("mystery",)
    assert bound.tokens == 10_000


def test_sdk_pricing_dict_form_is_accepted() -> None:
    cm = CostModel(prices={"m": {"input_per_1m": 2.0, "output_per_1m": 8.0}})

    assert cm.price_for("m") == ModelPrice(2.0, 8.0)


# ---------- the cost gate ----------


def test_plan_within_node_limits_but_over_budget_is_refused() -> None:
    spec = build_demo_spec()  # 4 nodes / 3 LLM calls: fine for the node limits
    policy = ExecutionPolicy(
        max_nodes=12, max_llm_calls=8, max_cost_usd=0.02, cost_model=_cost_model()
    )

    report = check_plan_policy(spec, policy)

    assert {d.gate: d.allowed for d in report.decisions}["structure"] is True
    assert not report.allowed
    (v,) = report.violations
    assert v.code == "cost_bound_exceeded"
    assert report.cost_bound is not None
    assert report.cost_bound.usd == pytest.approx(3 * PER_CALL)


def test_plan_under_budget_is_allowed_and_bound_is_reported() -> None:
    policy = ExecutionPolicy(max_cost_usd=1.0, cost_model=_cost_model())

    report = check_plan_policy(build_demo_spec(), policy)

    assert report.allowed
    cost = next(d for d in report.decisions if d.gate == "cost")
    assert cost.details["mode"] == "enforce"
    assert cost.details["usd"] == pytest.approx(3 * PER_CALL)


def test_token_ceiling_is_enforced() -> None:
    policy = ExecutionPolicy(max_tokens=20_000, cost_model=_cost_model())

    report = check_plan_policy(build_demo_spec(), policy)

    assert {v.code for v in report.violations} == {"token_bound_exceeded"}


def test_dollar_ceiling_without_cost_model_fails_closed() -> None:
    report = check_plan_policy(build_demo_spec(), ExecutionPolicy(max_cost_usd=100.0))

    assert {v.code for v in report.violations} == {"no_cost_model"}


def test_dollar_ceiling_with_unpriced_model_fails_closed() -> None:
    policy = ExecutionPolicy(
        max_cost_usd=100.0, cost_model=CostModel(prices=PRICES, default_model="mystery")
    )

    report = check_plan_policy(build_demo_spec(), policy)

    assert {v.code for v in report.violations} == {"unpriced_model"}


def test_cost_model_without_ceiling_only_estimates() -> None:
    report = check_plan_policy(
        build_demo_spec(), ExecutionPolicy(cost_model=_cost_model())
    )

    assert report.allowed
    assert report.cost_bound is not None


# ---------- refused before any model call ----------


def test_over_budget_plan_never_reaches_a_runner() -> None:
    calls: list[str] = []

    def runner(state, params):
        calls.append("called")
        return {}

    sup = Supervisor(
        planner=StaticPlanner(build_demo_spec()),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: runner}),
        recorder=NullRecorder(),
        policy=ExecutionPolicy(max_cost_usd=0.001, cost_model=_cost_model()),
    )

    result = sup.run("go", run_id="over-budget")

    assert result.status == "policy_denied"
    assert calls == []
    assert result.receipt is not None
    assert result.receipt.cost_bound is not None
    assert result.receipt.cost_bound.usd == pytest.approx(3 * PER_CALL)


def test_engine_uses_config_pricing_when_policy_has_no_cost_model() -> None:
    engine = DynamicSubgraphs(
        EngineConfig(
            planner="mock",
            policy=ExecutionPolicy(max_cost_usd=0.001),
            pricing={"mock-model": {"input_per_1m": 1.0, "output_per_1m": 4.0}},
        )
    )

    result = engine.run("x", fixed_spec=build_demo_spec())

    # EngineConfig.pricing became the cost model (no `no_cost_model`), but the
    # mock path has no role models to route nodes to, so they stay unpriced
    # and the dollar ceiling fails closed.
    codes = {i["code"] for i in result.errors[0]["issues"]}
    assert result.status == "policy_denied"
    assert codes == {"unpriced_model"}


def test_engine_refuses_over_budget_plan_and_returns_receipt() -> None:
    cm = CostModel(
        prices={"mock-model": ModelPrice(1.0, 4.0)}, default_model="mock-model"
    )
    engine = DynamicSubgraphs(
        EngineConfig(
            planner="mock", policy=ExecutionPolicy(max_cost_usd=0.01, cost_model=cm)
        )
    )

    result = engine.run("x", fixed_spec=build_demo_spec())

    assert result.status == "policy_denied"
    assert result.values == {}
    assert result.errors[0]["issues"][0]["code"] == "cost_bound_exceeded"
    assert result.to_dict()["receipt"]["status"] == "policy_denied"


def test_llm_runs_route_unset_cost_models_to_role_models() -> None:
    from app.assembly import RunConfig, _with_cost_defaults, run_model_names
    from app.runtime import ModelRef

    config = RunConfig(
        planner="llm",
        model="worker-m",
        strict_runners=True,
        planner_model=ModelRef("openai", "planner-m"),
        reducer_model=ModelRef("openai", "reducer-m"),
    )
    policy = ExecutionPolicy(cost_model=CostModel(prices=PRICES))

    cm = _with_cost_defaults(policy, config).cost_model

    assert cm is not None
    assert cm.default_model == "worker-m"
    assert cm.kind_models[NodeKind.REDUCE] == "reducer-m"
    assert "planner-m" in cm.extra_models
    assert run_model_names(config) == ("planner-m", "reducer-m", "worker-m")


def test_explicit_cost_model_routing_wins_over_role_defaults() -> None:
    from app.assembly import RunConfig, _with_cost_defaults

    config = RunConfig(planner="llm", model="worker-m", strict_runners=True)
    policy = ExecutionPolicy(cost_model=CostModel(prices=PRICES, default_model="cheap"))

    cm = _with_cost_defaults(policy, config).cost_model

    assert cm is not None and cm.default_model == "cheap"

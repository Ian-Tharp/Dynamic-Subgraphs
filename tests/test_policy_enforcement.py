"""ExecutionPolicy enforcement: the host caps the planner's budget at validation.

PR-1 added the policy types + resolver; this verifies they're actually *wired*:
the validator enforces the host ceiling (not the planner's self-declared budget)
and stamps the granted budget, and the engine surfaces it on the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import NodeKind
from app.models.graph_spec import GraphBudget
from app.policy import ExecutionPolicy
from app.registry import RegistryValidationError, validate_graph_spec
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model
from dynamic_subgraphs import ExecutionPolicy as PublicPolicy


def _linear(make_node, make_edge, n: int):
    nodes = [make_node(f"n{i}", outputs=[f"o{i}"]) for i in range(n)]
    edges = [make_edge("START", "n0")]
    edges += [make_edge(f"n{i}", f"n{i + 1}") for i in range(n - 1)]
    edges += [make_edge(f"n{n - 1}", "END")]
    return nodes, edges


# ---------- validator enforces the HOST ceiling, not the plan's self-grant ----------


def test_host_policy_caps_nodes_below_a_generous_request(
    spec_factory, make_node, make_edge
) -> None:
    # The plan self-grants a huge budget; the host policy must still cap it.
    nodes, edges = _linear(make_node, make_edge, 3)
    spec = spec_factory(
        nodes=nodes, edges=edges, budget=GraphBudget(max_nodes=99, max_llm_calls=99)
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec, policy=ExecutionPolicy(max_nodes=2))

    assert "budget_exceeded" in {i.code for i in exc.value.issues}


def test_default_policy_blocks_a_self_granted_budget(
    spec_factory, make_node, make_edge
) -> None:
    # 13 nodes with a planner-declared budget of 1000 — with NO policy passed,
    # the default ExecutionPolicy() host ceiling (12) still rejects it. This is
    # the core self-grant fix: the planner can't grant itself a larger budget.
    nodes, edges = _linear(make_node, make_edge, 13)
    spec = spec_factory(
        nodes=nodes, edges=edges, budget=GraphBudget(max_nodes=1000, max_llm_calls=1000)
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    assert "budget_exceeded" in {i.code for i in exc.value.issues}


def test_host_policy_caps_llm_calls(spec_factory, make_node, make_edge) -> None:
    nodes, edges = _linear(make_node, make_edge, 3)  # 3 llm_call nodes
    spec = spec_factory(
        nodes=nodes, edges=edges, budget=GraphBudget(max_nodes=99, max_llm_calls=99)
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec, policy=ExecutionPolicy(max_llm_calls=1))

    assert "budget_exceeded" in {i.code for i in exc.value.issues}


# ---------- the granted budget is stamped onto the validated spec ----------


def test_validator_stamps_min_of_host_and_request(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[make_node("step", outputs=["draft"])],
        edges=[make_edge("START", "step"), make_edge("step", "END")],
        budget=GraphBudget(
            max_nodes=12, max_llm_calls=8, max_depth=3, max_wall_seconds=90
        ),
    )

    validated = validate_graph_spec(
        spec,
        policy=ExecutionPolicy(
            max_nodes=5, max_llm_calls=4, max_depth=2, max_wall_seconds=30
        ),
    )

    assert validated.budget.max_nodes == 5
    assert validated.budget.max_llm_calls == 4
    assert validated.budget.max_depth == 2
    assert validated.budget.max_wall_seconds == 30


def test_planner_may_self_restrain_below_host(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[make_node("step", outputs=["draft"])],
        edges=[make_edge("START", "step"), make_edge("step", "END")],
        budget=GraphBudget(max_nodes=3, max_llm_calls=2),
    )

    validated = validate_graph_spec(
        spec, policy=ExecutionPolicy(max_nodes=12, max_llm_calls=8)
    )

    # min(host, request) — the smaller request wins, the host doesn't raise it.
    assert validated.budget.max_nodes == 3
    assert validated.budget.max_llm_calls == 2


# ---------- end-to-end through the SDK facade ----------


def test_execution_policy_is_publicly_exported() -> None:
    import dynamic_subgraphs as ds

    assert "ExecutionPolicy" in ds.__all__
    assert ds.ExecutionPolicy is ExecutionPolicy is PublicPolicy


def test_engine_surfaces_effective_budget(tmp_path: Path) -> None:
    # The mock demo spec requests max_nodes=8 with ~4 nodes. A host cap of 6 is
    # below the request (so it bites) but above the node count (so the run still
    # succeeds) — proving the host caps the granted budget end-to-end.
    engine = DynamicSubgraphs(
        EngineConfig(
            model=Model("openai", "x"),
            planner="mock",
            policy=PublicPolicy(max_nodes=6, max_llm_calls=6),
            runs_dir=tmp_path,
        )
    )
    result = engine.run("compare two things", run_id="pol-eff-001")

    assert result.ok
    assert result.effective_budget is not None
    # Granted = min(host 6, demo request 8) = 6 — the host capped it.
    assert result.effective_budget["max_nodes"] == 6
    assert result.effective_budget["max_llm_calls"] == 6
    # JSON-safe, including the new field.
    json.dumps(result.to_dict())


def test_engine_default_policy_runs_unchanged(tmp_path: Path) -> None:
    # No policy set: a normal mock run still succeeds (defaults are permissive
    # enough for typical plans), and effective_budget is populated.
    engine = DynamicSubgraphs(
        EngineConfig(model=Model("openai", "x"), planner="mock", runs_dir=tmp_path)
    )
    result = engine.run("compare two things", run_id="pol-default-001")

    assert result.ok
    assert result.effective_budget is not None
    assert result.effective_budget["max_nodes"] >= 1


# ---------- allow-set enforcement (host ∩ registry) ----------


def _tool_spec(spec_factory, make_node, make_edge):
    return spec_factory(
        nodes=[
            make_node(
                "search",
                NodeKind.TOOL_CALL,
                outputs=["r"],
                params={"tool_name": "web_search", "args": {"q": "x"}},
            )
        ],
        edges=[make_edge("START", "search"), make_edge("search", "END")],
    )


def test_policy_bans_a_tool_outside_the_allowlist(
    spec_factory, make_node, make_edge
) -> None:
    spec = _tool_spec(spec_factory, make_node, make_edge)

    # Host forbids all tools -> a tool_call naming web_search is rejected.
    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec, policy=ExecutionPolicy(allowed_tools=frozenset()))
    assert "tool_not_allowlisted" in {i.code for i in exc.value.issues}


def test_policy_allows_a_tool_inside_the_allowlist(
    spec_factory, make_node, make_edge
) -> None:
    spec = _tool_spec(spec_factory, make_node, make_edge)

    validated = validate_graph_spec(
        spec, policy=ExecutionPolicy(allowed_tools=frozenset({"web_search"}))
    )
    assert validated.nodes[0].params["tool_name"] == "web_search"


def test_policy_rejects_a_disallowed_node_kind(
    spec_factory, make_node, make_edge
) -> None:
    spec = _tool_spec(spec_factory, make_node, make_edge)

    # Only llm_call permitted -> the tool_call node is rejected.
    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(
            spec,
            policy=ExecutionPolicy(allowed_node_kinds=frozenset({NodeKind.LLM_CALL})),
        )
    assert "node_kind_not_allowed" in {i.code for i in exc.value.issues}

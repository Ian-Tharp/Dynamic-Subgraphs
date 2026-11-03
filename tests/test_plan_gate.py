"""Plan-level policy gate (app.governance.plan_gate): pure checks on a GraphSpec
before it compiles, and the supervisor wiring that refuses a denied plan."""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from app.governance import (
    PlanPolicyViolation,
    check_plan_policy,
    enforce_plan_policy,
)
from app.models import GraphSpec, NodeKind
from app.models.graph_spec import GraphBudget
from app.policy import ExecutionPolicy
from app.recording import NullRecorder
from app.runtime import LangGraphExecutor
from app.supervisor import StaticPlanner, Supervisor
from app.supervisor.planner import build_demo_spec

STATE_CHANGING = {"state_changing": frozenset({"create_follow_up_task"})}


def _codes(report) -> set[str]:
    return {v.code for v in report.violations}


def _tool(make_node, node_id: str, tool: str, **kw: Any):
    return make_node(node_id, NodeKind.TOOL_CALL, params={"tool_name": tool}, **kw)


@pytest.fixture
def chain(make_edge):
    """Build a linear START -> nodes... -> END spec."""

    def _chain(*nodes, budget: GraphBudget | None = None) -> GraphSpec:
        ids = ["START", *[n.id for n in nodes], "END"]
        return GraphSpec(
            graph_id="g",
            goal="test",
            budget=budget or GraphBudget(),
            nodes=list(nodes),
            edges=[make_edge(a, b) for a, b in itertools.pairwise(ids)],
        )

    return _chain


# ---------- defaults never deny ----------


def test_default_policy_allows_the_demo_plan() -> None:
    report = check_plan_policy(build_demo_spec(), ExecutionPolicy())

    assert report.allowed
    assert [d.gate for d in report.decisions] == [
        "structure",
        "capabilities",
        "data_flow",
        "catalog",
        "cost",
    ]
    assert report.plan_hash.startswith("sha256:")
    assert report.policy_hash.startswith("sha256:")


def test_check_is_deterministic() -> None:
    policy = ExecutionPolicy(max_nodes=3)

    a = check_plan_policy(build_demo_spec(), policy)
    b = check_plan_policy(build_demo_spec(), policy)

    assert a.as_dict() == b.as_dict()


# ---------- structure ----------


def test_node_count_over_policy_is_denied() -> None:
    report = check_plan_policy(build_demo_spec(), ExecutionPolicy(max_nodes=3))

    assert not report.allowed
    assert "max_nodes_exceeded" in _codes(report)


def test_longest_chain_over_limit_is_denied(make_node, chain) -> None:
    spec = chain(make_node("a"), make_node("b"), make_node("c"))

    report = check_plan_policy(spec, ExecutionPolicy(max_plan_length=2))

    assert "max_plan_length_exceeded" in _codes(report)
    structure = next(d for d in report.decisions if d.gate == "structure")
    assert structure.details["longest_chain"] == 3


def test_parallel_nodes_do_not_count_toward_chain_length() -> None:
    # demo: two parallel gathers -> join -> synthesize = chain of 3
    report = check_plan_policy(build_demo_spec(), ExecutionPolicy(max_plan_length=3))

    assert report.allowed


def test_spawn_subgraph_needs_depth_two(make_node, chain) -> None:
    spec = chain(
        make_node(
            "child",
            NodeKind.SPAWN_SUBGRAPH,
            params={"sub_goal": "dig", "name": "dig"},
        )
    )

    report = check_plan_policy(spec, ExecutionPolicy(max_depth=1))

    assert "nesting_depth_exceeded" in _codes(report)


# ---------- capabilities ----------


def test_disallowed_kind_tool_subagent_and_model_are_each_reported(
    make_node, chain
) -> None:
    spec = chain(
        _tool(make_node, "search", "web_search", outputs=["hits"]),
        make_node(
            "ask",
            NodeKind.SPAWN_SUBAGENT,
            params={"agent_name": "critic", "task": "review"},
        ),
        make_node("write", params={"instruction": "x", "model": "big-model"}),
    )
    policy = ExecutionPolicy(
        allowed_tools=frozenset({"policy_lookup"}),
        allowed_subagents=frozenset({"document_specialist"}),
        allowed_node_kinds=frozenset({NodeKind.LLM_CALL, NodeKind.TOOL_CALL}),
        allowed_models=frozenset({"small-model"}),
    )

    report = check_plan_policy(spec, policy)

    assert _codes(report) == {
        "tool_not_allowed",
        "subagent_not_allowed",
        "node_kind_not_allowed",
        "model_not_allowed",
    }
    by_code = {v.code: v for v in report.violations}
    assert by_code["tool_not_allowed"].node_id == "search"
    assert by_code["model_not_allowed"].field == "params.model"


def test_parallel_map_tool_child_is_checked(make_node, chain) -> None:
    spec = chain(
        make_node("seed", outputs=["items"]),
        make_node(
            "fan",
            NodeKind.PARALLEL_MAP,
            inputs=["items"],
            params={
                "over": "items",
                "child_kind": "tool_call",
                "child_params": {"tool_name": "web_search"},
            },
        ),
    )

    report = check_plan_policy(
        spec, ExecutionPolicy(allowed_tools=frozenset({"policy_lookup"}))
    )

    assert "tool_not_allowed" in _codes(report)


def test_run_models_outside_allow_set_are_denied_and_not_repairable() -> None:
    policy = ExecutionPolicy(allowed_models=frozenset({"gpt-small"}))

    report = check_plan_policy(
        build_demo_spec(), policy, run_models=["gpt-small", "gpt-huge"]
    )

    assert _codes(report) == {"run_model_not_allowed"}
    assert not report.repairable


def test_empty_model_allow_set_bans_everything() -> None:
    report = check_plan_policy(
        build_demo_spec(),
        ExecutionPolicy(allowed_models=frozenset()),
        run_models=["any"],
    )

    assert not report.allowed


# ---------- data flow: forbidden edges and required gates ----------


def test_tool_output_into_state_changing_tool_without_gate_is_denied(
    make_node, chain
) -> None:
    spec = chain(
        _tool(make_node, "search", "web_search", outputs=["hits"]),
        make_node("decide", inputs=["hits"], outputs=["plan"]),
        _tool(make_node, "act", "create_follow_up_task", inputs=["plan"]),
    )
    policy = ExecutionPolicy(
        tool_classes=STATE_CHANGING,
        state_changing_classes=frozenset({"state_changing"}),
    )

    report = check_plan_policy(spec, policy)

    assert not report.allowed
    (v,) = report.violations
    assert v.code == "ungated_tool_to_state_change"
    assert v.gate == "data_flow"
    assert v.node_id == "act"
    assert "search" in v.message


def test_gate_node_between_tool_and_state_change_allows(make_node, chain) -> None:
    spec = chain(
        _tool(make_node, "search", "web_search", outputs=["hits"]),
        make_node(
            "approve",
            NodeKind.WAIT_FOR_EVENT,
            inputs=["hits"],
            outputs=["approval"],
            params={"event_type": "human_approval"},
        ),
        _tool(make_node, "act", "create_follow_up_task", inputs=["approval"]),
    )
    policy = ExecutionPolicy(
        tool_classes=STATE_CHANGING,
        state_changing_classes=frozenset({"state_changing"}),
    )

    assert check_plan_policy(spec, policy).allowed


def test_a_side_path_around_the_gate_is_still_denied(make_node, make_edge) -> None:
    # search -> approve -> act, plus search -> act directly: the bypass counts.
    nodes = [
        _tool(make_node, "search", "web_search", outputs=["hits"]),
        make_node(
            "approve",
            NodeKind.WAIT_FOR_EVENT,
            inputs=["hits"],
            outputs=["approval"],
            params={"event_type": "human_approval"},
        ),
        _tool(make_node, "act", "create_follow_up_task", inputs=["approval"]),
    ]
    edges = [
        make_edge("START", "search"),
        make_edge("search", "approve"),
        make_edge("approve", "act"),
        make_edge("search", "act"),
        make_edge("act", "END"),
    ]
    spec = GraphSpec(graph_id="g", goal="t", nodes=nodes, edges=edges)
    policy = ExecutionPolicy(
        tool_classes=STATE_CHANGING,
        state_changing_classes=frozenset({"state_changing"}),
    )

    assert "ungated_tool_to_state_change" in _codes(check_plan_policy(spec, policy))


def test_state_changing_node_kind_is_protected(make_node, chain) -> None:
    spec = chain(
        _tool(make_node, "search", "web_search", outputs=["hits"]),
        make_node(
            "publish",
            NodeKind.EMIT_ARTIFACT,
            inputs=["hits"],
            params={"name": "r", "content_key": "hits"},
        ),
    )
    policy = ExecutionPolicy(
        state_changing_node_kinds=frozenset({NodeKind.EMIT_ARTIFACT})
    )

    assert "ungated_tool_to_state_change" in _codes(check_plan_policy(spec, policy))


def test_gate_required_class_needs_gate_on_every_path_from_start(
    make_node, chain
) -> None:
    ungated = chain(_tool(make_node, "pay", "create_follow_up_task"))
    gated = chain(
        make_node(
            "approve",
            NodeKind.WAIT_FOR_EVENT,
            outputs=["ok"],
            params={"event_type": "approval"},
        ),
        _tool(make_node, "pay", "create_follow_up_task"),
    )
    policy = ExecutionPolicy(
        tool_classes={"payments": frozenset({"create_follow_up_task"})},
        gate_required_classes=frozenset({"payments"}),
    )

    assert _codes(check_plan_policy(ungated, policy)) == {"missing_required_gate"}
    assert check_plan_policy(gated, policy).allowed


def test_unknown_tool_class_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="tool classes"):
        ExecutionPolicy(state_changing_classes=frozenset({"nope"}))


# ---------- fail closed ----------


def test_a_crashing_gate_denies(monkeypatch) -> None:
    import app.governance.plan_gate as pg

    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(pg, "_gate_data_flow", boom)

    report = check_plan_policy(build_demo_spec(), ExecutionPolicy())

    assert not report.allowed
    assert _codes(report) == {"gate_error"}
    assert not report.repairable


def test_enforce_raises_with_the_report() -> None:
    with pytest.raises(PlanPolicyViolation) as exc:
        enforce_plan_policy(build_demo_spec(), ExecutionPolicy(max_nodes=2))

    assert "max_nodes_exceeded" in {v.code for v in exc.value.report.violations}


def test_policy_hash_changes_with_any_rule() -> None:
    base = check_plan_policy(build_demo_spec(), ExecutionPolicy()).policy_hash
    tighter = check_plan_policy(
        build_demo_spec(), ExecutionPolicy(max_plan_length=9)
    ).policy_hash

    assert base != tighter


def test_policy_with_tool_classes_is_still_hashable() -> None:
    policy = ExecutionPolicy(tool_classes=STATE_CHANGING)

    assert hash(policy) == hash(ExecutionPolicy(tool_classes=STATE_CHANGING))


# ---------- supervisor wiring ----------


class _CountingRunner:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state, params) -> dict[str, Any]:
        from app.runtime.runners import run_llm_call

        self.calls += 1
        return run_llm_call(state, params)


def _supervisor(spec: GraphSpec, policy: ExecutionPolicy, runner, **kw) -> Supervisor:
    return Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: runner}),
        recorder=NullRecorder(),
        policy=policy,
        **kw,
    )


def test_supervisor_refuses_a_denied_plan_before_any_node_runs() -> None:
    runner = _CountingRunner()
    sup = _supervisor(build_demo_spec(), ExecutionPolicy(max_plan_length=1), runner)

    result = sup.run("go", run_id="denied-1")

    assert result.status == "policy_denied"
    assert runner.calls == 0
    assert result.result is None
    (err,) = result.errors
    assert err["stage"] == "policy_gate"
    assert err["issues"][0]["code"] == "max_plan_length_exceeded"
    assert result.receipt is not None
    assert result.receipt.status == "policy_denied"
    assert "deny" in {d.decision for d in result.receipt.gate_decisions}


def test_supervisor_repairs_a_repairable_gate_denial() -> None:
    small = build_demo_spec().model_copy(deep=True)
    big = small.model_copy(deep=True)
    plans = iter([big, small])
    runner = _CountingRunner()
    sup = Supervisor(
        planner=lambda prompt: next(plans),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: runner}),
        recorder=NullRecorder(),
        policy=ExecutionPolicy(max_plan_length=3),
        max_plan_attempts=2,
    )
    # First plan: add an extra node to the end of the chain -> length 4.
    from app.models import NodeSpec
    from app.models.graph_spec import EdgeSpec

    big.nodes.append(
        NodeSpec(
            id="extra",
            kind=NodeKind.LLM_CALL,
            inputs=["final_answer"],
            params={"instruction": "polish"},
        )
    )
    big.edges = [e for e in big.edges if e.to != "END"] + [
        EdgeSpec.model_validate({"from": "synthesize", "to": "extra"}),
        EdgeSpec.model_validate({"from": "extra", "to": "END"}),
    ]

    result = sup.run("go", run_id="repair-1")

    assert result.status == "ok"
    assert result.plan_attempts == 2


def test_supervisor_allows_and_logs_gate_decisions_in_the_trace() -> None:
    runner = _CountingRunner()
    sup = _supervisor(build_demo_spec(), ExecutionPolicy(), runner)

    result = sup.run("go", run_id="allowed-1")

    assert result.status == "ok"
    assert runner.calls == 3
    head = result.result.trace.events[:5]
    assert [e.kind.value for e in head] == ["policy_decision"] * 5
    assert {e.data["decision"] for e in head} == {"allow"}


def test_supervisor_checks_run_models_against_allowed_models() -> None:
    sup = _supervisor(
        build_demo_spec(),
        ExecutionPolicy(allowed_models=frozenset({"ok-model"})),
        _CountingRunner(),
        run_models=["ok-model", "forbidden-model"],
    )

    result = sup.run("go", run_id="models-1")

    assert result.status == "policy_denied"
    assert result.plan_attempts == 1  # not repairable: no re-plan

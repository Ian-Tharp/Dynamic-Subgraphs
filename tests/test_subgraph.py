"""spawn_subgraph: the re-entrant primitive that lets a node run a child graph.

Two layers under test:
- `build_spawn_subgraph_runner` — the NodeRunner: depth enforcement, child
  seeding, output hand-back. Driven with a stub launcher (deterministic).
- `make_child_launcher` — plan -> validate -> ban wait_for_event -> compile ->
  execute(initial_metadata) over a real LangGraphExecutor with mock runners.
"""

from __future__ import annotations

import pytest

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec, GraphBudget
from app.registry.validator import MAX_DEPTH_CEILING
from app.runtime.subgraph import (
    ChildResult,
    SubgraphChildFailed,
    SubgraphContainsWaitForEvent,
    SubgraphDepthExceeded,
    build_spawn_subgraph_runner,
    make_child_launcher,
)


def _state(*, values=None, graph_depth=0, run_id="root"):
    return {
        "values": dict(values or {}),
        "metadata": {"run_id": run_id, "graph_depth": graph_depth},
        "counters": {},
        "events": [],
        "errors": [],
    }


# ---------- the runner (stub launcher) ----------


def test_runner_runs_child_and_hands_back_its_values() -> None:
    captured: dict = {}

    def launcher(sub_goal, *, run_id, graph_depth, parent_run_id, inputs,
                 max_llm_calls=None):
        captured.update(
            sub_goal=sub_goal,
            run_id=run_id,
            graph_depth=graph_depth,
            parent_run_id=parent_run_id,
            inputs=inputs,
            max_llm_calls=max_llm_calls,
        )
        return ChildResult(
            values={"answer": 42}, counters={"llm_calls_consumed": 2}, status="ok"
        )

    runner = build_spawn_subgraph_runner(launcher)

    out = runner(
        _state(values={"x": 1, "y": 2}, run_id="parent", graph_depth=0),
        {"sub_goal": "do a thing", "name": "child1", "inputs_from": ["x"]},
    )

    assert out["result"] == {"answer": 42}
    assert out["__spend__"] == {"llm_calls_consumed": 2}  # child spend rolled up
    assert captured["sub_goal"] == "do a thing"
    assert captured["run_id"] == "parent__sg_child1"  # deterministic child id
    assert captured["graph_depth"] == 1  # one deeper than the parent
    assert captured["parent_run_id"] == "parent"
    assert captured["inputs"] == {"x": 1}  # only the inputs_from keys, seeded


def test_runner_enforces_depth_ceiling_before_launching() -> None:
    def launcher(*args, **kwargs):
        raise AssertionError("launcher must not run once the depth ceiling is hit")

    runner = build_spawn_subgraph_runner(launcher)

    with pytest.raises(SubgraphDepthExceeded):
        runner(
            _state(graph_depth=MAX_DEPTH_CEILING),
            {"sub_goal": "g", "name": "deep"},
        )


def test_runner_raises_when_child_fails() -> None:
    def launcher(*args, **kwargs):
        return ChildResult(values={}, counters={}, status="failed", response="boom")

    runner = build_spawn_subgraph_runner(launcher)

    with pytest.raises(SubgraphChildFailed):
        runner(_state(), {"sub_goal": "g", "name": "c"})


# ---------- the launcher (real executor, mock runners) ----------


def _child_spec(graph_id: str, *, nodes, edges, budget=None) -> GraphSpec:
    kwargs = {"graph_id": graph_id, "goal": graph_id, "nodes": nodes, "edges": edges}
    if budget is not None:
        kwargs["budget"] = budget
    return GraphSpec(**kwargs)


def test_launcher_plans_and_runs_a_child(tmp_path) -> None:
    from app.runtime.executor import LangGraphExecutor

    child = _child_spec(
        "child",
        nodes=[NodeSpec(id="step", kind=NodeKind.LLM_CALL, outputs=["draft"],
                        params={"instruction": "hi"})],
        edges=[EdgeSpec.model_validate({"from": "START", "to": "step"}),
               EdgeSpec.model_validate({"from": "step", "to": "END"})],
    )

    def planner(sub_goal):
        del sub_goal
        return child

    def echo(state, params):
        del state
        return {"result": params["instruction"]}

    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: echo})
    launcher = make_child_launcher(planner=planner, executor=executor)

    result = launcher(
        "any goal", run_id="p__sg_c", graph_depth=1, parent_run_id="p", inputs={}
    )

    assert result.status == "ok"
    assert result.values["draft"] == "hi"
    # the ledger from slice 2 is populated on the child run
    assert result.counters["nodes_executed"] == 1


def test_launcher_bans_wait_for_event_in_children() -> None:
    from app.runtime.executor import LangGraphExecutor

    child = _child_spec(
        "waiter",
        nodes=[NodeSpec(id="w", kind=NodeKind.WAIT_FOR_EVENT,
                        params={"event_type": "human"})],
        edges=[EdgeSpec.model_validate({"from": "START", "to": "w"}),
               EdgeSpec.model_validate({"from": "w", "to": "END"})],
    )

    def planner(sub_goal):
        del sub_goal
        return child

    launcher = make_child_launcher(planner=planner, executor=LangGraphExecutor())

    with pytest.raises(SubgraphContainsWaitForEvent):
        launcher("g", run_id="p__sg_w", graph_depth=1, parent_run_id="p", inputs={})


def test_launcher_seeds_graph_depth_into_child(tmp_path) -> None:
    from app.runtime.executor import LangGraphExecutor

    child = _child_spec(
        "child",
        nodes=[NodeSpec(id="step", kind=NodeKind.LLM_CALL, outputs=["depth_seen"],
                        params={"instruction": "x"})],
        edges=[EdgeSpec.model_validate({"from": "START", "to": "step"}),
               EdgeSpec.model_validate({"from": "step", "to": "END"})],
    )

    seen: dict = {}

    def planner(sub_goal):
        del sub_goal
        return child

    def depth_reader(state, params):
        del params
        seen["graph_depth"] = state["metadata"].get("graph_depth")
        seen["parent_run_id"] = state["metadata"].get("parent_run_id")
        return {"result": "ok"}

    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: depth_reader})
    launcher = make_child_launcher(planner=planner, executor=executor)

    launcher("g", run_id="p__sg_c", graph_depth=2, parent_run_id="p", inputs={})

    assert seen["graph_depth"] == 2
    assert seen["parent_run_id"] == "p"


# ---------- wrapper safeguard: interrupts must bubble, not be swallowed ----------


def test_node_wrapper_reraises_graph_interrupt_instead_of_swallowing() -> None:
    # LangGraph control-flow signals (GraphInterrupt/GraphBubbleUp) must
    # propagate, not get normalized into a NODE_ERROR. Otherwise a child that
    # somehow pauses would be silently mislabeled as a failure.
    from langgraph.errors import GraphInterrupt

    from app.runtime.wrappers import make_node_wrapper

    node = NodeSpec(id="n", kind=NodeKind.LLM_CALL, outputs=["o"],
                    params={"instruction": "x"})

    def interrupting_runner(state, params):
        del state, params
        raise GraphInterrupt()

    wrapper = make_node_wrapper(node, interrupting_runner)

    with pytest.raises(GraphInterrupt):
        wrapper(_state())


# ---------- full wired path (deterministic) ----------


def test_spawn_subgraph_end_to_end_merges_child_output_into_parent() -> None:
    # A parent graph whose only node is a spawn_subgraph: it plans + runs a
    # child, and the child's produced values land under the parent node's
    # declared output. Wired exactly as assembly does (late-bound launcher).
    from app.registry import validate_graph_spec
    from app.runtime.executor import LangGraphExecutor

    child = _child_spec(
        "child",
        nodes=[NodeSpec(id="answer", kind=NodeKind.LLM_CALL, outputs=["child_out"],
                        params={"instruction": "hello"})],
        edges=[EdgeSpec.model_validate({"from": "START", "to": "answer"}),
               EdgeSpec.model_validate({"from": "answer", "to": "END"})],
    )

    def child_planner(sub_goal):
        del sub_goal
        return child

    def echo(state, params):
        del state
        return {"result": params["instruction"]}

    runners = {NodeKind.LLM_CALL: echo}
    executor = LangGraphExecutor(runners=runners)
    runners[NodeKind.SPAWN_SUBGRAPH] = build_spawn_subgraph_runner(
        make_child_launcher(planner=child_planner, executor=executor)
    )

    parent = _child_spec(
        "parent",
        nodes=[NodeSpec(id="spawn", kind=NodeKind.SPAWN_SUBGRAPH,
                        outputs=["child_report"],
                        params={"sub_goal": "do the child thing", "name": "c"})],
        edges=[EdgeSpec.model_validate({"from": "START", "to": "spawn"}),
               EdgeSpec.model_validate({"from": "spawn", "to": "END"})],
    )
    validated = validate_graph_spec(parent)
    result = executor.execute(executor.compile(validated), run_id="parent")

    assert result.ok is True
    assert result.state["values"]["child_report"] == {"child_out": "hello"}
    # the parent counted the spawn node (conservative floor) via the ledger
    assert result.state["counters"]["llm_calls_consumed"] >= 1


def test_spawn_subgraph_rolls_up_child_spend_into_parent_ledger() -> None:
    # The parent ledger must reflect the child's ACTUAL spend, not a flat floor:
    # a child running two llm_call nodes contributes llm_calls_consumed == 2 and
    # nodes_executed == 2 to the parent, plus 1 node for the spawn itself.
    from app.registry import validate_graph_spec
    from app.runtime.executor import LangGraphExecutor

    child = _child_spec(
        "child",
        nodes=[
            NodeSpec(id="a", kind=NodeKind.LLM_CALL, outputs=["a"],
                     params={"instruction": "A"}),
            NodeSpec(id="b", kind=NodeKind.LLM_CALL, inputs=["a"], outputs=["b"],
                     params={"instruction": "B"}),
        ],
        edges=[EdgeSpec.model_validate({"from": "START", "to": "a"}),
               EdgeSpec.model_validate({"from": "a", "to": "b"}),
               EdgeSpec.model_validate({"from": "b", "to": "END"})],
    )

    def child_planner(sub_goal):
        del sub_goal
        return child

    def echo(state, params):
        del state
        return {"result": params["instruction"]}

    runners = {NodeKind.LLM_CALL: echo}
    executor = LangGraphExecutor(runners=runners)
    runners[NodeKind.SPAWN_SUBGRAPH] = build_spawn_subgraph_runner(
        make_child_launcher(planner=child_planner, executor=executor)
    )

    parent = _child_spec(
        "parent",
        nodes=[NodeSpec(id="spawn", kind=NodeKind.SPAWN_SUBGRAPH, outputs=["rep"],
                        params={"sub_goal": "g", "name": "c"})],
        edges=[EdgeSpec.model_validate({"from": "START", "to": "spawn"}),
               EdgeSpec.model_validate({"from": "spawn", "to": "END"})],
    )
    result = executor.execute(executor.compile(validate_graph_spec(parent)), run_id="p")

    assert result.ok is True
    assert result.state["counters"]["llm_calls_consumed"] == 2  # child's actual
    assert result.state["counters"]["nodes_executed"] == 3  # 1 spawn + 2 child


def test_spawn_subgraph_clamps_child_budget_to_parent_remaining() -> None:
    # The parent budget allows only 1 LLM call. The spawn's child wants 2, so
    # the child's budget is clamped to the parent's remaining (1) and the
    # oversized child fails closed — a nest can't overspend the root budget.
    from app.registry import validate_graph_spec
    from app.runtime.executor import LangGraphExecutor

    child = _child_spec(
        "child",
        nodes=[
            NodeSpec(id="a", kind=NodeKind.LLM_CALL, outputs=["a"],
                     params={"instruction": "A"}),
            NodeSpec(id="b", kind=NodeKind.LLM_CALL, inputs=["a"], outputs=["b"],
                     params={"instruction": "B"}),
        ],
        edges=[EdgeSpec.model_validate({"from": "START", "to": "a"}),
               EdgeSpec.model_validate({"from": "a", "to": "b"}),
               EdgeSpec.model_validate({"from": "b", "to": "END"})],
    )

    def child_planner(sub_goal):
        del sub_goal
        return child

    def echo(state, params):
        del state
        return {"result": params["instruction"]}

    runners = {NodeKind.LLM_CALL: echo}
    executor = LangGraphExecutor(runners=runners)
    runners[NodeKind.SPAWN_SUBGRAPH] = build_spawn_subgraph_runner(
        make_child_launcher(planner=child_planner, executor=executor)
    )

    parent = _child_spec(
        "parent",
        nodes=[NodeSpec(id="spawn", kind=NodeKind.SPAWN_SUBGRAPH, outputs=["rep"],
                        params={"sub_goal": "g", "name": "c"})],
        edges=[EdgeSpec.model_validate({"from": "START", "to": "spawn"}),
               EdgeSpec.model_validate({"from": "spawn", "to": "END"})],
        budget=GraphBudget(max_llm_calls=1),
    )
    result = executor.execute(executor.compile(validate_graph_spec(parent)), run_id="p")

    assert result.ok is False  # the oversized child was clamped and failed closed
    assert result.state["errors"]


def test_planner_advertises_spawn_subgraph_with_guidance() -> None:
    # Slice 4 un-gates nesting: now that child spend rolls up and child budgets
    # are clamped to the parent's remaining, the planner may reach for
    # spawn_subgraph. The system prompt must describe it (params + when to use).
    from app.supervisor.llm_planner import LLMPlanner

    planner = LLMPlanner(object())  # __init__ doesn't touch the model

    assert NodeKind.SPAWN_SUBGRAPH in planner._executable_kinds
    assert "spawn_subgraph" in planner._system_prompt
    assert "sub_goal" in planner._system_prompt  # params are documented

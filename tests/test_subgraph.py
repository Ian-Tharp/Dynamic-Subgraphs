"""spawn_subgraph: the re-entrant primitive that lets a node run a child graph.

Two layers under test:
- `build_spawn_subgraph_runner` — the NodeRunner: depth enforcement, child
  seeding, output hand-back. Driven with a stub launcher (deterministic).
- `make_child_launcher` — plan -> validate -> ban wait_for_event -> compile ->
  execute(initial_metadata) over a real LangGraphExecutor with mock runners.
"""

from __future__ import annotations

import json

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

    def launcher(
        sub_goal,
        *,
        run_id,
        graph_depth,
        parent_run_id,
        inputs,
        max_llm_calls=None,
        replay_of=None,
        **_extra,
    ):
        captured.update(
            sub_goal=sub_goal,
            run_id=run_id,
            graph_depth=graph_depth,
            parent_run_id=parent_run_id,
            inputs=inputs,
            max_llm_calls=max_llm_calls,
            replay_of=replay_of,
            **_extra,
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


def test_runner_pins_child_to_original_recorded_spec_during_replay() -> None:
    # On replay the runner reads `replay_of` (the ORIGINAL parent run id) from
    # metadata and tells the launcher which recorded child spec to reuse, while
    # still running the child under the NEW (replay) parent's id.
    captured: dict = {}

    def launcher(
        sub_goal,
        *,
        run_id,
        graph_depth,
        parent_run_id,
        inputs,
        max_llm_calls=None,
        replay_of=None,
        **_extra,
    ):
        captured.update(run_id=run_id, replay_of=replay_of)
        return ChildResult(values={}, counters={}, status="ok")

    runner = build_spawn_subgraph_runner(launcher)
    state = _state(run_id="Y", graph_depth=0)
    state["metadata"]["replay_of"] = "X"  # replaying original run X under Y

    runner(state, {"sub_goal": "g", "name": "c"})

    assert captured["run_id"] == "Y__sg_c"  # child runs under the replay parent
    assert captured["replay_of"] == "X__sg_c"  # but reuses the original's spec


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


def test_runner_composes_child_budget_against_parent_remaining() -> None:
    # The nest can't outspend the root: the child's node + LLM budgets are the
    # parent's REMAINING allowance, and depth is the tighter of host vs rail.
    captured: dict = {}

    def launcher(sub_goal, *, run_id, graph_depth, parent_run_id, inputs, **extra):
        captured.update(extra)
        return ChildResult(values={}, counters={}, status="ok")

    runner = build_spawn_subgraph_runner(launcher)
    state = {
        "values": {},
        "metadata": {
            "run_id": "p",
            "graph_depth": 0,
            "budget_max_nodes": 12,
            "budget_max_llm_calls": 8,
            "budget_max_depth": 2,
        },
        "counters": {"nodes_executed": 9, "llm_calls_consumed": 6},
    }

    runner(state, {"sub_goal": "g", "name": "c"})

    assert captured["max_nodes"] == 3  # 12 - 9 remaining
    assert captured["max_llm_calls"] == 2  # 8 - 6 remaining
    assert captured["max_depth"] == 2  # min(host 2, rail 3)


def test_runner_fails_closed_when_no_node_budget_remains() -> None:
    # Real launcher path: parent has spent its whole node budget -> the child is
    # refused rather than granted a fresh allowance.
    # executor/planner are never reached — the fail-closed guard fires first.
    runner = build_spawn_subgraph_runner(
        make_child_launcher(planner=lambda g: None, executor=None)
    )
    state = {
        "values": {},
        "metadata": {"run_id": "p", "graph_depth": 0, "budget_max_nodes": 4},
        "counters": {"nodes_executed": 4},  # 0 remaining
    }

    with pytest.raises(SubgraphChildFailed):
        runner(state, {"sub_goal": "g", "name": "c"})


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
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["draft"],
                params={"instruction": "hi"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
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


def test_launcher_records_child_run_with_lineage(tmp_path) -> None:
    # A nested child must be a first-class recorded run: its own spec + output
    # under runs/<child_run_id>/, with parent_run_id + graph_depth in its
    # metadata so the lineage forest assembles from a root run_id.
    from app.recording import FileRecorder
    from app.runtime.executor import LangGraphExecutor

    child = _child_spec(
        "child",
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["draft"],
                params={"instruction": "hi"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
    )

    def planner(sub_goal):
        del sub_goal
        return child

    def echo(state, params):
        del state
        return {"result": params["instruction"]}

    recorder = FileRecorder(tmp_path)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: echo})
    launcher = make_child_launcher(
        planner=planner, executor=executor, recorder=recorder
    )

    launcher(
        "any goal",
        run_id="parent__sg_c",
        graph_depth=1,
        parent_run_id="parent",
        inputs={},
    )

    child_dir = tmp_path / "parent__sg_c"
    assert (
        child_dir / "spec.json"
    ).exists()  # child spec persisted (replay foundation)
    out = json.loads((child_dir / "output.json").read_text(encoding="utf-8"))
    assert out["metadata"]["parent_run_id"] == "parent"
    assert out["metadata"]["graph_depth"] == 1


def test_launcher_pins_recorded_child_spec_on_replay(tmp_path) -> None:
    # Replay determinism: a child planned non-deterministically must, on replay,
    # reuse the ORIGINALLY RECORDED spec rather than re-planning. The planner
    # returns spec A first, spec B after — pinning must reuse A and never consult
    # the planner again.
    from app.recording import FileRecorder
    from app.runtime.executor import LangGraphExecutor

    spec_a = _child_spec(
        "child-a",
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["out"],
                params={"instruction": "alpha"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
    )
    spec_b = _child_spec(
        "child-b",
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["out"],
                params={"instruction": "beta"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
    )
    calls = {"n": 0}

    def planner(sub_goal):
        del sub_goal
        spec = spec_a if calls["n"] == 0 else spec_b
        calls["n"] += 1
        return spec

    def echo(state, params):
        del state
        return {"result": params["instruction"]}

    recorder = FileRecorder(tmp_path)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: echo})
    launcher = make_child_launcher(
        planner=planner, executor=executor, recorder=recorder
    )

    # Original run: plans spec_a and records it under X__sg_c.
    r1 = launcher("g", run_id="X__sg_c", graph_depth=1, parent_run_id="X", inputs={})
    assert r1.values["out"] == "alpha"
    assert calls["n"] == 1

    # Replay: pin to the recorded original child spec; the planner (which would
    # now hand back spec_b) must NOT be consulted.
    r2 = launcher(
        "g",
        run_id="Y__sg_c",
        graph_depth=1,
        parent_run_id="Y",
        inputs={},
        replay_of="X__sg_c",
    )
    assert r2.values["out"] == "alpha"  # reused recorded spec_a, not spec_b
    assert calls["n"] == 1  # planner not called again during replay


def test_launcher_bans_wait_for_event_in_children() -> None:
    from app.runtime.executor import LangGraphExecutor

    child = _child_spec(
        "waiter",
        nodes=[
            NodeSpec(
                id="w", kind=NodeKind.WAIT_FOR_EVENT, params={"event_type": "human"}
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "w"}),
            EdgeSpec.model_validate({"from": "w", "to": "END"}),
        ],
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
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["depth_seen"],
                params={"instruction": "x"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
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

    node = NodeSpec(
        id="n", kind=NodeKind.LLM_CALL, outputs=["o"], params={"instruction": "x"}
    )

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
        nodes=[
            NodeSpec(
                id="answer",
                kind=NodeKind.LLM_CALL,
                outputs=["child_out"],
                params={"instruction": "hello"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "answer"}),
            EdgeSpec.model_validate({"from": "answer", "to": "END"}),
        ],
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
        nodes=[
            NodeSpec(
                id="spawn",
                kind=NodeKind.SPAWN_SUBGRAPH,
                outputs=["child_report"],
                params={"sub_goal": "do the child thing", "name": "c"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "spawn"}),
            EdgeSpec.model_validate({"from": "spawn", "to": "END"}),
        ],
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
            NodeSpec(
                id="a",
                kind=NodeKind.LLM_CALL,
                outputs=["a"],
                params={"instruction": "A"},
            ),
            NodeSpec(
                id="b",
                kind=NodeKind.LLM_CALL,
                inputs=["a"],
                outputs=["b"],
                params={"instruction": "B"},
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "a"}),
            EdgeSpec.model_validate({"from": "a", "to": "b"}),
            EdgeSpec.model_validate({"from": "b", "to": "END"}),
        ],
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
        nodes=[
            NodeSpec(
                id="spawn",
                kind=NodeKind.SPAWN_SUBGRAPH,
                outputs=["rep"],
                params={"sub_goal": "g", "name": "c"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "spawn"}),
            EdgeSpec.model_validate({"from": "spawn", "to": "END"}),
        ],
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
            NodeSpec(
                id="a",
                kind=NodeKind.LLM_CALL,
                outputs=["a"],
                params={"instruction": "A"},
            ),
            NodeSpec(
                id="b",
                kind=NodeKind.LLM_CALL,
                inputs=["a"],
                outputs=["b"],
                params={"instruction": "B"},
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "a"}),
            EdgeSpec.model_validate({"from": "a", "to": "b"}),
            EdgeSpec.model_validate({"from": "b", "to": "END"}),
        ],
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
        nodes=[
            NodeSpec(
                id="spawn",
                kind=NodeKind.SPAWN_SUBGRAPH,
                outputs=["rep"],
                params={"sub_goal": "g", "name": "c"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "spawn"}),
            EdgeSpec.model_validate({"from": "spawn", "to": "END"}),
        ],
        budget=GraphBudget(max_llm_calls=1),
    )
    result = executor.execute(executor.compile(validate_graph_spec(parent)), run_id="p")

    assert result.ok is False  # the oversized child was clamped and failed closed
    assert result.state["errors"]


def test_supervisor_replay_pins_nested_child_to_recorded_spec(tmp_path) -> None:
    # Full e2e: a recorded nested run, on Supervisor.replay, reuses the
    # originally recorded child spec rather than re-planning — so the nested
    # shape reproduces deterministically (the canon-critical replay guarantee).
    from app.recording import FileRecorder
    from app.runtime.executor import LangGraphExecutor
    from app.supervisor import Supervisor

    parent = _child_spec(
        "parent",
        nodes=[
            NodeSpec(
                id="spawn",
                kind=NodeKind.SPAWN_SUBGRAPH,
                outputs=["rep"],
                params={"sub_goal": "child goal", "name": "c"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "spawn"}),
            EdgeSpec.model_validate({"from": "spawn", "to": "END"}),
        ],
    )
    child_a = _child_spec(
        "child-a",
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["out"],
                params={"instruction": "alpha"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
    )
    child_b = _child_spec(
        "child-b",
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["out"],
                params={"instruction": "beta"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
    )
    child_calls = {"n": 0}

    def planner(prompt):
        if prompt == "PARENT GOAL":
            return parent
        spec = child_a if child_calls["n"] == 0 else child_b
        child_calls["n"] += 1
        return spec

    def echo(state, params):
        del state
        return {"result": params["instruction"]}

    recorder = FileRecorder(tmp_path)
    runners = {NodeKind.LLM_CALL: echo}
    executor = LangGraphExecutor(runners=runners)
    runners[NodeKind.SPAWN_SUBGRAPH] = build_spawn_subgraph_runner(
        make_child_launcher(planner=planner, executor=executor, recorder=recorder)
    )
    supervisor = Supervisor(planner=planner, executor=executor, recorder=recorder)

    original = supervisor.run("PARENT GOAL", run_id="p1")
    assert original.status == "ok"
    assert original.result.state["values"]["rep"] == {"out": "alpha"}
    assert child_calls["n"] == 1  # child planned once

    replayed = supervisor.replay("p1", new_run_id="p1_replay")
    assert replayed.status == "ok"
    # Reused the recorded child spec (alpha), NOT a fresh plan (which is beta).
    assert replayed.result.state["values"]["rep"] == {"out": "alpha"}
    assert child_calls["n"] == 1  # planner NOT consulted again for the child


def test_planner_advertises_spawn_subgraph_with_guidance() -> None:
    # Slice 4 un-gates nesting: now that child spend rolls up and child budgets
    # are clamped to the parent's remaining, the planner may reach for
    # spawn_subgraph. The system prompt must describe it (params + when to use).
    from app.supervisor.llm_planner import LLMPlanner

    planner = LLMPlanner(object())  # __init__ doesn't touch the model

    assert NodeKind.SPAWN_SUBGRAPH in planner._executable_kinds
    assert "spawn_subgraph" in planner._system_prompt
    assert "sub_goal" in planner._system_prompt  # params are documented


# ---------- concurrent-sibling budget (TOCTOU reserve-and-refund) ----------


def _two_llm_child() -> GraphSpec:
    return _child_spec(
        "child",
        nodes=[
            NodeSpec(
                id="a",
                kind=NodeKind.LLM_CALL,
                outputs=["a"],
                params={"instruction": "A"},
            ),
            NodeSpec(
                id="b",
                kind=NodeKind.LLM_CALL,
                inputs=["a"],
                outputs=["b"],
                params={"instruction": "B"},
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "a"}),
            EdgeSpec.model_validate({"from": "a", "to": "b"}),
            EdgeSpec.model_validate({"from": "b", "to": "END"}),
        ],
    )


def _run_two_spawns(*, max_llm_calls: int, sequential: bool):
    # Two spawn_subgraph siblings, each child wanting 2 LLM calls. `sequential`
    # orders them (spawn_a -> spawn_b, distinct supersteps); otherwise both edge
    # straight off START and LangGraph runs them in ONE superstep (concurrently,
    # on its thread pool) — the case the TOCTOU is about.
    from app.registry import validate_graph_spec
    from app.runtime.executor import LangGraphExecutor

    def child_planner(sub_goal):
        del sub_goal
        return _two_llm_child()

    def echo(state, params):
        del state
        return {"result": params["instruction"]}

    runners = {NodeKind.LLM_CALL: echo}
    executor = LangGraphExecutor(runners=runners)
    runners[NodeKind.SPAWN_SUBGRAPH] = build_spawn_subgraph_runner(
        make_child_launcher(planner=child_planner, executor=executor),
        ledger_registry=executor.ledgers,
    )
    nodes = [
        NodeSpec(
            id="spawn_a",
            kind=NodeKind.SPAWN_SUBGRAPH,
            outputs=["rep_a"],
            params={"sub_goal": "g", "name": "ca"},
        ),
        NodeSpec(
            id="spawn_b",
            kind=NodeKind.SPAWN_SUBGRAPH,
            outputs=["rep_b"],
            params={"sub_goal": "g", "name": "cb"},
        ),
    ]
    if sequential:
        raw_edges = [
            {"from": "START", "to": "spawn_a"},
            {"from": "spawn_a", "to": "spawn_b"},
            {"from": "spawn_b", "to": "END"},
        ]
    else:
        raw_edges = [
            {"from": "START", "to": "spawn_a"},
            {"from": "START", "to": "spawn_b"},
            {"from": "spawn_a", "to": "END"},
            {"from": "spawn_b", "to": "END"},
        ]
    parent = _child_spec(
        "parent",
        nodes=nodes,
        edges=[EdgeSpec.model_validate(e) for e in raw_edges],
        budget=GraphBudget(max_llm_calls=max_llm_calls),
    )
    return executor.execute(executor.compile(validate_graph_spec(parent)), run_id="p")


@pytest.mark.parametrize("trial", range(5))
def test_concurrent_sibling_spawns_cannot_overspend_the_budget(trial: int) -> None:
    # The regression. Two spawn_subgraph siblings run in ONE superstep; each child
    # wants 2 LLM calls but the parent budget is only 2. Without the shared
    # BudgetLedger both read the same pre-merge counters, each clamp their child to
    # the full remaining (2), and jointly spend 4. With it, the lock serializes the
    # reservations: the first sibling reserves the whole remaining and the second is
    # starved to zero and fails closed (allocation is first-come-greedy). Run a few
    # trials so a lock regression can't hide behind a single lucky interleaving.
    del trial
    result = _run_two_spawns(max_llm_calls=2, sequential=False)
    counters = result.state["counters"]
    values = result.state.get("values", {}) or {}

    # EXACTLY the budget is spent — never 4 (the over-spend this fixes) and never
    # 0/1 (the winner runs its 2-call child in full). A `<= 2` assertion would also
    # pass with the lock removed under common schedules; `== 2` is what pins the fix.
    assert counters["llm_calls_consumed"] == 2
    # One child completed and produced its report; the other was refused outright.
    produced = [key for key in ("rep_a", "rep_b") if key in values]
    assert len(produced) == 1
    # The starved sibling failed CLOSED (recorded an error), not silently — fail-
    # closed under contention is the whole governance contract here.
    assert result.ok is False
    assert result.state.get("errors")


def test_sequential_spawns_share_the_budget_across_supersteps() -> None:
    # Two spawns in series: the second runs a later superstep, so counters have
    # absorbed the first's spend. The ledger's baseline reconciliation must let the
    # second see the real remaining and run — both 2-call children fit a budget of 4.
    result = _run_two_spawns(max_llm_calls=4, sequential=True)

    assert result.ok is True
    assert result.state["counters"]["llm_calls_consumed"] == 4  # 2 + 2
    # Both children completed — neither is starved when the budget genuinely fits.
    values = result.state.get("values", {}) or {}
    assert "rep_a" in values and "rep_b" in values

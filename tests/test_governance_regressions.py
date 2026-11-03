"""Regressions for the governance/contract fixes.

Each section pins one repaired contract from the code review:

1. Budgets are enforced at *runtime*, not just at validation — the node
   wrapper's spend gate, the parallel_map dispatcher's atomic ledger charge,
   and the spawn_subgraph planner-call charge together make `max_llm_calls` a
   hard ceiling on actual spend.
2. A resumed run gets a fresh wall-clock deadline (a hung post-resume runner
   can't block forever).
3. A completed run cannot be "resumed" again (no event replay / re-execution),
   and a replay cannot clobber an existing run's recording.
4. Errors surface: node-level failures land on `SupervisorResult.errors`, and
   the SDK's `run()` returns a failed result instead of raising.
5. Failed attempts (plan/validate/compile) leave a record, and a recording
   failure never masks the run's own outcome.
"""

from __future__ import annotations

import json
import time

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Command, Send

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec, GraphBudget
from app.recording import FileRecorder
from app.runtime import LangGraphExecutor
from app.runtime.budget_ledger import BudgetLedger
from app.runtime.parallel_map import make_parallel_map_dispatcher
from app.runtime.subgraph import (
    SubgraphChildFailed,
    build_spawn_subgraph_runner,
    make_child_launcher,
)
from app.runtime.wrappers import make_node_wrapper
from app.supervisor import StaticPlanner, Supervisor


def _edge(from_: str, to: str) -> EdgeSpec:
    return EdgeSpec.model_validate({"from": from_, "to": to})


def _llm_node(node_id: str, output: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        kind=NodeKind.LLM_CALL,
        outputs=[output],
        params={"instruction": node_id},
    )


# ---------- 1. runtime budget enforcement ----------


def test_ledger_try_charge_grants_within_budget_and_refuses_beyond() -> None:
    ledger = BudgetLedger()

    assert ledger.try_charge(key="llm_calls", amount=8, budget=10, consumed=0)
    # Same stale counters snapshot (a concurrent dispatcher): 8 pending + 8 > 10.
    assert not ledger.try_charge(key="llm_calls", amount=8, budget=10, consumed=0)
    # But the leftover 2 are still grantable.
    assert ledger.try_charge(key="llm_calls", amount=2, budget=10, consumed=0)


def test_ledger_try_charge_unbounded_budget_always_grants() -> None:
    ledger = BudgetLedger()
    assert ledger.try_charge(key="llm_calls", amount=999, budget=None, consumed=0)


def test_ledger_try_charge_resets_pending_when_counters_advance() -> None:
    ledger = BudgetLedger()
    assert ledger.try_charge(key="llm_calls", amount=4, budget=8, consumed=0)
    # A new superstep absorbed the spend into counters: consumed advanced to 4,
    # so the pending tally resets and the remaining 4 are grantable.
    assert ledger.try_charge(key="llm_calls", amount=4, budget=8, consumed=4)
    assert not ledger.try_charge(key="llm_calls", amount=1, budget=8, consumed=4)


def test_wrapper_refuses_llm_call_once_budget_exhausted() -> None:
    node = _llm_node("over", "draft")
    invoked = []

    def runner(state, params):
        invoked.append(True)
        return {"result": "should not run"}

    wrapper = make_node_wrapper(node, runner, counts_as_llm_call=True)
    state = {
        "values": {},
        "metadata": {"budget_max_llm_calls": 2},
        "counters": {"llm_calls_consumed": 2},
    }

    update = wrapper(state)

    assert not invoked  # fail-closed: the call never fired
    assert isinstance(update, Command)
    assert update.goto == END
    errors = update.update["errors"]
    assert errors[0]["type"] == "LlmCallBudgetExceeded"
    assert errors[0]["node_id"] == "over"


def test_wrapper_allows_llm_call_within_budget() -> None:
    node = _llm_node("within", "draft")

    def runner(state, params):
        return {"result": "ran"}

    wrapper = make_node_wrapper(node, runner, counts_as_llm_call=True)
    state = {
        "values": {},
        "metadata": {"budget_max_llm_calls": 2},
        "counters": {"llm_calls_consumed": 1},
    }

    update = wrapper(state)

    assert not isinstance(update, Command)
    assert update["values"] == {"draft": "ran"}
    assert update["counters"]["llm_calls_consumed"] == 1


def _pm_node(node_id: str, output: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        kind=NodeKind.PARALLEL_MAP,
        outputs=[output],
        params={
            "over": "items",
            "child_kind": "llm_call",
            "child_params": {"instruction": "x"},
        },
    )


def test_concurrent_dispatchers_cannot_jointly_overrun_llm_budget() -> None:
    # Two parallel_map dispatchers in one superstep read the SAME pre-merge
    # counters snapshot. With the shared ledger they must not jointly dispatch
    # more llm-call workers than the budget allows.
    ledger = BudgetLedger()
    registry = {"r": ledger}
    state = {
        "values": {"items": [f"i{i}" for i in range(8)]},
        "metadata": {
            "run_id": "r",
            "budget_max_llm_calls": 10,
            "budget_max_fanout": 64,
        },
        "counters": {},
    }

    def dispatcher(pm_id: str):
        return make_parallel_map_dispatcher(
            _pm_node(pm_id, f"{pm_id}_out"),
            worker_id=f"{pm_id}__pm_worker",
            join_id=f"{pm_id}__pm_join",
            child_counts_as_llm_call=True,
            ledger_registry=registry,
        )

    first = dispatcher("pm_a")(dict(state))
    second = dispatcher("pm_b")(dict(state))

    # First fan-out (8 <= 10) dispatches; the second (8 + 8 pending > 10)
    # must halt fail-closed even though its counters snapshot still reads 0.
    assert isinstance(first.goto, list)
    assert all(isinstance(send, Send) for send in first.goto)
    assert second.goto == END
    assert second.update["errors"][0]["type"] == "LlmCallBudgetExceeded"


def test_child_launcher_charges_the_planner_llm_call() -> None:
    child = GraphSpec(
        graph_id="child",
        goal="child",
        budget=GraphBudget(max_llm_calls=8),
        nodes=[_llm_node("step", "draft")],
        edges=[_edge("START", "step"), _edge("step", "END")],
    )

    def planner(sub_goal):
        return child

    def echo(state, params):
        return {"result": params["instruction"]}

    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: echo})
    launcher = make_child_launcher(
        planner=planner, executor=executor, planner_counts_as_llm_call=True
    )

    result = launcher(
        "goal",
        run_id="p__sg_c",
        graph_depth=1,
        parent_run_id="p",
        inputs={},
        max_llm_calls=2,
    )

    # 1 child llm call + 1 planning call — both charged against the nest.
    assert result.status == "ok"
    assert result.counters["llm_calls_consumed"] == 2


def test_child_launcher_refuses_to_plan_on_empty_llm_budget() -> None:
    planned = []

    def planner(sub_goal):
        planned.append(sub_goal)
        raise AssertionError("planner must not be called with no budget")

    executor = LangGraphExecutor(runners={})
    launcher = make_child_launcher(
        planner=planner, executor=executor, planner_counts_as_llm_call=True
    )

    with pytest.raises(SubgraphChildFailed, match="LLM-call budget"):
        launcher(
            "goal",
            run_id="p__sg_c",
            graph_depth=1,
            parent_run_id="p",
            inputs={},
            max_llm_calls=0,
        )
    assert not planned


def test_parent_halts_when_child_rollup_exhausts_llm_budget() -> None:
    # The nesting overspend regression: the child legitimately consumes the
    # whole remaining allowance, so the parent's own later llm node must halt
    # at the runtime gate instead of overspending the granted budget.
    child = GraphSpec(
        graph_id="child",
        goal="child",
        budget=GraphBudget(max_llm_calls=8),
        nodes=[_llm_node("inner", "inner_out")],
        edges=[_edge("START", "inner"), _edge("inner", "END")],
    )

    def child_planner(sub_goal):
        return child

    def echo(state, params):
        return {"result": params["instruction"]}

    runners = {NodeKind.LLM_CALL: echo}
    executor = LangGraphExecutor(runners=runners)
    runners[NodeKind.SPAWN_SUBGRAPH] = build_spawn_subgraph_runner(
        make_child_launcher(planner=child_planner, executor=executor),
        ledger_registry=executor.ledgers,
    )

    parent = GraphSpec(
        graph_id="parent",
        goal="parent",
        budget=GraphBudget(max_llm_calls=1),
        nodes=[
            NodeSpec(
                id="nest",
                kind=NodeKind.SPAWN_SUBGRAPH,
                outputs=["nest_out"],
                params={"name": "c", "sub_goal": "dig"},
            ),
            _llm_node("after", "after_out"),
        ],
        edges=[
            _edge("START", "nest"),
            _edge("nest", "after"),
            _edge("after", "END"),
        ],
    )

    result = executor.execute(executor.compile(parent), run_id="parent-run")

    # The child's rolled-up call consumed the budget of 1; the parent's later
    # llm node must NOT run — total spend stays within the granted budget.
    assert result.ok is False
    assert result.state["counters"]["llm_calls_consumed"] == 1
    assert any(
        e["type"] == "LlmCallBudgetExceeded" and e["node_id"] == "after"
        for e in result.state["errors"]
    )
    assert "after_out" not in result.state.get("values", {})


# ---------- 2. resume wall-clock deadline ----------


def _wait_then_step_spec(*, max_wall_seconds: int = 90) -> GraphSpec:
    return GraphSpec(
        graph_id="wait-graph",
        goal="pause then finish",
        budget=GraphBudget(max_nodes=8, max_wall_seconds=max_wall_seconds),
        nodes=[
            NodeSpec(
                id="hold",
                kind=NodeKind.WAIT_FOR_EVENT,
                outputs=["signal"],
                params={"event_type": "human_input", "output_key": "signal"},
            ),
            _llm_node("finish", "final"),
        ],
        edges=[
            _edge("START", "hold"),
            _edge("hold", "finish"),
            _edge("finish", "END"),
        ],
    )


def test_resume_applies_a_fresh_wall_clock_deadline() -> None:
    def hanging_runner(state, params):
        time.sleep(3.0)
        return {"result": "too late"}

    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: hanging_runner},
        checkpointer=MemorySaver(),
    )
    spec = _wait_then_step_spec(max_wall_seconds=1)
    compiled = executor.compile(spec)

    started = executor.execute(compiled, run_id="hang-resume")
    assert started.paused is True

    began = time.monotonic()
    resumed = executor.resume(compiled, run_id="hang-resume", event={"value": "go"})
    elapsed = time.monotonic() - began

    assert resumed.ok is False
    assert "wall-clock deadline" in (resumed.error or "")
    assert elapsed < 2.5  # abandoned at the ~1s deadline, not the 3s hang


# ---------- 3. resume/replay integrity ----------


def test_completed_run_cannot_be_resumed_again(tmp_path) -> None:
    executed = []

    def counting_runner(state, params):
        executed.append(params["instruction"])
        return {"result": "done"}

    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: counting_runner},
        checkpointer=MemorySaver(),
    )
    sup = Supervisor(
        planner=StaticPlanner(_wait_then_step_spec()),
        executor=executor,
        recorder=FileRecorder(root_dir=tmp_path, overwrite=True),
    )

    assert sup.run("go", run_id="w1").status == "paused"
    first = sup.resume("w1", event={"event_type": "human_input", "value": "a"})
    assert first.status == "ok"
    runs_after_first = len(executed)

    second = sup.resume("w1", event={"event_type": "human_input", "value": "b"})

    assert second.status == "resume_failed"
    assert "not paused" in second.response
    assert len(executed) == runs_after_first  # nothing re-executed
    assert second.errors[-1]["type"] == "NotResumable"


# ---------- 4. error contract ----------


def test_node_failures_surface_on_supervisor_errors(tmp_path) -> None:
    def failing_runner(state, params):
        raise RuntimeError("runner exploded")

    spec = GraphSpec(
        graph_id="boom",
        goal="fail in a node",
        nodes=[_llm_node("step", "draft")],
        edges=[_edge("START", "step"), _edge("step", "END")],
    )
    sup = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: failing_runner}),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    result = sup.run("go", run_id="err-surface")

    assert result.status == "execution_failed"
    node_errors = [e for e in result.errors if e.get("node_id") == "step"]
    assert node_errors, f"expected a node-level entry in errors: {result.errors}"
    assert node_errors[0]["stage"] == "execute"
    assert "runner exploded" in node_errors[0]["message"]


def test_sdk_run_returns_failed_result_instead_of_raising() -> None:
    from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

    engine = DynamicSubgraphs(
        EngineConfig(model=Model("no-such-provider", "some-model"))
    )

    result = engine.run("hello")  # must not raise

    assert result.ok is False
    assert result.status == "engine_failed"
    assert result.errors[0]["stage"] == "engine"
    assert "no-such-provider" in result.errors[0]["message"]


# ---------- 5. failure recording & status masking ----------


class _ExplodingRecorder(FileRecorder):
    def record(self, **kwargs):
        raise OSError("disk full")


def test_record_failure_does_not_mask_execution_failure(tmp_path) -> None:
    def failing_runner(state, params):
        raise RuntimeError("runner exploded")

    spec = GraphSpec(
        graph_id="boom",
        goal="fail then fail to record",
        nodes=[_llm_node("step", "draft")],
        edges=[_edge("START", "step"), _edge("step", "END")],
    )
    sup = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: failing_runner}),
        recorder=_ExplodingRecorder(root_dir=tmp_path),
    )

    result = sup.run("go", run_id="mask-check")

    # The run's own outcome wins; the record failure is appended, not a mask.
    assert result.status == "execution_failed"
    assert any(e.get("stage") == "record" for e in result.errors)


def test_record_failure_still_downgrades_an_ok_run(tmp_path) -> None:
    def ok_runner(state, params):
        return {"result": "fine"}

    spec = GraphSpec(
        graph_id="fine",
        goal="succeed but fail to record",
        nodes=[_llm_node("step", "draft")],
        edges=[_edge("START", "step"), _edge("step", "END")],
    )
    sup = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: ok_runner}),
        recorder=_ExplodingRecorder(root_dir=tmp_path),
    )

    result = sup.run("go", run_id="downgrade-check")

    assert result.status == "record_failed"


def test_failure_record_respects_artifact_selection(tmp_path) -> None:
    recorder = FileRecorder(
        root_dir=tmp_path, overwrite=True, selection=frozenset({"graph.mmd"})
    )

    record = recorder.record_failure(
        run_id="narrow", status="plan_failed", prompt="p", errors=[]
    )

    # Nothing this path can produce was selected: no directory, no receipt.
    assert record is None
    assert not (tmp_path / "narrow").exists()


def test_failure_record_writes_output_and_summary(tmp_path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)

    record = recorder.record_failure(
        run_id="fdir",
        status="plan_failed",
        prompt="do something",
        errors=[{"stage": "plan", "type": "RuntimeError", "message": "nope"}],
    )

    assert record is not None
    output = json.loads((tmp_path / "fdir" / "output.json").read_text("utf-8"))
    assert output["ok"] is False
    assert output["status"] == "plan_failed"
    assert output["error"] == "nope"
    assert (tmp_path / "fdir" / "prompt.md").read_text("utf-8") == "do something"
    assert "plan_failed" in (tmp_path / "fdir" / "summary.md").read_text("utf-8")

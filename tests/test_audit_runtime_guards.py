"""Regression tests for two runtime-governance holes found in the 2026-09 audit.

1. A failed node must halt the run: `Command(goto=END)` alone does not stop the
   node's *static* downstream edges, so later nodes (and their LLM spend) used to
   keep running after a failure.
2. `max_llm_calls` must hold for plain LLM nodes scheduled in the same superstep
   as a `spawn_subgraph` (or each other): the per-node gate read only a stale
   counters snapshot, so concurrent siblings could jointly overshoot.
"""

from __future__ import annotations

import threading
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.models import NodeKind
from app.registry import validate_graph_spec
from app.runtime.budget_ledger import BudgetLedger
from app.runtime.executor import LangGraphExecutor
from app.runtime.wrappers import make_node_wrapper


def _run(executor: LangGraphExecutor, spec, run_id: str = "audit"):
    validated = validate_graph_spec(spec)
    return executor.execute(executor.compile(validated), run_id=run_id)


# --- 1. failure halts the run ------------------------------------------------


def test_failed_node_does_not_run_downstream_nodes(
    spec_factory, make_node, make_edge
) -> None:
    ran: list[str] = []

    def runner(state: Any, params: dict[str, Any]) -> dict[str, Any]:
        ran.append(params["instruction"])
        if params["instruction"] == "boom":
            raise RuntimeError("boom")
        return {"result": params["instruction"]}

    spec = spec_factory(
        nodes=[
            make_node("a", outputs=["a"], params={"instruction": "boom"}),
            make_node("b", inputs=["a"], outputs=["b"], params={"instruction": "B"}),
            make_node("c", inputs=["b"], outputs=["c"], params={"instruction": "C"}),
        ],
        edges=[
            make_edge("START", "a"),
            make_edge("a", "b"),
            make_edge("b", "c"),
            make_edge("c", "END"),
        ],
    )
    result = _run(LangGraphExecutor(runners={NodeKind.LLM_CALL: runner}), spec)

    assert result.ok is False
    assert ran == ["boom"], f"downstream nodes ran after failure: {ran}"
    # Exactly one failing node is reported; skipped nodes add no errors/spend.
    assert [e["node_id"] for e in result.state["errors"]] == ["a"]
    assert result.state["counters"]["llm_calls_consumed"] == 1


def test_failed_node_does_not_reach_wait_for_event(
    spec_factory, make_node, make_edge
) -> None:
    def runner(state: Any, params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    spec = spec_factory(
        nodes=[
            make_node("a", outputs=["a"]),
            make_node(
                "wait",
                NodeKind.WAIT_FOR_EVENT,
                inputs=["a"],
                outputs=["approval"],
                params={"event_type": "approval"},
            ),
        ],
        edges=[
            make_edge("START", "a"),
            make_edge("a", "wait"),
            make_edge("wait", "END"),
        ],
    )
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: runner}, checkpointer=MemorySaver()
    )
    result = _run(executor, spec)

    assert result.ok is False
    assert result.paused is False, "a failed run must not pause on a later wait node"


def test_failed_node_does_not_run_branch_node(
    spec_factory, make_node, make_edge
) -> None:
    def runner(state: Any, params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    spec = spec_factory(
        nodes=[
            make_node("a", outputs=["route"]),
            make_node(
                "gate",
                NodeKind.BRANCH,
                inputs=["route"],
                params={"branches": ["x", "y"], "decision_key": "route"},
            ),
            make_node("x", outputs=["x"], params={"instruction": "X"}),
            make_node("y", outputs=["y"], params={"instruction": "Y"}),
        ],
        edges=[
            make_edge("START", "a"),
            make_edge("a", "gate"),
            make_edge("gate", "x"),
            make_edge("gate", "y"),
            make_edge("x", "END"),
            make_edge("y", "END"),
        ],
    )
    result = _run(LangGraphExecutor(runners={NodeKind.LLM_CALL: runner}), spec)

    assert result.ok is False
    assert [e["node_id"] for e in result.state["errors"]] == ["a"]


# --- 2. LLM budget holds across same-superstep siblings ----------------------


def _wrapper_state(consumed: int, budget: int, run_id: str = "r") -> dict[str, Any]:
    return {
        "values": {},
        "errors": [],
        "counters": {"llm_calls_consumed": consumed},
        "metadata": {"budget_max_llm_calls": budget, "run_id": run_id},
    }


def test_llm_node_gate_is_atomic_across_same_superstep_siblings(make_node) -> None:
    ledger = BudgetLedger()
    registry = {"r": ledger}

    def runner(state: Any, params: dict[str, Any]) -> dict[str, Any]:
        return {"result": "ok"}

    a = make_node("a", outputs=["a"])
    b = make_node("b", outputs=["b"])
    wa = make_node_wrapper(a, runner, counts_as_llm_call=True, ledger_registry=registry)
    wb = make_node_wrapper(b, runner, counts_as_llm_call=True, ledger_registry=registry)

    # Both siblings read the SAME pre-merge snapshot: 0 consumed, budget 1.
    snapshot = _wrapper_state(consumed=0, budget=1)
    first = wa(snapshot)
    second = wb(snapshot)

    outcomes = [
        "LlmCallBudgetExceeded" if _is_budget_halt(r) else "ran"
        for r in (first, second)
    ]
    assert sorted(outcomes) == ["LlmCallBudgetExceeded", "ran"], outcomes


def _reserved_ledger(budget: int) -> BudgetLedger:
    ledger = BudgetLedger()
    # A concurrent spawn_subgraph sibling already reserved the whole remainder.
    ledger.reserve(budgets={"llm_calls": budget}, consumed={"llm_calls": 0})
    return ledger


def _plain_wrapper(make_node, ledger: BudgetLedger, ran: list[str]):
    def runner(state: Any, params: dict[str, Any]) -> dict[str, Any]:
        ran.append("ran")
        return {"result": "ok"}

    return make_node_wrapper(
        make_node("a", outputs=["a"]),
        runner,
        counts_as_llm_call=True,
        ledger_registry={"r": ledger},
    )


def test_llm_node_waits_for_spawn_refund_instead_of_failing(make_node) -> None:
    ledger = _reserved_ledger(3)
    ran: list[str] = []
    # The spawn child finishes having spent 1 of its 3-call reservation.
    threading.Timer(
        0.1,
        ledger.refund,
        kwargs={"reserved": {"llm_calls": 3}, "spent": {"llm_calls": 1}},
    ).start()

    result = _plain_wrapper(make_node, ledger, ran)(_wrapper_state(0, 3))

    assert ran == ["ran"] and not _is_budget_halt(result)


def test_llm_node_refused_when_spawn_child_spent_the_whole_budget(make_node) -> None:
    ledger = _reserved_ledger(3)
    ran: list[str] = []
    threading.Timer(
        0.1,
        ledger.refund,
        kwargs={"reserved": {"llm_calls": 3}, "spent": {"llm_calls": 3}},
    ).start()

    result = _plain_wrapper(make_node, ledger, ran)(_wrapper_state(0, 3))

    assert ran == [] and _is_budget_halt(result)


def _is_budget_halt(result: Any) -> bool:
    if not isinstance(result, Command):
        return False
    update = result.update
    return any(e.get("type") == "LlmCallBudgetExceeded" for e in update["errors"])


def test_spawn_and_gauntlet_names_are_capped() -> None:
    import pytest
    from pydantic import ValidationError

    from app.registry.params import SpawnSubgraphParams

    SpawnSubgraphParams(sub_goal="g", name="n" * 32)
    with pytest.raises(ValidationError):
        SpawnSubgraphParams(sub_goal="g", name="n" * 33)

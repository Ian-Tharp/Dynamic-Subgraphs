"""Plan-repair loop: a rejected plan is re-planned with the validator's issues
+ the host limits fed back, bounded by `max_plan_attempts`.
"""

from __future__ import annotations

from pathlib import Path

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec, GraphBudget
from app.policy import ExecutionPolicy
from app.recording import FileRecorder
from app.runtime import LangGraphExecutor
from app.supervisor import Supervisor
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model


def _mock_llm(state, params):
    return {"result": params["instruction"]}


def _linear_spec(n_nodes: int, *, budget_nodes: int) -> GraphSpec:
    nodes = [
        NodeSpec(
            id=f"n{i}",
            kind=NodeKind.LLM_CALL,
            outputs=[f"o{i}"],
            params={"instruction": f"step {i}"},
        )
        for i in range(n_nodes)
    ]
    edges = [EdgeSpec.model_validate({"from": "START", "to": "n0"})]
    edges += [
        EdgeSpec.model_validate({"from": f"n{i}", "to": f"n{i + 1}"})
        for i in range(n_nodes - 1)
    ]
    edges += [EdgeSpec.model_validate({"from": f"n{n_nodes - 1}", "to": "END"})]
    return GraphSpec(
        graph_id="repair-test",
        goal="g",
        nodes=nodes,
        edges=edges,
        budget=GraphBudget(max_nodes=budget_nodes, max_llm_calls=99),
    )


class _RepairPlanner:
    """Over-budget on the first pass; a valid small plan once it sees the repair
    prompt (which contains "REJECTED")."""

    def __init__(self, big: GraphSpec, small: GraphSpec) -> None:
        self.big = big
        self.small = small
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> GraphSpec:
        self.prompts.append(prompt)
        return self.small if "REJECTED" in prompt else self.big


class _StubbornPlanner:
    """Always returns the same over-budget plan, ignoring repair feedback."""

    def __init__(self, big: GraphSpec) -> None:
        self.big = big
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> GraphSpec:
        self.prompts.append(prompt)
        return self.big


def _supervisor(planner, tmp_path: Path, *, max_plan_attempts: int) -> Supervisor:
    return Supervisor(
        planner=planner,
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm}),
        recorder=FileRecorder(root_dir=tmp_path),
        policy=ExecutionPolicy(max_nodes=3),
        max_plan_attempts=max_plan_attempts,
    )


def test_repair_recovers_an_over_budget_plan(tmp_path: Path) -> None:
    planner = _RepairPlanner(
        big=_linear_spec(5, budget_nodes=99),  # 5 nodes > host max_nodes=3
        small=_linear_spec(2, budget_nodes=99),  # 2 nodes, fits
    )
    sup = _supervisor(planner, tmp_path, max_plan_attempts=2)

    result = sup.run("do the thing", run_id="repair-ok")

    assert result.status == "ok"
    assert result.plan_attempts == 2
    # The second prompt is the repair prompt: it carries the issue + host limit.
    assert len(planner.prompts) == 2
    assert "REJECTED" in planner.prompts[1]
    assert "at most 3 nodes" in planner.prompts[1]
    # A repaired-then-successful run carries NO terminal errors.
    assert result.errors == []


def test_strict_mode_blocks_on_first_failure(tmp_path: Path) -> None:
    planner = _RepairPlanner(
        big=_linear_spec(5, budget_nodes=99),
        small=_linear_spec(2, budget_nodes=99),
    )
    sup = _supervisor(planner, tmp_path, max_plan_attempts=1)

    result = sup.run("do the thing", run_id="repair-strict")

    assert result.status == "validation_failed"
    assert result.plan_attempts == 1
    assert len(planner.prompts) == 1  # no retry
    codes = {i.get("code") for e in result.errors for i in (e.get("issues") or [])}
    assert "budget_exceeded" in codes


def test_repair_loop_is_bounded_and_reports_when_never_valid(tmp_path: Path) -> None:
    planner = _StubbornPlanner(_linear_spec(5, budget_nodes=99))
    sup = _supervisor(planner, tmp_path, max_plan_attempts=2)

    result = sup.run("do the thing", run_id="repair-exhausted")

    assert result.status == "validation_failed"
    assert result.plan_attempts == 2  # bounded — exactly max_plan_attempts
    assert len(planner.prompts) == 2


# ---------- SDK surface ----------


def test_engine_config_default_repairs_once() -> None:
    assert EngineConfig().max_plan_attempts == 2


def test_run_result_surfaces_plan_attempts(tmp_path: Path) -> None:
    # A normal mock run (valid first try) reports a single attempt.
    engine = DynamicSubgraphs(
        EngineConfig(model=Model("openai", "x"), planner="mock", runs_dir=tmp_path)
    )
    result = engine.run("compare two things", run_id="repair-sdk")

    assert result.ok
    assert result.plan_attempts == 1
    assert result.to_dict()["plan_attempts"] == 1

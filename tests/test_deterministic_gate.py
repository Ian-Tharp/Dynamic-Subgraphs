"""DeterministicEvalGate — structural scoring behavior (Slice 7, PR2).

Offline and token-free: the gate is a pure function of a completed run's
artifacts, so every case here builds a `GraphSpec` + `ExecutionResult` by hand
and asserts the scored `EvalResult`. AAA throughout.
"""

from __future__ import annotations

import json

from app.models import RunTrace
from app.models.graph_spec import EdgeSpec, GraphSpec, GraphSpecMetadata, NodeSpec
from app.models.node_kinds import NodeKind
from app.runtime import ExecutionResult
from dynamic_subgraphs.eval import EvalContext, EvalReference
from dynamic_subgraphs.eval.gates import DeterministicEvalGate
from dynamic_subgraphs.usage import TokenUsage

GOAL = "Compare solar and nuclear and recommend one"


def _llm(node_id: str, instruction: str = "do work") -> NodeSpec:
    return NodeSpec(
        id=node_id, kind=NodeKind.LLM_CALL, params={"instruction": instruction}
    )


def _web_search(node_id: str) -> NodeSpec:
    return NodeSpec(id=node_id, kind=NodeKind.TOOL_CALL, params={"tool": "web_search"})


def _edge(src: str, dst: str) -> EdgeSpec:
    return EdgeSpec(from_=src, to=dst)


def _spec(
    nodes: list[NodeSpec], edges: list[EdgeSpec], *, goal: str = GOAL
) -> GraphSpec:
    return GraphSpec(
        graph_id="g1",
        goal=goal,
        nodes=nodes,
        edges=edges,
        metadata=GraphSpecMetadata(planner_model="planner-x"),
    )


def _context(
    spec: GraphSpec,
    values: dict,
    *,
    ok: bool = True,
    status: str = "ok",
    total_tokens: int = 1000,
    cost: float | None = 0.001,
    reference: EvalReference | None = None,
    paused: bool = False,
) -> EvalContext:
    state = {"values": values, "events": [], "errors": []}
    result = ExecutionResult(
        state=state,  # type: ignore[arg-type]
        trace=RunTrace(run_id="r1", graph_id=spec.graph_id, events=[]),
        ok=ok,
        error=None,
        paused=paused,
    )
    half = total_tokens // 2
    usage = TokenUsage(
        input_tokens=half,
        output_tokens=total_tokens - half,
        total_tokens=total_tokens,
        by_model={
            "planner-x": {
                "input_tokens": half,
                "output_tokens": total_tokens - half,
                "total_tokens": total_tokens,
            }
        },
    )
    return EvalContext(
        run_id="r1",
        spec=spec,
        result=result,
        prompt="Compare solar and nuclear",
        status=status,  # type: ignore[arg-type]
        usage=usage,
        cost=cost,
        reference=reference,
        planner_model_name="planner-x",
    )


def test_scores_a_successful_run_across_four_dimensions() -> None:
    # Arrange
    spec = _spec(
        [_llm("plan"), _llm("write")],
        [_edge("START", "plan"), _edge("plan", "write"), _edge("write", "END")],
    )
    ctx = _context(
        spec, {"write": "Solar is cleaner; nuclear is denser. Recommend solar."}
    )

    # Act
    result = DeterministicEvalGate().evaluate(ctx)

    # Assert
    assert result.deterministic is True
    assert {c.dimension for c in result.components} == {
        "plan_validity",
        "grounding",
        "goal_completion",
        "cost_efficiency",
    }
    assert 0.0 <= result.quality <= 1.0
    assert result.total_tokens == 1000
    assert result.value_per_ktok is not None
    assert result.fingerprint is not None
    assert result.fingerprint.node_count == 2
    assert result.overall_verdict in {"pass", "weak", "fail"}


def test_grounding_inapplicable_drops_out_of_the_score() -> None:
    # Arrange — a pure llm plan with no reference: grounding does not apply.
    spec = _spec([_llm("write")], [_edge("START", "write"), _edge("write", "END")])
    ctx = _context(spec, {"write": "Solar wins."})

    # Act
    result = DeterministicEvalGate().evaluate(ctx)
    grounding = next(c for c in result.components if c.dimension == "grounding")

    # Assert
    assert grounding.applicable is False
    assert grounding.verdict == "skipped"
    assert "grounding" not in result.applicable_dimensions
    assert result.fingerprint is not None
    assert result.fingerprint.has_grounded_tool is False


def test_grounding_applies_and_passes_when_web_search_returns_results() -> None:
    # Arrange — a web_search node plus a real grounded output in state values.
    spec = _spec(
        [_web_search("search"), _llm("write")],
        [_edge("START", "search"), _edge("search", "write"), _edge("write", "END")],
    )
    values = {
        "search": {
            "tool": "web_search",
            "answer": "Solar is cheaper per kWh in 2026.",
            "results": [{"title": "t", "url": "u", "snippet": "s"}],
        },
        "write": "Recommend solar.",
    }
    ctx = _context(spec, values)

    # Act
    result = DeterministicEvalGate().evaluate(ctx)
    grounding = next(c for c in result.components if c.dimension == "grounding")

    # Assert
    assert grounding.applicable is True
    assert grounding.score == 1.0
    assert grounding.verdict == "pass"
    assert "grounding" in result.applicable_dimensions
    assert result.fingerprint is not None
    assert result.fingerprint.has_grounded_tool is True


def test_grounding_required_but_missing_fails() -> None:
    # Arrange — reference demands grounding, but no web_search output exists.
    spec = _spec([_llm("write")], [_edge("START", "write"), _edge("write", "END")])
    ctx = _context(
        spec, {"write": "Solar wins."}, reference=EvalReference(grounding_required=True)
    )

    # Act
    result = DeterministicEvalGate().evaluate(ctx)
    grounding = next(c for c in result.components if c.dimension == "grounding")

    # Assert
    assert grounding.applicable is True
    assert grounding.score == 0.0
    assert grounding.verdict == "fail"


def test_reference_coverage_scores_goal_completion() -> None:
    # Arrange — output covers one of two required points.
    spec = _spec([_llm("write")], [_edge("START", "write"), _edge("write", "END")])
    reference = EvalReference(checklist=["solar", "nuclear"])
    ctx = _context(
        spec, {"write": "Only solar is discussed here."}, reference=reference
    )

    # Act
    result = DeterministicEvalGate().evaluate(ctx)
    goal = next(c for c in result.components if c.dimension == "goal_completion")

    # Assert
    assert goal.method == "reference"
    assert goal.signals["covered"] == 1.0
    assert goal.signals["required"] == 2.0
    assert 0.4 <= goal.score <= 0.6


def test_must_not_include_penalizes_goal_completion() -> None:
    # Arrange — required point covered, but a forbidden term appears.
    spec = _spec([_llm("write")], [_edge("START", "write"), _edge("write", "END")])
    reference = EvalReference(checklist=["solar"], must_not_include=["coal"])
    ctx = _context(spec, {"write": "Solar beats coal."}, reference=reference)

    # Act
    result = DeterministicEvalGate().evaluate(ctx)
    goal = next(c for c in result.components if c.dimension == "goal_completion")

    # Assert
    assert goal.signals["must_not_hits"] == 1.0
    assert goal.score < 1.0


def test_failed_run_scores_zero_without_a_judge() -> None:
    # Arrange
    spec = _spec([_llm("write")], [_edge("START", "write"), _edge("write", "END")])
    ctx = _context(spec, {}, ok=False, status="execution_failed")

    # Act
    result = DeterministicEvalGate().evaluate(ctx)

    # Assert
    assert result.quality == 0.0
    assert result.overall_verdict == "fail"
    assert result.components == []
    assert result.quality_floor_met is False
    assert result.deterministic is True
    assert result.fingerprint is not None


def test_paused_run_is_skipped() -> None:
    # Arrange
    spec = _spec([_llm("wait")], [_edge("START", "wait"), _edge("wait", "END")])
    ctx = _context(spec, {}, ok=False, status="paused", paused=True)

    # Act
    result = DeterministicEvalGate().evaluate(ctx)

    # Assert
    assert result.overall_verdict == "skipped"
    assert result.quality == 0.0


def test_all_passing_run_meets_quality_floor() -> None:
    # Arrange — valid plan, grounded, full reference coverage, cheap.
    spec = _spec(
        [_web_search("search"), _llm("write")],
        [_edge("START", "search"), _edge("search", "write"), _edge("write", "END")],
    )
    values = {
        "search": {
            "tool": "web_search",
            "answer": "solar facts",
            "results": [{"x": 1}],
        },
        "write": "Solar is the recommendation.",
    }
    reference = EvalReference(checklist=["solar"], grounding_required=True)
    ctx = _context(spec, values, total_tokens=500, reference=reference)

    # Act
    result = DeterministicEvalGate().evaluate(ctx)

    # Assert
    assert result.quality_floor_met is True
    assert result.overall_verdict == "pass"


def test_scoring_is_deterministic() -> None:
    # Arrange
    spec = _spec(
        [_llm("plan"), _llm("write")],
        [_edge("START", "plan"), _edge("plan", "write"), _edge("write", "END")],
    )
    ctx = _context(spec, {"write": "Solar and nuclear compared."})
    gate = DeterministicEvalGate()

    # Act
    first = gate.evaluate(ctx).model_dump(mode="json")
    second = gate.evaluate(ctx).model_dump(mode="json")
    first.pop("scored_at")
    second.pop("scored_at")

    # Assert
    assert first == second


def test_eval_result_is_json_serializable_and_round_trips() -> None:
    # Arrange
    spec = _spec([_llm("write")], [_edge("START", "write"), _edge("write", "END")])
    ctx = _context(spec, {"write": "Solar."})

    # Act
    result = DeterministicEvalGate().evaluate(ctx)
    dumped = result.model_dump(mode="json")
    restored = type(result).model_validate(dumped)

    # Assert
    json.dumps(dumped)  # must not raise
    assert restored.quality == result.quality
    assert restored.fingerprint == result.fingerprint

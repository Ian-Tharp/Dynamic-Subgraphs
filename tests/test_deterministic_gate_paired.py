"""DeterministicEvalGate paired-mode (reference_only) grounding applicability.

Tests for the new grounding_applicability parameter that enables benchmark
paired-mode where applicability is a TASK property (from EvalReference) not
dependent on voluntary plan choices. Also includes regression test for the
latent bug in _web_search_nodes (now recognizing tool_name contract key).
"""

from __future__ import annotations

from app.models import RunTrace
from app.models.graph_spec import EdgeSpec, GraphSpec, GraphSpecMetadata, NodeSpec
from app.models.node_kinds import NodeKind
from app.runtime import ExecutionResult
from dynamic_subgraphs.eval import EvalContext, EvalReference
from dynamic_subgraphs.eval.gates import DeterministicEvalGate, _web_search_nodes
from dynamic_subgraphs.usage import TokenUsage

GOAL = "Compare solar and nuclear and recommend one"


def _llm(node_id: str, instruction: str = "do work") -> NodeSpec:
    return NodeSpec(
        id=node_id, kind=NodeKind.LLM_CALL, params={"instruction": instruction}
    )


def _web_search_with_tool_name(node_id: str) -> NodeSpec:
    """web_search node using the contract-correct tool_name key."""
    return NodeSpec(
        id=node_id,
        kind=NodeKind.TOOL_CALL,
        params={"tool_name": "web_search", "args": {}},
    )


def _web_search_legacy(node_id: str) -> NodeSpec:
    """web_search node using legacy tool key (fallback support)."""
    return NodeSpec(
        id=node_id,
        kind=NodeKind.TOOL_CALL,
        params={"tool": "web_search"},
    )


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


def test_paired_mode_ignores_voluntary_grounding() -> None:
    """reference_only: task says grounding NOT required -> dimension inapplicable
    even though the plan contains a web_search node.
    """
    # Arrange
    gate = DeterministicEvalGate(grounding_applicability="reference_only")
    spec = _spec(
        [_web_search_with_tool_name("search"), _llm("write")],
        [_edge("START", "search"), _edge("search", "write"), _edge("write", "END")],
    )
    ctx = _context(
        spec,
        {"search": {"tool": "web_search", "results": []}, "write": "Solar wins."},
        reference=EvalReference(grounding_required=False),
    )

    # Act
    result = gate.evaluate(ctx)
    grounding = next(c for c in result.components if c.dimension == "grounding")

    # Assert
    assert grounding.applicable is False
    assert grounding.verdict == "skipped"
    assert "grounding" not in result.applicable_dimensions


def test_paired_mode_required_dimension_zero() -> None:
    """reference_only with grounding_required=True but no grounded output -> 0.0 fail."""
    # Arrange
    gate = DeterministicEvalGate(grounding_applicability="reference_only")
    spec = _spec(
        [_llm("write")],
        [_edge("START", "write"), _edge("write", "END")],
    )
    ctx = _context(
        spec,
        {"write": "Solar wins."},
        reference=EvalReference(grounding_required=True),
    )

    # Act
    result = gate.evaluate(ctx)
    grounding = next(c for c in result.components if c.dimension == "grounding")

    # Assert
    assert grounding.applicable is True
    assert grounding.score == 0.0
    assert grounding.verdict == "fail"


def test_default_mode_unchanged() -> None:
    """default "auto" keeps today's behavior: voluntary grounding -> applicable."""
    # Arrange
    gate = DeterministicEvalGate()  # default is "auto"
    spec = _spec(
        [_web_search_with_tool_name("search"), _llm("write")],
        [_edge("START", "search"), _edge("search", "write"), _edge("write", "END")],
    )
    ctx = _context(
        spec,
        {"search": {"tool": "web_search", "results": []}, "write": "Solar wins."},
        reference=EvalReference(grounding_required=False),
    )

    # Act
    result = gate.evaluate(ctx)
    grounding = next(c for c in result.components if c.dimension == "grounding")

    # Assert
    assert grounding.applicable is True


def test_web_search_nodes_recognizes_tool_name_key() -> None:
    """Latent-bug regression: the contract key is tool_name; legacy tool/name
    keys remain as fallbacks.
    """
    # Arrange — contract-correct tool_name key
    spec_correct = _spec(
        [_web_search_with_tool_name("search")],
        [_edge("START", "search"), _edge("search", "END")],
    )

    # Act
    count_correct = _web_search_nodes(spec_correct)

    # Assert
    assert count_correct == 1

    # Arrange — legacy tool key (fallback support)
    spec_legacy = _spec(
        [_web_search_legacy("search")],
        [_edge("START", "search"), _edge("search", "END")],
    )

    # Act
    count_legacy = _web_search_nodes(spec_legacy)

    # Assert
    assert count_legacy == 1


def test_paired_mode_both_arms_same_applicability() -> None:
    """Benchmark scenario: reference_only ensures both arms have identical
    applicable_dimensions regardless of plan shape.
    """
    # Arrange
    gate = DeterministicEvalGate(grounding_applicability="reference_only")
    reference = EvalReference(grounding_required=True)

    # Arm 1: uses web_search (voluntary)
    spec_with_search = _spec(
        [_web_search_with_tool_name("search"), _llm("write")],
        [_edge("START", "search"), _edge("search", "write"), _edge("write", "END")],
    )
    ctx1 = _context(
        spec_with_search,
        {
            "search": {
                "tool": "web_search",
                "answer": "Solar facts",
                "results": [{"title": "t"}],
            },
            "write": "Recommend solar.",
        },
        reference=reference,
    )

    # Arm 2: no web_search
    spec_no_search = _spec(
        [_llm("write")],
        [_edge("START", "write"), _edge("write", "END")],
    )
    ctx2 = _context(
        spec_no_search,
        {"write": "Recommend solar."},
        reference=reference,
    )

    # Act
    result1 = gate.evaluate(ctx1)
    result2 = gate.evaluate(ctx2)

    # Assert — both arms have identical applicable_dimensions
    assert result1.applicable_dimensions == result2.applicable_dimensions
    assert "grounding" in result1.applicable_dimensions
    # Arm 1 passes grounding (has results), Arm 2 fails (no results)
    grounding1 = next(c for c in result1.components if c.dimension == "grounding")
    grounding2 = next(c for c in result2.components if c.dimension == "grounding")
    assert grounding1.verdict == "pass"
    assert grounding2.verdict == "fail"

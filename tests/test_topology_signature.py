"""Fingerprint shape-signature canaries (Slice 7, PR2).

`topology_signature` must be shape-aware (a branch DAG never collides with a
linear chain of the same kinds) yet invariant to node ids and params; the
separate `instruction_sha256` carries the params["instruction"] text. These are
asserted through the public gate output (`EvalResult.fingerprint`).
"""

from __future__ import annotations

from app.models import RunTrace
from app.models.graph_spec import EdgeSpec, GraphSpec, NodeSpec
from app.models.node_kinds import NodeKind
from app.runtime import ExecutionResult
from dynamic_subgraphs.eval import EvalContext
from dynamic_subgraphs.eval.gates import DeterministicEvalGate
from dynamic_subgraphs.usage import TokenUsage


def _llm(node_id: str, instruction: str = "do work") -> NodeSpec:
    return NodeSpec(
        id=node_id, kind=NodeKind.LLM_CALL, params={"instruction": instruction}
    )


def _edge(src: str, dst: str) -> EdgeSpec:
    return EdgeSpec(from_=src, to=dst)


def _fingerprint(nodes: list[NodeSpec], edges: list[EdgeSpec]):
    spec = GraphSpec(graph_id="g", goal="goal", nodes=nodes, edges=edges)
    state = {"values": {"out": "text"}, "events": [], "errors": []}
    result = ExecutionResult(
        state=state,  # type: ignore[arg-type]
        trace=RunTrace(run_id="r", graph_id="g", events=[]),
        ok=True,
    )
    ctx = EvalContext(
        run_id="r",
        spec=spec,
        result=result,
        prompt="p",
        status="ok",
        usage=TokenUsage(total_tokens=10),
        cost=None,
    )
    fp = DeterministicEvalGate().evaluate(ctx).fingerprint
    assert fp is not None
    return fp


def test_branch_dag_differs_from_linear_chain_of_same_kinds() -> None:
    # Arrange — same three llm nodes, different edge structure.
    linear = _fingerprint(
        [_llm("a"), _llm("b"), _llm("c")],
        [_edge("START", "a"), _edge("a", "b"), _edge("b", "c"), _edge("c", "END")],
    )
    fan = _fingerprint(
        [_llm("a"), _llm("b"), _llm("c")],
        [
            _edge("START", "a"),
            _edge("a", "b"),
            _edge("a", "c"),
            _edge("b", "END"),
            _edge("c", "END"),
        ],
    )

    # Assert — shape difference must change the signature.
    assert linear.topology_signature != fan.topology_signature


def test_topology_signature_invariant_to_ids_and_params() -> None:
    # Arrange — same 2-node linear shape, different ids and instructions.
    first = _fingerprint(
        [_llm("a", "instruction one"), _llm("b", "instruction one")],
        [_edge("START", "a"), _edge("a", "b"), _edge("b", "END")],
    )
    second = _fingerprint(
        [_llm("x", "instruction two"), _llm("y", "instruction two")],
        [_edge("START", "x"), _edge("x", "y"), _edge("y", "END")],
    )

    # Assert — topology is the same; the instruction hash separates them.
    assert first.topology_signature == second.topology_signature
    assert first.instruction_sha256 != second.instruction_sha256


def test_graph_depth_tracks_the_longest_chain() -> None:
    # Arrange / Act
    chain = _fingerprint(
        [_llm("a"), _llm("b"), _llm("c")],
        [_edge("START", "a"), _edge("a", "b"), _edge("b", "c"), _edge("c", "END")],
    )

    # Assert — a->b->c->END is the longest directed path.
    assert chain.max_depth == 3

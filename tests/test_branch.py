"""End-to-end coverage for branch: routing, trace, validator, and runtime errors."""

from __future__ import annotations

from typing import Any

import pytest

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.registry import RegistryValidationError, validate_graph_spec
from app.runtime import LangGraphExecutor


# ---------- runners used in tests ----------


def _routing_runner(decision: str | None):
    """Runner whose seed node emits the chosen decision, and identity-paths for downstream."""

    def runner(state, params):
        values = state.get("values", {}) or {}
        if "decision" in values:
            # Downstream paths just echo their instruction
            return {"result": params["instruction"]}
        return {"result": decision}

    return runner


# ---------- spec builders ----------


def _branch_spec(branches: list[str] | None = None) -> GraphSpec:
    """seed -> decide -> {happy | sad} -> END."""
    branches = branches or ["happy", "sad"]

    nodes = [
        NodeSpec(
            id="seed",
            kind=NodeKind.LLM_CALL,
            outputs=["decision"],
            params={"instruction": "emit a routing string"},
        ),
        NodeSpec(
            id="decide",
            kind=NodeKind.BRANCH,
            inputs=["decision"],
            outputs=[],
            params={"branches": branches, "decision_key": "decision"},
        ),
        NodeSpec(
            id="happy",
            kind=NodeKind.LLM_CALL,
            outputs=["happy_result"],
            params={"instruction": "we picked happy"},
        ),
        NodeSpec(
            id="sad",
            kind=NodeKind.LLM_CALL,
            outputs=["sad_result"],
            params={"instruction": "we picked sad"},
        ),
    ]
    edges = [
        EdgeSpec.model_validate({"from": "START", "to": "seed"}),
        EdgeSpec.model_validate({"from": "seed", "to": "decide"}),
        EdgeSpec.model_validate({"from": "decide", "to": "happy"}),
        EdgeSpec.model_validate({"from": "decide", "to": "sad"}),
        EdgeSpec.model_validate({"from": "happy", "to": "END"}),
        EdgeSpec.model_validate({"from": "sad", "to": "END"}),
    ]
    return GraphSpec(
        graph_id="branch-test",
        goal="branch e2e",
        nodes=nodes,
        edges=edges,
    )


# ---------- routing ----------


def test_branch_routes_to_happy_when_decision_is_happy() -> None:
    spec = _branch_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _routing_runner("happy")})

    result = executor.execute(executor.compile(validated), run_id="br-happy")

    assert result.ok is True
    assert result.state["values"]["happy_result"] == "we picked happy"
    assert "sad_result" not in result.state["values"]


def test_branch_routes_to_sad_when_decision_is_sad() -> None:
    spec = _branch_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _routing_runner("sad")})

    result = executor.execute(executor.compile(validated), run_id="br-sad")

    assert result.ok is True
    assert result.state["values"]["sad_result"] == "we picked sad"
    assert "happy_result" not in result.state["values"]


def test_branch_with_unknown_decision_halts_with_error() -> None:
    spec = _branch_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: _routing_runner("definitely_not_a_branch")}
    )

    result = executor.execute(executor.compile(validated), run_id="br-bad")

    assert result.ok is False
    assert "not in branches" in result.error
    # No downstream value was produced.
    assert "happy_result" not in result.state["values"]
    assert "sad_result" not in result.state["values"]


def test_branch_with_missing_decision_halts_with_error() -> None:
    spec = _branch_spec()
    validated = validate_graph_spec(spec)
    # Seed produces no "decision" key; downstream branch will see None.
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _routing_runner(None)})

    result = executor.execute(executor.compile(validated), run_id="br-missing")

    assert result.ok is False
    assert "not in branches" in result.error


# ---------- trace ----------


def test_branch_trace_emits_start_and_finish_with_decision_on_success() -> None:
    spec = _branch_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _routing_runner("happy")})

    result = executor.execute(executor.compile(validated), run_id="br-trace-ok")

    branch_events = [e for e in result.trace.events if e.node_id == "decide"]
    kinds = [e.kind.value for e in branch_events]
    assert kinds == ["node_start", "node_finish"]
    assert branch_events[1].data.get("decision") == "happy"


def test_branch_trace_emits_start_and_error_on_invalid_decision() -> None:
    spec = _branch_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _routing_runner("nope")})

    result = executor.execute(executor.compile(validated), run_id="br-trace-fail")

    branch_events = [e for e in result.trace.events if e.node_id == "decide"]
    kinds = [e.kind.value for e in branch_events]
    assert kinds == ["node_start", "node_error"]


# ---------- validator ----------


def test_validator_rejects_branch_target_that_is_not_a_node_id() -> None:
    spec = _branch_spec()
    # Change one branch name to something that isn't a node id
    spec.nodes[1].params["branches"] = ["happy", "ghost"]

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "branch_target_unknown" in codes


def test_validator_rejects_branch_with_missing_edge_to_declared_target() -> None:
    spec = _branch_spec()
    # Drop the edge to "sad"; "sad" is still in branches.
    spec.edges = [e for e in spec.edges if not (e.from_ == "decide" and e.to == "sad")]

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "branch_edges_mismatch" in codes


def test_validator_rejects_branch_with_extra_outgoing_edge() -> None:
    spec = _branch_spec()
    # Add an unexpected outgoing edge from the branch.
    spec.nodes.append(
        NodeSpec(
            id="extra",
            kind=NodeKind.LLM_CALL,
            outputs=["extra_out"],
            params={"instruction": "extra"},
        )
    )
    spec.edges.append(EdgeSpec.model_validate({"from": "decide", "to": "extra"}))
    spec.edges.append(EdgeSpec.model_validate({"from": "extra", "to": "END"}))

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "branch_edges_mismatch" in codes


def test_validator_rejects_branch_when_decision_key_not_in_inputs() -> None:
    spec = _branch_spec()
    # Strip the decision_key from inputs.
    spec.nodes[1].inputs = []

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "missing_decision_key_in_inputs" in codes

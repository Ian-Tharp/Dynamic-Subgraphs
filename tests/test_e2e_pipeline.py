"""End-to-end pipeline coverage: validate -> compile -> execute.

These tests run a non-trivial graph through every layer and assert the
properties that matter for downstream callers (supervisor, recorder):

- the final ExecutionResult reports ok=True when no node fails;
- every node's declared outputs land in state.values;
- the trace contains exactly one start+finish per successfully-run node;
- trace events appear in topological order (no NODE_FINISH before its
  NODE_START, no event from a downstream node before its upstream's finish);
- state propagates: nodes can read prior node outputs at execution time.

The LLM is mocked via an injected runner so this stays token-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.models import DynamicRunState, GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.recording import FileRecorder
from app.registry import validate_graph_spec
from app.runtime import LangGraphExecutor


def _mock_llm(state: DynamicRunState, params: Mapping[str, Any]) -> dict[str, Any]:
    """Stand-in for an LLM call. Echoes instruction and surfaces upstream keys."""

    upstream = sorted(state.get("values", {}).keys())
    return {"result": f"<mock>{params['instruction']}|seen={','.join(upstream)}</mock>"}


@pytest.fixture
def research_spec() -> GraphSpec:
    """Fan-out gather -> reduce concat -> downstream tool_call -> synthesize."""

    nodes = [
        NodeSpec(
            id="gather_a",
            kind=NodeKind.LLM_CALL,
            outputs=["a"],
            params={"instruction": "fetch A"},
        ),
        NodeSpec(
            id="gather_b",
            kind=NodeKind.LLM_CALL,
            outputs=["b"],
            params={"instruction": "fetch B"},
        ),
        NodeSpec(
            id="join",
            kind=NodeKind.REDUCE,
            inputs=["a", "b"],
            outputs=["joined"],
            params={
                "strategy": "concat",
                "input_keys": ["a", "b"],
                "output_key": "joined",
            },
        ),
        NodeSpec(
            id="search",
            kind=NodeKind.TOOL_CALL,
            outputs=["search_result"],
            params={"tool_name": "web_search", "args": {"q": "context"}},
        ),
        NodeSpec(
            id="synthesize",
            kind=NodeKind.LLM_CALL,
            inputs=["joined", "search_result"],
            outputs=["final"],
            params={"instruction": "recommend one"},
        ),
    ]
    edges = [
        EdgeSpec.model_validate({"from": "START", "to": "gather_a"}),
        EdgeSpec.model_validate({"from": "START", "to": "gather_b"}),
        EdgeSpec.model_validate({"from": "gather_a", "to": "join"}),
        EdgeSpec.model_validate({"from": "gather_b", "to": "join"}),
        EdgeSpec.model_validate({"from": "join", "to": "search"}),
        EdgeSpec.model_validate({"from": "search", "to": "synthesize"}),
        EdgeSpec.model_validate({"from": "synthesize", "to": "END"}),
    ]
    return GraphSpec(
        graph_id="e2e-research-001",
        goal="end-to-end pipeline coverage",
        nodes=nodes,
        edges=edges,
    )


# ---------- happy path ----------


def test_full_pipeline_succeeds_and_reports_ok(research_spec: GraphSpec) -> None:
    validated = validate_graph_spec(research_spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm})

    result = executor.execute(executor.compile(validated), run_id="e2e-run-001")

    assert result.ok is True
    assert result.error is None


def test_every_declared_output_lands_in_state_values(research_spec: GraphSpec) -> None:
    validated = validate_graph_spec(research_spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm})

    result = executor.execute(executor.compile(validated), run_id="e2e-outputs")

    declared = {out for node in validated.nodes for out in node.outputs}
    assert declared.issubset(result.state["values"].keys())


def test_synthesize_sees_all_upstream_outputs(research_spec: GraphSpec) -> None:
    validated = validate_graph_spec(research_spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm})

    result = executor.execute(executor.compile(validated), run_id="e2e-state-prop")

    final = result.state["values"]["final"]
    # synthesize ran last; its mock should have seen every prior output key
    for key in ("a", "b", "joined", "search_result"):
        assert key in final


# ---------- trace shape ----------


def test_trace_has_start_and_finish_per_successful_node(
    research_spec: GraphSpec,
) -> None:
    validated = validate_graph_spec(research_spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm})

    result = executor.execute(executor.compile(validated), run_id="e2e-trace-shape")

    node_ids = [node.id for node in validated.nodes]
    for nid in node_ids:
        node_events = [e for e in result.trace.events if e.node_id == nid]
        kinds = [e.kind.value for e in node_events]
        assert kinds == ["node_start", "node_finish"], f"node {nid} produced {kinds}"


def test_trace_events_respect_topological_ordering(
    research_spec: GraphSpec,
) -> None:
    validated = validate_graph_spec(research_spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm})

    result = executor.execute(executor.compile(validated), run_id="e2e-trace-order")

    # For each (upstream, downstream) edge, upstream's finish must precede
    # downstream's start in the linear event stream.
    finish_index = {
        e.node_id: i
        for i, e in enumerate(result.trace.events)
        if e.kind.value == "node_finish"
    }
    start_index = {
        e.node_id: i
        for i, e in enumerate(result.trace.events)
        if e.kind.value == "node_start"
    }
    for edge in validated.edges:
        if edge.from_ in finish_index and edge.to in start_index:
            assert finish_index[edge.from_] < start_index[edge.to], (
                f"{edge.from_} finished after {edge.to} started"
            )


def test_every_finish_event_carries_duration(research_spec: GraphSpec) -> None:
    validated = validate_graph_spec(research_spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm})

    result = executor.execute(executor.compile(validated), run_id="e2e-trace-time")

    finishes = [e for e in result.trace.events if e.kind.value == "node_finish"]
    assert all("duration_ms" in e.data for e in finishes)
    assert all(e.data["duration_ms"] >= 0 for e in finishes)


# ---------- failure surface ----------


def test_runner_failure_produces_failed_result_and_halts_downstream(
    research_spec: GraphSpec,
) -> None:
    def explode(state, params):
        raise RuntimeError("upstream failure")

    # Replace LLM_CALL runner so the very first gather node fails.
    validated = validate_graph_spec(research_spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: explode})

    result = executor.execute(executor.compile(validated), run_id="e2e-failure")

    assert result.ok is False
    assert result.error == "upstream failure"
    # synthesize is downstream of failing nodes and must not have run.
    finished_nodes = {
        e.node_id for e in result.trace.events if e.kind.value == "node_finish"
    }
    assert "synthesize" not in finished_nodes


# ---------- executor isolation guarantee ----------


def test_executor_rejects_cross_executor_compiled_handle(
    research_spec: GraphSpec,
) -> None:
    class FakeCompiled:
        spec = research_spec

    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm})

    with pytest.raises(TypeError, match="another executor"):
        executor.execute(FakeCompiled(), run_id="e2e-bad-handle")  # type: ignore[arg-type]


# ---------- recording integration ----------


def test_full_pipeline_including_recording_produces_run_artifacts(
    research_spec: GraphSpec,
    tmp_path: Path,
) -> None:
    validated = validate_graph_spec(research_spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm})
    recorder = FileRecorder(root_dir=tmp_path)

    result = executor.execute(executor.compile(validated), run_id="e2e-recorded")
    record = recorder.record(spec=validated, result=result, prompt="demo prompt")

    assert record.directory == tmp_path / "e2e-recorded"
    for name in (
        "spec.json",
        "trace.jsonl",
        "output.json",
        "graph.mmd",
        "summary.md",
        "prompt.md",
    ):
        assert (record.directory / name).is_file(), f"missing {name}"

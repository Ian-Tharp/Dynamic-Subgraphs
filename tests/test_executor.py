"""Validated GraphSpec -> compile -> execute integration coverage."""

from __future__ import annotations

import pytest

from app.compiler import GraphCompilationError
from app.models import GraphSpec, NodeKind, TraceEventKind
from app.registry import validate_graph_spec
from app.runtime.executor import LangGraphExecutor, _LangGraphCompiledGraph


class _RecordingGraph:
    """Test double for a compiled LangGraph: records the invoke config + state."""

    def __init__(self) -> None:
        self.config: dict | None = None
        self.state: dict | None = None

    def invoke(self, state, *, config=None):
        self.state = state
        self.config = config
        return state


def test_executor_passes_recursion_limit_to_invoke(minimal_spec: GraphSpec) -> None:
    # The executor must bound super-steps per graph with an explicit
    # recursion_limit — the steps-per-graph half of the recursion rail. The
    # limit must be at least the node budget so a legitimate graph isn't
    # strangled.
    validated = validate_graph_spec(minimal_spec)
    recorder = _RecordingGraph()
    compiled = _LangGraphCompiledGraph(spec=validated, graph=recorder)
    executor = LangGraphExecutor()

    executor.execute(compiled, run_id="run-reclimit")

    assert recorder.config is not None
    limit = recorder.config["recursion_limit"]
    assert isinstance(limit, int) and limit > 0
    assert limit >= validated.budget.max_nodes


def test_executor_seeds_initial_metadata_alongside_run_id(
    minimal_spec: GraphSpec,
) -> None:
    # The execute() seam must let callers seed extra metadata (e.g. graph_depth
    # for nested subgraphs) into the initial state WITHOUT dropping run_id.
    validated = validate_graph_spec(minimal_spec)
    recorder = _RecordingGraph()
    compiled = _LangGraphCompiledGraph(spec=validated, graph=recorder)
    executor = LangGraphExecutor()

    result = executor.execute(
        compiled, run_id="r", initial_metadata={"graph_depth": 2}
    )

    assert recorder.state is not None
    assert recorder.state["metadata"]["run_id"] == "r"
    assert recorder.state["metadata"]["graph_depth"] == 2
    assert result.state["metadata"]["graph_depth"] == 2
    assert result.state["metadata"]["run_id"] == "r"


def test_executor_runs_validated_single_node_spec(minimal_spec: GraphSpec) -> None:
    validated = validate_graph_spec(minimal_spec)
    executor = LangGraphExecutor()

    compiled = executor.compile(validated)
    result = executor.execute(compiled, run_id="run-single")

    assert result.ok is True
    assert result.error is None
    assert result.state["values"]["draft"] == "do the thing"
    assert [event.kind for event in result.trace.events] == [
        TraceEventKind.NODE_START,
        TraceEventKind.NODE_FINISH,
    ]


def test_executor_merges_parallel_state_updates(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        graph_id="parallel-root-fanout",
        nodes=[
            make_node("left", outputs=["left"], params={"instruction": "L"}),
            make_node("right", outputs=["right"], params={"instruction": "R"}),
        ],
        edges=[
            make_edge("START", "left"),
            make_edge("START", "right"),
            make_edge("left", "END"),
            make_edge("right", "END"),
        ],
    )
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor()

    result = executor.execute(executor.compile(validated), run_id="run-parallel")

    assert result.ok is True
    assert result.state["values"] == {"left": "L", "right": "R"}
    assert len(result.trace.events) == 4


def test_ledger_counts_every_node_in_linear_run(
    spec_factory, make_node, make_edge
) -> None:
    # The spend/depth ledger must increment nodes_executed once per node that
    # runs. A three-node linear chain therefore reports nodes_executed == 3.
    spec = spec_factory(
        graph_id="ledger-linear",
        nodes=[
            make_node("a", outputs=["a"], params={"instruction": "A"}),
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
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor()

    result = executor.execute(executor.compile(validated), run_id="ledger-linear")

    assert result.ok is True
    assert result.state["counters"]["nodes_executed"] == 3


def test_ledger_counts_llm_calls_consumed(minimal_spec: GraphSpec) -> None:
    # A run that executes an llm_call node must report at least one consumed
    # LLM call, derived from the registry's notion of which kinds count.
    validated = validate_graph_spec(minimal_spec)
    executor = LangGraphExecutor()

    result = executor.execute(executor.compile(validated), run_id="ledger-llm")

    assert result.ok is True
    assert result.state["counters"]["llm_calls_consumed"] >= 1


def test_ledger_sums_increments_across_parallel_fanout(
    spec_factory, make_node, make_edge
) -> None:
    # Concurrent branches each increment the counters; LangGraph merges their
    # state updates. A plain last-writer-wins field would LOSE one branch's
    # increment, so this guards the additive reducer: two parallel llm nodes
    # must sum to nodes_executed == 2 and llm_calls_consumed == 2.
    spec = spec_factory(
        graph_id="ledger-fanout",
        nodes=[
            make_node("left", outputs=["left"], params={"instruction": "L"}),
            make_node("right", outputs=["right"], params={"instruction": "R"}),
        ],
        edges=[
            make_edge("START", "left"),
            make_edge("START", "right"),
            make_edge("left", "END"),
            make_edge("right", "END"),
        ],
    )
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor()

    result = executor.execute(executor.compile(validated), run_id="ledger-fanout")

    assert result.ok is True
    assert result.state["counters"]["nodes_executed"] == 2
    assert result.state["counters"]["llm_calls_consumed"] == 2


def test_executor_runs_concat_reduce(spec_factory, make_node, make_edge) -> None:
    spec = spec_factory(
        graph_id="concat-reduce",
        nodes=[
            make_node("left", outputs=["left"], params={"instruction": "alpha"}),
            make_node("right", outputs=["right"], params={"instruction": "beta"}),
            make_node(
                "join",
                NodeKind.REDUCE,
                inputs=["left", "right"],
                outputs=["joined"],
                params={
                    "strategy": "concat",
                    "input_keys": ["left", "right"],
                    "output_key": "joined",
                },
            ),
        ],
        edges=[
            make_edge("START", "left"),
            make_edge("left", "right"),
            make_edge("right", "join"),
            make_edge("join", "END"),
        ],
    )
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor()

    result = executor.execute(executor.compile(validated), run_id="run-reduce")

    assert result.ok is True
    assert result.state["values"]["joined"] == "alpha\nbeta"


def test_executor_returns_failed_result_when_runner_raises(
    minimal_spec: GraphSpec,
) -> None:
    def explode(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("boom")

    validated = validate_graph_spec(minimal_spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: explode})

    result = executor.execute(executor.compile(validated), run_id="run-fail")

    assert result.ok is False
    assert result.error == "boom"
    assert result.state["errors"] == [
        {"node_id": "step", "message": "boom", "type": "RuntimeError"}
    ]
    assert [event.kind for event in result.trace.events] == [
        TraceEventKind.NODE_START,
        TraceEventKind.NODE_ERROR,
    ]


def test_strict_runners_rejects_placeholder_defaults(
    minimal_spec: GraphSpec,
) -> None:
    validated = validate_graph_spec(minimal_spec)
    executor = LangGraphExecutor(strict_runners=True)

    with pytest.raises(GraphCompilationError, match="llm_call"):
        executor.compile(validated)


def test_strict_runners_accepts_explicit_runner(
    minimal_spec: GraphSpec,
) -> None:
    def run_step(state, params):
        del state
        return {"result": f"explicit: {params['instruction']}"}

    validated = validate_graph_spec(minimal_spec)
    executor = LangGraphExecutor(
        strict_runners=True,
        runners={NodeKind.LLM_CALL: run_step},
    )

    result = executor.execute(executor.compile(validated), run_id="run-strict-ok")

    assert result.ok is True
    assert result.state["values"]["draft"] == "explicit: do the thing"


def test_every_registry_kind_has_an_executable_path() -> None:
    """Invariant: every `NodeKind` in the registry can be compiled.

    Replaces the older 'compiler rejects unsupported kinds' tests — those
    asserted that emit_artifact (then parallel_map, then wait_for_event)
    would fail at compile because no runner existed. As of the
    emit_artifact slice, every registry kind has either a runner in
    `default_runners()` or a handler in `COMPILER_HANDLED_KINDS`. This
    test guards against regressions: if someone adds a new `NodeKind`
    to the enum without wiring it, this fails immediately.
    """
    from app.compiler.build import COMPILER_HANDLED_KINDS
    from app.runtime.runners import default_runners

    executable = set(default_runners().keys()) | set(COMPILER_HANDLED_KINDS)
    missing = set(NodeKind) - executable

    assert (
        not missing
    ), f"these NodeKinds have no executable path: {sorted(k.value for k in missing)}"

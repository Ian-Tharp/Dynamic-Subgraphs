"""Validated GraphSpec -> compile -> execute integration coverage."""

from __future__ import annotations

import pytest

from app.compiler import GraphCompilationError
from app.models import GraphSpec, NodeKind, TraceEventKind
from app.registry import validate_graph_spec
from app.runtime.executor import LangGraphExecutor


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

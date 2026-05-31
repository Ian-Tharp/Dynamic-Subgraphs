"""Node wrapper unit tests — isolated from the compiler and executor.

The wrapper is the seam where tracing, timing, error normalization, and
output-mapping happen. These tests exercise it directly with synthetic runners.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END
from langgraph.types import Command

from app.models import NodeKind, NodeSpec, TraceEvent, TraceEventKind
from app.runtime.state import make_initial_state
from app.runtime.wrappers import make_node_wrapper, map_outputs


# ---------- success path ----------


def test_wrapper_emits_start_then_finish_with_duration() -> None:
    node = NodeSpec(
        id="ok", kind=NodeKind.LLM_CALL, outputs=["draft"], params={"instruction": "x"}
    )

    def runner(state, params):
        return {"result": "done"}

    wrapper = make_node_wrapper(node, runner)
    update = wrapper(make_initial_state())

    events = [TraceEvent.model_validate(e) for e in update["events"]]
    assert [e.kind for e in events] == [
        TraceEventKind.NODE_START,
        TraceEventKind.NODE_FINISH,
    ]
    assert all(e.node_id == "ok" for e in events)
    assert events[1].data["duration_ms"] >= 0


def test_wrapper_passes_state_to_runner() -> None:
    node = NodeSpec(
        id="reader",
        kind=NodeKind.LLM_CALL,
        outputs=["draft"],
        params={"instruction": "x"},
    )
    captured: dict[str, Any] = {}

    def runner(state, params):
        captured["state_keys"] = sorted(state.keys())
        captured["values"] = dict(state.get("values", {}))
        return {"result": "ok"}

    wrapper = make_node_wrapper(node, runner)
    seeded = make_initial_state()
    seeded["values"]["upstream"] = "data"

    wrapper(seeded)

    assert "values" in captured["state_keys"]
    assert captured["values"] == {"upstream": "data"}


# ---------- output mapping ----------


def test_wrapper_maps_result_to_single_declared_output() -> None:
    node = NodeSpec(
        id="single",
        kind=NodeKind.LLM_CALL,
        outputs=["draft"],
        params={"instruction": "x"},
    )

    def runner(state, params):
        return {"result": "the answer"}

    wrapper = make_node_wrapper(node, runner)
    update = wrapper(make_initial_state())

    assert update["values"] == {"draft": "the answer"}


def test_wrapper_maps_named_outputs_directly() -> None:
    node = NodeSpec(
        id="multi",
        kind=NodeKind.REDUCE,
        outputs=["alpha", "beta"],
        params={
            "strategy": "concat",
            "input_keys": ["a"],
            "output_key": "alpha",
        },
    )

    def runner(state, params):
        return {"alpha": 1, "beta": 2}

    wrapper = make_node_wrapper(node, runner)
    update = wrapper(make_initial_state())

    assert update["values"] == {"alpha": 1, "beta": 2}


def test_wrapper_produces_no_value_updates_when_no_outputs_declared() -> None:
    node = NodeSpec(
        id="noout",
        kind=NodeKind.LLM_CALL,
        outputs=[],
        params={"instruction": "x"},
    )

    def runner(state, params):
        return {"result": "ignored"}

    wrapper = make_node_wrapper(node, runner)
    update = wrapper(make_initial_state())

    assert update["values"] == {}


def test_map_outputs_rejects_missing_declared_output() -> None:
    node = NodeSpec(
        id="bad",
        kind=NodeKind.LLM_CALL,
        outputs=["alpha", "beta"],
        params={"instruction": "x"},
    )

    with pytest.raises(ValueError, match="did not produce declared output"):
        map_outputs(node, {"alpha": 1})


def test_map_outputs_rejects_non_mapping_return() -> None:
    node = NodeSpec(
        id="bad",
        kind=NodeKind.LLM_CALL,
        outputs=["x"],
        params={"instruction": "x"},
    )

    with pytest.raises(TypeError, match="must return a mapping"):
        map_outputs(node, "not a mapping")  # type: ignore[arg-type]


# ---------- error path ----------


def test_wrapper_normalizes_runner_exception_into_command_goto_end() -> None:
    node = NodeSpec(
        id="boom",
        kind=NodeKind.LLM_CALL,
        outputs=["draft"],
        params={"instruction": "x"},
    )

    def runner(state, params):
        raise RuntimeError("kaboom")

    wrapper = make_node_wrapper(node, runner)
    result = wrapper(make_initial_state())

    assert isinstance(result, Command)
    assert result.goto == END
    update = result.update
    assert update["errors"] == [
        {"node_id": "boom", "message": "kaboom", "type": "RuntimeError"}
    ]
    events = [TraceEvent.model_validate(e) for e in update["events"]]
    assert [e.kind for e in events] == [
        TraceEventKind.NODE_START,
        TraceEventKind.NODE_ERROR,
    ]
    assert events[1].message == "kaboom"
    assert events[1].data["duration_ms"] >= 0


def test_wrapper_records_output_mapping_failure_as_error() -> None:
    node = NodeSpec(
        id="contract",
        kind=NodeKind.LLM_CALL,
        outputs=["draft"],
        params={"instruction": "x"},
    )

    def runner(state, params):
        return {"unrelated": "value"}

    wrapper = make_node_wrapper(node, runner)
    result = wrapper(make_initial_state())

    assert isinstance(result, Command)
    assert result.goto == END
    assert result.update["errors"][0]["type"] == "ValueError"
    assert "draft" in result.update["errors"][0]["message"]

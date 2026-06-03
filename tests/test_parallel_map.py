"""End-to-end coverage for parallel_map: dispatch + Send + worker + join.

These tests build a real spec (producer -> parallel_map -> consumer),
validate it, compile it through LangGraphExecutor, and assert that:

- N items in produce N results in declared order;
- the empty list short-circuits to an empty result;
- a non-list `over` source halts with a clear TypeError;
- worker exceptions halt the graph (no downstream finish);
- the worker sees the current item under `state.values["item"]`;
- the final output_key is the ordered list (slot keys exist but are an
  implementation detail of the merge_dicts reducer).
"""

from __future__ import annotations

from typing import Any

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.registry import validate_graph_spec
from app.runtime import LangGraphExecutor

# ---------- runners used in tests ----------


def _identity_llm_runner(state, params):
    """Returns the current item formatted with the instruction."""
    item = state.get("values", {}).get("item")
    return {"result": f"{params['instruction']}::{item}"}


def _exploding_llm_runner(state, params):
    raise RuntimeError("worker boom")


# ---------- helpers ----------


def _spec_with_parallel_map(
    *,
    items_node_outputs: list[Any] | None = None,
    child_kind: str = "llm_call",
    child_params: dict[str, Any] | None = None,
) -> GraphSpec:
    """Build a producer -> parallel_map -> consumer spec.

    Producer emits a fixed list under `queries`. The parallel_map fans out
    over it. A trivial consumer reduces the results back into one key so
    we can confirm propagation past the join.
    """
    if items_node_outputs is None:
        items_node_outputs = ["a", "b", "c"]
    if child_params is None:
        child_params = {"instruction": "summarize"}

    producer = NodeSpec(
        id="seed",
        kind=NodeKind.LLM_CALL,
        outputs=["queries"],
        params={"instruction": "emit queries"},
    )
    fanout = NodeSpec(
        id="fan",
        kind=NodeKind.PARALLEL_MAP,
        inputs=["queries"],
        outputs=["fan_results"],
        params={
            "over": "queries",
            "child_kind": child_kind,
            "child_params": child_params,
        },
    )
    consumer = NodeSpec(
        id="collect",
        kind=NodeKind.REDUCE,
        inputs=["fan_results"],
        outputs=["collected"],
        params={
            "strategy": "concat",
            "input_keys": ["fan_results"],
            "output_key": "collected",
        },
    )

    edges = [
        EdgeSpec.model_validate({"from": "START", "to": "seed"}),
        EdgeSpec.model_validate({"from": "seed", "to": "fan"}),
        EdgeSpec.model_validate({"from": "fan", "to": "collect"}),
        EdgeSpec.model_validate({"from": "collect", "to": "END"}),
    ]

    return GraphSpec(
        graph_id="pm-test",
        goal="parallel_map e2e",
        nodes=[producer, fanout, consumer],
        edges=edges,
    )


def _seed_with_items(items: Any):
    """Replace the seed's runner so it produces a specific value under 'queries'."""

    def _runner(state, params):
        return {"result": items}

    return _runner


# ---------- happy path ----------


def test_parallel_map_fans_out_and_assembles_ordered_list() -> None:
    spec = _spec_with_parallel_map()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={
            NodeKind.LLM_CALL: _seed_for_node_then_identity(["a", "b", "c"]),
        }
    )

    result = executor.execute(executor.compile(validated), run_id="pm-ok")

    assert result.ok is True
    assert result.state["values"]["fan_results"] == [
        "summarize::a",
        "summarize::b",
        "summarize::c",
    ]


def test_parallel_map_preserves_order_for_larger_input() -> None:
    spec = _spec_with_parallel_map()
    validated = validate_graph_spec(spec)
    items = [f"item-{i}" for i in range(6)]
    executor = LangGraphExecutor(
        runners={
            NodeKind.LLM_CALL: _seed_for_node_then_identity(items),
        }
    )

    result = executor.execute(executor.compile(validated), run_id="pm-order")

    assert result.ok is True
    assert result.state["values"]["fan_results"] == [f"summarize::{i}" for i in items]


def test_parallel_map_emits_trace_start_and_finish_with_pm_node_id() -> None:
    spec = _spec_with_parallel_map()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: _seed_for_node_then_identity(["a", "b"])}
    )

    result = executor.execute(executor.compile(validated), run_id="pm-trace")

    pm_events = [e for e in result.trace.events if e.node_id == "fan"]
    kinds = [e.kind.value for e in pm_events]
    assert kinds == ["node_start", "node_finish"]


def test_parallel_map_passes_item_under_item_key_to_llm_child() -> None:
    spec = _spec_with_parallel_map(child_params={"instruction": "process"})
    validated = validate_graph_spec(spec)
    seen_items: list[Any] = []

    def capturing_runner(state, params):
        # Only capture on parallel_map child invocations (where 'item' is set).
        values = state.get("values", {})
        if "item" in values:
            seen_items.append(values["item"])
            return {"result": f"{params['instruction']}::{values['item']}"}
        return {"result": ["q1", "q2", "q3"]}

    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: capturing_runner})

    executor.execute(executor.compile(validated), run_id="pm-item-key")

    assert sorted(seen_items) == ["q1", "q2", "q3"]


def test_parallel_map_strips_internal_pm_keys_before_child_runner() -> None:
    """The child should not see _pm_item / _pm_index in its upstream state."""
    spec = _spec_with_parallel_map()
    validated = validate_graph_spec(spec)
    captured_state_keys: list[set[str]] = []

    def runner(state, params):
        values = state.get("values", {})
        if "item" in values:
            captured_state_keys.append(set(values.keys()))
            return {"result": values["item"]}
        return {"result": ["x"]}

    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: runner})

    executor.execute(executor.compile(validated), run_id="pm-stripped")

    assert captured_state_keys, "child runner was not invoked"
    for keys in captured_state_keys:
        assert "_pm_item" not in keys
        assert "_pm_index" not in keys
        assert "item" in keys


# ---------- empty input ----------


def test_parallel_map_with_empty_list_routes_directly_to_join() -> None:
    spec = _spec_with_parallel_map()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: _seed_for_node_then_identity([])}
    )

    result = executor.execute(executor.compile(validated), run_id="pm-empty")

    assert result.ok is True
    assert result.state["values"]["fan_results"] == []


# ---------- failure paths ----------


def test_parallel_map_decodes_json_encoded_list_strings() -> None:
    """LLM-call producers emit lists as JSON strings; parallel_map should accept them."""
    spec = _spec_with_parallel_map()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={
            NodeKind.LLM_CALL: _seed_for_node_then_identity('["a", "b", "c"]'),
        }
    )

    result = executor.execute(executor.compile(validated), run_id="pm-json-list")

    assert result.ok is True
    assert result.state["values"]["fan_results"] == [
        "summarize::a",
        "summarize::b",
        "summarize::c",
    ]


def test_parallel_map_decodes_single_key_object_wrapping_the_list() -> None:
    """LLMs often wrap requested lists as `{"<over_key>": [...]}`. Accept that too."""
    spec = _spec_with_parallel_map()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={
            NodeKind.LLM_CALL: _seed_for_node_then_identity('{"queries": ["a", "b"]}'),
        }
    )

    result = executor.execute(executor.compile(validated), run_id="pm-json-obj")

    assert result.ok is True
    assert result.state["values"]["fan_results"] == [
        "summarize::a",
        "summarize::b",
    ]


def test_parallel_map_halts_when_over_key_is_not_a_list() -> None:
    spec = _spec_with_parallel_map()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: _seed_for_node_then_identity("not a list")}
    )

    result = executor.execute(executor.compile(validated), run_id="pm-bad-source")

    assert result.ok is False
    assert "is not a list" in result.error
    assert result.state["errors"][0]["type"] == "TypeError"


def test_parallel_map_halts_when_a_worker_raises() -> None:
    spec = _spec_with_parallel_map()
    validated = validate_graph_spec(spec)

    def runner(state, params):
        values = state.get("values", {})
        if "item" in values:
            raise RuntimeError(f"boom on {values['item']}")
        return {"result": ["a", "b"]}

    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: runner})

    result = executor.execute(executor.compile(validated), run_id="pm-worker-fail")

    assert result.ok is False
    assert "parallel_map worker" in result.error
    # The collect (downstream) node must not have finished.
    finished = {e.node_id for e in result.trace.events if e.kind.value == "node_finish"}
    assert "collect" not in finished


def test_parallel_map_ledger_sums_llm_calls_across_workers() -> None:
    # Each fan-out worker makes one child llm_call and runs concurrently with
    # its siblings. LangGraph merges every worker's state update; the ledger's
    # additive reducer must SUM their increments rather than overwrite. With
    # three items, the workers alone contribute llm_calls_consumed == 3 (plus
    # the seed node), so the total must be >= 3 — a last-writer-wins counter
    # would collapse the three concurrent worker increments down to 1.
    spec = _spec_with_parallel_map()  # three items: ["a", "b", "c"]
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: _seed_for_node_then_identity(["a", "b", "c"])}
    )

    result = executor.execute(executor.compile(validated), run_id="pm-ledger")

    assert result.ok is True
    counters = result.state["counters"]
    # seed (1) + three workers (3) = 4 llm calls consumed.
    assert counters["llm_calls_consumed"] == 4
    # nodes_executed counts the seed, the three worker invocations, and the
    # reduce consumer (the dispatcher/joiner are infrastructure, not runners).
    assert counters["nodes_executed"] == 5


# ---------- helpers (lower) ----------


def _seed_for_node_then_identity(seed_value: Any):
    """Runner that emits `seed_value` for the seed node and identity-formats for workers.

    Distinguishes by whether the state contains the 'item' key (workers do; seed doesn't).
    """

    def runner(state, params):
        values = state.get("values", {})
        if "item" in values:
            return {"result": f"{params['instruction']}::{values['item']}"}
        return {"result": seed_value}

    return runner

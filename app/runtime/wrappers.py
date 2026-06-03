"""Node wrappers — the executable seam between a NodeSpec and a runner.

A wrapper is the function LangGraph actually calls per node. Its job:

- emit `NODE_START` before invoking the runner;
- time the runner with `perf_counter`;
- catch any exception, normalize it into an `errors` entry plus a `NODE_ERROR`
  trace event, and short-circuit the graph to `END`;
- on success, map the runner's raw outputs into the node's declared output keys
  and emit `NODE_FINISH` with the elapsed duration.

Runners stay ignorant of state shape, tracing, and error handling — this
wrapper is the one place that knows about all three.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END
from langgraph.types import Command

from app.models import (
    DynamicRunState,
    NodeSpec,
    TraceEvent,
    TraceEventKind,
)
from app.models.run_state import add_counters
from app.runtime.runners import NodeRunner


def make_node_wrapper(
    node: NodeSpec,
    runner: NodeRunner,
    *,
    counts_as_llm_call: bool = False,
) -> Callable[[DynamicRunState], DynamicRunState | Command]:
    """Build the LangGraph-callable wrapper for one NodeSpec + runner pair.

    `counts_as_llm_call` is the registry's per-node notion of whether running
    this node spends an LLM call (see `Registry.counts_as_llm_call_for_node`).
    It feeds the additive spend ledger in `state["counters"]`.
    """

    def _run(state: DynamicRunState) -> DynamicRunState | Command:
        started_at = datetime.now(UTC)
        started_perf = perf_counter()
        started = TraceEvent(
            kind=TraceEventKind.NODE_START,
            timestamp=started_at,
            node_id=node.id,
        )

        try:
            raw_outputs = runner(state, node.params)
            value_updates = map_outputs(node, raw_outputs)
        except GraphBubbleUp:
            # LangGraph control-flow signals (interrupt/bubble-up) are not
            # errors — let them propagate so a pause is honored, never
            # mislabeled as a NODE_ERROR by the generic handler below.
            raise
        except Exception as exc:
            duration_ms = round((perf_counter() - started_perf) * 1000, 3)
            failed = TraceEvent(
                kind=TraceEventKind.NODE_ERROR,
                node_id=node.id,
                message=str(exc),
                data={"duration_ms": duration_ms},
            )
            return Command(
                update={
                    "errors": [
                        {
                            "node_id": node.id,
                            "message": str(exc),
                            "type": type(exc).__name__,
                        }
                    ],
                    "events": [
                        started.model_dump(mode="json"),
                        failed.model_dump(mode="json"),
                    ],
                    "counters": node_counter_delta(counts_as_llm_call),
                },
                goto=END,
            )

        duration_ms = round((perf_counter() - started_perf) * 1000, 3)
        finished = TraceEvent(
            kind=TraceEventKind.NODE_FINISH,
            node_id=node.id,
            data={"duration_ms": duration_ms},
        )
        # A runner may report spend it incurred internally (e.g. spawn_subgraph
        # running a whole child graph) via the reserved `__spend__` key; fold it
        # into this node's ledger delta so nested cost rolls up to the parent.
        counters = node_counter_delta(counts_as_llm_call)
        extra_spend = (
            raw_outputs.get("__spend__") if isinstance(raw_outputs, Mapping) else None
        )
        if extra_spend:
            counters = add_counters(counters, dict(extra_spend))
        return {
            "values": value_updates,
            "events": [
                started.model_dump(mode="json"),
                finished.model_dump(mode="json"),
            ],
            "counters": counters,
        }

    return _run


def node_counter_delta(counts_as_llm_call: bool) -> dict[str, int]:
    """One node's contribution to the additive spend ledger.

    Always counts the node execution; counts an LLM call only when the node's
    kind consumes one. Returned deltas are summed across concurrent branches by
    the `add_counters` reducer on `DynamicRunState["counters"]`.
    """
    delta = {"nodes_executed": 1}
    if counts_as_llm_call:
        delta["llm_calls_consumed"] = 1
    return delta


def map_outputs(node: NodeSpec, raw_outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Map a runner's return dict into the node's declared output keys.

    - If the runner returned every declared output by name, use them directly.
    - If exactly one output is declared and the runner returned a generic
      `"result"` key, route that to the declared output.
    - Otherwise, raise — the runner did not satisfy its contract.
    """

    if not isinstance(raw_outputs, Mapping):
        raise TypeError(f"Node '{node.id}' runner must return a mapping")

    if not node.outputs:
        return {}

    direct = {
        output_key: raw_outputs[output_key]
        for output_key in node.outputs
        if output_key in raw_outputs
    }
    if len(direct) == len(node.outputs):
        return direct

    if len(node.outputs) == 1 and "result" in raw_outputs:
        return {node.outputs[0]: raw_outputs["result"]}

    missing = ", ".join(key for key in node.outputs if key not in direct)
    raise ValueError(
        f"Node '{node.id}' runner did not produce declared output(s): {missing}"
    )

"""Runtime support for `parallel_map` nodes.

A `parallel_map` `NodeSpec` cannot be expressed as a single `NodeRunner`: it
needs to emit `Send` commands for runtime fan-out and converge across the
results. The compiler expands one PM `NodeSpec` into three LangGraph nodes:

    <pm>  →  Send fan-out  →  <pm>__pm_worker  ⤵
                                                <pm>__pm_join  →  (downstream)

- **Dispatcher (`<pm>`)** reads `state.values[over]`, emits one `Send` per
  item carrying the item + index in the worker's payload. For an empty
  input list, it routes directly to the joiner with `state.values[output_key] = []`.
- **Worker (`<pm>__pm_worker`)** invokes the child runner with the current
  item exposed under `state.values["item"]` (and `args["item"]` for
  `tool_call` children). It writes its result to a per-index slot key
  (`<output_key>__<idx>`) so parallel writes don't collide under the
  `merge_dicts` reducer.
- **Joiner (`<pm>__pm_join`)** scans the slot keys, orders them by index,
  and writes `state.values[output_key]` as the assembled list.

Trace events use `node_id = <pm>.id` for both START (from the dispatcher)
and FINISH (from the joiner) so the parallel_map appears as one logical
node in the trace, with duration spanning the whole fan-out.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from langgraph.graph import END
from langgraph.types import Command, Send

from app.models import DynamicRunState, NodeKind, NodeSpec, TraceEvent, TraceEventKind
from app.runtime.runners import NodeRunner
from app.runtime.wrappers import node_counter_delta

# Per-Send payload keys (live in state.values for the worker's branch only)
# Fallback fan-out cap when metadata carries none (e.g. a direct-executor test
# that didn't go through the engine). Mirrors the ExecutionPolicy default.
_DEFAULT_MAX_FANOUT = 64

_PM_ITEM_KEY = "_pm_item"
_PM_INDEX_KEY = "_pm_index"

# Metadata key prefix the dispatcher uses to stash start time for the joiner
_PM_START_META_PREFIX = "__pm_started_perf__"


def parallel_map_worker_name(pm_id: str) -> str:
    return f"{pm_id}__pm_worker"


def parallel_map_join_name(pm_id: str) -> str:
    return f"{pm_id}__pm_join"


def _slot_prefix(output_key: str) -> str:
    return f"{output_key}__"


def make_parallel_map_dispatcher(
    node: NodeSpec,
    *,
    worker_id: str,
    join_id: str,
    child_counts_as_llm_call: bool = False,
) -> Callable[[DynamicRunState], Command]:
    """Build the LangGraph node function for the parallel_map dispatcher.

    `child_counts_as_llm_call` lets the dispatcher debit the fan-out against the
    host `max_llm_calls` budget at dispatch (fail-closed) when each worker spends
    an LLM call — the width cap alone (`max_fanout`) does not bound LLM spend.
    """

    over_key = node.params["over"]
    output_key = node.outputs[0] if node.outputs else f"{node.id}_results"

    def dispatch(state: DynamicRunState) -> Command:
        started_at = datetime.now(UTC)
        started_perf = perf_counter()
        start_event = TraceEvent(
            kind=TraceEventKind.NODE_START,
            timestamp=started_at,
            node_id=node.id,
        )

        values = state.get("values", {}) or {}
        items = values.get(over_key)

        # Upstream `llm_call` nodes return text, so a list emitted by an LLM
        # arrives here as a JSON-encoded string. Opportunistically decode two
        # common shapes so parallel_map composes naturally with llm_call
        # producers:
        #   1. A bare JSON array: `["a", "b", "c"]`
        #   2. A single-key object wrapping the list under the same key the
        #      `over` param names: `{"<over_key>": ["a", "b", "c"]}`
        if isinstance(items, str):
            stripped = items.strip()
            decoded: Any = None
            if (stripped.startswith("[") and stripped.endswith("]")) or (
                stripped.startswith("{") and stripped.endswith("}")
            ):
                try:
                    decoded = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    decoded = None
            if isinstance(decoded, list):
                items = decoded
            elif isinstance(decoded, dict) and isinstance(decoded.get(over_key), list):
                items = decoded[over_key]

        if not isinstance(items, list):
            return _halt_with_error(
                node=node,
                start_event=start_event,
                started_perf=started_perf,
                error_type="TypeError",
                error_message=(
                    f"parallel_map source key '{over_key}' is not a list "
                    f"(got {type(items).__name__})"
                ),
            )

        # Host fan-out cap: enforced *after* decode (so a JSON-string list is
        # measured by its real length), *before* any Send is emitted. Fail-closed
        # — never truncate. The cap rides in metadata (seeded by the executor from
        # the granted budget); fall back to the policy default if unseeded.
        max_fanout = int(
            (state.get("metadata", {}) or {}).get(
                "budget_max_fanout", _DEFAULT_MAX_FANOUT
            )
        )
        if len(items) > max_fanout:
            return _halt_with_error(
                node=node,
                start_event=start_event,
                started_perf=started_perf,
                error_type="FanoutExceeded",
                error_message=(
                    f"parallel_map fan-out {len(items)} exceeds the host "
                    f"max_fanout {max_fanout}"
                ),
            )

        # Host LLM-call budget: a fan-out *within* the width cap can still blow
        # the graph's max_llm_calls when every worker spends a call. Debit the
        # fan-out against the *remaining* llm budget here, before any Send, and
        # halt fail-closed — never dispatch a fan-out that would overrun the
        # host's granted call budget. Only llm-call children are charged;
        # tool_call fan-outs spend no LLM calls. The budget rides in metadata
        # (seeded by the executor); absent it (a direct-executor unit test), the
        # check is skipped, mirroring the max_fanout fallback above.
        if child_counts_as_llm_call:
            budget_raw = (state.get("metadata", {}) or {}).get("budget_max_llm_calls")
            if budget_raw is not None:
                max_llm_calls = int(budget_raw)
                consumed = int(
                    (state.get("counters", {}) or {}).get("llm_calls_consumed", 0)
                )
                if consumed + len(items) > max_llm_calls:
                    return _halt_with_error(
                        node=node,
                        start_event=start_event,
                        started_perf=started_perf,
                        error_type="LlmCallBudgetExceeded",
                        error_message=(
                            f"parallel_map fan-out of {len(items)} llm_call workers "
                            f"would consume {consumed + len(items)} LLM calls, "
                            f"exceeding the host max_llm_calls {max_llm_calls} "
                            f"({consumed} already spent)"
                        ),
                    )

        base_update: dict[str, Any] = {
            "events": [start_event.model_dump(mode="json")],
            "metadata": {f"{_PM_START_META_PREFIX}{node.id}": started_perf},
        }

        if not items:
            # Empty input: skip workers, hand an empty list to the joiner.
            return Command(
                update={
                    **base_update,
                    "values": {output_key: []},
                },
                goto=join_id,
            )

        sends: list[Send] = []
        for idx, item in enumerate(items):
            payload_values = {
                **values,
                _PM_ITEM_KEY: item,
                _PM_INDEX_KEY: idx,
            }
            payload: DynamicRunState = {
                "values": payload_values,
                "metadata": dict(state.get("metadata", {}) or {}),
            }
            sends.append(Send(worker_id, payload))

        return Command(update=base_update, goto=sends)

    return dispatch


def make_parallel_map_worker(
    node: NodeSpec,
    *,
    child_kind: NodeKind,
    child_runner: NodeRunner,
    child_counts_as_llm_call: bool = False,
) -> Callable[[DynamicRunState], Any]:
    """Build the LangGraph node function for one parallel_map worker invocation.

    Each worker invocation is one node execution that runs the child kind once.
    `child_counts_as_llm_call` reflects whether that child consumes an LLM call;
    both feed the additive `state["counters"]` ledger. Because workers run
    concurrently and LangGraph merges their updates via the `add_counters`
    reducer, per-worker deltas SUM across the whole fan-out.
    """

    output_key = node.outputs[0] if node.outputs else f"{node.id}_results"
    child_params_template: dict[str, Any] = dict(node.params.get("child_params", {}))

    def worker(state: DynamicRunState) -> Any:
        values = state.get("values", {}) or {}
        item = values.get(_PM_ITEM_KEY)
        idx = values.get(_PM_INDEX_KEY, 0)

        # Build a clean view of state.values for the child runner: strip
        # internal `_pm_*` keys and expose the current item under "item".
        child_values = {
            key: value for key, value in values.items() if not key.startswith("_pm_")
        }
        child_values["item"] = item
        child_state: DynamicRunState = dict(state)
        child_state["values"] = child_values

        # Inject item into params based on child kind.
        if child_kind == NodeKind.TOOL_CALL:
            local_params = {
                **child_params_template,
                "args": {
                    **dict(child_params_template.get("args", {})),
                    "item": item,
                },
            }
        else:
            # For llm_call, the runner reads `item` from state.values directly.
            local_params = dict(child_params_template)

        # Count the worker execution (and its child LLM call) even if the child
        # raises: the spend was incurred. Summed across workers by add_counters.
        counter_delta = node_counter_delta(child_counts_as_llm_call)

        try:
            raw = child_runner(child_state, local_params)
        except Exception as exc:
            return Command(
                update={
                    "errors": [
                        {
                            "node_id": node.id,
                            "message": (f"parallel_map worker [{idx}] failed: {exc}"),
                            "type": type(exc).__name__,
                        }
                    ],
                    "counters": counter_delta,
                },
                goto=END,
            )

        if not isinstance(raw, dict):
            return Command(
                update={
                    "errors": [
                        {
                            "node_id": node.id,
                            "message": (
                                f"parallel_map worker [{idx}] returned non-dict: "
                                f"{type(raw).__name__}"
                            ),
                            "type": "TypeError",
                        }
                    ],
                    "counters": counter_delta,
                },
                goto=END,
            )

        # Use the runner's `result` key by convention; fall back to whole dict.
        result = raw.get("result", raw)
        slot_key = f"{_slot_prefix(output_key)}{idx}"
        return {"values": {slot_key: result}, "counters": counter_delta}

    return worker


def make_parallel_map_joiner(
    node: NodeSpec,
) -> Callable[[DynamicRunState], dict[str, Any]]:
    """Build the LangGraph node function that converges parallel results."""

    output_key = node.outputs[0] if node.outputs else f"{node.id}_results"
    slot_prefix = _slot_prefix(output_key)
    start_meta_key = f"{_PM_START_META_PREFIX}{node.id}"

    def join(state: DynamicRunState) -> Any:
        values = state.get("values", {}) or {}

        metadata = state.get("metadata", {}) or {}
        started_perf = metadata.get(start_meta_key)
        duration_ms: float
        if isinstance(started_perf, (int, float)):
            duration_ms = round((perf_counter() - started_perf) * 1000, 3)
        else:
            duration_ms = 0.0

        # If any worker for *this* parallel_map failed, halt the whole graph.
        # Workers return Command(goto=END) but LangGraph still fires the
        # static worker->join edge per Send branch, so the join is the only
        # place we can reliably stop downstream nodes from running.
        worker_errors = [
            err
            for err in state.get("errors", []) or []
            if isinstance(err, dict) and err.get("node_id") == node.id
        ]
        if worker_errors:
            error_event = TraceEvent(
                kind=TraceEventKind.NODE_ERROR,
                node_id=node.id,
                message=worker_errors[0].get("message", "parallel_map worker failed"),
                data={"duration_ms": duration_ms},
            )
            return Command(
                update={"events": [error_event.model_dump(mode="json")]},
                goto=END,
            )

        slots: dict[int, Any] = {}
        for key, value in values.items():
            if not key.startswith(slot_prefix):
                continue
            suffix = key[len(slot_prefix) :]
            try:
                idx = int(suffix)
            except ValueError:
                continue
            slots[idx] = value

        ordered = [slots[i] for i in sorted(slots.keys())]

        finish_event = TraceEvent(
            kind=TraceEventKind.NODE_FINISH,
            node_id=node.id,
            data={"duration_ms": duration_ms},
        )

        return {
            "values": {output_key: ordered},
            "events": [finish_event.model_dump(mode="json")],
        }

    return join


def _halt_with_error(
    *,
    node: NodeSpec,
    start_event: TraceEvent,
    started_perf: float,
    error_type: str,
    error_message: str,
) -> Command:
    duration_ms = round((perf_counter() - started_perf) * 1000, 3)
    error_event = TraceEvent(
        kind=TraceEventKind.NODE_ERROR,
        node_id=node.id,
        message=error_message,
        data={"duration_ms": duration_ms},
    )
    return Command(
        update={
            "errors": [
                {
                    "node_id": node.id,
                    "message": error_message,
                    "type": error_type,
                }
            ],
            "events": [
                start_event.model_dump(mode="json"),
                error_event.model_dump(mode="json"),
            ],
        },
        goto=END,
    )

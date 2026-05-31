"""Runtime support for `branch` nodes — conditional routing on state.

A `branch` `NodeSpec` is compiler-handled. The compiler emits:

- A **passthrough node** at `branch.id` that emits trace events. It reads
  `state.values[decision_key]` and, when the decision is missing or
  unknown, attaches a structured error so the router below halts cleanly.
- A **conditional edge** out of that node that routes based on the same
  decision key. Valid decisions route to the matching name (which must be
  a node id in the spec); invalid decisions route to `END`.

The branch node does NOT use `Command(goto=...)`. Routing lives entirely
in the conditional edge so the compiler can express the targets
declaratively. The earlier gotcha — static edges firing alongside a goto
— doesn't apply here because the branch has no static outgoing edges.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from langgraph.graph import END

from app.models import DynamicRunState, NodeSpec, TraceEvent, TraceEventKind


def make_branch_node(
    node: NodeSpec,
) -> Callable[[DynamicRunState], dict[str, Any]]:
    """Build the LangGraph node function for a branch (passthrough + trace)."""

    branches: set[str] = set(node.params["branches"])
    decision_key: str = node.params["decision_key"]

    def branch_fn(state: DynamicRunState) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        started_perf = perf_counter()
        start_event = TraceEvent(
            kind=TraceEventKind.NODE_START,
            timestamp=started_at,
            node_id=node.id,
        )

        decision = (state.get("values") or {}).get(decision_key)

        if not isinstance(decision, str) or decision not in branches:
            duration_ms = round((perf_counter() - started_perf) * 1000, 3)
            message = (
                f"branch decision_key '{decision_key}' has value "
                f"{decision!r} which is not in branches {sorted(branches)}"
            )
            error_event = TraceEvent(
                kind=TraceEventKind.NODE_ERROR,
                node_id=node.id,
                message=message,
                data={"duration_ms": duration_ms},
            )
            return {
                "errors": [
                    {
                        "node_id": node.id,
                        "message": message,
                        "type": "ValueError",
                    }
                ],
                "events": [
                    start_event.model_dump(mode="json"),
                    error_event.model_dump(mode="json"),
                ],
            }

        duration_ms = round((perf_counter() - started_perf) * 1000, 3)
        finish_event = TraceEvent(
            kind=TraceEventKind.NODE_FINISH,
            node_id=node.id,
            data={"duration_ms": duration_ms, "decision": decision},
        )
        return {
            "events": [
                start_event.model_dump(mode="json"),
                finish_event.model_dump(mode="json"),
            ],
        }

    return branch_fn


def make_branch_router(node: NodeSpec) -> Callable[[DynamicRunState], str]:
    """Conditional-edge router: pick a target name from state, or END on mismatch."""

    branches: set[str] = set(node.params["branches"])
    decision_key: str = node.params["decision_key"]

    def router(state: DynamicRunState) -> str:
        decision = (state.get("values") or {}).get(decision_key)
        if isinstance(decision, str) and decision in branches:
            return decision
        return END

    return router

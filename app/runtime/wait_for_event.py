"""Runtime support for `wait_for_event` nodes — durable pause/resume.

A `wait_for_event` `NodeSpec` is compiler-handled. The node calls
`langgraph.types.interrupt(...)` which, on first execution, raises
`GraphInterrupt`. LangGraph catches that internally, persists state to the
configured checkpointer, and returns control to the caller. Execution can
later be resumed by invoking the graph with `Command(resume=value)`; on
the resume pass, `interrupt()` *returns* the resume value rather than
raising.

Semantics:

- The interrupt payload broadcast to the caller is
  `{"event_type": ..., "node_id": ..., "timeout_seconds": ...}`. The
  caller (typically the supervisor) reads this to know *what kind* of
  event the workflow is waiting for.
- On resume, the value passed to `Command(resume=...)` becomes the node's
  output under `outputs[0]`.
- Trace `NODE_START` and `NODE_FINISH` events are both emitted **on the
  resume pass** (timestamps will be near-identical). The original
  first-pass execution doesn't get a chance to emit state updates before
  `interrupt()` raises, so we accept that the pause itself is recorded
  by the supervisor as a separate trace event rather than by this node.

Requirements:

- The executor must compile this graph with a `checkpointer` — the
  compiler enforces this at compile time so missing-checkpointer bugs
  surface immediately, not at runtime when the first interrupt fires.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langgraph.types import interrupt

from app.models import DynamicRunState, NodeSpec, TraceEvent, TraceEventKind


def make_wait_for_event_node(
    node: NodeSpec,
) -> Callable[[DynamicRunState], dict[str, Any]]:
    """Build the LangGraph node function for one `wait_for_event` node."""

    event_type = str(node.params["event_type"])
    timeout_seconds = node.params.get("timeout_seconds")
    output_key = node.outputs[0] if node.outputs else f"{node.id}_event"

    def wait_fn(state: DynamicRunState) -> dict[str, Any]:
        # On first pass: interrupt() raises GraphInterrupt; LangGraph captures
        # it, persists state via the checkpointer, and returns to the caller.
        # On resume: interrupt() returns the value passed via Command(resume=...).
        payload = interrupt(
            {
                "event_type": event_type,
                "node_id": node.id,
                "timeout_seconds": timeout_seconds,
            }
        )

        # We only reach here on the resume pass. Emit START + FINISH together
        # so the trace shows the wait completed; the supervisor/recorder are
        # responsible for marking the pause itself.
        now = datetime.now(UTC)
        start_event = TraceEvent(
            kind=TraceEventKind.NODE_START,
            timestamp=now,
            node_id=node.id,
        )
        finish_event = TraceEvent(
            kind=TraceEventKind.NODE_FINISH,
            timestamp=now,
            node_id=node.id,
            data={"event_type": event_type},
        )

        return {
            "values": {output_key: payload},
            "events": [
                start_event.model_dump(mode="json"),
                finish_event.model_dump(mode="json"),
            ],
        }

    return wait_fn

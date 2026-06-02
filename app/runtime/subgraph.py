"""spawn_subgraph: a node that synthesizes and runs a bounded child graph.

The recursion primitive. A `spawn_subgraph` node hands a sub-goal to a
`ChildLauncher`, which plans -> validates -> compiles -> executes a fresh child
GraphSpec on its own state envelope, one level deeper. Depth is capped so a nest
can never exceed the validated ceiling; durable pause (`wait_for_event`) is
refused inside children (synchronous children only in v1).

The registry vocabulary stays frozen: spawning a graph adds zero capability —
the child is re-validated through the same registry, allowlists, and cycle
checks as any other graph. Only *composition* goes fractal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.models import NodeKind
from app.registry.validator import MAX_DEPTH_CEILING, validate_graph_spec
from app.runtime.runners import NodeRunner

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from app.models import GraphSpec
    from app.registry import Registry
    from app.runtime.executor import GraphExecutor


class SubgraphError(RuntimeError):
    """Base for spawn_subgraph failures — each halts the parent node fail-closed."""


class SubgraphDepthExceeded(SubgraphError):
    """A spawn would exceed the nesting depth ceiling."""


class SubgraphChildFailed(SubgraphError):
    """The child graph ran but ended in a non-ok status."""


class SubgraphContainsWaitForEvent(SubgraphError):
    """A child graph tried to use durable pause, which is banned in children."""


@dataclass(frozen=True)
class ChildResult:
    """What a `ChildLauncher` hands back to the spawn_subgraph runner."""

    values: dict[str, Any]
    counters: dict[str, int] = field(default_factory=dict)
    status: str = "ok"
    response: str = ""


class ChildLauncher:
    """Structural type: plans and runs one child graph one level deeper.

    Any callable with this shape works; defined as a class only for the type
    hint. See `make_child_launcher` for the production implementation.
    """

    def __call__(
        self,
        sub_goal: str,
        *,
        run_id: str,
        graph_depth: int,
        parent_run_id: str,
        inputs: dict[str, Any],
    ) -> ChildResult:  # pragma: no cover - structural
        ...


def build_spawn_subgraph_runner(
    launcher: ChildLauncher,
    *,
    depth_ceiling: int = MAX_DEPTH_CEILING,
) -> NodeRunner:
    """Build the NodeRunner for `spawn_subgraph` nodes.

    Reads the current `graph_depth` from state metadata and refuses to spawn
    past `depth_ceiling` (fail-closed, before any child runs). Seeds the child
    with depth+1, the parent run id, and the `inputs_from` slice of parent
    values, then hands the child's produced values back under `result` (which
    the wrapper maps to the node's declared output, if any).
    """

    def _runner(state: "Mapping[str, Any]", params: "Mapping[str, Any]") -> dict[str, Any]:
        metadata = state.get("metadata", {}) or {}
        depth = int(metadata.get("graph_depth", 0))
        if depth + 1 > depth_ceiling:
            raise SubgraphDepthExceeded(
                f"spawn_subgraph at depth {depth} would exceed the nesting "
                f"ceiling {depth_ceiling}"
            )

        parent_run_id = str(metadata.get("run_id", "root"))
        name = str(params["name"])
        sub_goal = str(params["sub_goal"])
        inputs_from = list(params.get("inputs_from", []) or [])
        parent_values = state.get("values", {}) or {}
        inputs = {key: parent_values[key] for key in inputs_from if key in parent_values}

        child_run_id = f"{parent_run_id}__sg_{name}"
        result = launcher(
            sub_goal,
            run_id=child_run_id,
            graph_depth=depth + 1,
            parent_run_id=parent_run_id,
            inputs=inputs,
        )
        if result.status != "ok":
            raise SubgraphChildFailed(
                f"child subgraph {child_run_id!r} ended with status "
                f"{result.status!r}: {result.response}"
            )
        return {"result": dict(result.values)}

    return _runner


def make_child_launcher(
    *,
    planner: "Callable[[str], GraphSpec]",
    executor: "GraphExecutor",
    registry: "Registry | None" = None,
) -> ChildLauncher:
    """Build a `ChildLauncher` from a planner + executor.

    Reuses the `execute(initial_metadata=...)` seam to seed the child's
    depth/lineage. The child runs synchronously on its own fresh state envelope;
    its produced values and spend counters come back in the `ChildResult`.
    `wait_for_event` anywhere in the child is refused before compile — nested
    durable pause is a later, larger effort.
    """

    def launch(
        sub_goal: str,
        *,
        run_id: str,
        graph_depth: int,
        parent_run_id: str,
        inputs: dict[str, Any],
    ) -> ChildResult:
        validated = validate_graph_spec(planner(sub_goal), registry)
        if any(node.kind == NodeKind.WAIT_FOR_EVENT for node in validated.nodes):
            raise SubgraphContainsWaitForEvent(
                f"child subgraph {run_id!r} contains a wait_for_event node; "
                "durable pause is not allowed inside a nested subgraph "
                "(synchronous children only in v1)"
            )

        result = executor.execute(
            executor.compile(validated),
            run_id=run_id,
            inputs=inputs or None,
            initial_metadata={
                "graph_depth": graph_depth,
                "parent_run_id": parent_run_id,
            },
        )
        state = result.state or {}
        return ChildResult(
            values=dict(state.get("values", {}) or {}),
            counters=dict(state.get("counters", {}) or {}),
            status="ok" if result.ok else "failed",
            response=result.error or "",
        )

    return launch

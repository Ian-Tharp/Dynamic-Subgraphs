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

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.models import NodeKind
from app.registry.validator import MAX_DEPTH_CEILING, validate_graph_spec
from app.runtime.budget_ledger import BudgetLedger
from app.runtime.runners import NodeRunner

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from app.models import GraphSpec
    from app.policy import ExecutionPolicy
    from app.recording import Recorder
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
        max_llm_calls: int | None = None,
        max_nodes: int | None = None,
        max_depth: int | None = None,
        wall_deadline: float | None = None,
        replay_of: str | None = None,
    ) -> ChildResult:  # pragma: no cover - structural
        ...


def build_spawn_subgraph_runner(
    launcher: ChildLauncher,
    *,
    depth_ceiling: int = MAX_DEPTH_CEILING,
    ledger_registry: Mapping[str, BudgetLedger] | None = None,
) -> NodeRunner:
    """Build the NodeRunner for `spawn_subgraph` nodes.

    Reads the current `graph_depth` from state metadata and refuses to spawn
    past `depth_ceiling` (fail-closed, before any child runs). Seeds the child
    with depth+1, the parent run id, and the `inputs_from` slice of parent
    values, then hands the child's produced values back under `result` (which
    the wrapper maps to the node's declared output, if any).
    """

    def _runner(state: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
        metadata = state.get("metadata", {}) or {}
        depth = int(metadata.get("graph_depth", 0))
        # The effective ceiling is the tighter of the host policy's max_depth and
        # the hard rail, so a host that sets max_depth=2 actually caps nesting at
        # 2 (not the rail of 3).
        effective_ceiling = min(
            int(metadata.get("budget_max_depth", depth_ceiling)), depth_ceiling
        )
        if depth + 1 > effective_ceiling:
            raise SubgraphDepthExceeded(
                f"spawn_subgraph at depth {depth} would exceed the nesting "
                f"ceiling {effective_ceiling}"
            )

        parent_run_id = str(metadata.get("run_id", "root"))
        name = str(params["name"])
        sub_goal = str(params["sub_goal"])
        inputs_from = list(params.get("inputs_from", []) or [])
        parent_values = state.get("values", {}) or {}
        inputs = {
            key: parent_values[key] for key in inputs_from if key in parent_values
        }

        # Compose the child's budget against the parent's REMAINING node and LLM
        # allowance so a nest can't outspend the root. The per-graph `BudgetLedger`
        # (owned by the executor, looked up by this graph's run_id) reserves the
        # remaining allowance under a lock, so CONCURRENT spawn_subgraph siblings in
        # one superstep can't each claim the full remaining — the documented TOCTOU.
        # Whatever the child doesn't spend is refunded in the `finally`.
        counters = state.get("counters", {}) or {}
        budgets: dict[str, int | None] = {
            "llm_calls": metadata.get("budget_max_llm_calls"),
            "nodes": metadata.get("budget_max_nodes"),
        }
        consumed = {
            "llm_calls": int(counters.get("llm_calls_consumed", 0)),
            "nodes": int(counters.get("nodes_executed", 0)),
        }
        ledger = (
            ledger_registry.get(parent_run_id) if ledger_registry is not None else None
        )
        if ledger is not None:
            grant = ledger.reserve(budgets=budgets, consumed=consumed)
        else:
            # No ledger seeded (a bare-executor unit test): plain remaining — a
            # single spawn with no concurrent-sibling enforcement.
            grant = {
                key: (None if cap is None else max(0, cap - consumed[key]))
                for key, cap in budgets.items()
            }

        child_run_id = f"{parent_run_id}__sg_{name}"
        # If this run is itself a replay (metadata carries the ORIGINAL parent's
        # run id), pin the child to the original's recorded spec by deriving the
        # original child run id; the launcher loads it instead of re-planning.
        replay_of = metadata.get("replay_of")
        original_child_run_id = f"{replay_of}__sg_{name}" if replay_of else None

        spent = {"llm_calls": 0, "nodes": 0}
        try:
            result = launcher(
                sub_goal,
                run_id=child_run_id,
                graph_depth=depth + 1,
                parent_run_id=parent_run_id,
                inputs=inputs,
                max_llm_calls=grant["llm_calls"],
                max_nodes=grant["nodes"],
                max_depth=effective_ceiling,
                wall_deadline=metadata.get("wall_deadline_monotonic"),
                replay_of=original_child_run_id,
            )
            spent = {
                "llm_calls": int(result.counters.get("llm_calls_consumed", 0)),
                "nodes": int(result.counters.get("nodes_executed", 0)),
            }
            if result.status != "ok":
                raise SubgraphChildFailed(
                    f"child subgraph {child_run_id!r} ended with status "
                    f"{result.status!r}: {result.response}"
                )
            # Hand the child's values back under `result` (mapped to the node's
            # declared output) and roll the child's ACTUAL spend up to the parent
            # ledger via the reserved `__spend__` key (see make_node_wrapper). The
            # spawn node makes no *direct* LLM call, so all llm spend here is the
            # child's — no floor double-count.
            return {"result": dict(result.values), "__spend__": dict(result.counters)}
        finally:
            # Return the unused portion of the reservation so an under-spending or
            # failed child frees budget for later siblings (spent stays 0 if the
            # launcher raised before the child ran — a full refund).
            if ledger is not None:
                ledger.refund(reserved=grant, spent=spent)

    return _runner


def make_child_launcher(
    *,
    planner: Callable[[str], GraphSpec],
    executor: GraphExecutor,
    registry: Registry | None = None,
    recorder: Recorder | None = None,
    policy: ExecutionPolicy | None = None,
) -> ChildLauncher:
    """Build a `ChildLauncher` from a planner + executor.

    Reuses the `execute(initial_metadata=...)` seam to seed the child's
    depth/lineage. The child runs synchronously on its own fresh state envelope;
    its produced values and spend counters come back in the `ChildResult`.
    `wait_for_event` anywhere in the child is refused before compile — nested
    durable pause is a later, larger effort.

    When a `recorder` is given, each child run is persisted as its own run dir
    (`runs/<child_run_id>/`) — child spec + trace + output, with parent_run_id
    and graph_depth in its metadata — so a nested run is fully inspectable and
    its synthesized spec is durable. Recording failures never break the child
    run (they're swallowed), matching the supervisor's chain-record behavior.
    """

    def launch(
        sub_goal: str,
        *,
        run_id: str,
        graph_depth: int,
        parent_run_id: str,
        inputs: dict[str, Any],
        max_llm_calls: int | None = None,
        max_nodes: int | None = None,
        max_depth: int | None = None,
        wall_deadline: float | None = None,
        replay_of: str | None = None,
    ) -> ChildResult:
        # Fail closed before planning if the parent has no remaining node budget
        # — a nest out of budget must not get a fresh allowance (or burn a
        # planner call).
        if max_nodes is not None and max_nodes < 1:
            raise SubgraphChildFailed(
                f"child subgraph {run_id!r} has no remaining node budget "
                f"(the nest is out of budget)"
            )

        # Replay determinism: when replaying, reuse the originally recorded child
        # spec (`replay_of` is its run id) instead of re-planning, so the nested
        # shape reproduces. Fall back to planning if it can't be loaded.
        spec = None
        if replay_of is not None and recorder is not None:
            try:
                spec = recorder.load_validated_spec(replay_of)
            except Exception:
                spec = None
        if spec is None:
            spec = planner(sub_goal)
        # Cap the child's budget to the parent's REMAINING allowance (nodes, LLM
        # calls, depth) before validation; an oversized child then fails closed
        # against the host policy rather than overspending the nest.
        budget_update: dict[str, Any] = {}
        if max_llm_calls is not None:
            budget_update["max_llm_calls"] = min(
                spec.budget.max_llm_calls, max_llm_calls
            )
        if max_nodes is not None:
            budget_update["max_nodes"] = min(spec.budget.max_nodes, max_nodes)
        if max_depth is not None:
            budget_update["max_depth"] = min(spec.budget.max_depth, max_depth)
        if budget_update:
            spec = spec.model_copy(
                update={"budget": spec.budget.model_copy(update=budget_update)}
            )
        # The host policy applies to the child too: a nested graph can never
        # grant itself a larger budget than the host (the parent's remaining LLM
        # allowance is already folded into spec.budget above).
        validated = validate_graph_spec(spec, registry, policy=policy)
        if any(node.kind == NodeKind.WAIT_FOR_EVENT for node in validated.nodes):
            raise SubgraphContainsWaitForEvent(
                f"child subgraph {run_id!r} contains a wait_for_event node; "
                "durable pause is not allowed inside a nested subgraph "
                "(synchronous children only in v1)"
            )

        child_metadata: dict[str, Any] = {
            "graph_depth": graph_depth,
            "parent_run_id": parent_run_id,
        }
        if wall_deadline is not None:
            # One deadline spans the whole nest: the child reuses the parent's
            # absolute deadline rather than minting a fresh max_wall_seconds.
            child_metadata["wall_deadline_monotonic"] = wall_deadline
        if replay_of is not None:
            # Propagate replay so a grandchild also pins to its recorded spec.
            child_metadata["replay_of"] = replay_of
        result = executor.execute(
            executor.compile(validated),
            run_id=run_id,
            inputs=inputs or None,
            initial_metadata=child_metadata,
        )
        if recorder is not None:
            with contextlib.suppress(Exception):
                recorder.record(
                    spec=validated, result=result, prompt=sub_goal, overwrite=True
                )
        state = result.state or {}
        return ChildResult(
            values=dict(state.get("values", {}) or {}),
            counters=dict(state.get("counters", {}) or {}),
            status="ok" if result.ok else "failed",
            response=result.error or "",
        )

    return launch

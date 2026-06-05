"""LangGraphExecutor — compiles a validated GraphSpec and runs it, one level deep.

The runtime seam that keeps LangGraph behind a stable interface: `compile()` turns a
validated spec into a transient graph, and `execute()` runs it on a fresh state
envelope with the host-granted budget, an absolute wall-clock deadline, and a
per-run `BudgetLedger` (so concurrent `spawn_subgraph` siblings can't overspend).
Each invocation runs on an isolated daemon thread so an inner `interrupt()` stays
contained and a hung runner can't block the caller past the deadline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.models import DynamicRunState, GraphSpec, NodeKind, RunTrace, TraceEvent
from app.registry import Registry
from app.runtime.budget_ledger import BudgetLedger
from app.runtime.runners import NodeRunner
from app.runtime.state import make_initial_state

# NOTE: `app.compiler.build` is imported lazily inside compile() to break a
# package-init cycle. runtime/__init__.py exports the executor; the compiler
# imports runtime.wrappers; importing wrappers triggers runtime/__init__,
# which would re-enter the executor while the compiler is still loading.
# Resolving the import at call time avoids the import-time cycle without
# loosening the architectural seam.

# Steps-per-graph rail: every execution carries an explicit recursion_limit so
# runaway graphs (and, once spawn_subgraph lands, runaway nests) hit a ceiling.
_DEFAULT_RECURSION_LIMIT = 25
_STEPS_PER_NODE = 4


class _DeadlineExceeded(Exception):
    """Raised when a graph execution outruns its wall-clock budget."""


def _run_graph_isolated(
    graph: Any,
    payload: Any,
    *,
    config: dict[str, Any],
    inspect_config: dict[str, Any] | None,
    hard_timeout_s: float | None = None,
) -> tuple[Any, bool, list[Any]]:
    """Invoke a compiled graph in a dedicated thread; return state + pause info.

    LangGraph (>=1.2.x) routes an `interrupt()` to the *ambient* Pregel
    execution it discovers via a context variable. The Supervisor runs each
    transient graph inside a node of its own static graph, and a spawn_subgraph
    node runs a child graph inside a parent transient graph — so a naive
    `graph.invoke(...)` lets an inner interrupt bubble up and halt the outer
    graph (it returns with no result, stuck "pending"). A fresh thread does not
    inherit that context variable, so the interrupt stays contained to the
    graph that raised it.

    The pending-interrupt inspection (`get_state`) must run in the **same**
    isolated thread: called from the outer Pregel context it reports no
    interrupts, so the pause would be lost. Returns
    ``(state, paused, interrupt_payloads)``. Exceptions are re-raised on the
    calling thread, preserving error behavior.
    """

    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            state = graph.invoke(payload, config=config)
            if inspect_config is not None:
                paused, payloads = _inspect_pending_interrupts(graph, inspect_config)
            else:
                paused, payloads = False, []
            box["ok"] = (state, paused, payloads)
        except BaseException as exc:  # re-raised on the caller thread below
            box["error"] = exc

    # Daemon thread + bounded join: a runner that hangs (e.g. an LLM HTTP call
    # with no client timeout) no longer blocks the caller forever and dies with
    # the process. On timeout the caller regains control; the abandoned thread
    # may keep running until its in-flight call returns (documented limitation —
    # wall-clock bounds *latency*, the llm-call budget bounds *spend*).
    thread = threading.Thread(target=_run, name="ds-graph-invoke", daemon=True)
    thread.start()
    thread.join(hard_timeout_s)
    if thread.is_alive():
        raise _DeadlineExceeded()
    if "error" in box:
        raise box["error"]
    return box["ok"]


class CompiledGraph(Protocol):
    """Opaque handle for a compiled transient graph."""

    spec: GraphSpec


@dataclass(frozen=True)
class ExecutionResult:
    state: DynamicRunState
    trace: RunTrace
    ok: bool
    error: str | None = None
    paused: bool = False
    interrupt_payloads: list[Any] = field(default_factory=list)


class GraphExecutor(Protocol):
    def compile(self, spec: GraphSpec) -> CompiledGraph: ...

    def execute(
        self,
        compiled: CompiledGraph,
        *,
        inputs: dict[str, Any] | None = None,
        run_id: str,
        initial_metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult: ...

    def resume(
        self,
        compiled: CompiledGraph,
        *,
        run_id: str,
        event: Any,
    ) -> ExecutionResult: ...


@dataclass(frozen=True)
class _LangGraphCompiledGraph:
    spec: GraphSpec
    graph: Any


class LangGraphExecutor:
    """Concrete executor; LangGraph stays behind this runtime seam."""

    def __init__(
        self,
        *,
        registry: Registry | None = None,
        runners: dict[NodeKind, NodeRunner] | None = None,
        checkpointer: Any = None,
        strict_runners: bool = False,
    ) -> None:
        self._registry = registry or Registry()
        self._runners = runners
        self._checkpointer = checkpointer
        self._strict_runners = strict_runners
        # Per-graph budget ledgers keyed by run_id, kept OFF the serialized state
        # (the checkpointer msgpack-serializes state, which a live object breaks) —
        # the spawn_subgraph runner looks one up by run_id. Populated per execute(),
        # removed when that graph finishes. Concurrent runs (the API runs jobs on a
        # thread pool) mutate this map from different threads, so insert/pop are
        # guarded by a lock; reads (`.get`) are single GIL-atomic ops, and run_ids
        # are globally unique (JobStore.create rejects collisions), so a live entry
        # is never overwritten.
        self._ledgers: dict[str, BudgetLedger] = {}
        self._ledgers_lock = threading.Lock()

    @property
    def ledgers(self) -> dict[str, BudgetLedger]:
        """run_id -> BudgetLedger registry the spawn_subgraph runner reads from."""
        return self._ledgers

    def compile(self, spec: GraphSpec) -> CompiledGraph:
        from app.compiler.build import build_graph
        from app.compiler.errors import GraphCompilationError

        has_wait = any(n.kind == NodeKind.WAIT_FOR_EVENT for n in spec.nodes)
        if has_wait and self._checkpointer is None:
            raise GraphCompilationError(
                "Graph contains wait_for_event nodes but the executor has no "
                "checkpointer. Construct LangGraphExecutor(checkpointer=...) — "
                "use langgraph.checkpoint.memory.MemorySaver() for in-process / "
                "tests or langgraph.checkpoint.sqlite.SqliteSaver(...) for "
                "durable resume across process restarts."
            )

        builder = build_graph(
            spec,
            registry=self._registry,
            runners=self._runners,
            use_default_runners=not self._strict_runners,
        )
        compile_kwargs: dict[str, Any] = {}
        if self._checkpointer is not None:
            compile_kwargs["checkpointer"] = self._checkpointer
        return _LangGraphCompiledGraph(
            spec=spec,
            graph=builder.compile(**compile_kwargs),
        )

    def execute(
        self,
        compiled: CompiledGraph,
        *,
        inputs: dict[str, Any] | None = None,
        run_id: str,
        initial_metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        concrete = _coerce_compiled_graph(compiled)
        config = self._invoke_config(concrete.spec, run_id)
        md_in = initial_metadata or {}
        # An absolute monotonic deadline bounds the whole run. A nested child
        # reuses the parent's deadline (propagated in metadata) so the deadline
        # spans the tree; the root mints it from its granted max_wall_seconds.
        deadline = md_in.get("wall_deadline_monotonic")
        if deadline is None:
            deadline = time.monotonic() + concrete.spec.budget.max_wall_seconds
        # Seed caller metadata (e.g. graph_depth for nested subgraphs) while
        # keeping run_id and this graph's own granted budget authoritative —
        # set after the caller's dict so a caller can't override them. The
        # budget lets a spawn_subgraph node clamp a child (budget_max_llm_calls
        # - llm_calls_consumed) and parallel_map cap its fan-out.
        metadata = {
            **md_in,
            "budget_max_nodes": concrete.spec.budget.max_nodes,
            "budget_max_llm_calls": concrete.spec.budget.max_llm_calls,
            "budget_max_depth": concrete.spec.budget.max_depth,
            "budget_max_fanout": concrete.spec.budget.max_fanout,
            "wall_deadline_monotonic": deadline,
            "run_id": run_id,
        }
        # A fresh budget ledger for this graph, registered by run_id so the
        # spawn_subgraph runner can reserve against ONE pool across concurrent
        # siblings (the TOCTOU fix) — kept off the serialized state. A nested child
        # gets its own ledger from its own execute(); removed when this graph ends.
        with self._ledgers_lock:
            self._ledgers[run_id] = BudgetLedger()
        # Only the checkpointer path can be inspected for interrupts (it carries
        # the thread_id); without one there's no persisted state to query.
        inspect_config = config if self._checkpointer is not None else None
        try:
            try:
                state, paused, payloads = _run_graph_isolated(
                    concrete.graph,
                    make_initial_state(inputs=inputs, metadata=metadata),
                    config=config,
                    inspect_config=inspect_config,
                    hard_timeout_s=max(0.0, deadline - time.monotonic()),
                )
            except _DeadlineExceeded:
                return self._timeout_result(concrete, run_id)
            return self._build_result(
                concrete, state, run_id, paused=paused, interrupt_payloads=payloads
            )
        finally:
            with self._ledgers_lock:
                self._ledgers.pop(run_id, None)

    def _timeout_result(
        self, concrete: _LangGraphCompiledGraph, run_id: str
    ) -> ExecutionResult:
        trace = RunTrace(run_id=run_id, graph_id=concrete.spec.graph_id, events=[])
        return ExecutionResult(
            state={},  # type: ignore[typeddict-item]
            trace=trace,
            ok=False,
            error=(
                f"wall-clock deadline ({concrete.spec.budget.max_wall_seconds}s) "
                f"exceeded"
            ),
            paused=False,
            interrupt_payloads=[],
        )

    def resume(
        self,
        compiled: CompiledGraph,
        *,
        run_id: str,
        event: Any,
    ) -> ExecutionResult:
        from langgraph.types import Command

        if self._checkpointer is None:
            raise RuntimeError(
                "Cannot resume: executor has no checkpointer. The run's state "
                "was never persisted. Construct LangGraphExecutor with a "
                "checkpointer if you intend to support resume."
            )

        concrete = _coerce_compiled_graph(compiled)
        config = self._invoke_config(concrete.spec, run_id)
        state, paused, payloads = _run_graph_isolated(
            concrete.graph,
            Command(resume=event),
            config=config,
            inspect_config=config,
        )
        return self._build_result(
            concrete, state, run_id, paused=paused, interrupt_payloads=payloads
        )

    def _invoke_config(self, spec: GraphSpec, run_id: str) -> dict[str, Any]:
        """Build the LangGraph invoke config: an explicit recursion_limit plus,
        when a checkpointer is wired, the thread_id needed for resume."""
        config: dict[str, Any] = {"recursion_limit": _recursion_limit_for(spec)}
        if self._checkpointer is not None:
            config.update(_config_for(run_id))
        return config

    def _build_result(
        self,
        concrete: _LangGraphCompiledGraph,
        state: DynamicRunState,
        run_id: str,
        *,
        paused: bool,
        interrupt_payloads: list[Any],
    ) -> ExecutionResult:
        trace = RunTrace(
            run_id=run_id,
            graph_id=concrete.spec.graph_id,
            events=[
                TraceEvent.model_validate(event) for event in state.get("events", [])
            ],
        )
        errors = state.get("errors") or []
        first_error = errors[0] if errors else None

        return ExecutionResult(
            state=state,
            trace=trace,
            ok=(first_error is None) and not paused,
            error=first_error["message"] if first_error else None,
            paused=paused,
            interrupt_payloads=interrupt_payloads,
        )


def _coerce_compiled_graph(compiled: CompiledGraph) -> _LangGraphCompiledGraph:
    if not isinstance(compiled, _LangGraphCompiledGraph):
        raise TypeError(
            "LangGraphExecutor received a graph compiled by another executor"
        )
    return compiled


def _config_for(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def _recursion_limit_for(spec: GraphSpec) -> int:
    """Explicit super-step cap for one graph's execution.

    Scales with the node budget — a generous factor over node count covers
    fan-out and branch re-entry — but never drops below LangGraph's default, so
    small graphs keep ample headroom while runaway graphs still hit a ceiling.
    Bounds steps per graph, not nesting depth (that is the depth budget's job).
    """
    return max(_DEFAULT_RECURSION_LIMIT, spec.budget.max_nodes * _STEPS_PER_NODE)


def _inspect_pending_interrupts(
    graph: Any, config: dict[str, Any]
) -> tuple[bool, list[Any]]:
    """Ask the compiled graph whether it has pending interrupts after invoke.

    Returns (paused, payloads). `payloads` lists the values passed to each
    pending `interrupt(...)` call — these tell the caller what kind of event
    each paused node is waiting for.
    """
    try:
        snapshot = graph.get_state(config)
    except Exception:
        return False, []

    payloads: list[Any] = []
    # LangGraph >=1.2.x exposes pending interrupts at the top level of the
    # state snapshot; older versions hung them off each pending task. Prefer
    # the top-level field, then fall back for backward compatibility.
    for itr in getattr(snapshot, "interrupts", None) or ():
        payloads.append(getattr(itr, "value", itr))
    if not payloads:
        for task in getattr(snapshot, "tasks", None) or ():
            for itr in getattr(task, "interrupts", None) or ():
                payloads.append(getattr(itr, "value", itr))

    return bool(payloads), payloads

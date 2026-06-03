from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.models import DynamicRunState, GraphSpec, NodeKind, RunTrace, TraceEvent
from app.registry import Registry
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


def _run_graph_isolated(
    graph: Any,
    payload: Any,
    *,
    config: dict[str, Any],
    inspect_config: dict[str, Any] | None,
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
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            box["error"] = exc

    thread = threading.Thread(target=_run, name="ds-graph-invoke")
    thread.start()
    thread.join()
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
        # Seed caller-supplied metadata (e.g. graph_depth for nested subgraphs)
        # while keeping run_id and this graph's own budget authoritative — both
        # are set after the caller's dict so a caller can't override them. The
        # budget lets a spawn_subgraph node clamp a child to the parent's
        # remaining (budget_max_llm_calls - llm_calls_consumed).
        metadata = {
            **(initial_metadata or {}),
            "budget_max_llm_calls": concrete.spec.budget.max_llm_calls,
            "run_id": run_id,
        }
        # Only the checkpointer path can be inspected for interrupts (it carries
        # the thread_id); without one there's no persisted state to query.
        inspect_config = config if self._checkpointer is not None else None
        state, paused, payloads = _run_graph_isolated(
            concrete.graph,
            make_initial_state(inputs=inputs, metadata=metadata),
            config=config,
            inspect_config=inspect_config,
        )
        return self._build_result(
            concrete, state, run_id, paused=paused, interrupt_payloads=payloads
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

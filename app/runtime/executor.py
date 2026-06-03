from __future__ import annotations

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
        state = concrete.graph.invoke(
            make_initial_state(inputs=inputs, metadata=metadata),
            config=config,
        )
        # Only the checkpointer path needs the config back for interrupt
        # inspection (it carries the thread_id); without one there's no state to
        # query, so don't hand it down.
        result_config = config if self._checkpointer is not None else None
        return self._build_result(concrete, state, run_id, config=result_config)

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
        state = concrete.graph.invoke(Command(resume=event), config=config)
        return self._build_result(concrete, state, run_id, config=config)

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
        config: dict[str, Any] | None,
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

        paused = False
        interrupt_payloads: list[Any] = []
        if config is not None:
            paused, interrupt_payloads = _inspect_pending_interrupts(
                concrete.graph, config
            )

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
    tasks = getattr(snapshot, "tasks", None) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or ()
        for itr in interrupts:
            value = getattr(itr, "value", itr)
            payloads.append(value)

    return bool(payloads), payloads

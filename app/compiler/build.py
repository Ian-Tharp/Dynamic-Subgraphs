"""GraphSpec -> StateGraph wiring.

This layer's only job is to turn a validated `GraphSpec` into a configured
LangGraph `StateGraph`. It owns no execution logic for ordinary nodes — node
wrappers, tracing, and error normalization live in `app.runtime`.

`parallel_map` is the one kind the compiler handles specially: it can't be
expressed as a single `NodeRunner`, so the compiler expands each `parallel_map`
NodeSpec into three LangGraph nodes (dispatcher / worker / join) wired with an
internal worker->join edge. Outgoing spec edges from the `parallel_map` are
rewritten to originate from its join node.
"""

from __future__ import annotations

from collections.abc import Mapping

from langgraph.graph import END, START, StateGraph

from app.compiler.errors import GraphCompilationError
from app.models import DynamicRunState, GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import SPECIAL_NODE_END, SPECIAL_NODE_START
from app.registry import Registry
from app.runtime.branch import make_branch_node, make_branch_router
from app.runtime.parallel_map import (
    make_parallel_map_dispatcher,
    make_parallel_map_joiner,
    make_parallel_map_worker,
    parallel_map_join_name,
    parallel_map_worker_name,
)
from app.runtime.runners import NodeRunner, default_runners
from app.runtime.wait_for_event import make_wait_for_event_node
from app.runtime.wrappers import make_node_wrapper

# Kinds whose execution is implemented inside the compiler/runtime infrastructure
# itself rather than as a `NodeRunner` in `default_runners()`.
COMPILER_HANDLED_KINDS: frozenset[NodeKind] = frozenset(
    {NodeKind.PARALLEL_MAP, NodeKind.BRANCH, NodeKind.WAIT_FOR_EVENT}
)


def build_graph(
    spec: GraphSpec,
    *,
    registry: Registry | None = None,
    runners: Mapping[NodeKind, NodeRunner] | None = None,
    use_default_runners: bool = True,
) -> StateGraph:
    """Build a transient LangGraph StateGraph from an already-validated GraphSpec.

    The validator is the trust boundary: this function assumes every NodeSpec is
    registry-approved and every edge is well-formed. It only checks that each
    node kind is executable in the current slice (either via a registered runner
    or via compiler-native handling).
    """

    reg = registry or Registry()
    available_runners = dict(default_runners()) if use_default_runners else {}
    if runners:
        available_runners.update(runners)

    unsupported = [
        node.kind.value
        for node in spec.nodes
        if reg.get_definition(node.kind) is None
        or (
            node.kind not in available_runners
            and node.kind not in COMPILER_HANDLED_KINDS
        )
    ]
    if unsupported:
        kinds = ", ".join(sorted(set(unsupported)))
        raise GraphCompilationError(
            f"Node kind(s) not executable in the current compiler slice: {kinds}"
        )

    graph = StateGraph(DynamicRunState)
    pm_ids = {n.id for n in spec.nodes if n.kind == NodeKind.PARALLEL_MAP}
    branch_ids = {n.id for n in spec.nodes if n.kind == NodeKind.BRANCH}
    branch_nodes = [n for n in spec.nodes if n.kind == NodeKind.BRANCH]

    for node in spec.nodes:
        if node.kind == NodeKind.PARALLEL_MAP:
            _add_parallel_map(graph, node, available_runners, reg)
        elif node.kind == NodeKind.BRANCH:
            graph.add_node(node.id, make_branch_node(node))
        elif node.kind == NodeKind.WAIT_FOR_EVENT:
            graph.add_node(node.id, make_wait_for_event_node(node))
        else:
            graph.add_node(
                node.id,
                make_node_wrapper(
                    node,
                    available_runners[node.kind],
                    counts_as_llm_call=reg.counts_as_llm_call_for_node(node),
                ),
            )

    for edge in spec.edges:
        source = START if edge.from_ == SPECIAL_NODE_START else edge.from_
        target = END if edge.to == SPECIAL_NODE_END else edge.to
        # The parallel_map's "exit" is its join node, not the dispatcher.
        if source in pm_ids:
            source = parallel_map_join_name(source)
        # Branch routing is via add_conditional_edges below; skip static edges
        # from branch nodes so they don't fire in parallel with the router.
        if source in branch_ids:
            continue
        graph.add_edge(source, target)

    # Conditional edges for branches must be registered after their source
    # node is added; do this in a second pass.
    for node in branch_nodes:
        router = make_branch_router(node)
        allowed_targets = [*list(node.params["branches"]), END]
        graph.add_conditional_edges(node.id, router, allowed_targets)

    return graph


def _add_parallel_map(
    graph: StateGraph,
    node: NodeSpec,
    available_runners: Mapping[NodeKind, NodeRunner],
    registry: Registry,
) -> None:
    child_kind_str = node.params.get("child_kind")
    if not isinstance(child_kind_str, str):
        raise GraphCompilationError(
            f"parallel_map node '{node.id}' is missing 'child_kind' in params"
        )
    try:
        child_kind = NodeKind(child_kind_str)
    except ValueError as exc:
        raise GraphCompilationError(
            f"parallel_map node '{node.id}' has unknown child_kind '{child_kind_str}'"
        ) from exc

    child_runner = available_runners.get(child_kind)
    if child_runner is None:
        raise GraphCompilationError(
            f"parallel_map node '{node.id}' references child kind "
            f"'{child_kind_str}' which has no runner registered"
        )

    worker_id = parallel_map_worker_name(node.id)
    join_id = parallel_map_join_name(node.id)

    graph.add_node(
        node.id,
        make_parallel_map_dispatcher(node, worker_id=worker_id, join_id=join_id),
    )
    child_definition = registry.get_definition(child_kind)
    child_counts_as_llm_call = bool(
        child_definition and child_definition.counts_as_llm_call
    )
    graph.add_node(
        worker_id,
        make_parallel_map_worker(
            node,
            child_kind=child_kind,
            child_runner=child_runner,
            child_counts_as_llm_call=child_counts_as_llm_call,
        ),
    )
    graph.add_node(join_id, make_parallel_map_joiner(node))
    graph.add_edge(worker_id, join_id)

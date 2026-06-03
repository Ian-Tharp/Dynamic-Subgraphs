from __future__ import annotations

from collections import defaultdict, deque

from app.models.graph_spec import (
    GRAPH_SPEC_SCHEMA_VERSION,
    SPECIAL_NODE_END,
    SPECIAL_NODE_START,
    EdgeSpec,
    GraphSpec,
    NodeSpec,
)
from app.models.node_kinds import NodeKind
from app.registry.errors import RegistryValidationError, RegistryValidationIssue
from app.registry.registry import Registry

# Hard ceiling on a graph's declared recursion depth. Nested subgraphs may never
# request more than this, so the recursion rail holds regardless of what a
# planner emits. Matches the canonical design's shallow-depth guidance.
MAX_DEPTH_CEILING = 3


def validate_graph_spec(spec: GraphSpec, registry: Registry | None = None) -> GraphSpec:
    """
    Validate a GraphSpec against registry policy and graph topology (v1 §9).
    Returns a copy with normalized node params on success.
    """
    reg = registry or Registry()
    issues: list[RegistryValidationIssue] = []

    if spec.schema_version != GRAPH_SPEC_SCHEMA_VERSION:
        issues.append(
            RegistryValidationIssue(
                code="unsupported_schema",
                message=f"Unsupported schema_version {spec.schema_version}",
            )
        )

    node_ids = [n.id for n in spec.nodes]
    if len(node_ids) != len(set(node_ids)):
        seen: set[str] = set()
        for nid in node_ids:
            if nid in seen:
                issues.append(
                    RegistryValidationIssue(
                        code="duplicate_node_id",
                        message=f"Duplicate node id '{nid}'",
                        node_id=nid,
                    )
                )
            seen.add(nid)
        if issues:
            raise RegistryValidationError(issues)

    normalized_nodes: list[NodeSpec] = []
    for node in spec.nodes:
        try:
            normalized_nodes.append(reg.validate_node_params(node))
        except RegistryValidationError as exc:
            issues.extend(exc.issues)

    if issues:
        raise RegistryValidationError(issues)

    if len(normalized_nodes) > spec.budget.max_nodes:
        issues.append(
            RegistryValidationIssue(
                code="budget_exceeded",
                message=f"Node count {len(normalized_nodes)} exceeds max_nodes {spec.budget.max_nodes}",
            )
        )

    llm_count = reg.count_llm_calls(normalized_nodes)
    if llm_count > spec.budget.max_llm_calls:
        issues.append(
            RegistryValidationIssue(
                code="budget_exceeded",
                message=f"LLM call count {llm_count} exceeds max_llm_calls {spec.budget.max_llm_calls}",
            )
        )

    if spec.budget.max_depth > MAX_DEPTH_CEILING:
        issues.append(
            RegistryValidationIssue(
                code="depth_ceiling_exceeded",
                message=(
                    f"max_depth {spec.budget.max_depth} exceeds the hard ceiling "
                    f"{MAX_DEPTH_CEILING}"
                ),
                field="budget.max_depth",
            )
        )

    issues.extend(_validate_edges(spec, {n.id for n in normalized_nodes}))
    issues.extend(_validate_inputs(normalized_nodes, spec.edges))
    issues.extend(_detect_cycles(spec, {n.id for n in normalized_nodes}))
    issues.extend(_validate_branches(normalized_nodes, spec.edges))

    if issues:
        raise RegistryValidationError(issues)

    return spec.model_copy(update={"nodes": normalized_nodes})


def _validate_edges(
    spec: GraphSpec, node_ids: set[str]
) -> list[RegistryValidationIssue]:
    issues: list[RegistryValidationIssue] = []
    adj: dict[str, list[str]] = defaultdict(list)

    for edge in spec.edges:
        src, dst = edge.from_, edge.to
        if src != SPECIAL_NODE_START and src not in node_ids:
            issues.append(
                RegistryValidationIssue(
                    code="dangling_edge",
                    message=f"Edge source '{src}' does not exist",
                    field="edges",
                )
            )
        if dst != SPECIAL_NODE_END and dst not in node_ids:
            issues.append(
                RegistryValidationIssue(
                    code="dangling_edge",
                    message=f"Edge target '{dst}' does not exist",
                    field="edges",
                )
            )
        if src != SPECIAL_NODE_END and dst != SPECIAL_NODE_START:
            adj[src].append(dst)

    if not _has_path(adj, SPECIAL_NODE_START, SPECIAL_NODE_END, node_ids):
        issues.append(
            RegistryValidationIssue(
                code="no_start_to_end_path",
                message="No path from START to END",
            )
        )

    unreachable = node_ids - _reachable_from_start(adj, node_ids)
    if unreachable:
        for nid in sorted(unreachable):
            issues.append(
                RegistryValidationIssue(
                    code="unreachable_node",
                    message=f"Node '{nid}' is not reachable from START",
                    node_id=nid,
                )
            )

    return issues


def _has_path(
    adj: dict[str, list[str]],
    start: str,
    end: str,
    node_ids: set[str],
) -> bool:
    """BFS from START; END may be reached directly or via planner nodes."""
    queue: deque[str] = deque(adj.get(start, []))
    visited: set[str] = set()
    while queue:
        current = queue.popleft()
        if current == end:
            return True
        if current in visited:
            continue
        visited.add(current)
        if current in node_ids:
            queue.extend(adj.get(current, []))
    return False


def _reachable_from_start(adj: dict[str, list[str]], node_ids: set[str]) -> set[str]:
    reachable: set[str] = set()
    queue = deque(adj.get(SPECIAL_NODE_START, []))
    while queue:
        current = queue.popleft()
        if current == SPECIAL_NODE_END or current in reachable:
            continue
        if current in node_ids:
            reachable.add(current)
        queue.extend(adj.get(current, []))
    return reachable


def _validate_inputs(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec],
) -> list[RegistryValidationIssue]:
    """Flag inputs that are not produced by any upstream node in topological order."""
    available: set[str] = set()
    issues: list[RegistryValidationIssue] = []

    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}
    for edge in edges:
        src, dst = edge.from_, edge.to
        if src in in_degree and dst in in_degree:
            adj[src].append(dst)
            in_degree[dst] += 1

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    node_by_id = {n.id: n for n in nodes}

    while queue:
        nid = queue.popleft()
        node = node_by_id[nid]
        for inp in node.inputs:
            if inp not in available:
                issues.append(
                    RegistryValidationIssue(
                        code="missing_upstream_input",
                        message=f"Input '{inp}' is not available before node '{nid}'",
                        node_id=nid,
                        field=f"inputs.{inp}",
                    )
                )
        for out_key in node.outputs:
            available.add(out_key)
        for nxt in adj[nid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    return issues


def _validate_branches(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec],
) -> list[RegistryValidationIssue]:
    """Branch-specific topology checks (run after node-level param normalization).

    For each `branch` node:
    - every name in `params.branches` must be an existing node id;
    - the spec's outgoing edges from the branch must exactly match `params.branches`;
    - `params.decision_key` must be declared in the branch's `inputs`.
    """
    issues: list[RegistryValidationIssue] = []
    branch_nodes = [n for n in nodes if n.kind == NodeKind.BRANCH]
    if not branch_nodes:
        return issues

    all_node_ids = {n.id for n in nodes}
    out_targets_by_source: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        out_targets_by_source[edge.from_].add(edge.to)

    for branch in branch_nodes:
        params = branch.params
        declared_branches = list(params.get("branches", []))
        branches_set = set(declared_branches)
        decision_key = params.get("decision_key", "")

        # (a) Every branch name must refer to a real node id.
        for bname in declared_branches:
            if bname not in all_node_ids:
                issues.append(
                    RegistryValidationIssue(
                        code="branch_target_unknown",
                        message=(
                            f"branch '{branch.id}' references non-existent target "
                            f"'{bname}'"
                        ),
                        node_id=branch.id,
                        field="params.branches",
                    )
                )

        # (b) Outgoing edges must exactly match the declared branches.
        actual_targets = out_targets_by_source.get(branch.id, set())
        missing_edges = branches_set - actual_targets
        extra_edges = actual_targets - branches_set
        if missing_edges or extra_edges:
            parts: list[str] = []
            if missing_edges:
                parts.append(f"missing edges to {sorted(missing_edges)}")
            if extra_edges:
                parts.append(f"unexpected edges to {sorted(extra_edges)}")
            issues.append(
                RegistryValidationIssue(
                    code="branch_edges_mismatch",
                    message=(
                        f"branch '{branch.id}' outgoing edges don't match its "
                        f"branches: " + "; ".join(parts)
                    ),
                    node_id=branch.id,
                    field="edges",
                )
            )

        # (c) decision_key must be declared in inputs so input-provenance catches
        #     a missing upstream producer.
        if decision_key and decision_key not in branch.inputs:
            issues.append(
                RegistryValidationIssue(
                    code="missing_decision_key_in_inputs",
                    message=(
                        f"branch '{branch.id}' has decision_key '{decision_key}' "
                        f"but does not declare it in inputs"
                    ),
                    node_id=branch.id,
                    field="inputs",
                )
            )

    return issues


def _detect_cycles(
    spec: GraphSpec, node_ids: set[str]
) -> list[RegistryValidationIssue]:
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in spec.edges:
        src, dst = edge.from_, edge.to
        if src in node_ids and dst in node_ids:
            adj[src].append(dst)

    visiting: set[str] = set()
    visited: set[str] = set()
    issues: list[RegistryValidationIssue] = []

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in adj[node]:
            if dfs(nxt):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    for nid in node_ids:
        if dfs(nid):
            issues.append(
                RegistryValidationIssue(
                    code="cycle_detected",
                    message="Graph contains a cycle among planner nodes",
                )
            )
            break

    return issues

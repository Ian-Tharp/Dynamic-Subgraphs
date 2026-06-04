from __future__ import annotations

import re
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
from app.policy import MAX_DEPTH_CEILING, ExecutionPolicy, resolve_effective_budget
from app.registry.errors import RegistryValidationError, RegistryValidationIssue
from app.registry.registry import Registry

# `MAX_DEPTH_CEILING` is defined in `app.policy` (host-governance source of
# truth) and re-exported here so `from app.registry.validator import
# MAX_DEPTH_CEILING` keeps working for callers like `subgraph.py`.
__all__ = ["MAX_DEPTH_CEILING", "validate_graph_spec"]


# Node ids must be simple identifiers and must not collide with reserved names:
# the graph terminals (START/END) or the parallel_map internal marker, since the
# compiler derives internal node names like "<pm>__pm_worker" / "<pm>__pm_join"
# from user-supplied ids. A planner-chosen id like "x__pm_join" could otherwise
# shadow a generated node.
_VALID_NODE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_RESERVED_ID_MARKER = "__pm_"


def validate_graph_spec(
    spec: GraphSpec,
    registry: Registry | None = None,
    *,
    policy: ExecutionPolicy | None = None,
) -> GraphSpec:
    """
    Validate a GraphSpec against registry policy and graph topology (v1 §9).

    Numeric budgets are enforced against the **host-owned** `ExecutionPolicy`,
    not the planner's self-declared budget: the effective limit per field is
    `min(host ceiling, planner request)`. The resolved budget is stamped onto
    the returned spec so the executor's recursion rail, the nested-subgraph
    clamp, recording, and the API all read the *granted* limits. With no policy
    passed, the default `ExecutionPolicy()` applies (its ceilings match the
    historical `GraphBudget` defaults).

    Returns a copy with normalized node params and the granted budget.
    """
    reg = registry or Registry()
    effective = resolve_effective_budget(
        policy or ExecutionPolicy(),
        spec.budget,
        registry_tools=reg.tools,
        registry_subagents=reg.subagents,
        registry_kinds=reg.allowed_kinds(),
    )
    # Enforce the host allow-set. Narrow the registry to the effective
    # (host ∩ registry) tools/subagents so the existing per-node allowlist gates
    # reject anything the policy forbids; node kinds are checked separately
    # below against effective.allowed_node_kinds. (When the policy adds no
    # narrowing, effective.* equals the registry's own sets and this is a no-op.)
    if (
        effective.allowed_tools != reg.tools
        or effective.allowed_subagents != reg.subagents
    ):
        reg = Registry(
            tools=effective.allowed_tools,
            subagents=effective.allowed_subagents,
        )
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

    issues.extend(_validate_node_ids(spec.nodes))

    normalized_nodes: list[NodeSpec] = []
    for node in spec.nodes:
        try:
            normalized_nodes.append(reg.validate_node_params(node))
        except RegistryValidationError as exc:
            issues.extend(exc.issues)

    if issues:
        raise RegistryValidationError(issues)

    if len(normalized_nodes) > effective.max_nodes:
        issues.append(
            RegistryValidationIssue(
                code="budget_exceeded",
                message=(
                    f"Node count {len(normalized_nodes)} exceeds the effective "
                    f"max_nodes {effective.max_nodes} "
                    f"(host policy ∧ requested {spec.budget.max_nodes})"
                ),
            )
        )

    llm_count = reg.count_llm_calls(normalized_nodes)
    if llm_count > effective.max_llm_calls:
        issues.append(
            RegistryValidationIssue(
                code="budget_exceeded",
                message=(
                    f"LLM call count {llm_count} exceeds the effective "
                    f"max_llm_calls {effective.max_llm_calls} "
                    f"(host policy ∧ requested {spec.budget.max_llm_calls})"
                ),
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

    # Host policy may restrict which node kinds a plan may use (intersection with
    # the registry). When the policy adds no narrowing this set is the full
    # registry vocabulary, so nothing is rejected.
    for node in normalized_nodes:
        if node.kind not in effective.allowed_node_kinds:
            issues.append(
                RegistryValidationIssue(
                    code="node_kind_not_allowed",
                    message=(
                        f"Node kind '{node.kind.value}' is not allowed by the "
                        f"host policy"
                    ),
                    node_id=node.id,
                    field="kind",
                )
            )

    issues.extend(_validate_edges(spec, {n.id for n in normalized_nodes}))
    issues.extend(_validate_inputs(normalized_nodes, spec.edges))
    issues.extend(_detect_cycles(spec, {n.id for n in normalized_nodes}))
    issues.extend(_validate_branches(normalized_nodes, spec.edges))

    if issues:
        raise RegistryValidationError(issues)

    # Stamp the granted budget so the executor's recursion rail, the nested
    # clamp, recording, and the API all read the host-enforced limits — there is
    # no separate copy of spec.budget for anything downstream to trust.
    granted = spec.budget.model_copy(
        update={
            "max_nodes": effective.max_nodes,
            "max_llm_calls": effective.max_llm_calls,
            "max_depth": effective.max_depth,
            "max_wall_seconds": effective.max_wall_seconds,
            "max_fanout": effective.max_fanout,
        }
    )
    return spec.model_copy(update={"nodes": normalized_nodes, "budget": granted})


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


def _validate_node_ids(nodes: list[NodeSpec]) -> list[RegistryValidationIssue]:
    """Reject reserved / collision-prone / malformed node ids.

    `NodeSpec.id` is an open string, but the compiler treats ids as the system's
    addressing scheme: it reserves `START`/`END` for the graph terminals and
    derives internal parallel_map node names by suffixing user ids (so an id
    containing `__pm_` could shadow a generated node). Ids are also used in
    Mermaid, file paths, and edge wiring, so they must be simple identifiers.
    """
    issues: list[RegistryValidationIssue] = []
    for node in nodes:
        nid = node.id
        if nid in (SPECIAL_NODE_START, SPECIAL_NODE_END):
            issues.append(
                RegistryValidationIssue(
                    code="reserved_node_id",
                    message=f"Node id '{nid}' is reserved for the graph terminals",
                    node_id=nid,
                    field="id",
                )
            )
        elif _RESERVED_ID_MARKER in nid:
            issues.append(
                RegistryValidationIssue(
                    code="reserved_node_id",
                    message=(
                        f"Node id '{nid}' contains the reserved marker "
                        f"'{_RESERVED_ID_MARKER}' used for parallel_map internal nodes"
                    ),
                    node_id=nid,
                    field="id",
                )
            )
        elif not _VALID_NODE_ID.match(nid):
            issues.append(
                RegistryValidationIssue(
                    code="invalid_node_id",
                    message=(
                        f"Node id '{nid}' must be a simple identifier matching "
                        f"[A-Za-z0-9_-]+ (no spaces or special characters)"
                    ),
                    node_id=nid,
                    field="id",
                )
            )
    return issues


def _validate_inputs(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec],
) -> list[RegistryValidationIssue]:
    """Flag inputs not produced by any *ancestor* of the consuming node.

    Availability is per-node: an input is satisfiable only if some node on a
    directed path *into* this node produces it. Using each node's real ancestors
    (not a single set of all previously-visited outputs) means a value produced
    only by a sibling branch — with no edge guaranteeing it runs first — is
    correctly rejected rather than accepted by topological luck.
    """
    issues: list[RegistryValidationIssue] = []
    node_by_id = {n.id: n for n in nodes}

    predecessors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.from_ in node_by_id and edge.to in node_by_id:
            predecessors[edge.to].append(edge.from_)

    def _ancestor_outputs(node_id: str) -> set[str]:
        produced: set[str] = set()
        seen: set[str] = set()
        stack = list(predecessors[node_id])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            produced.update(node_by_id[current].outputs)
            stack.extend(predecessors[current])
        return produced

    for node in nodes:
        available = _ancestor_outputs(node.id)
        for inp in node.inputs:
            if inp not in available:
                issues.append(
                    RegistryValidationIssue(
                        code="missing_upstream_input",
                        message=(
                            f"Input '{inp}' is not produced by any ancestor of "
                            f"node '{node.id}'"
                        ),
                        node_id=node.id,
                        field=f"inputs.{inp}",
                    )
                )

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

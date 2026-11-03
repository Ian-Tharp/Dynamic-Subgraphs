"""Plan-level policy gate: check a synthesized GraphSpec before it compiles.

Call-level governance (model-call limits, HITL on a single tool call, PII
filters) inspects one call at a time. This gate inspects the **whole plan**
against the host ``ExecutionPolicy`` before anything runs:

``structure``
    node count, longest START->END chain (``max_plan_length``), and nesting
    (a ``spawn_subgraph``/``gauntlet`` needs ``max_depth >= 2``).
``capabilities``
    node kinds, tools (incl. ``parallel_map`` tool children), subagents, and
    models (node ``params.model`` plus the run's role models).
``data_flow``
    forbidden edges: output of any tool may not reach a state-changing tool or
    node unless a gate node (``wait_for_event`` by default) sits on every path
    in between; tools in a gate-required class need a gate on every path from
    START.
``catalog``
    in catalog mode, the plan must be a filled approved entry.
``cost``
    the worst-case token / USD bound (``estimate_plan_cost``) must fit
    ``max_tokens`` / ``max_cost_usd``. A dollar ceiling with no cost model, or
    a counted model with no price, fails closed.

``check_plan_policy`` is pure: no I/O, no model calls, deterministic output.
It never raises for a bad plan — it returns a ``PlanGateReport`` whose
``allowed`` is False and whose ``violations`` say why. An exception inside a
gate becomes a ``gate_error`` violation (fail closed).
``enforce_plan_policy`` raises ``PlanPolicyViolation`` instead.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from app.errors import DynamicSubgraphsError
from app.governance.cost import CostBound, CostModel, estimate_plan_cost
from app.governance.hashing import plan_hash, sha256_hex
from app.models.graph_spec import (
    SPECIAL_NODE_END,
    SPECIAL_NODE_START,
    GraphSpec,
    NodeSpec,
)
from app.models.node_kinds import NodeKind
from app.policy import ExecutionPolicy

GATES: tuple[str, ...] = ("structure", "capabilities", "data_flow", "catalog", "cost")

# Violations a re-plan cannot fix: the planner can't change the run's models
# or the catalog, and a gate crash is a host bug.
NON_REPAIRABLE_CODES: frozenset[str] = frozenset(
    {"run_model_not_allowed", "plan_not_in_catalog", "gate_error", "no_cost_model"}
)


@dataclass(frozen=True)
class PolicyViolation:
    """One structured reason a plan was refused."""

    gate: str
    code: str
    message: str
    node_id: str | None = None
    field: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "code": self.code,
            "message": self.message,
            "node_id": self.node_id,
            "field": self.field,
        }


@dataclass(frozen=True)
class GateDecision:
    """The outcome of one gate."""

    gate: str
    allowed: bool
    violations: tuple[PolicyViolation, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "decision": "allow" if self.allowed else "deny",
            "violations": [v.as_dict() for v in self.violations],
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PlanGateReport:
    """All gate decisions for one plan. ``allowed`` iff every gate allowed."""

    plan_hash: str
    policy_hash: str
    decisions: tuple[GateDecision, ...]
    cost_bound: CostBound | None = None
    catalog_entry_id: str | None = None
    catalog_entry_hash: str | None = None

    @property
    def allowed(self) -> bool:
        return all(d.allowed for d in self.decisions)

    @property
    def violations(self) -> tuple[PolicyViolation, ...]:
        return tuple(v for d in self.decisions for v in d.violations)

    @property
    def repairable(self) -> bool:
        """True if some violation is something a re-plan could fix."""
        codes = {v.code for v in self.violations}
        return bool(codes - NON_REPAIRABLE_CODES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "plan_hash": self.plan_hash,
            "policy_hash": self.policy_hash,
            "decisions": [d.as_dict() for d in self.decisions],
            "cost_bound": self.cost_bound.as_dict() if self.cost_bound else None,
            "catalog_entry_id": self.catalog_entry_id,
            "catalog_entry_hash": self.catalog_entry_hash,
        }


class PlanPolicyViolation(DynamicSubgraphsError):
    """Raised by ``enforce_plan_policy`` when a plan is refused."""

    def __init__(self, report: PlanGateReport) -> None:
        self.report = report
        msgs = [v.message for v in report.violations]
        summary = "; ".join(msgs[:5])
        if len(msgs) > 5:
            summary += f" (+{len(msgs) - 5} more)"
        super().__init__(summary or "plan refused by policy")


def policy_hash(policy: ExecutionPolicy) -> str:
    """Content hash of a policy (budgets, allow-sets, classes, prices, catalog)."""
    return sha256_hex(policy.as_dict())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def node_tools(node: NodeSpec) -> tuple[str, ...]:
    """Tool names a node invokes (``tool_call`` or a ``parallel_map`` tool child)."""
    params = node.params
    if node.kind == NodeKind.TOOL_CALL and params.get("tool_name"):
        return (str(params["tool_name"]),)
    if node.kind == NodeKind.PARALLEL_MAP and params.get("child_kind") == "tool_call":
        child = params.get("child_params") or {}
        if child.get("tool_name"):
            return (str(child["tool_name"]),)
    return ()


def node_models(node: NodeSpec) -> tuple[str, ...]:
    """Models a node pins explicitly via ``params.model``."""
    params = node.params
    out: list[str] = []
    if params.get("model"):
        out.append(str(params["model"]))
    if node.kind == NodeKind.PARALLEL_MAP:
        child = params.get("child_params") or {}
        if child.get("model"):
            out.append(str(child["model"]))
    return tuple(out)


def _adjacency(spec: GraphSpec, node_ids: set[str]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in spec.edges:
        src, dst = edge.from_, edge.to
        if (src in node_ids or src == SPECIAL_NODE_START) and (
            dst in node_ids or dst == SPECIAL_NODE_END
        ):
            adj[src].append(dst)
    return adj


def _reachable(adj: dict[str, list[str]], start: str, *, blocked: set[str]) -> set[str]:
    """Nodes reachable from ``start`` without passing through ``blocked``."""
    seen: set[str] = set()
    stack = [n for n in adj.get(start, []) if n not in blocked]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(n for n in adj.get(cur, []) if n not in blocked and n not in seen)
    return seen


def _longest_chain(adj: dict[str, list[str]], node_ids: set[str]) -> int:
    """Nodes on the longest START->END path (0 on a cycle; the validator rejects those)."""
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(n: str) -> int:
        if n == SPECIAL_NODE_END:
            return 0
        if n in memo:
            return memo[n]
        if n in visiting:
            raise ValueError("cycle")
        visiting.add(n)
        best = max((depth(m) for m in adj.get(n, [])), default=0)
        visiting.discard(n)
        memo[n] = best + (1 if n in node_ids else 0)
        return memo[n]

    try:
        return depth(SPECIAL_NODE_START)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _gate_structure(spec: GraphSpec, policy: ExecutionPolicy) -> GateDecision:
    out: list[PolicyViolation] = []
    node_ids = {n.id for n in spec.nodes}
    max_nodes = min(policy.max_nodes, spec.budget.max_nodes)
    if len(spec.nodes) > max_nodes:
        out.append(
            PolicyViolation(
                "structure",
                "max_nodes_exceeded",
                f"plan has {len(spec.nodes)} nodes; the limit is {max_nodes}",
            )
        )
    length = _longest_chain(_adjacency(spec, node_ids), node_ids)
    if policy.max_plan_length is not None and length > policy.max_plan_length:
        out.append(
            PolicyViolation(
                "structure",
                "max_plan_length_exceeded",
                f"longest START->END chain has {length} nodes; the limit is "
                f"{policy.max_plan_length}",
            )
        )
    nesting = [
        n.id
        for n in spec.nodes
        if n.kind in (NodeKind.SPAWN_SUBGRAPH, NodeKind.GAUNTLET)
    ]
    depth = 2 if nesting else 1
    if depth > policy.max_depth:
        for nid in nesting:
            out.append(
                PolicyViolation(
                    "structure",
                    "nesting_depth_exceeded",
                    f"node '{nid}' spawns a child graph but the policy allows "
                    f"depth {policy.max_depth}",
                    node_id=nid,
                )
            )
    return GateDecision(
        "structure",
        not out,
        tuple(out),
        {"nodes": len(spec.nodes), "longest_chain": length, "static_depth": depth},
    )


def _gate_capabilities(
    spec: GraphSpec, policy: ExecutionPolicy, run_models: Iterable[str]
) -> GateDecision:
    out: list[PolicyViolation] = []
    for node in spec.nodes:
        if (
            policy.allowed_node_kinds is not None
            and node.kind not in policy.allowed_node_kinds
        ):
            out.append(
                PolicyViolation(
                    "capabilities",
                    "node_kind_not_allowed",
                    f"node kind '{node.kind.value}' is not allowed",
                    node_id=node.id,
                    field="kind",
                )
            )
        if policy.allowed_tools is not None:
            for tool in node_tools(node):
                if tool not in policy.allowed_tools:
                    out.append(
                        PolicyViolation(
                            "capabilities",
                            "tool_not_allowed",
                            f"tool '{tool}' is not allowed",
                            node_id=node.id,
                            field="params.tool_name",
                        )
                    )
        if (
            policy.allowed_subagents is not None
            and node.kind == NodeKind.SPAWN_SUBAGENT
            and node.params.get("agent_name") not in policy.allowed_subagents
        ):
            out.append(
                PolicyViolation(
                    "capabilities",
                    "subagent_not_allowed",
                    f"subagent '{node.params.get('agent_name')}' is not allowed",
                    node_id=node.id,
                    field="params.agent_name",
                )
            )
        if policy.allowed_models is not None:
            for model in node_models(node):
                if model not in policy.allowed_models:
                    out.append(
                        PolicyViolation(
                            "capabilities",
                            "model_not_allowed",
                            f"model '{model}' is not allowed",
                            node_id=node.id,
                            field="params.model",
                        )
                    )
    if policy.allowed_models is not None:
        for model in sorted(set(run_models)):
            if model not in policy.allowed_models:
                out.append(
                    PolicyViolation(
                        "capabilities",
                        "run_model_not_allowed",
                        f"the run is configured with model '{model}', which is "
                        "not allowed",
                    )
                )
    return GateDecision("capabilities", not out, tuple(out))


def _gate_data_flow(spec: GraphSpec, policy: ExecutionPolicy) -> GateDecision:
    out: list[PolicyViolation] = []
    node_ids = {n.id for n in spec.nodes}
    adj = _adjacency(spec, node_ids)
    gates = {n.id for n in spec.nodes if n.kind in policy.gate_node_kinds}
    state_tools = policy.tools_in_classes(policy.state_changing_classes)
    gated_tools = policy.tools_in_classes(policy.gate_required_classes)

    def is_state_changing(node: NodeSpec) -> bool:
        return node.kind in policy.state_changing_node_kinds or any(
            t in state_tools for t in node_tools(node)
        )

    sinks = [n for n in spec.nodes if is_state_changing(n) and n.id not in gates]
    sources = [n for n in spec.nodes if node_tools(n)]
    for sink in sinks:
        for src in sources:
            if src.id == sink.id:
                continue
            if sink.id in _reachable(adj, src.id, blocked=gates):
                out.append(
                    PolicyViolation(
                        "data_flow",
                        "ungated_tool_to_state_change",
                        f"output of tool node '{src.id}' can reach state-changing "
                        f"node '{sink.id}' without passing a gate node "
                        f"({', '.join(sorted(k.value for k in policy.gate_node_kinds)) or 'none configured'})",
                        node_id=sink.id,
                        field="edges",
                    )
                )
    from_start = _reachable(adj, SPECIAL_NODE_START, blocked=gates)
    for node in spec.nodes:
        needs = [t for t in node_tools(node) if t in gated_tools]
        if needs and node.id in from_start:
            out.append(
                PolicyViolation(
                    "data_flow",
                    "missing_required_gate",
                    f"tool '{needs[0]}' on node '{node.id}' needs a gate node on "
                    "every path from START",
                    node_id=node.id,
                    field="edges",
                )
            )
    return GateDecision(
        "data_flow",
        not out,
        tuple(out),
        {
            "gate_nodes": sorted(gates),
            "state_changing_nodes": sorted(n.id for n in sinks),
        },
    )


def _gate_catalog(
    spec: GraphSpec, policy: ExecutionPolicy
) -> tuple[GateDecision, str | None, str | None]:
    catalog = policy.approved_catalog
    if catalog is None:
        return GateDecision("catalog", True, (), {"mode": "off"}), None, None
    entry = catalog.match(spec)
    if entry is None:
        v = PolicyViolation(
            "catalog",
            "plan_not_in_catalog",
            "plan does not match any approved catalog entry "
            f"({', '.join(catalog.ids)})",
        )
        return GateDecision("catalog", False, (v,), {"mode": "on"}), None, None
    return (
        GateDecision(
            "catalog",
            True,
            (),
            {"mode": "on", "entry_id": entry.entry_id, "entry_hash": entry.spec_hash},
        ),
        entry.entry_id,
        entry.spec_hash,
    )


def _gate_cost(
    spec: GraphSpec, policy: ExecutionPolicy, cost_model: CostModel | None
) -> tuple[GateDecision, CostBound | None]:
    has_ceiling = policy.max_cost_usd is not None or policy.max_tokens is not None
    cost_model = cost_model or policy.cost_model
    if cost_model is None:
        if policy.max_cost_usd is not None:
            v = PolicyViolation(
                "cost",
                "no_cost_model",
                "max_cost_usd is set but the policy has no cost_model to price "
                "the plan; refusing (fail closed)",
            )
            return GateDecision("cost", False, (v,)), None
        if policy.max_tokens is None:
            return GateDecision("cost", True, (), {"mode": "off"}), None
    max_llm_calls = min(policy.max_llm_calls, spec.budget.max_llm_calls)
    max_fanout = min(policy.max_fanout, spec.budget.max_fanout)
    bound = estimate_plan_cost(
        spec,
        cost_model or CostModel(),
        max_llm_calls=max_llm_calls,
        max_fanout=max_fanout,
    )
    out: list[PolicyViolation] = []
    if policy.max_tokens is not None and bound.tokens > policy.max_tokens:
        out.append(
            PolicyViolation(
                "cost",
                "token_bound_exceeded",
                f"worst-case tokens {bound.tokens} exceed the budget "
                f"{policy.max_tokens}",
            )
        )
    if policy.max_cost_usd is not None:
        if bound.usd is None:
            out.append(
                PolicyViolation(
                    "cost",
                    "unpriced_model",
                    "cannot bound the plan's dollar cost: no price for "
                    f"{', '.join(bound.unpriced_models)}",
                )
            )
        elif bound.usd > policy.max_cost_usd:
            out.append(
                PolicyViolation(
                    "cost",
                    "cost_bound_exceeded",
                    f"worst-case cost ${bound.usd:.6f} exceeds the budget "
                    f"${policy.max_cost_usd:.6f}",
                )
            )
    details = {
        "mode": "enforce" if has_ceiling else "estimate",
        **bound.as_dict(),
        # The exact prices/caps/routing used, so the bound can be recomputed.
        "cost_model_hash": sha256_hex(cost_model.as_dict()) if cost_model else None,
    }
    return GateDecision("cost", not out, tuple(out), details), bound


def _guard(gate: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # fail closed: a crashing gate denies
        v = PolicyViolation(
            gate, "gate_error", f"gate '{gate}' failed: {type(exc).__name__}: {exc}"
        )
        return GateDecision(gate, False, (v,))


def check_plan_policy(
    spec: GraphSpec,
    policy: ExecutionPolicy,
    *,
    run_models: Iterable[str] = (),
    gates: Iterable[str] = GATES,
    cost_model: CostModel | None = None,
) -> PlanGateReport:
    """Run the plan gate. Pure and deterministic; never raises for a bad plan.

    Args:
        spec: the plan to check (the planner's output or a fixed spec).
        policy: the host policy.
        run_models: model names the run is configured to use (planner, worker,
            reducer, ...); checked against ``policy.allowed_models``.
        gates: subset of ``GATES`` to run (nested children skip ``cost``, which
            the root bound already covers).
        cost_model: the cost model to price with, overriding
            ``policy.cost_model`` (the SDK passes one with the run's role models
            filled in). The report's ``policy_hash`` is always of ``policy`` as
            given, so it matches the policy the host configured.
    """
    selected = set(gates)
    decisions: list[GateDecision] = []
    bound: CostBound | None = None
    entry_id: str | None = None
    entry_hash: str | None = None
    run_models = tuple(run_models)

    if "structure" in selected:
        decisions.append(_guard("structure", lambda: _gate_structure(spec, policy)))
    if "capabilities" in selected:
        decisions.append(
            _guard("capabilities", lambda: _gate_capabilities(spec, policy, run_models))
        )
    if "data_flow" in selected:
        decisions.append(_guard("data_flow", lambda: _gate_data_flow(spec, policy)))
    if "catalog" in selected:
        res = _guard("catalog", lambda: _gate_catalog(spec, policy))
        if isinstance(res, GateDecision):
            decisions.append(res)
        else:
            decisions.append(res[0])
            entry_id, entry_hash = res[1], res[2]
    if "cost" in selected:
        res = _guard("cost", lambda: _gate_cost(spec, policy, cost_model))
        if isinstance(res, GateDecision):
            decisions.append(res)
        else:
            decisions.append(res[0])
            bound = res[1]

    try:
        p_hash = policy_hash(policy)
    except Exception as exc:
        p_hash = "unhashable"
        decisions.append(
            GateDecision(
                "policy",
                False,
                (
                    PolicyViolation(
                        "policy", "gate_error", f"policy could not be hashed: {exc}"
                    ),
                ),
            )
        )
    return PlanGateReport(
        plan_hash=plan_hash(spec),
        policy_hash=p_hash,
        decisions=tuple(decisions),
        cost_bound=bound,
        catalog_entry_id=entry_id,
        catalog_entry_hash=entry_hash,
    )


def enforce_plan_policy(
    spec: GraphSpec,
    policy: ExecutionPolicy,
    *,
    run_models: Iterable[str] = (),
    gates: Iterable[str] = GATES,
    cost_model: CostModel | None = None,
) -> PlanGateReport:
    """``check_plan_policy`` that raises ``PlanPolicyViolation`` on a deny."""
    report = check_plan_policy(
        spec, policy, run_models=run_models, gates=gates, cost_model=cost_model
    )
    if not report.allowed:
        raise PlanPolicyViolation(report)
    return report

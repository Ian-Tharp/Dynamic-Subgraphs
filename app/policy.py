"""Host-owned execution governance.

The budget that bounds a run must belong to the **host**, not the planner. A
planner emits a `GraphBudget` as a *request*; this module resolves it against a
host-owned `ExecutionPolicy` into an immutable `EffectiveBudget` — the single
source of truth every enforcement site reads. The rule is uniform:

    effective = min(planner request, host ceiling[, parent remaining])  # per numeric field
    allowed   = host allow-set ∩ registry                                # per vocabulary

so a planner can only ever *narrow* what the host permits, never widen it.

This is the foundation slice (types + the pure resolver). It changes no
behavior on its own; nothing reads an `ExecutionPolicy` until later slices wire
it into the validator, executor, `parallel_map`, and `spawn_subgraph`.

Layering: this lives in `app` and imports only `app.models.*` (never the
registry/validator/runtime), so the facade can re-export `ExecutionPolicy`
without a reverse import and there is no cycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.models.graph_spec import GraphBudget
from app.models.node_kinds import NodeKind

# Absolute nesting rail. Nested subgraphs may never exceed this regardless of
# what a host or planner asks for. This is the single source of truth; the
# validator re-exports it so `from app.registry.validator import
# MAX_DEPTH_CEILING` keeps working.
MAX_DEPTH_CEILING = 3

# Allow-set sentinel contract (tools / subagents / node kinds):
#   None        -> host imposes NO narrowing  => effective set = the full registry set
#   frozenset() -> host forbids ALL of that category (fail-closed ban-all)
# These are DISTINCT. Never treat a falsy value as "no narrowing" — that would
# turn a deliberate ban-all into fail-open.


@dataclass(frozen=True)
class ExecutionPolicy:
    """Host-owned ceilings for a run.

    Numeric fields are hard upper bounds. The allow-set fields, when not
    ``None``, are the host's permitted vocabulary; the planner may narrow
    further (by naming a subset in its nodes) but can never widen them.
    Immutable and safe to share across runs.
    """

    max_nodes: int = 12
    max_llm_calls: int = 8
    max_depth: int = 2
    max_depth_ceiling: int = MAX_DEPTH_CEILING
    max_fanout: int = 64
    max_wall_seconds: int = 90
    allowed_tools: frozenset[str] | None = None
    allowed_subagents: frozenset[str] | None = None
    allowed_node_kinds: frozenset[NodeKind] | None = None

    def __post_init__(self) -> None:
        # Coerce any iterable allow-set to frozenset so equality / hashing /
        # as_dict are stable even if a caller passes a plain set or list.
        for name in ("allowed_tools", "allowed_subagents", "allowed_node_kinds"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, frozenset):
                object.__setattr__(self, name, frozenset(value))
        for name in (
            "max_nodes",
            "max_depth",
            "max_fanout",
            "max_wall_seconds",
            "max_depth_ceiling",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"ExecutionPolicy.{name} must be >= 1")
        if self.max_llm_calls < 0:
            raise ValueError("ExecutionPolicy.max_llm_calls must be >= 0")
        # A host can soften the per-run depth cap below the rail, never above it.
        if self.max_depth > self.max_depth_ceiling:
            object.__setattr__(self, "max_depth", self.max_depth_ceiling)


@dataclass(frozen=True)
class RemainingBudget:
    """What a parent run has left for a child. Carried in the summed counters
    ledger (never in last-writer-wins metadata), so concurrent siblings cannot
    both claim the full remainder."""

    nodes: int
    llm_calls: int
    depth: int


@dataclass(frozen=True)
class EffectiveBudget:
    """The resolved limits for ONE graph execution — the single source of truth
    every enforcement site reads. JSON-safe via :meth:`as_dict` so the executor
    can seed it into run metadata and children can reload it."""

    max_nodes: int
    max_llm_calls: int
    max_depth: int
    max_fanout: int
    max_wall_seconds: int
    allowed_tools: frozenset[str]
    allowed_subagents: frozenset[str]
    allowed_node_kinds: frozenset[NodeKind]

    def as_dict(self) -> dict[str, object]:
        return {
            "max_nodes": self.max_nodes,
            "max_llm_calls": self.max_llm_calls,
            "max_depth": self.max_depth,
            "max_fanout": self.max_fanout,
            "max_wall_seconds": self.max_wall_seconds,
            "allowed_tools": sorted(self.allowed_tools),
            "allowed_subagents": sorted(self.allowed_subagents),
            "allowed_node_kinds": sorted(k.value for k in self.allowed_node_kinds),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EffectiveBudget:
        """Reconstruct from :meth:`as_dict` output (e.g. seeded run metadata)."""

        def _ints(key: str) -> int:
            return int(data[key])  # type: ignore[arg-type]

        def _strs(key: str) -> frozenset[str]:
            return frozenset(str(x) for x in (data.get(key) or []))  # type: ignore[union-attr]

        kinds = data.get("allowed_node_kinds") or []
        return cls(
            max_nodes=_ints("max_nodes"),
            max_llm_calls=_ints("max_llm_calls"),
            max_depth=_ints("max_depth"),
            max_fanout=_ints("max_fanout"),
            max_wall_seconds=_ints("max_wall_seconds"),
            allowed_tools=_strs("allowed_tools"),
            allowed_subagents=_strs("allowed_subagents"),
            allowed_node_kinds=frozenset(NodeKind(str(k)) for k in kinds),  # type: ignore[union-attr]
        )


def _intersect(
    host: frozenset[str] | None, registry: frozenset[str]
) -> frozenset[str]:
    """None => no host narrowing (full registry set); else host ∩ registry."""
    return registry if host is None else (host & registry)


def _intersect_kinds(
    host: frozenset[NodeKind] | None, registry: frozenset[NodeKind]
) -> frozenset[NodeKind]:
    return registry if host is None else (host & registry)


def resolve_effective_budget(
    policy: ExecutionPolicy,
    request: GraphBudget,
    *,
    registry_tools: Iterable[str],
    registry_subagents: Iterable[str],
    registry_kinds: Iterable[NodeKind],
    remaining: RemainingBudget | None = None,
    is_child: bool = False,
) -> EffectiveBudget:
    """Resolve a planner request against the host policy (and, for a child, the
    parent's remaining budget) into an immutable :class:`EffectiveBudget`.

    For every numeric field the effective value is ``min(host, request[,
    remaining])``; vocabularies are the intersection of the host allow-set with
    the registry. A child resolution (``is_child=True``) **must** pass
    ``remaining`` — there is no safe full-budget fallback, since that would let
    an N-level nest re-mint N fresh budgets.
    """
    if is_child and remaining is None:
        raise ValueError(
            "resolve_effective_budget: a child resolution (is_child=True) "
            "requires `remaining`; refusing to grant a child the full host budget."
        )

    reg_tools = frozenset(registry_tools)
    reg_subs = frozenset(registry_subagents)
    reg_kinds = frozenset(registry_kinds)

    node_cap = policy.max_nodes
    llm_cap = policy.max_llm_calls
    depth_cap = policy.max_depth
    if remaining is not None:
        node_cap = min(node_cap, remaining.nodes)
        llm_cap = min(llm_cap, remaining.llm_calls)
        depth_cap = min(depth_cap, remaining.depth)

    # GraphBudget has no max_fanout field yet (added in a later slice); fall back
    # to the host cap so old recorded specs resolve unchanged.
    request_fanout = getattr(request, "max_fanout", policy.max_fanout)

    return EffectiveBudget(
        max_nodes=min(node_cap, request.max_nodes),
        max_llm_calls=min(llm_cap, request.max_llm_calls),
        max_depth=min(depth_cap, request.max_depth, policy.max_depth_ceiling),
        max_fanout=min(policy.max_fanout, request_fanout),
        max_wall_seconds=min(policy.max_wall_seconds, request.max_wall_seconds),
        allowed_tools=_intersect(policy.allowed_tools, reg_tools),
        allowed_subagents=_intersect(policy.allowed_subagents, reg_subs),
        allowed_node_kinds=_intersect_kinds(policy.allowed_node_kinds, reg_kinds),
    )

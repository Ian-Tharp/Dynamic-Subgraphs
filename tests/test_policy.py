"""Host-owned ExecutionPolicy: resolution math + invariants (PR-1).

Types and the pure resolver only — no wiring yet. These lock the contract every
later enforcement site depends on: effective = min(host, request[, remaining]),
vocabularies = host ∩ registry, and children can never fall back to a full budget.
"""

from __future__ import annotations

import pytest

from app.models.graph_spec import GraphBudget
from app.models.node_kinds import NodeKind
from app.policy import (
    MAX_DEPTH_CEILING,
    EffectiveBudget,
    ExecutionPolicy,
    RemainingBudget,
    resolve_effective_budget,
)

# Full registry sets used as the outer bound in most tests.
ALL_TOOLS = frozenset({"web_search", "policy_lookup", "create_follow_up_task"})
ALL_SUBS = frozenset({"document_specialist", "critic"})
ALL_KINDS = frozenset(NodeKind)


def _resolve(policy: ExecutionPolicy, request: GraphBudget, **kw) -> EffectiveBudget:
    return resolve_effective_budget(
        policy,
        request,
        registry_tools=ALL_TOOLS,
        registry_subagents=ALL_SUBS,
        registry_kinds=ALL_KINDS,
        **kw,
    )


# ---------- numeric min() semantics ----------


def test_planner_asking_more_is_clamped_to_host() -> None:
    policy = ExecutionPolicy(max_nodes=12, max_llm_calls=8)
    req = GraphBudget(max_nodes=1000, max_llm_calls=1000, max_wall_seconds=99999)
    eff = _resolve(policy, req)
    assert eff.max_nodes == 12
    assert eff.max_llm_calls == 8
    assert eff.max_wall_seconds == 90  # host default


def test_planner_asking_less_is_honored() -> None:
    policy = ExecutionPolicy(max_nodes=12, max_llm_calls=8, max_wall_seconds=90)
    req = GraphBudget(max_nodes=3, max_llm_calls=2, max_wall_seconds=30)
    eff = _resolve(policy, req)
    assert (eff.max_nodes, eff.max_llm_calls, eff.max_wall_seconds) == (3, 2, 30)


def test_depth_is_triple_min_of_host_request_and_ceiling() -> None:
    # host wants 3, planner asks 5, ceiling 3 -> 3
    eff = _resolve(ExecutionPolicy(max_depth=3), GraphBudget(max_depth=5))
    assert eff.max_depth == 3
    # planner asks 1 (less than host) -> 1
    eff2 = _resolve(ExecutionPolicy(max_depth=3), GraphBudget(max_depth=1))
    assert eff2.max_depth == 1


def test_fanout_is_host_owned() -> None:
    # GraphBudget has no max_fanout today; effective uses the host cap.
    eff = _resolve(ExecutionPolicy(max_fanout=64), GraphBudget())
    assert eff.max_fanout == 64


# ---------- __post_init__ invariants ----------


def test_host_max_depth_is_clamped_to_ceiling() -> None:
    policy = ExecutionPolicy(max_depth=99)
    assert (
        policy.max_depth == MAX_DEPTH_CEILING
    )  # a misconfigured host can't raise the rail


def test_allow_sets_are_coerced_to_frozenset() -> None:
    policy = ExecutionPolicy(
        allowed_tools={"web_search"},  # plain set
        allowed_node_kinds=[NodeKind.LLM_CALL],  # list
    )
    assert isinstance(policy.allowed_tools, frozenset)
    assert isinstance(policy.allowed_node_kinds, frozenset)


def test_invalid_numeric_fields_raise() -> None:
    with pytest.raises(ValueError):
        ExecutionPolicy(max_nodes=0)
    with pytest.raises(ValueError):
        ExecutionPolicy(max_llm_calls=-1)
    # max_llm_calls == 0 is allowed (a no-LLM policy)
    assert ExecutionPolicy(max_llm_calls=0).max_llm_calls == 0


# ---------- intersection semantics (None vs frozenset() vs subset) ----------


def test_none_allow_set_means_full_registry() -> None:
    eff = _resolve(ExecutionPolicy(allowed_tools=None), GraphBudget())
    assert eff.allowed_tools == ALL_TOOLS


def test_subset_allow_set_intersects_with_registry() -> None:
    policy = ExecutionPolicy(allowed_tools=frozenset({"web_search", "not_a_real_tool"}))
    eff = _resolve(policy, GraphBudget())
    # host-named tool absent from the registry is NOT granted (intersection).
    assert eff.allowed_tools == frozenset({"web_search"})


def test_empty_frozenset_is_ban_all_not_no_narrowing() -> None:
    # Resolver semantics only: empty set => ban-all (distinct from None =
    # no-narrowing). That the validator actually *rejects* a banned tool is
    # covered by test_policy_enforcement.test_policy_bans_a_tool_outside_the_allowlist.
    eff = _resolve(ExecutionPolicy(allowed_tools=frozenset()), GraphBudget())
    assert eff.allowed_tools == frozenset()


def test_node_kind_intersection() -> None:
    policy = ExecutionPolicy(allowed_node_kinds=frozenset({NodeKind.LLM_CALL}))
    eff = _resolve(policy, GraphBudget())
    assert eff.allowed_node_kinds == frozenset({NodeKind.LLM_CALL})


# ---------- child resolution against remaining ----------


def test_child_without_remaining_raises() -> None:
    with pytest.raises(ValueError, match="requires `remaining`"):
        _resolve(ExecutionPolicy(), GraphBudget(), is_child=True)


def test_child_caps_are_driven_by_remaining() -> None:
    policy = ExecutionPolicy(max_nodes=12, max_llm_calls=8, max_depth=3)
    req = GraphBudget(max_nodes=12, max_llm_calls=8, max_depth=3)
    remaining = RemainingBudget(nodes=4, llm_calls=2, depth=1)
    eff = _resolve(policy, req, remaining=remaining, is_child=True)
    assert eff.max_nodes == 4  # min(host 12, request 12, remaining 4)
    assert eff.max_llm_calls == 2
    assert eff.max_depth == 1


def test_child_cannot_exceed_remaining_even_if_request_is_large() -> None:
    policy = ExecutionPolicy(max_nodes=12)
    req = GraphBudget(max_nodes=1000)
    eff = _resolve(
        policy,
        req,
        remaining=RemainingBudget(nodes=1, llm_calls=0, depth=1),
        is_child=True,
    )
    assert eff.max_nodes == 1
    assert eff.max_llm_calls == 0


# ---------- EffectiveBudget round-trip ----------


def test_effective_budget_dict_round_trip() -> None:
    eff = _resolve(
        ExecutionPolicy(
            allowed_node_kinds=frozenset({NodeKind.LLM_CALL, NodeKind.REDUCE})
        ),
        GraphBudget(max_nodes=5),
    )
    restored = EffectiveBudget.from_dict(eff.as_dict())
    assert restored == eff
    # JSON-safe: as_dict values are ints and sorted lists of strings.
    import json

    json.dumps(eff.as_dict())

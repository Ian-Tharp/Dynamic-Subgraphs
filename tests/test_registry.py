"""Registry surface: kind admission, param validation, allowlists, LLM accounting."""

from __future__ import annotations

import pytest

from app.models import NodeKind, NodeSpec
from app.registry import (
    FORBIDDEN_KINDS,
    Registry,
    RegistryValidationError,
    default_registry,
)


# ---------- kind admission ----------


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_KINDS))
def test_forbidden_kinds_are_never_allowed(forbidden: str) -> None:
    assert default_registry.is_allowed_kind(forbidden) is False


@pytest.mark.parametrize("kind", list(NodeKind))
def test_every_declared_node_kind_is_allowed(kind: NodeKind) -> None:
    assert default_registry.is_allowed_kind(kind) is True


def test_unknown_string_kind_is_rejected() -> None:
    assert default_registry.is_allowed_kind("not_a_kind") is False


# ---------- param validation: happy path ----------


def test_validate_node_returns_node_with_normalized_params(registry: Registry, make_node) -> None:
    node = make_node(
        "n1",
        NodeKind.TOOL_CALL,
        params={"tool_name": "web_search", "args": {"q": "x"}},
    )

    normalized = registry.validate_node_params(node)

    assert normalized.params["tool_name"] == "web_search"
    assert normalized.params["args"] == {"q": "x"}


def test_validation_does_not_mutate_input_node(registry: Registry, make_node) -> None:
    original_params = {"tool_name": "web_search"}
    node = make_node("n1", NodeKind.TOOL_CALL, params=original_params)

    registry.validate_node_params(node)

    assert original_params == {"tool_name": "web_search"}


# ---------- param validation: rejections ----------


def test_tool_call_with_unallowlisted_tool_is_rejected(
    registry: Registry, make_node
) -> None:
    node = make_node("n1", NodeKind.TOOL_CALL, params={"tool_name": "rm_rf_everything"})

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "tool_not_allowlisted" in codes


def test_spawn_subagent_with_unknown_name_is_rejected(
    registry: Registry, make_node
) -> None:
    node = make_node(
        "n1",
        NodeKind.SPAWN_SUBAGENT,
        params={"agent_name": "ghost_agent", "task": "do stuff"},
    )

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "subagent_not_allowlisted" in codes


def test_llm_call_requires_non_empty_instruction(
    registry: Registry, make_node
) -> None:
    node = make_node("n1", NodeKind.LLM_CALL, params={"instruction": ""})

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "invalid_params" in codes


# ---------- parallel_map child constraints ----------


def test_parallel_map_with_unallowlisted_child_tool_is_rejected(
    registry: Registry, make_node
) -> None:
    node = make_node(
        "fan",
        NodeKind.PARALLEL_MAP,
        params={
            "over": "queries",
            "child_kind": "tool_call",
            "child_params": {"tool_name": "not_a_real_tool"},
        },
    )

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "tool_not_allowlisted" in codes


def test_parallel_map_child_tool_call_requires_tool_name(
    registry: Registry, make_node
) -> None:
    node = make_node(
        "fan",
        NodeKind.PARALLEL_MAP,
        params={
            "over": "queries",
            "child_kind": "tool_call",
            "child_params": {},
        },
    )

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "missing_child_tool" in codes


def test_parallel_map_child_llm_call_requires_instruction(
    registry: Registry, make_node
) -> None:
    node = make_node(
        "fan",
        NodeKind.PARALLEL_MAP,
        params={
            "over": "queries",
            "child_kind": "llm_call",
            "child_params": {},
        },
    )

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "missing_child_instruction" in codes


# ---------- typed param model invariants ----------


def test_reduce_with_llm_summarize_requires_instruction(
    registry: Registry, make_node
) -> None:
    node = make_node(
        "r",
        NodeKind.REDUCE,
        params={
            "strategy": "llm_summarize",
            "input_keys": ["a", "b"],
            "output_key": "summary",
        },
    )

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "invalid_params" in codes


def test_reduce_with_concat_strategy_does_not_require_instruction(
    registry: Registry, make_node
) -> None:
    node = make_node(
        "r",
        NodeKind.REDUCE,
        params={
            "strategy": "concat",
            "input_keys": ["a", "b"],
            "output_key": "joined",
        },
    )

    normalized = registry.validate_node_params(node)

    assert normalized.params["strategy"] == "concat"


def test_branch_requires_at_least_two_branches(
    registry: Registry, make_node
) -> None:
    node = make_node(
        "b",
        NodeKind.BRANCH,
        params={"branches": ["only_one"], "decision_key": "route"},
    )

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "invalid_params" in codes


def test_wait_for_event_accepts_no_timeout(registry: Registry, make_node) -> None:
    node = make_node(
        "w", NodeKind.WAIT_FOR_EVENT, params={"event_type": "callback"}
    )

    normalized = registry.validate_node_params(node)

    assert normalized.params["event_type"] == "callback"
    assert normalized.params["timeout_seconds"] is None


def test_wait_for_event_rejects_non_positive_timeout(
    registry: Registry, make_node
) -> None:
    node = make_node(
        "w",
        NodeKind.WAIT_FOR_EVENT,
        params={"event_type": "callback", "timeout_seconds": 0},
    )

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "invalid_params" in codes


def test_emit_artifact_requires_name_and_content_key(
    registry: Registry, make_node
) -> None:
    node = make_node("e", NodeKind.EMIT_ARTIFACT, params={"artifact_type": "report"})

    with pytest.raises(RegistryValidationError) as exc:
        registry.validate_node_params(node)

    codes = {i.code for i in exc.value.issues}
    assert "invalid_params" in codes


# ---------- custom registry composition ----------


def test_custom_registry_accepts_custom_tools(make_node) -> None:
    reg = Registry(tools=frozenset({"custom_tool"}))
    node = make_node("t", NodeKind.TOOL_CALL, params={"tool_name": "custom_tool"})

    normalized = reg.validate_node_params(node)

    assert normalized.params["tool_name"] == "custom_tool"


def test_custom_registry_rejects_default_tools_not_in_its_allowlist(make_node) -> None:
    reg = Registry(tools=frozenset({"only_this"}))
    node = make_node("t", NodeKind.TOOL_CALL, params={"tool_name": "web_search"})

    with pytest.raises(RegistryValidationError):
        reg.validate_node_params(node)


# ---------- count_llm_calls ----------


def test_count_llm_calls_counts_direct_llm_node(
    registry: Registry, make_node
) -> None:
    nodes = [
        make_node("a", NodeKind.LLM_CALL, params={"instruction": "x"}),
        make_node(
            "b", NodeKind.TOOL_CALL, params={"tool_name": "web_search"}
        ),
    ]

    count = registry.count_llm_calls(nodes)

    assert count == 1


def test_count_llm_calls_counts_reduce_only_when_llm_strategy(
    registry: Registry, make_node
) -> None:
    deterministic = make_node(
        "r1",
        NodeKind.REDUCE,
        params={"strategy": "concat", "input_keys": ["a"], "output_key": "o"},
    )
    llm_backed = make_node(
        "r2",
        NodeKind.REDUCE,
        params={
            "strategy": "llm_summarize",
            "input_keys": ["a"],
            "output_key": "o",
            "instruction": "summarize",
        },
    )

    count = registry.count_llm_calls([deterministic, llm_backed])

    assert count == 1


def test_count_llm_calls_counts_parallel_map_llm_child(
    registry: Registry, make_node
) -> None:
    nodes = [
        make_node(
            "fan",
            NodeKind.PARALLEL_MAP,
            params={
                "over": "queries",
                "child_kind": "llm_call",
                "child_params": {"instruction": "summarize each"},
            },
        )
    ]

    count = registry.count_llm_calls(nodes)

    assert count == 1


def test_count_llm_calls_does_not_count_parallel_map_tool_child(
    registry: Registry, make_node
) -> None:
    nodes = [
        make_node(
            "fan",
            NodeKind.PARALLEL_MAP,
            params={
                "over": "queries",
                "child_kind": "tool_call",
                "child_params": {"tool_name": "web_search"},
            },
        )
    ]

    count = registry.count_llm_calls(nodes)

    assert count == 0

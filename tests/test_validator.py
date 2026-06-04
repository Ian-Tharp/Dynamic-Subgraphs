"""Graph-level validation: topology, budget, reachability, cycles, input provenance."""

from __future__ import annotations

import pytest

from app.models import GraphSpec
from app.models.graph_spec import GraphBudget
from app.registry import RegistryValidationError, validate_graph_spec
from app.registry.validator import MAX_DEPTH_CEILING

# ---------- happy path ----------


def test_minimal_spec_is_accepted_and_returns_normalized_copy(
    minimal_spec: GraphSpec,
) -> None:
    validated = validate_graph_spec(minimal_spec)

    assert validated.nodes[0].params["instruction"] == "do the thing"
    assert validated is not minimal_spec


# ---------- topology errors ----------


def test_duplicate_node_ids_are_rejected(spec_factory, make_node, make_edge) -> None:
    spec = spec_factory(
        nodes=[
            make_node("dup", params={"instruction": "a"}),
            make_node("dup", params={"instruction": "b"}),
        ],
        edges=[make_edge("START", "dup"), make_edge("dup", "END")],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "duplicate_node_id" in codes


def test_dangling_edge_target_is_rejected(spec_factory, make_node, make_edge) -> None:
    spec = spec_factory(
        nodes=[make_node("step", outputs=["draft"])],
        edges=[make_edge("START", "missing"), make_edge("step", "END")],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "dangling_edge" in codes


def test_dangling_edge_source_is_rejected(spec_factory, make_node, make_edge) -> None:
    spec = spec_factory(
        nodes=[make_node("step", outputs=["draft"])],
        edges=[
            make_edge("START", "step"),
            make_edge("ghost", "END"),
            make_edge("step", "END"),
        ],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "dangling_edge" in codes


def test_unreachable_node_is_flagged(spec_factory, make_node, make_edge) -> None:
    spec = spec_factory(
        nodes=[
            make_node("reachable", outputs=["a"]),
            make_node("orphan", outputs=["b"]),
        ],
        edges=[
            make_edge("START", "reachable"),
            make_edge("reachable", "END"),
            make_edge("orphan", "END"),
        ],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    issue_node_ids = {
        i.node_id for i in exc.value.issues if i.code == "unreachable_node"
    }
    assert "orphan" in issue_node_ids


def test_graph_with_no_start_to_end_path_is_rejected(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[make_node("step", outputs=["draft"])],
        edges=[make_edge("START", "step")],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "no_start_to_end_path" in codes


def test_cycle_among_planner_nodes_is_rejected(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[
            make_node("a", outputs=["x"]),
            make_node("b", outputs=["y"]),
        ],
        edges=[
            make_edge("START", "a"),
            make_edge("a", "b"),
            make_edge("b", "a"),
            make_edge("b", "END"),
        ],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "cycle_detected" in codes


# ---------- input provenance ----------


def test_node_input_must_be_produced_upstream(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[
            make_node("consumer", inputs=["sources"], outputs=["draft"]),
        ],
        edges=[
            make_edge("START", "consumer"),
            make_edge("consumer", "END"),
        ],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "missing_upstream_input" in codes


def test_input_satisfied_by_upstream_output_is_accepted(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[
            make_node("producer", outputs=["sources"]),
            make_node("consumer", inputs=["sources"], outputs=["draft"]),
        ],
        edges=[
            make_edge("START", "producer"),
            make_edge("producer", "consumer"),
            make_edge("consumer", "END"),
        ],
    )

    validated = validate_graph_spec(spec)

    assert {n.id for n in validated.nodes} == {"producer", "consumer"}


def test_sibling_branch_output_is_not_a_valid_input(
    spec_factory, make_node, make_edge
) -> None:
    # producer and consumer are siblings off START — no edge guarantees producer
    # runs before consumer, so consumer's input is NOT satisfied. This used to
    # pass by topological luck (global available set); now it is correctly flagged.
    spec = spec_factory(
        nodes=[
            make_node("producer", outputs=["sources"]),
            make_node("consumer", inputs=["sources"], outputs=["draft"]),
        ],
        edges=[
            make_edge("START", "producer"),
            make_edge("START", "consumer"),
            make_edge("producer", "END"),
            make_edge("consumer", "END"),
        ],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    issue_nodes = {
        i.node_id for i in exc.value.issues if i.code == "missing_upstream_input"
    }
    assert "consumer" in issue_nodes


def test_input_from_transitive_ancestor_is_accepted(
    spec_factory, make_node, make_edge
) -> None:
    # x is produced by `a`, consumed by `c`, with `b` in between: availability
    # must follow the full ancestor chain, not just the direct predecessor.
    spec = spec_factory(
        nodes=[
            make_node("a", outputs=["x"]),
            make_node("b", outputs=["y"]),
            make_node("c", inputs=["x"], outputs=["z"]),
        ],
        edges=[
            make_edge("START", "a"),
            make_edge("a", "b"),
            make_edge("b", "c"),
            make_edge("c", "END"),
        ],
    )

    validated = validate_graph_spec(spec)
    assert {n.id for n in validated.nodes} == {"a", "b", "c"}


# ---------- node id validation ----------


def test_node_id_with_reserved_pm_marker_is_rejected(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[make_node("worker__pm_join", outputs=["x"])],
        edges=[
            make_edge("START", "worker__pm_join"),
            make_edge("worker__pm_join", "END"),
        ],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "reserved_node_id" in codes


def test_node_id_matching_a_terminal_is_rejected(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[make_node("START", outputs=["x"])],
        edges=[make_edge("START", "END")],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "reserved_node_id" in codes


def test_node_id_with_invalid_characters_is_rejected(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[make_node("has space", outputs=["x"])],
        edges=[make_edge("START", "has space"), make_edge("has space", "END")],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "invalid_node_id" in codes


# ---------- budget enforcement ----------


def test_budget_rejects_too_many_nodes(spec_factory, make_node, make_edge) -> None:
    nodes = [make_node(f"n{i}", outputs=[f"o{i}"]) for i in range(3)]
    edges = [
        make_edge("START", "n0"),
        make_edge("n0", "n1"),
        make_edge("n1", "n2"),
        make_edge("n2", "END"),
    ]
    spec = spec_factory(
        nodes=nodes,
        edges=edges,
        budget=GraphBudget(max_nodes=2, max_llm_calls=10),
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "budget_exceeded" in codes


def test_budget_rejects_too_many_llm_calls(spec_factory, make_node, make_edge) -> None:
    spec = spec_factory(
        nodes=[make_node("step", outputs=["draft"])],
        edges=[make_edge("START", "step"), make_edge("step", "END")],
        budget=GraphBudget(max_nodes=10, max_llm_calls=0),
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "budget_exceeded" in codes


def test_budget_rejects_max_depth_above_ceiling(
    spec_factory, make_node, make_edge
) -> None:
    # The depth budget caps how deep nested subgraphs may ever recurse. A spec
    # declaring max_depth above the hard ceiling must be rejected before it can
    # run, so the recursion rail holds once spawn_subgraph exists. One over the
    # ceiling pins the exact boundary.
    spec = spec_factory(
        nodes=[make_node("step", outputs=["draft"])],
        edges=[make_edge("START", "step"), make_edge("step", "END")],
        budget=GraphBudget(max_depth=MAX_DEPTH_CEILING + 1),
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "depth_ceiling_exceeded" in codes


def test_budget_allows_max_depth_at_ceiling(spec_factory, make_node, make_edge) -> None:
    # The ceiling itself is allowed — the rail rejects only what exceeds it.
    spec = spec_factory(
        nodes=[make_node("step", outputs=["draft"])],
        edges=[make_edge("START", "step"), make_edge("step", "END")],
        budget=GraphBudget(max_depth=MAX_DEPTH_CEILING),
    )

    validate_graph_spec(spec)  # does not raise


# ---------- schema versioning ----------


def test_unsupported_schema_version_is_rejected(
    minimal_spec: GraphSpec,
) -> None:
    raw = minimal_spec.model_dump(mode="json", by_alias=True)
    raw["schema_version"] = 99

    with pytest.raises(Exception):
        # Pydantic Literal[1] will reject before validator even runs;
        # this guards against the literal being widened by accident.
        GraphSpec.model_validate(raw)


# ---------- error aggregation ----------


def test_multiple_issues_are_aggregated_into_one_exception(
    spec_factory, make_node, make_edge
) -> None:
    spec = spec_factory(
        nodes=[
            make_node("dup", params={"instruction": "a"}),
            make_node("dup", params={"instruction": "b"}),
        ],
        edges=[
            make_edge("START", "dup"),
            make_edge("dup", "missing"),
        ],
    )

    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(spec)

    codes = {i.code for i in exc.value.issues}
    assert "duplicate_node_id" in codes

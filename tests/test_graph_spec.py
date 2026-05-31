"""GraphSpec model serialization round-trips."""

from __future__ import annotations

from app.models import GraphSpec, NodeKind
from app.models.graph_spec import GRAPH_SPEC_SCHEMA_VERSION


def test_spec_round_trips_through_json_with_aliases(minimal_spec: GraphSpec) -> None:
    raw = minimal_spec.model_dump(mode="json", by_alias=True)

    restored = GraphSpec.model_validate(raw)

    assert restored.goal == minimal_spec.goal
    assert restored.nodes[0].kind == NodeKind.LLM_CALL
    assert restored.edges[0].from_ == "START"


def test_edge_serializes_with_from_alias_not_from_underscore(minimal_spec: GraphSpec) -> None:
    raw = minimal_spec.model_dump(mode="json", by_alias=True)

    edge_keys = set(raw["edges"][0].keys())

    assert "from" in edge_keys
    assert "from_" not in edge_keys


def test_schema_version_defaults_to_one(minimal_spec: GraphSpec) -> None:
    assert minimal_spec.schema_version == GRAPH_SPEC_SCHEMA_VERSION == 1

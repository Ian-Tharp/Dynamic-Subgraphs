"""Shared fixtures for the test suite.

Conventions:
- Tests follow AAA structure (arrange / act / assert separated by blank lines).
- Each test names one behavior; use parametrize for axis variations.
- Fixtures return fresh instances per test to avoid hidden coupling.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec, GraphBudget
from app.registry import Registry


@pytest.fixture
def registry() -> Registry:
    return Registry()


@pytest.fixture
def make_node():
    def _make(
        node_id: str,
        kind: NodeKind = NodeKind.LLM_CALL,
        *,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> NodeSpec:
        return NodeSpec(
            id=node_id,
            kind=kind,
            inputs=inputs or [],
            outputs=outputs or [],
            params=params if params is not None else {"instruction": "do the thing"},
        )

    return _make


@pytest.fixture
def make_edge():
    def _make(src: str, dst: str) -> EdgeSpec:
        return EdgeSpec.model_validate({"from": src, "to": dst})

    return _make


@pytest.fixture
def minimal_spec(make_node, make_edge) -> GraphSpec:
    """Smallest valid graph: START -> step -> END with one LLM node."""
    return GraphSpec(
        graph_id="spec-minimal",
        goal="run a single step",
        nodes=[make_node("step", outputs=["draft"])],
        edges=[make_edge("START", "step"), make_edge("step", "END")],
    )


@pytest.fixture
def spec_factory(make_node, make_edge):
    """Factory for ad-hoc specs that lets tests override nodes/edges/budget."""

    def _factory(
        *,
        nodes: list[NodeSpec] | None = None,
        edges: list[EdgeSpec] | None = None,
        budget: GraphBudget | None = None,
        graph_id: str = "spec-factory",
        goal: str = "factory spec",
    ) -> GraphSpec:
        if nodes is None:
            nodes = [make_node("step", outputs=["draft"])]
        if edges is None:
            edges = [make_edge("START", "step"), make_edge("step", "END")]
        kwargs: dict[str, Any] = {
            "graph_id": graph_id,
            "goal": goal,
            "nodes": nodes,
            "edges": edges,
        }
        if budget is not None:
            kwargs["budget"] = budget
        return GraphSpec(**kwargs)

    return _factory

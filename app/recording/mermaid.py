"""Render a GraphSpec as a Mermaid `graph TD` diagram.

The output is deterministic given the spec: nodes appear in spec order,
edges in spec order. That makes recorded `graph.mmd` files diffable across
re-runs of the same spec.
"""

from __future__ import annotations

from app.models import GraphSpec
from app.models.graph_spec import SPECIAL_NODE_END, SPECIAL_NODE_START


def render_mermaid(spec: GraphSpec) -> str:
    """Return the Mermaid source for one GraphSpec."""

    lines = [
        "graph TD",
        f"    {SPECIAL_NODE_START}([{SPECIAL_NODE_START}])",
        f"    {SPECIAL_NODE_END}([{SPECIAL_NODE_END}])",
    ]
    for node in spec.nodes:
        label = f"{node.id}<br/>{node.kind.value}"
        lines.append(f'    {node.id}["{label}"]')
    for edge in spec.edges:
        lines.append(f"    {edge.from_} --> {edge.to}")
    return "\n".join(lines) + "\n"

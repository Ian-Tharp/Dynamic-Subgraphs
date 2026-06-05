"""Planner abstraction — the seam where an LLM-based planner will plug in.

For v1, the planner is a plain `Callable[[str], GraphSpec]`. `StaticPlanner`
returns a single pre-built spec regardless of prompt; useful for tests and
the hardcoded-spec demo. A real LLM planner (phase 5 of the MVP sequence)
will satisfy the same type by emitting a structured GraphSpec.
"""

from __future__ import annotations

from collections.abc import Callable

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec, GraphBudget

Planner = Callable[[str], GraphSpec]


class StaticPlanner:
    """Always returns the same spec. The prompt is ignored."""

    def __init__(self, spec: GraphSpec) -> None:
        self._spec = spec

    def __call__(self, prompt: str) -> GraphSpec:
        del prompt
        return self._spec


def build_demo_spec() -> GraphSpec:
    """The static demo spec used by the mock planner and the CLI fallback.

    A small parallel-gather -> concat -> synthesize research graph. Lives here
    (next to `StaticPlanner`, its consumer) rather than in the CLI entrypoint, so
    `assembly` and `main` both import it from a neutral module instead of the
    `app.main` __main__ — keeping the dependency direction pointing downward.
    """

    nodes = [
        NodeSpec(
            id="gather_left",
            kind=NodeKind.LLM_CALL,
            outputs=["left_evidence"],
            params={"instruction": "summarize source A"},
        ),
        NodeSpec(
            id="gather_right",
            kind=NodeKind.LLM_CALL,
            outputs=["right_evidence"],
            params={"instruction": "summarize source B"},
        ),
        NodeSpec(
            id="join",
            kind=NodeKind.REDUCE,
            inputs=["left_evidence", "right_evidence"],
            outputs=["combined_evidence"],
            params={
                "strategy": "concat",
                "input_keys": ["left_evidence", "right_evidence"],
                "output_key": "combined_evidence",
            },
        ),
        NodeSpec(
            id="synthesize",
            kind=NodeKind.LLM_CALL,
            inputs=["combined_evidence"],
            outputs=["final_answer"],
            params={"instruction": "produce a recommendation"},
        ),
    ]

    edges = [
        EdgeSpec.model_validate({"from": "START", "to": "gather_left"}),
        EdgeSpec.model_validate({"from": "START", "to": "gather_right"}),
        EdgeSpec.model_validate({"from": "gather_left", "to": "join"}),
        EdgeSpec.model_validate({"from": "gather_right", "to": "join"}),
        EdgeSpec.model_validate({"from": "join", "to": "synthesize"}),
        EdgeSpec.model_validate({"from": "synthesize", "to": "END"}),
    ]

    return GraphSpec(
        graph_id="demo-research-001",
        goal="compare two evidence sources and recommend one",
        rationale="parallel gather then concat then synthesize",
        budget=GraphBudget(max_nodes=8, max_llm_calls=6),
        nodes=nodes,
        edges=edges,
    )

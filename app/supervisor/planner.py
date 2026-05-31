"""Planner abstraction — the seam where an LLM-based planner will plug in.

For v1, the planner is a plain `Callable[[str], GraphSpec]`. `StaticPlanner`
returns a single pre-built spec regardless of prompt; useful for tests and
the hardcoded-spec demo. A real LLM planner (phase 5 of the MVP sequence)
will satisfy the same type by emitting a structured GraphSpec.
"""

from __future__ import annotations

from collections.abc import Callable

from app.models import GraphSpec

Planner = Callable[[str], GraphSpec]


class StaticPlanner:
    """Always returns the same spec. The prompt is ignored."""

    def __init__(self, spec: GraphSpec) -> None:
        self._spec = spec

    def __call__(self, prompt: str) -> GraphSpec:
        del prompt
        return self._spec

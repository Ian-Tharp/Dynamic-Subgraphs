from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge state maps written by separate nodes in the same graph step."""

    return {**left, **right}


def add_counters(
    left: dict[str, int] | None, right: dict[str, int] | None
) -> dict[str, int]:
    """Additively reduce per-run counters: sum int values by key.

    Unlike `merge_dicts` (last-writer-wins), this SUMS overlapping keys so
    concurrent increments survive a fan-out merge. When two worker branches in
    a `parallel_map` (or two parallel root nodes) each report
    `{"nodes_executed": 1}`, LangGraph reduces them to `{"nodes_executed": 2}`
    instead of dropping one. Keys present in only one side pass through.
    """

    merged: dict[str, int] = dict(left or {})
    for key, value in (right or {}).items():
        merged[key] = merged.get(key, 0) + value
    return merged


class DynamicRunState(TypedDict, total=False):
    """Generic runtime envelope for transient graphs (v1)."""

    inputs: dict[str, Any]
    values: Annotated[dict[str, Any], merge_dicts]
    artifacts: Annotated[dict[str, Any], merge_dicts]
    errors: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
    metadata: Annotated[dict[str, Any], merge_dicts]
    # Spend/depth ledger. Additively reduced (see `add_counters`) so increments
    # from concurrent branches sum instead of overwriting. Distinct field from
    # `metadata` precisely because metadata uses last-writer-wins semantics that
    # run_id/graph_depth rely on — counters must not.
    counters: Annotated[dict[str, int], add_counters]

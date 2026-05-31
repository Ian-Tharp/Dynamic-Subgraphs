from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge state maps written by separate nodes in the same graph step."""

    return {**left, **right}


class DynamicRunState(TypedDict, total=False):
    """Generic runtime envelope for transient graphs (v1)."""

    inputs: dict[str, Any]
    values: Annotated[dict[str, Any], merge_dicts]
    artifacts: Annotated[dict[str, Any], merge_dicts]
    errors: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
    metadata: Annotated[dict[str, Any], merge_dicts]

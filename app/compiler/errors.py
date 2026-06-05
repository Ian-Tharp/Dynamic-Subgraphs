from __future__ import annotations

from app.errors import DynamicSubgraphsError


class GraphCompilationError(DynamicSubgraphsError, ValueError):
    """Raised when a validated spec cannot be compiled by the current slice."""

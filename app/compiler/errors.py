from __future__ import annotations


class GraphCompilationError(ValueError):
    """Raised when a validated spec cannot be compiled by the current slice."""

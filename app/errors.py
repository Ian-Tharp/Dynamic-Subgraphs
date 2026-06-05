"""Common base for the Dynamic Subgraphs error hierarchy.

Every error the engine raises on its own behalf — registry validation, graph
compilation, planning, subgraph nesting — derives from `DynamicSubgraphsError`,
so a host can catch the whole family with one `except DynamicSubgraphsError`.
Specific errors additionally keep the closest stdlib base (e.g. `ValueError`,
`RuntimeError`) so existing `except ValueError`/`except RuntimeError` handling and
the semantic intent are preserved — the marker root is added, nothing is removed.
"""

from __future__ import annotations


class DynamicSubgraphsError(Exception):
    """Root of the Dynamic Subgraphs error hierarchy."""

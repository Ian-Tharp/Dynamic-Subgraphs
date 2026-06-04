from __future__ import annotations

import json
from typing import Any

from app.models import DynamicRunState


def make_initial_state(
    *,
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DynamicRunState:
    """Create the full runtime envelope expected by compiled transient graphs."""

    return {
        "inputs": dict(inputs or {}),
        "values": {},
        "artifacts": {},
        "errors": [],
        "events": [],
        "metadata": dict(metadata or {}),
        "counters": {},
    }


def render_value_for_prompt(value: Any) -> str:
    """Render an arbitrary state value compactly for inclusion in an LLM prompt.

    Strings pass through verbatim. Everything else is JSON-encoded with a
    `default=str` fallback. If JSON encoding itself fails, fall back to
    `repr()`. The output is suitable for embedding under a labeled bullet
    in a Human message (e.g., `- key: <rendered value>`).
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return repr(value)

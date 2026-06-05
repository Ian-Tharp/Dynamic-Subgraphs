"""Shared rendering for run-outcome response strings (run / resume / replay).

The supervisor's static graph and its resume/replay paths each turn a finished
`ExecutionResult` into a human-readable response. The per-path framing is
deliberately different (resume says "again", replay names the new run id), but the
two mechanical pieces — the paused-interrupt label and the produced-values summary
— were copied verbatim across three sites, so a change to either had to be made
three times. They live here so each is defined once.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# How many value keys to name in an ok-response preview before truncating.
_PREVIEW_LIMIT = 6


def render_interrupt_label(payloads: list[Any]) -> str:
    """Human label for what a paused run is waiting on (its interrupt payloads)."""
    event_types = [
        str(p.get("event_type")) if isinstance(p, dict) else str(p) for p in payloads
    ]
    return ", ".join(event_types) if event_types else "an unspecified event"


def render_value_summary(values: Mapping[str, Any]) -> str:
    """`"N output value(s): k1, k2, ..."` — the shared tail of an ok response."""
    preview_keys = sorted(values.keys())[:_PREVIEW_LIMIT]
    listed = ", ".join(preview_keys) if preview_keys else "(none)"
    return f"{len(values)} output value(s): {listed}"

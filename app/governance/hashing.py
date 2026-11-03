"""Canonical hashing for plans, policies, and traces.

Every hash a provenance receipt carries is ``sha256:`` over **canonical JSON**
(sorted keys, no whitespace, UTF-8), so the same logical object always hashes
the same regardless of dict ordering or how it was loaded from disk. Pure; no
I/O.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from app.models.graph_spec import GraphSpec
from app.models.trace import TraceEvent

HASH_PREFIX = "sha256:"


def canonical_json(value: Any) -> str:
    """Deterministic JSON text: sorted keys, compact separators, ``str`` fallback."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_hex(value: Any) -> str:
    """``sha256:<hex>`` of the canonical JSON form of ``value``."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return HASH_PREFIX + digest


def spec_payload(spec: GraphSpec) -> dict[str, Any]:
    """The hashed view of a spec: everything except ``metadata``.

    ``metadata`` (planner model name, ``created_at``) is descriptive, not
    behavioral — two plans that differ only there run identically.
    """
    data = spec.model_dump(mode="json", by_alias=True)
    data.pop("metadata", None)
    return data


def plan_hash(spec: GraphSpec) -> str:
    """Content hash of a GraphSpec (nodes, edges, params, budget, goal)."""
    return sha256_hex(spec_payload(spec))


def plan_hash_from_json(raw: Mapping[str, Any]) -> str:
    """Hash a spec loaded from disk (e.g. ``spec.json``) the same way."""
    return plan_hash(GraphSpec.model_validate(dict(raw)))


def trace_digest(events: Iterable[TraceEvent | Mapping[str, Any]]) -> str:
    """Digest of an ordered trace — the in-memory events or ``trace.jsonl`` rows.

    Both forms normalize to the JSON dict each event serializes to, so a digest
    taken before recording matches one recomputed from the recorder log.
    """
    rows = [
        e.model_dump(mode="json") if isinstance(e, TraceEvent) else dict(e)
        for e in events
    ]
    return sha256_hex(rows)

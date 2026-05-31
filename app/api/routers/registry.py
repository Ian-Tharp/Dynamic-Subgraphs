# app/api/routers/registry.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.registry import (
    DEFAULT_SUBAGENTS,
    DEFAULT_TOOLS,
    FORBIDDEN_KINDS,
    default_kind_definitions,
)

router = APIRouter(tags=["registry"])


@router.get("/registry")
def get_registry() -> dict[str, Any]:
    kinds: list[dict[str, Any]] = []
    for kind, definition in default_kind_definitions().items():
        kinds.append(
            {
                "kind": kind.value,
                "description": definition.description,
                "counts_as_llm_call": definition.counts_as_llm_call,
                "has_side_effects": definition.has_side_effects,
                "requires_tool_allowlist": definition.requires_tool_allowlist,
                "requires_subagent_allowlist": definition.requires_subagent_allowlist,
                "param_schema": definition.param_model.model_json_schema(),
            }
        )
    return {
        "node_kinds": kinds,
        "tools": sorted(DEFAULT_TOOLS),
        "subagents": sorted(DEFAULT_SUBAGENTS),
        "forbidden_kinds": sorted(FORBIDDEN_KINDS),
    }

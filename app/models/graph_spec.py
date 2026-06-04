from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.node_kinds import NodeKind

GRAPH_SPEC_SCHEMA_VERSION = 1
SPECIAL_NODE_START = "START"
SPECIAL_NODE_END = "END"


class GraphBudget(BaseModel):
    max_nodes: int = Field(default=12, ge=1)
    max_depth: int = Field(default=2, ge=1)
    max_wall_seconds: int = Field(default=90, ge=1)
    max_llm_calls: int = Field(default=8, ge=0)
    # Max items a single parallel_map node may fan out to. Enforced at dispatch
    # against the host ExecutionPolicy (effective = min(host, request)).
    max_fanout: int = Field(default=64, ge=1)


class NodeSpec(BaseModel):
    id: str
    kind: NodeKind
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class EdgeSpec(BaseModel):
    from_: str = Field(alias="from")
    to: str

    model_config = {"populate_by_name": True}


class GraphSpecMetadata(BaseModel):
    planner_model: str | None = None
    purpose: str | None = None
    created_at: datetime | None = None


class GraphSpec(BaseModel):
    schema_version: Literal[1] = GRAPH_SPEC_SCHEMA_VERSION
    graph_id: str
    goal: str
    rationale: str = ""
    budget: GraphBudget = Field(default_factory=GraphBudget)
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    metadata: GraphSpecMetadata = Field(default_factory=GraphSpecMetadata)

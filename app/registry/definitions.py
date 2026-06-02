from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from app.models.node_kinds import NodeKind


@dataclass(frozen=True)
class NodeKindDefinition:
    kind: NodeKind
    description: str
    param_model: type[BaseModel]
    counts_as_llm_call: bool = False
    has_side_effects: bool = False
    requires_tool_allowlist: bool = False
    requires_subagent_allowlist: bool = False


def default_kind_definitions() -> dict[NodeKind, NodeKindDefinition]:
    from app.registry.params import (
        BranchParams,
        EmitArtifactParams,
        LlmCallParams,
        ParallelMapParams,
        ReduceParams,
        SpawnSubagentParams,
        SpawnSubgraphParams,
        ToolCallParams,
        WaitForEventParams,
    )

    return {
        NodeKind.LLM_CALL: NodeKindDefinition(
            kind=NodeKind.LLM_CALL,
            description="Ask a model to transform or reason over inputs.",
            param_model=LlmCallParams,
            counts_as_llm_call=True,
        ),
        NodeKind.TOOL_CALL: NodeKindDefinition(
            kind=NodeKind.TOOL_CALL,
            description="Invoke an allowlisted tool.",
            param_model=ToolCallParams,
            has_side_effects=True,
            requires_tool_allowlist=True,
        ),
        NodeKind.SPAWN_SUBAGENT: NodeKindDefinition(
            kind=NodeKind.SPAWN_SUBAGENT,
            description="Delegate to a named specialist subagent.",
            param_model=SpawnSubagentParams,
            counts_as_llm_call=True,
            requires_subagent_allowlist=True,
        ),
        NodeKind.PARALLEL_MAP: NodeKindDefinition(
            kind=NodeKind.PARALLEL_MAP,
            description="Fan one child operation over many inputs.",
            param_model=ParallelMapParams,
        ),
        NodeKind.REDUCE: NodeKindDefinition(
            kind=NodeKind.REDUCE,
            description="Merge prior outputs into one value.",
            param_model=ReduceParams,
            # LLM cost depends on params.strategy; counted conditionally in Registry.count_llm_calls.
            counts_as_llm_call=False,
        ),
        NodeKind.BRANCH: NodeKindDefinition(
            kind=NodeKind.BRANCH,
            description="Choose among named exits (no arbitrary conditions).",
            param_model=BranchParams,
        ),
        NodeKind.WAIT_FOR_EVENT: NodeKindDefinition(
            kind=NodeKind.WAIT_FOR_EVENT,
            description="Suspend until durable external input.",
            param_model=WaitForEventParams,
        ),
        NodeKind.EMIT_ARTIFACT: NodeKindDefinition(
            kind=NodeKind.EMIT_ARTIFACT,
            description="Write a file, message, or report (explicit side effect).",
            param_model=EmitArtifactParams,
            has_side_effects=True,
        ),
        NodeKind.SPAWN_SUBGRAPH: NodeKindDefinition(
            kind=NodeKind.SPAWN_SUBGRAPH,
            description="Synthesize and run a bounded child graph for a sub-goal.",
            param_model=SpawnSubgraphParams,
            # Conservative floor: a spawned child runs at least one model call.
            # Precise child-spend roll-up arrives with budget clamping (slice 4).
            counts_as_llm_call=True,
        ),
    }

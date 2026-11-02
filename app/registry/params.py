from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.models.node_kinds import NodeKind

# Planner-chosen child names are embedded in the child run_id
# (`<parent>__sg_<name>`), which becomes a directory name, ledger key and
# checkpointer thread_id — keep them to the safe run-id charset.
_SAFE_NAME_PATTERN = r"^[A-Za-z0-9_-]+$"
_MAX_NAME_LENGTH = (
    32  # nested run_ids grow ~name+6 per level; keep Windows paths < MAX_PATH
)


class LlmCallParams(BaseModel):
    instruction: str = Field(min_length=1)
    model: str | None = None


class ToolCallParams(BaseModel):
    tool_name: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class SpawnSubagentParams(BaseModel):
    agent_name: str = Field(min_length=1)
    task: str = Field(min_length=1)


class ParallelMapParams(BaseModel):
    over: str = Field(min_length=1, description="State key to fan out over")
    child_kind: Literal["tool_call", "llm_call"]
    child_params: dict[str, Any] = Field(default_factory=dict)


class ReduceParams(BaseModel):
    strategy: Literal["concat", "merge_dict", "llm_summarize"] = "concat"
    input_keys: list[str] = Field(min_length=1)
    output_key: str = Field(min_length=1)
    instruction: str | None = None

    @model_validator(mode="after")
    def instruction_required_for_llm(self) -> ReduceParams:
        if self.strategy == "llm_summarize" and not self.instruction:
            raise ValueError("instruction is required when strategy is llm_summarize")
        return self


class BranchParams(BaseModel):
    branches: list[str] = Field(min_length=2)
    decision_key: str = Field(
        min_length=1,
        description="State key holding which named branch to take",
    )


class WaitForEventParams(BaseModel):
    event_type: str = Field(min_length=1)
    timeout_seconds: int | None = Field(default=None, ge=1)


class EmitArtifactParams(BaseModel):
    artifact_type: Literal["file", "message", "report"] = "report"
    name: str = Field(min_length=1)
    content_key: str = Field(min_length=1)


class SpawnSubgraphParams(BaseModel):
    """Synthesize and run a bounded child graph for a sub-goal.

    `sub_goal` is the natural-language goal the child planner expands into a
    fresh GraphSpec. `name` identifies the child run deterministically
    (child run_id = `<parent>__sg_<name>`). `inputs_from` lists parent state
    value keys to seed into the child's initial inputs. No embedded GraphSpec —
    the child is planned at runtime, so "plans, not code" holds at both levels.
    """

    sub_goal: str = Field(min_length=1)
    name: str = Field(pattern=_SAFE_NAME_PATTERN, max_length=_MAX_NAME_LENGTH)
    inputs_from: list[str] = Field(default_factory=list)


class GauntletParams(BaseModel):
    """Run a bounded builder/critic loop against a cached reference.

    `sub_goal` briefs the builder child graph (planned at runtime, one level
    deeper, like `spawn_subgraph`). `reference_key` names the parent state
    value holding the reference artifact to beat — it must be in the node's
    `inputs` so provenance is validated. Each round a fresh-context critic
    compares the builder's artifact with the reference blind and returns the
    winner plus one gap; the gap briefs the next round. `artifact_key` is the
    child value treated as the artifact (falls back to all child values).
    """

    sub_goal: str = Field(min_length=1)
    name: str = Field(pattern=_SAFE_NAME_PATTERN, max_length=_MAX_NAME_LENGTH)
    reference_key: str = Field(min_length=1)
    artifact_key: str = Field(default="artifact", min_length=1)
    max_rounds: int = Field(default=3, ge=1, le=5)
    inputs_from: list[str] = Field(default_factory=list)


PARAM_MODEL_BY_KIND: dict[NodeKind, type[BaseModel]] = {
    NodeKind.LLM_CALL: LlmCallParams,
    NodeKind.TOOL_CALL: ToolCallParams,
    NodeKind.SPAWN_SUBAGENT: SpawnSubagentParams,
    NodeKind.PARALLEL_MAP: ParallelMapParams,
    NodeKind.REDUCE: ReduceParams,
    NodeKind.BRANCH: BranchParams,
    NodeKind.WAIT_FOR_EVENT: WaitForEventParams,
    NodeKind.EMIT_ARTIFACT: EmitArtifactParams,
    NodeKind.SPAWN_SUBGRAPH: SpawnSubgraphParams,
    NodeKind.GAUNTLET: GauntletParams,
}

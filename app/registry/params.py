from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.models.node_kinds import NodeKind

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


PARAM_MODEL_BY_KIND: dict[NodeKind, type[BaseModel]] = {
    NodeKind.LLM_CALL: LlmCallParams,
    NodeKind.TOOL_CALL: ToolCallParams,
    NodeKind.SPAWN_SUBAGENT: SpawnSubagentParams,
    NodeKind.PARALLEL_MAP: ParallelMapParams,
    NodeKind.REDUCE: ReduceParams,
    NodeKind.BRANCH: BranchParams,
    NodeKind.WAIT_FOR_EVENT: WaitForEventParams,
    NodeKind.EMIT_ARTIFACT: EmitArtifactParams,
}

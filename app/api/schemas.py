# app/api/schemas.py
"""HTTP request/response contract models. Distinct from internal app/models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ExecutionMode = Literal["sync", "async", "auto"]
PlannerChoice = Literal["mock", "llm", "openai"]
ChainDeciderChoice = Literal["status", "llm"]

# Upper bound on the free-text prompt. It is the one large, attacker-controllable
# input that drives a paid LLM call and is persisted to disk, so it carries an
# explicit ceiling (generous — ~25k tokens) instead of relying on a downstream
# limit. `success_criteria` is already capped; this removes that asymmetry.
MAX_PROMPT_CHARS = 100_000


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    run_id: str | None = None
    mode: ExecutionMode = "auto"
    planner: PlannerChoice | None = None
    provider: str | None = None
    model: str | None = None


class ChainRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    run_id: str | None = None
    mode: ExecutionMode = "auto"
    planner: PlannerChoice | None = None
    provider: str | None = None
    model: str | None = None
    max_iterations: int = Field(default=3, ge=1, le=10)
    decider: ChainDeciderChoice = "status"
    success_criteria: str | None = Field(default=None, min_length=1, max_length=4000)
    judge_failed_runs: bool = False


class ResumeRequest(BaseModel):
    event: Any


class ReplayRequest(BaseModel):
    new_run_id: str | None = None


class RunAck(BaseModel):
    run_id: str
    status: str
    state: str
    links: dict[str, str]


class RunResult(BaseModel):
    run_id: str
    status: str
    response: str
    spec: dict[str, Any] | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    record_dir: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    links: dict[str, str] = Field(default_factory=dict)


class RunStatus(BaseModel):
    run_id: str
    state: str
    status: str | None = None
    response: str | None = None
    budget_wall_seconds: int | None = None
    on_disk: bool = False
    links: dict[str, str] = Field(default_factory=dict)


class RunListItem(BaseModel):
    run_id: str
    status: str
    nodes: int


class RunList(BaseModel):
    runs: list[RunListItem]
    count: int


class ErrorEnvelope(BaseModel):
    error: dict[str, Any]

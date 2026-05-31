from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TraceEventKind(StrEnum):
    NODE_START = "node_start"
    NODE_FINISH = "node_finish"
    NODE_ERROR = "node_error"
    PLAN_SELECTED = "plan_selected"
    SPEC_VALIDATED = "spec_validated"
    POLICY_DECISION = "policy_decision"
    EVAL_RESULT = "eval_result"


class TraceEvent(BaseModel):
    kind: TraceEventKind
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node_id: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class RunTrace(BaseModel):
    run_id: str
    graph_id: str
    events: list[TraceEvent] = Field(default_factory=list)

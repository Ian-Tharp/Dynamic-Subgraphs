"""SupervisorState — the outer durable graph's state envelope.

This is distinct from `DynamicRunState`, which is the *inner* transient graph's
state. The supervisor's state holds the artifacts that flow between supervisor
stages (prompt, spec, validated_spec, result, record, response, status).

Only `errors` uses a reducer (append). Every other key is written by exactly
one supervisor node, so LangGraph's default last-write-wins is fine.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.models import GraphSpec
from app.recording import RunRecord
from app.runtime import ExecutionResult


class SupervisorState(TypedDict, total=False):
    """State that flows through the supervisor StateGraph.

    `status` is a single string indicating the most recent outcome. Allowed:
        - "pending"              initial
        - "ok"                   full pipeline succeeded
        - "paused"               graph hit a wait_for_event and is awaiting resume
        - "plan_failed"          planner raised
        - "validation_failed"    GraphSpec rejected by validator
        - "compile_failed"       compiler rejected (e.g. unsupported kind)
        - "execution_failed"     runner exception inside a node
        - "record_failed"        result exists but couldn't be persisted
        - "resume_failed"        executor or recorder errored during resume
        - "replay_failed"        executor or recorder errored during replay
    """

    prompt: str
    run_id: str
    spec: GraphSpec | None
    validated_spec: GraphSpec | None
    result: ExecutionResult | None
    record: RunRecord | None
    response: str
    status: str
    errors: Annotated[list[dict[str, Any]], operator.add]

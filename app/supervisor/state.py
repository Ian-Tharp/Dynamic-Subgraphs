"""SupervisorState — the outer durable graph's state envelope.

This is distinct from `DynamicRunState`, which is the *inner* transient graph's
state. The supervisor's state holds the artifacts that flow between supervisor
stages (prompt, spec, validated_spec, result, record, response, status).

Only `errors` uses a reducer (append). Every other key is written by exactly
one supervisor node, so LangGraph's default last-write-wins is fine.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from app.models import GraphSpec
from app.recording import RunRecord
from app.runtime import ExecutionResult


class RunStatus(StrEnum):
    """Canonical supervisor run-status vocabulary — the single source of truth.

    `StrEnum` members ARE plain strings, so `status == "ok"`, set membership, and
    JSON serialization behave exactly as the bare literals did; this only gives the
    vocabulary one definition instead of scattering it across literals, named
    constants in `graph.py`, and a hardcoded set in `iteration.py`.
    """

    PENDING = "pending"
    OK = "ok"
    PAUSED = "paused"
    PLAN_FAILED = "plan_failed"
    VALIDATION_FAILED = "validation_failed"
    COMPILE_FAILED = "compile_failed"
    EXECUTION_FAILED = "execution_failed"
    RECORD_FAILED = "record_failed"
    RESUME_FAILED = "resume_failed"
    REPLAY_FAILED = "replay_failed"
    # Transient: routes straight back to `plan`, never a final status.
    PLAN_REPAIR_NEEDED = "plan_repair_needed"


# Statuses the LLM iteration judge defers on: the framework already classified the
# outcome, so it shouldn't spend tokens re-deciding. `EXECUTION_FAILED` is excluded
# on purpose — `judge_failed_runs` decides whether to judge those. Derived from the
# enum so the defer-set can't desync from the vocabulary.
DEFER_STATUSES: frozenset[RunStatus] = frozenset(
    {
        RunStatus.PAUSED,
        RunStatus.PLAN_FAILED,
        RunStatus.VALIDATION_FAILED,
        RunStatus.COMPILE_FAILED,
        RunStatus.RECORD_FAILED,
        RunStatus.RESUME_FAILED,
        RunStatus.REPLAY_FAILED,
    }
)


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

    Transient (never a final status — always routes straight back to `plan`):
        - "plan_repair_needed"   validation failed but a repair attempt remains
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
    # Plan-repair loop: how many times `plan` has run, and the (transient)
    # context a repair re-plan reads. Kept out of `errors` so a run that is
    # repaired and then succeeds doesn't carry a non-terminal failure.
    plan_attempts: int
    last_validation_issues: list[dict[str, Any]] | None
    last_rejected_spec: GraphSpec | None

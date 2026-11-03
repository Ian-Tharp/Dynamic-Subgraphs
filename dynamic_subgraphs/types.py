"""Closed-set vocabularies for the SDK — typed once, introspectable at runtime.

Each `Literal` is paired with a runtime tuple derived from it via
`typing.get_args`, so static type-checkers and `DynamicSubgraphs.capabilities()`
share one definition and can never drift. Coding agents should call
`capabilities()` (or read these tuples) instead of guessing option strings.
"""

from __future__ import annotations

from typing import Literal, get_args

from app.models import NodeKind
from dynamic_subgraphs.recording import Artifact

# Run outcome — mirrors app/supervisor/state.py (single-run statuses), plus
# "engine_failed": the run never reached the supervisor (e.g. a provider SDK
# raised while building the models — missing credentials, unknown provider).
RunStatus = Literal[
    "pending",
    "ok",
    "paused",
    "plan_failed",
    "validation_failed",
    "compile_failed",
    "execution_failed",
    "record_failed",
    "resume_failed",
    "replay_failed",
    "engine_failed",
    "unknown",
]
RUN_STATUSES: tuple[str, ...] = get_args(RunStatus)

# Planner mode accepted by the SDK facade.
Planner = Literal["llm", "mock"]
PLANNERS: tuple[str, ...] = get_args(Planner)

# How a provider coerces structured output. Local OpenAI-compatible servers
# (LM Studio, vLLM) usually need "json_schema"; hosted OpenAI prefers
# "function_calling" for GraphSpec's open params dict.
StructuredMethod = Literal["function_calling", "json_schema"]
STRUCTURED_METHODS: tuple[str, ...] = get_args(StructuredMethod)

# The per-role model slots a run can target (see ModelSelection).
MODEL_ROLES: tuple[str, ...] = (
    "planner",
    "worker",
    "reducer",
    "subagent",
    "judge",
)

# Selectable artifact identifiers (the on-disk filenames; "emitted" is the
# emit_artifact sink toggle).
ARTIFACT_KINDS: tuple[str, ...] = tuple(a.value for a in Artifact)

# The bounded vocabulary of node kinds a plan may use — sourced directly from
# the registry's `NodeKind` enum so the SDK surface (and `capabilities()`) can
# never drift from what the compiler actually accepts.
NODE_KINDS: tuple[str, ...] = tuple(k.value for k in NodeKind)

__all__ = [
    "ARTIFACT_KINDS",
    "MODEL_ROLES",
    "NODE_KINDS",
    "PLANNERS",
    "RUN_STATUSES",
    "STRUCTURED_METHODS",
    "Planner",
    "RunStatus",
    "StructuredMethod",
]

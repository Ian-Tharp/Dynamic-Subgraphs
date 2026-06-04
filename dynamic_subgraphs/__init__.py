"""Dynamic Subgraphs — public SDK.

A governed dynamic-graph runtime: plan, validate, and run transient LangGraph
workflows from a bounded node registry, with full run recording. This package
is the importable facade; the engine implementation currently lives under the
`app` namespace and is re-exported here.

    from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

    engine = DynamicSubgraphs(EngineConfig(model=Model("openai", "gpt-5.4-nano")))
    result = engine.run("Compare two sources and recommend one.")

Bring-your-own model/key/endpoint, including local servers:

    DynamicSubgraphs(EngineConfig(model=Model.lmstudio("openai/gpt-oss-20b")))
    DynamicSubgraphs(EngineConfig(model=Model.ollama("llama3.1")))
    DynamicSubgraphs(EngineConfig(model=Model.openai_compatible(
        "my-model", base_url="http://my-host/v1", api_key="...")))

Note: use a capable model for the planner role — small local models often
emit invalid plans. See `docs/recipes.md`.
"""

from __future__ import annotations

from app.policy import ExecutionPolicy
from app.runtime import ModelRef
from dynamic_subgraphs.engine import (
    DynamicSubgraphs,
    EngineConfig,
    Model,
    ModelSelection,
    RunResult,
    TokenUsage,
)
from dynamic_subgraphs.eval import (
    EvalContext,
    EvalGate,
    EvalReference,
    EvalResult,
    EvalTags,
    RunFingerprint,
    ScoreComponent,
)
from dynamic_subgraphs.recording import Artifact, Recording
from dynamic_subgraphs.types import (
    ARTIFACT_KINDS,
    MODEL_ROLES,
    PLANNERS,
    RUN_STATUSES,
    STRUCTURED_METHODS,
    Planner,
    RunStatus,
    StructuredMethod,
)

__all__ = [
    "ARTIFACT_KINDS",
    "MODEL_ROLES",
    "PLANNERS",
    "RUN_STATUSES",
    "STRUCTURED_METHODS",
    "Artifact",
    "DynamicSubgraphs",
    "EngineConfig",
    "EvalContext",
    "EvalGate",
    "EvalReference",
    "EvalResult",
    "EvalTags",
    "ExecutionPolicy",
    "Model",
    "ModelRef",
    "ModelSelection",
    "Planner",
    "Recording",
    "RunFingerprint",
    "RunResult",
    "RunStatus",
    "ScoreComponent",
    "StructuredMethod",
    "TokenUsage",
]
